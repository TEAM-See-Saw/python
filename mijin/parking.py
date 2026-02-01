import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================================
# [0] PORT / BAUD
# ==========================================================
PORT = "COM4"
LIDAR_PORT = "COM3"
BAUD = 115200

# ==========================================================
# [1] SPEED / SERVO
# ==========================================================
SPEED_SEARCH = 80
SPEED_SETUP  = 80
SPEED_PARK   = 75
SPEED_EXIT   = 80

SERVO_CENTER    = 570
SERVO_RIGHT_MAX = 480   # ✅ 너가 말한 최신 최대치(오른쪽)
SERVO_LEFT_MAX  = 680

STEER_WAIT_TIME = 0.6

# ==========================================================
# [2] “즉시 판단” 임계값 (오른쪽=0~90 기준)
# ==========================================================
# 오른쪽에 장애물이 "있다"
RIGHT_OCC_TH = 850    # rs_right < 850mm -> 옆에 차/장애물 있음
# 오른쪽이 "비었다"(빈 공간)
RIGHT_EMPTY_TH = 1150 # rs_right > 1150mm -> 옆이 비었음(GAP)

# GAP(오른쪽 비었을 때)에는 우전방에서 CAR2를 봐야 한다
CAR2_RF_TH = 950      # rf_front < 950mm -> 우전방에 두번째 장애물(앞차) 있다고 간주

# ==========================================================
# [3] 주차 제어 (간단 P-control)
# ==========================================================
RS_TARGET = 550
KP_RS = 0.25

RR_MIN_SAFE = 350
RR_RELEASE  = 25

# ==========================================================
# [4] STOP (요구 반영: rear 90~270에서 "안 찍히면" 정지)
#  - rear_all 값이 매우 크면(장애물 거의 없음) -> 정지
# ==========================================================
STOP_CLEAR_TH = 1800

# ==========================================================
# [5] EXIT safety
# ==========================================================
FRONT_SAFE_STOP = 350

# ==========================================================
# [6] SONAR (optional safety)
# ==========================================================
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
US_VALID_MIN = 50
US_VALID_MAX = 5000

def valid_us(d):
    return (d is not None) and (US_VALID_MIN <= d <= US_VALID_MAX)

# ==========================================================
# LiDAR helper
# ==========================================================
def get_lidar_dist_in_sector(scan, ang_min, ang_max, invalid=9999, percentile=30):
    """
    percentile:
      - 너무 낮으면(10~20) '가장 가까운 점' 위주라 튐
      - 너무 높으면(70~90) '먼 점' 위주라 빈공간 판단이 둔해짐
    실사용에서는 25~40 구간이 안정적이라 기본 30으로 둠.
    """
    dists = []
    for m in scan:
        if len(m) != 3:
            continue
        _, angle, dist = m
        if dist <= 0:
            continue

        if ang_min <= ang_max:
            in_range = (ang_min <= angle <= ang_max)
        else:
            in_range = (angle >= ang_min or angle <= ang_max)

        if in_range:
            dists.append(dist)

    if not dists:
        return invalid
    return float(np.percentile(dists, percentile))

# ==========================================================
# SERIAL helpers
# ==========================================================
ser = None
lidar = None

def read_sensors():
    global sonar_data, ser
    if ser is None:
        return
    # 최신값으로 갱신
    for _ in range(5):
        if ser.in_waiting <= 0:
            break
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass

def send_cmd(servo, speed):
    global ser
    try:
        ser.write(f"S,{int(servo)}\n".encode())
        ser.write(f"D,{int(speed)}\n".encode())
    except:
        pass

def smooth_brake(from_speed, steps=6, dt=0.05):
    global ser
    if ser is None:
        return
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

# ==========================================================
# STATE (프레임 카운트 없음)
# ==========================================================
STATE_SEARCH = 0       # CAR1 옆을 따라 주행
STATE_GAP    = 1       # 오른쪽이 빈공간(GAP)인 상태
STATE_SETUP  = 2       # 주차 각 만들기(좌대각 전진)
STATE_REVERSE= 3       # 후진 주차
STATE_WAIT   = 4       # 4초 정차
STATE_EXIT   = 5       # 출차(우회전)
STATE_DONE   = 6

def main():
    global ser, lidar, sonar_data

    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 성공")
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    setup_t0 = None
    wait_t0 = None

    print("🚀 START")

    try:
        scan_iter = lidar.iter_scans()

        while True:
            # --- scan ---
            try:
                scan = next(scan_iter)
            except Exception as e:
                print(f"[WARN] LiDAR scan error: {e}")
                send_cmd(SERVO_CENTER, 0)
                break

            read_sensors()

            # ======================================================
            # 오른쪽=0~90 기준 (너 말 반영)
            # ======================================================
            # ✅ 오른쪽 측면(빈공간/장애물 판단) : 넓게 잡아서 안 찍히는 문제 줄임
            rs_right = get_lidar_dist_in_sector(scan, 40, 90, percentile=30)

            # ✅ GAP일 때 CAR2 탐지용(우전방) : 0~45로 넓힘
            rf_front = get_lidar_dist_in_sector(scan, 0, 45, percentile=30)

            # 주차 제어용
            rf = get_lidar_dist_in_sector(scan, 0, 60, percentile=30)
            rs = get_lidar_dist_in_sector(scan, 60, 90, percentile=30)
            rr = get_lidar_dist_in_sector(scan, 90, 130, percentile=30)

            # 후방(요구: 90~270)
            rear_all = get_lidar_dist_in_sector(scan, 90, 270, percentile=20)

            # 전방 중앙(랩어라운드)
            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=20)

            # sonar safety(옵션)
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]
            if not valid_us(dist_LT): dist_LT = 9999
            if not valid_us(dist_RT): dist_RT = 9999

            # ======================================================
            # 즉시 판단 플래그 (프레임 카운트 없음)
            # ======================================================
            right_has_obs = (rs_right < RIGHT_OCC_TH)
            right_empty   = (rs_right > RIGHT_EMPTY_TH)
            car2_seen     = (rf_front < CAR2_RF_TH)

            # 기본 명령
            cmd_servo = SERVO_CENTER
            cmd_speed = 0
            msg = ""

            # ✅ 터미널에서 상태가 안 바뀌는지 바로 확인하려고 매번 찍음
            print(f"STATE:{state} | rs_right:{rs_right:.0f} rf_front:{rf_front:.0f} rear_all:{rear_all:.0f} front:{front_center:.0f}")

            # ======================================================
            # [S0] SEARCH: 오른쪽에 장애물이 있는 동안은 계속 직진
            # ======================================================
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = "SEARCH (FOLLOW CAR1)"

                # ✅ 오른쪽이 비는 순간 = GAP
                if right_empty:
                    state = STATE_GAP

            # ======================================================
            # [S1] GAP: 오른쪽이 비어 있는 동안, 우전방(CAR2) 보이면 정지 -> SETUP
            # ======================================================
            elif state == STATE_GAP:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = "GAP (RIGHT EMPTY)"

                # 만약 다시 오른쪽에 장애물이 생기면(측정 흔들림 포함) SEARCH로 복귀
                if right_has_obs:
                    state = STATE_SEARCH

                # ✅ 오른쪽이 빈 상태에서 CAR2가 보이면 = 슬롯 끝 도달 -> 주차 시작
                if right_empty and car2_seen:
                    smooth_brake(SPEED_SEARCH)
                    send_cmd(SERVO_CENTER, 0)
                    time.sleep(0.15)

                    state = STATE_SETUP
                    setup_t0 = time.time()

            # ======================================================
            # [S2] SETUP: 좌대각 전진으로 각도 만들기 (시간만 사용, 짧게)
            # ======================================================
            elif state == STATE_SETUP:
                msg = "SETUP (FORWARD-LEFT)"
                cmd_servo = SERVO_LEFT_MAX

                if time.time() - setup_t0 < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "SETUP (STEER WAIT)"
                else:
                    cmd_speed = SPEED_SETUP

                # ✅ 오른쪽이 충분히 열리면(빈공간) 바로 후진 주차로
                # (프레임 카운트 없이 즉시)
                if right_empty:
                    send_cmd(SERVO_CENTER, 0)
                    time.sleep(0.1)
                    state = STATE_REVERSE

            # ======================================================
            # [S3] REVERSE: 후진 주차 (rs_target P-control) + rear_all “안찍히면 stop”
            # ======================================================
            elif state == STATE_REVERSE:
                msg = "REVERSE PARK"

                # (1) 후방이 “안 찍힘”이면 정지
                if rear_all > STOP_CLEAR_TH:
                    send_cmd(SERVO_CENTER, 0)
                    state = STATE_WAIT
                    wait_t0 = time.time()
                else:
                    # (2) 우측 간격 유지 P-control
                    err = RS_TARGET - rs
                    servo = SERVO_CENTER - (KP_RS * err)

                    if rr < RR_MIN_SAFE:
                        servo += RR_RELEASE

                    cmd_servo = int(clamp(servo, SERVO_RIGHT_MAX, SERVO_LEFT_MAX))
                    cmd_speed = -SPEED_PARK

            # ======================================================
            # [S4] WAIT: 4초 정차
            # ======================================================
            elif state == STATE_WAIT:
                msg = "WAIT 4s"
                cmd_servo = SERVO_CENTER
                cmd_speed = 0

                if time.time() - wait_t0 >= 4.0:
                    state = STATE_EXIT

            # ======================================================
            # [S5] EXIT: 전진 우회전(진입방향과 반대로 나가기)
            # ======================================================
            elif state == STATE_EXIT:
                msg = "EXIT (FORWARD RIGHT)"
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = SPEED_EXIT

                # 전방 안전
                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP"

                # 종료 조건(간단): 전방이 충분히 열리면 DONE
                if front_center > 2000 and rs_right > 1500:
                    state = STATE_DONE

            elif state == STATE_DONE:
                msg = "DONE"
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                send_cmd(SERVO_CENTER, 0)
                break

            # sonar safety(옵션)
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                msg = "SAFETY STOP (US)"

            send_cmd(cmd_servo, cmd_speed)

            # debug view
            img = np.zeros((360, 1100, 3), dtype=np.uint8)
            cv2.putText(img, f"STATE:{state}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(img, msg, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, f"rs_right:{int(rs_right)}  rf_front:{int(rf_front)}  rear_all:{int(rear_all)}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
            cv2.putText(img, f"right_empty:{right_empty} car2_seen:{car2_seen} right_has_obs:{right_has_obs}",
                        (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
            cv2.putText(img, f"cmd_servo:{cmd_servo} cmd_speed:{cmd_speed}",
                        (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)

            cv2.imshow("Parking Monitor", img)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        try:
            if ser:
                send_cmd(SERVO_CENTER, 0)
                ser.close()
        except:
            pass
        try:
            if lidar:
                lidar.stop()
                lidar.disconnect()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()