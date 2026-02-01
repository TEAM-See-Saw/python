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
SERVO_RIGHT_MAX = 480
SERVO_LEFT_MAX  = 680

STEER_WAIT_TIME = 0.6  # 서보가 꺾일 시간(필수)

# ==========================================================
# [2] “즉시 판단” 임계값 (오른쪽=0~90 기준)
# ==========================================================
RIGHT_OCC_TH   = 850     # rs_right < 850  -> 오른쪽에 장애물(=CAR1)
RIGHT_EMPTY_TH = 1150    # rs_right > 1150 -> 오른쪽 빈공간(=GAP)
CAR2_RF_TH     = 950     # rf_front < 950  -> GAP 상태에서 CAR2(우전방 장애물)

# GAP 진입 직후 CAR2가 너무 빨리 잡히는 튐 방지(프레임 카운트 X, 시간 1개만)
ARM_TIME_SEC = 0.15      # 0.12~0.20 추천(속도 80 기준)

# ==========================================================
# [3] 주차 제어 (간단 P-control)
# ==========================================================
RS_TARGET = 550
KP_RS = 0.25

RR_MIN_SAFE = 350
RR_RELEASE  = 25

# SETUP 종료 조건(센서 기반): 오른쪽이 충분히 열렸다고 판단
SETUP_RS_TH = 1250       # RIGHT_EMPTY_TH보다 약간 크게(현장에 따라 1200~1400)
SETUP_MIN_TIME = 0.30    # 바로 튀지 않게 최소 시간

# ==========================================================
# [4] STOP (rear 90~270에서 "안 찍히면" 정지)
# ==========================================================
STOP_CLEAR_TH = 1800     # rear_all이 이보다 크면 "후방에 장애물 안 찍힘"으로 간주
REV_MIN_TIME  = 0.80     # 후진 시작 직후 바로 stop되는 걸 막기 위한 최소시간

# ==========================================================
# [5] EXIT safety
# ==========================================================
FRONT_SAFE_STOP = 350

# 출차 완료(대략) 조건: 전방이 충분히 열리고 오른쪽도 충분히 열림을 일정시간 유지
EXIT_OPEN_FRONT = 2000
EXIT_OPEN_RIGHT = 1500
EXIT_DONE_HOLD  = 0.7

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
    for _ in range(5):
        if ser.in_waiting <= 0:
            break
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data[:] = [int(p) for p in parts]
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
    if ser is None:
        return
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

# ==========================================================
# STATE
# ==========================================================
STATE_SEARCH = 0   # CAR1 찾기(오른쪽에 장애물 나타날 때까지)
STATE_GAP    = 1   # CAR1 본 이후에만, 오른쪽이 비면 GAP 진입 + CAR2 탐색
STATE_SETUP  = 2   # 좌대각 전진(각 만들기)
STATE_REVERSE= 3   # 후진 주차
STATE_WAIT   = 4   # 4초 정차
STATE_EXIT   = 5   # 출차(우회전)
STATE_DONE   = 6

def main():
    global ser, lidar, sonar_data

    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 성공")
        time.sleep(1.2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH

    car1_seen = False
    gap_t0 = None
    setup_t0 = None
    rev_t0 = None
    wait_t0 = None
    exit_open_t0 = None

    prev_t = time.time()

    try:
        scan_iter = lidar.iter_scans()

        while True:
            try:
                scan = next(scan_iter)
            except Exception as e:
                print(f"[WARN] LiDAR scan error: {e}")
                send_cmd(SERVO_CENTER, 0)
                break

            now = time.time()
            dt = now - prev_t
            prev_t = now

            read_sensors()

            # ------------------------------------------------------
            # 오른쪽=0~90 기준 섹터 재정의
            # ------------------------------------------------------
            # 오른쪽 측면(탐색/갭 판단): 넓게 잡아 안정화
            rs_right = get_lidar_dist_in_sector(scan, 65, 90, percentile=30)
            # 우전방(CAR2 감지)
            rf_front = get_lidar_dist_in_sector(scan, 15, 40, percentile=30)

            # 주차 제어용 섹터(조금 넓게)
            rf = get_lidar_dist_in_sector(scan, 15, 60, percentile=30)
            rs = get_lidar_dist_in_sector(scan, 60, 90, percentile=30)
            rr = get_lidar_dist_in_sector(scan, 90, 130, percentile=30)

            # 후방 전체(요구: 90~270)
            rear_all = get_lidar_dist_in_sector(scan, 90, 270, percentile=20)

            # 전방 중앙(랩어라운드)
            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=20)

            # sonar 안전(옵션)
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]
            if not valid_us(dist_LT): dist_LT = 9999
            if not valid_us(dist_RT): dist_RT = 9999

            # invalid 방어
            rs_ok = (rs_right < 9000)
            rf_ok = (rf_front < 9000)

            cmd_servo = SERVO_CENTER
            cmd_speed = 0
            msg = ""

            # ======================================================
            # [S0] SEARCH: CAR1을 먼저 봐야 함
            # ======================================================
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = "SEARCH: find CAR1"

                # CAR1 인식: 오른쪽이 가까워짐
                if (not car1_seen) and rs_ok and (rs_right < RIGHT_OCC_TH):
                    car1_seen = True
                    msg = "SEARCH: CAR1 seen -> keep going"
                    # car1_seen만 켜고, GAP 판단은 다음부터

                # CAR1을 본 이후에만 GAP 판단으로 넘어감
                if car1_seen and rs_ok and (rs_right > RIGHT_EMPTY_TH):
                    state = STATE_GAP
                    gap_t0 = time.time()
                    msg = "SEARCH -> GAP"

            # ======================================================
            # [S1] GAP: 오른쪽이 비어있는 상태에서 CAR2(우전방) 찾기
            # ======================================================
            elif state == STATE_GAP:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = "GAP: searching CAR2"

                # 만약 다시 오른쪽이 막히면(갭이 아니면) SEARCH로 복귀 (현장 튐 방지)
                if rs_ok and (rs_right < RIGHT_OCC_TH):
                    state = STATE_SEARCH
                    msg = "GAP -> SEARCH (right occupied again)"
                else:
                    # ARM_TIME 이후에만 CAR2 판정
                    if (gap_t0 is not None) and ((time.time() - gap_t0) >= ARM_TIME_SEC):
                        if rf_ok and (rf_front < CAR2_RF_TH):
                            msg = "CAR2 detected -> STOP & SETUP"
                            smooth_brake(SPEED_SEARCH)
                            send_cmd(SERVO_CENTER, 0)
                            time.sleep(0.15)

                            state = STATE_SETUP
                            setup_t0 = time.time()

            # ======================================================
            # [S2] SETUP: 좌대각 전진으로 각 만들기(센서 기반 종료)
            # ======================================================
            elif state == STATE_SETUP:
                cmd_servo = SERVO_LEFT_MAX

                # 서보 기다리기
                if (setup_t0 is not None) and ((time.time() - setup_t0) < STEER_WAIT_TIME):
                    cmd_speed = 0
                    msg = "SETUP: steer left wait"
                else:
                    cmd_speed = SPEED_SETUP
                    msg = "SETUP: forward-left"

                # 센서 기반 종료: 오른쪽이 충분히 열림 + 최소시간 만족
                if (setup_t0 is not None) and ((time.time() - setup_t0) >= (STEER_WAIT_TIME + SETUP_MIN_TIME)):
                    if rs > SETUP_RS_TH:
                        send_cmd(SERVO_CENTER, 0)
                        time.sleep(0.12)
                        state = STATE_REVERSE
                        rev_t0 = time.time()
                        msg = "SETUP -> REVERSE"

            # ======================================================
            # [S3] REVERSE: P-control + rear_all clear stop
            # ======================================================
            elif state == STATE_REVERSE:
                msg = "REVERSE: parking"

                # P-control로 rs_target 유지하며 후진
                err = RS_TARGET - rs
                servo = SERVO_CENTER - (KP_RS * err)

                if rr < RR_MIN_SAFE:
                    servo += RR_RELEASE

                cmd_servo = int(clamp(servo, SERVO_RIGHT_MAX, SERVO_LEFT_MAX))
                cmd_speed = -SPEED_PARK

                # rear_all "안 찍히면" stop (단, 후진 시작 직후 즉시 stop 방지)
                if (rev_t0 is not None) and ((time.time() - rev_t0) >= REV_MIN_TIME):
                    if rear_all > STOP_CLEAR_TH:
                        send_cmd(SERVO_CENTER, 0)
                        state = STATE_WAIT
                        wait_t0 = time.time()
                        msg = "REVERSE -> WAIT (rear clear)"

            # ======================================================
            # [S4] WAIT: 4초 정차
            # ======================================================
            elif state == STATE_WAIT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = "WAIT: 4s"

                if (wait_t0 is not None) and ((time.time() - wait_t0) >= 4.0):
                    state = STATE_EXIT
                    exit_open_t0 = None
                    msg = "WAIT -> EXIT"

            # ======================================================
            # [S5] EXIT: 전진 우회전(진입 방향과 반대로)
            # ======================================================
            elif state == STATE_EXIT:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = SPEED_EXIT
                msg = "EXIT: forward right"

                # 전방 안전
                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT: SAFETY STOP (front)"

                # 출차 완료 판정(시간 기반 hold)
                if (front_center > EXIT_OPEN_FRONT) and (rs_right > EXIT_OPEN_RIGHT):
                    if exit_open_t0 is None:
                        exit_open_t0 = time.time()
                    elif (time.time() - exit_open_t0) >= EXIT_DONE_HOLD:
                        state = STATE_DONE
                else:
                    exit_open_t0 = None

            # ======================================================
            # [DONE]
            # ======================================================
            elif state == STATE_DONE:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = "DONE"
                send_cmd(cmd_servo, cmd_speed)
                break

            # (옵션) 초음파 안전: 너무 가까우면 정지
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                msg = "SAFETY STOP (US)"

            send_cmd(cmd_servo, cmd_speed)

            # debug view
            sub = (
                f"dt:{dt*1000:.0f}ms | state:{state} car1:{car1_seen} | "
                f"rsR:{int(rs_right)} rfF:{int(rf_front)} | "
                f"rf:{int(rf)} rs:{int(rs)} rr:{int(rr)} | "
                f"rearAll:{int(rear_all)} front:{int(front_center)} | "
                f"servo:{cmd_servo} spd:{cmd_speed}"
            )

            img = np.zeros((380, 1120, 3), dtype=np.uint8)
            cv2.putText(img, msg, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(img, sub, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
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