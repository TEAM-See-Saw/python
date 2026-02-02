import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================================
# [0] PORT / BAUD
# ==========================================================
PORT = "COM4"       # 아두이노 포트
LIDAR_PORT = "COM3" # 라이다 포트
BAUD = 115200

# ==========================================================
# [1] SPEED / SERVO
# ==========================================================
SPEED_SEARCH = 65
SPEED_SETUP = 60
SPEED_PARK = 65
SPEED_EXIT = 65

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480
SERVO_LEFT_MAX = 680

STEER_WAIT_TIME = 0.5

# ==========================================================
# [2] 판단 임계값 (THRESHOLD)
# ==========================================================
# [보완] CAR1을 더 잘 잡기 위해 OCC 임계값을 약간 완화(현장에 맞게 조정 가능)
RIGHT_OCC_TH = 1100     # 이보다 작으면 장애물(차/벽)로 간주
RIGHT_EMPTY_TH = 1150   # 이보다 크면 빈공간(GAP)로 간주

CAR2_RF_TH = 950        # GAP 주행 중 전방(우전방) CAR2 감지 거리

# [Skipping 방지]
GAP_MIN_TIME = 0.8
SETUP_MIN_TIME = 0.6

# ==========================================================
# [2-1] 디바운싱(연속 프레임 확정)
# ==========================================================
CAR_OCC_N = 3       # CAR1 확정: 연속 3프레임 장애물
CAR_EMPTY_N = 4     # GAP 확정: 연속 4프레임 빈공간

# ==========================================================
# [3] 주차 제어
# ==========================================================
RS_TARGET = 550
KP_RS = 0.25
RR_MIN_SAFE = 350
RR_RELEASE = 25
SETUP_RS_TH = 1250

# ==========================================================
# [4] STOP / EXIT
# ==========================================================
STOP_CLEAR_TH = 1800
REV_MIN_TIME = 0.8
FRONT_SAFE_STOP = 350
EXIT_OPEN_FRONT = 2000
EXIT_OPEN_RIGHT = 1500
EXIT_DONE_HOLD = 0.7

# ==========================================================
# [5] SONAR
# ==========================================================
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
US_VALID_MIN = 50
US_VALID_MAX = 5000

def valid_us(d):
    return (d is not None) and (US_VALID_MIN <= d <= US_VALID_MAX)

# ==========================================================
# LiDAR Helper
# ==========================================================
def get_lidar_dist_in_sector(scan, ang_min, ang_max, invalid=9999, percentile=20):
    """
    percentile를 20으로 기본 설정(하위 분위수) -> 장애물 '놓침' 줄이는 방향
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

def flush_lidar(iterator, count=5):
    """상태 전환 시 구형 데이터 삭제"""
    try:
        for _ in range(count):
            next(iterator)
    except:
        pass

# ==========================================================
# SERIAL Helper
# ==========================================================
ser = None
lidar = None

def read_sensors():
    global sonar_data, ser
    if ser is None:
        return
    for _ in range(10):
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

def smooth_brake(from_speed, steps=5, dt=0.05):
    if ser is None:
        return
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)

def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))

# ==========================================================
# MAIN
# ==========================================================
STATE_NAMES = ["SEARCH", "GAP", "SETUP", "REVERSE", "WAIT", "EXIT", "DONE"]
STATE_SEARCH = 0
STATE_GAP = 1
STATE_SETUP = 2
STATE_REVERSE = 3
STATE_WAIT = 4
STATE_EXIT = 5
STATE_DONE = 6

def main():
    global ser, lidar, sonar_data

    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 성공")
        time.sleep(1.0)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    car1_seen = False

    # [보완] 디바운싱 카운터
    car_occ_cnt = 0
    car_empty_cnt = 0

    gap_t0 = None
    setup_t0 = None
    rev_t0 = None
    wait_t0 = None
    exit_open_t0 = None

    scan_iter = lidar.iter_scans()

    print("\n========== STARTING PARKING LOGIC ==========\n")

    try:
        while True:
            try:
                scan = next(scan_iter)
            except Exception:
                scan_iter = lidar.iter_scans()
                continue

            read_sensors()

            # ------------------------------------------------------
            # [센서 값 처리]
            # ------------------------------------------------------
            # 오른쪽은 설치각/차체간섭 때문에 한 섹터만 쓰면 놓칠 수 있어
            # -> 3섹터(40~60, 60~90, 90~120) 중 가장 가까운 값(min) 사용
            rs_r1 = get_lidar_dist_in_sector(scan, 40, 60, percentile=20)
            rs_r2 = get_lidar_dist_in_sector(scan, 60, 90, percentile=20)
            rs_r3 = get_lidar_dist_in_sector(scan, 90, 120, percentile=20)
            rs_right = min(rs_r1, rs_r2, rs_r3)

            rf_front = get_lidar_dist_in_sector(scan, 15, 40, percentile=20)

            rf = get_lidar_dist_in_sector(scan, 15, 60, percentile=20)
            rs = get_lidar_dist_in_sector(scan, 60, 90, percentile=20)
            rr = get_lidar_dist_in_sector(scan, 90, 130, percentile=20)
            rear_all = get_lidar_dist_in_sector(scan, 90, 270, percentile=20)
            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=20)

            dist_LT = sonar_data[IDX_LT] if valid_us(sonar_data[IDX_LT]) else 9999
            dist_RT = sonar_data[IDX_RT] if valid_us(sonar_data[IDX_RT]) else 9999

            rs_ok = (rs_right < 8000)
            rf_ok = (rf_front < 8000)

            cmd_servo = SERVO_CENTER
            cmd_speed = 0
            msg = ""

            # ======================================================
            # [S0] SEARCH
            # ======================================================
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                is_occ = rs_ok and (rs_right < RIGHT_OCC_TH)
                is_empty = rs_ok and (rs_right > RIGHT_EMPTY_TH)

                if not car1_seen:
                    # CAR1 확정: 연속 N프레임 장애물
                    car_occ_cnt = car_occ_cnt + 1 if is_occ else 0
                    if car_occ_cnt >= CAR_OCC_N:
                        car1_seen = True
                        msg = f"CAR1 CONFIRMED! (cnt={car_occ_cnt})"
                        car_empty_cnt = 0
                else:
                    # GAP 확정: 연속 N프레임 빈공간
                    car_empty_cnt = car_empty_cnt + 1 if is_empty else 0
                    if car_empty_cnt >= CAR_EMPTY_N:
                        msg = f">> GAP CONFIRMED! (cnt={car_empty_cnt})"
                        state = STATE_GAP
                        gap_t0 = time.time()
                        flush_lidar(scan_iter)

                        # 카운터 리셋
                        car_occ_cnt = 0
                        car_empty_cnt = 0

            # ======================================================
            # [S1] GAP (CAR2 탐색)
            # ======================================================
            elif state == STATE_GAP:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                elapsed = time.time() - gap_t0
                if elapsed < GAP_MIN_TIME:
                    msg = f"GAP Force Move ({elapsed:.1f}s)"
                else:
                    msg = "Searching CAR2..."
                    # 오른쪽이 다시 막히면(빈공간 아님) SEARCH로 복귀
                    if rs_ok and (rs_right < RIGHT_OCC_TH):
                        state = STATE_SEARCH
                        msg = "<< Reset to SEARCH"
                        # 복귀 시 카운터 초기화
                        car_occ_cnt = 0
                        car_empty_cnt = 0
                    elif rf_ok and (rf_front < CAR2_RF_TH):
                        msg = ">> CAR2 DETECTED! STOP."
                        smooth_brake(SPEED_SEARCH)
                        time.sleep(0.4)
                        flush_lidar(scan_iter)
                        state = STATE_SETUP
                        setup_t0 = time.time()

            # ======================================================
            # [S2] SETUP
            # ======================================================
            elif state == STATE_SETUP:
                cmd_servo = SERVO_LEFT_MAX
                elapsed = time.time() - setup_t0

                if elapsed < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "Steer Wait"
                elif elapsed < (STEER_WAIT_TIME + SETUP_MIN_TIME):
                    cmd_speed = SPEED_SETUP
                    msg = "Force Forward"
                else:
                    cmd_speed = SPEED_SETUP
                    msg = "Check Right Side..."
                    if rs > SETUP_RS_TH:
                        send_cmd(SERVO_CENTER, 0)
                        time.sleep(0.2)
                        flush_lidar(scan_iter)
                        state = STATE_REVERSE
                        rev_t0 = time.time()
                        msg = ">> REVERSE START"

            # ======================================================
            # [S3] REVERSE
            # ======================================================
            elif state == STATE_REVERSE:
                msg = "Parking..."
                err = RS_TARGET - rs
                servo = SERVO_CENTER - (KP_RS * err)
                if rr < RR_MIN_SAFE:
                    servo += RR_RELEASE

                cmd_servo = int(clamp(servo, SERVO_RIGHT_MAX, SERVO_LEFT_MAX))
                cmd_speed = -SPEED_PARK

                if (time.time() - rev_t0) >= REV_MIN_TIME:
                    if rear_all > STOP_CLEAR_TH:
                        send_cmd(SERVO_CENTER, 0)
                        state = STATE_WAIT
                        wait_t0 = time.time()
                        msg = ">> PARKING DONE (Wait)"

            # ======================================================
            # [S4] WAIT
            # ======================================================
            elif state == STATE_WAIT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                elapsed = time.time() - wait_t0
                msg = f"Waiting... {4.0 - elapsed:.1f}s"

                if elapsed >= 4.0:
                    state = STATE_EXIT
                    exit_open_t0 = None
                    flush_lidar(scan_iter)
                    msg = ">> EXIT START"

            # ======================================================
            # [S5] EXIT
            # ======================================================
            elif state == STATE_EXIT:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = SPEED_EXIT
                msg = "Exiting..."

                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP!"

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
                msg = "ALL DONE"
                send_cmd(cmd_servo, cmd_speed)

            # ------------------------------------------------------
            # [터미널 출력]
            # ------------------------------------------------------
            st_name = STATE_NAMES[state]
            print(
                f"[{st_name}] rsR:{int(rs_right)} (r1:{int(rs_r1)} r2:{int(rs_r2)} r3:{int(rs_r3)}) "
                f"rfF:{int(rf_front)} | RF:{int(rf)} RS:{int(rs)} RR:{int(rr)} | Rear:{int(rear_all)} | {msg}"
            )

            # 안전 정지(초음파)
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                print("!!! SONAR EMERGENCY STOP !!!")

            send_cmd(cmd_servo, cmd_speed)

            # OpenCV 표시
            img = np.zeros((320, 1050, 3), dtype=np.uint8)
            cv2.putText(img, f"State: {st_name}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(img, msg, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
            cv2.putText(img, f"rsR:{int(rs_right)} rfF:{int(rf_front)} Rear:{int(rear_all)}",
                        (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(img, f"Right sectors: r1(40-60)={int(rs_r1)} r2(60-90)={int(rs_r2)} r3(90-120)={int(rs_r3)}",
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(img, f"OCC<{RIGHT_OCC_TH} EMPTY>{RIGHT_EMPTY_TH} | occ_cnt={car_occ_cnt} empty_cnt={car_empty_cnt}",
                        (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Parking Monitor", img)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        send_cmd(SERVO_CENTER, 0)
        if ser:
            ser.close()
        if lidar:
            lidar.stop()
            lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()