import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================================
# [0] PORT / BAUD
# ==========================================================
PORT = "COM4"  # 아두이노 포트
LIDAR_PORT = "COM3"  # 라이다 포트
BAUD = 115200

# ==========================================================
# [1] SPEED / SERVO
# ==========================================================
# 속도를 조금 낮춰서(60~65) 센서 처리 시간을 확보함
SPEED_SEARCH = 65
SPEED_SETUP = 60
SPEED_PARK = 65
SPEED_EXIT = 65

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480
SERVO_LEFT_MAX = 680

STEER_WAIT_TIME = 0.5  # 서보 꺾고 대기하는 시간

# ==========================================================
# [2] 판단 임계값 (THRESHOLD)
# ==========================================================
# [핵심 수정] GAP 인식을 더 빠르게!
# 오른쪽 장애물이 850보다 작으면 CAR1
# 오른쪽 거리가 1000보다 크면 GAP (기존 1150에서 1000으로 낮춤 -> 더 빨리 반응)
RIGHT_OCC_TH = 850
RIGHT_EMPTY_TH = 1000

CAR2_RF_TH = 950  # GAP 주행 중 이 거리 안으로 들어오면 정지

# [핵심 수정] 최소 강제 주행 시간 (Skipping 방지)
GAP_MIN_TIME = 0.8  # GAP 진입 후 0.8초간은 무조건 직진 (CAR2 탐색 금지)
SETUP_MIN_TIME = 0.6  # SETUP 회전 후 0.6초간은 무조건 전진 (센서 무시)

# ==========================================================
# [3] 주차 제어 (P-control)
# ==========================================================
RS_TARGET = 550
KP_RS = 0.25

RR_MIN_SAFE = 350
RR_RELEASE = 25

# SETUP 종료 조건: 오른쪽이 뻥 뚫리면 정지
SETUP_RS_TH = 1250

# ==========================================================
# [4] STOP / EXIT
# ==========================================================
STOP_CLEAR_TH = 1800  # 후방에 아무것도 없으면(1.8m 이상) 주차 완료로 판단
REV_MIN_TIME = 0.8  # 후진 시작 후 최소 0.8초는 정지 판단 금지

FRONT_SAFE_STOP = 350
EXIT_OPEN_FRONT = 2000
EXIT_OPEN_RIGHT = 1500
EXIT_DONE_HOLD = 0.7

# ==========================================================
# [5] SONAR (안전장치)
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
        if len(m) != 3: continue
        _, angle, dist = m
        if dist <= 0: continue

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
    """
    [중요] 버퍼에 쌓인 구형 데이터를 강제로 읽어버림.
    상태가 바뀔 때나 sleep 후에 호출하여 '현재' 데이터를 보게 함.
    """
    try:
        for _ in range(count):
            next(iterator)
    except:
        pass


# ==========================================================
# SERIAL helpers
# ==========================================================
ser = None
lidar = None


def read_sensors():
    global sonar_data, ser
    if ser is None: return
    # 시리얼 버퍼에 쌓인거 최대한 최신꺼 읽기
    for _ in range(10):
        if ser.in_waiting <= 0: break
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
    """부드럽게 정지"""
    if ser is None: return
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


# ==========================================================
# STATE DEFINITIONS
# ==========================================================
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
        time.sleep(1.2)
        lidar.clean_input()  # 시작 전 청소
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH

    # 상태 관리 변수들
    car1_seen = False

    gap_t0 = None
    setup_t0 = None
    rev_t0 = None
    wait_t0 = None
    exit_open_t0 = None

    prev_t = time.time()
    scan_iter = lidar.iter_scans()

    try:
        while True:
            try:
                scan = next(scan_iter)
            except Exception as e:
                print(f"[WARN] LiDAR scan error: {e}")
                # 에러 발생시 이터레이터 재생성
                scan_iter = lidar.iter_scans()
                continue

            now = time.time()
            dt = now - prev_t
            prev_t = now

            read_sensors()  # 초음파 읽기

            # ------------------------------------------------------
            # [센서 데이터 가공]
            # ------------------------------------------------------
            # SEARCH 및 GAP 판단용 (오른쪽 65~90도)
            rs_right = get_lidar_dist_in_sector(scan, 65, 90, percentile=30)

            # CAR2 감지용 (우전방 15~40도)
            rf_front = get_lidar_dist_in_sector(scan, 15, 40, percentile=30)

            # 주차 제어용
            rf = get_lidar_dist_in_sector(scan, 15, 60, percentile=30)
            rs = get_lidar_dist_in_sector(scan, 60, 90, percentile=30)
            rr = get_lidar_dist_in_sector(scan, 90, 130, percentile=30)

            # 주차 완료(Stop) 판단용 (후방 90~270도 전체)
            rear_all = get_lidar_dist_in_sector(scan, 90, 270, percentile=20)

            # 출차 안전 확인용
            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=20)

            # 초음파 안전
            dist_LT = sonar_data[IDX_LT] if valid_us(sonar_data[IDX_LT]) else 9999
            dist_RT = sonar_data[IDX_RT] if valid_us(sonar_data[IDX_RT]) else 9999

            # 유효성 체크
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

                # 1. CAR1 감지
                if (not car1_seen) and rs_ok and (rs_right < RIGHT_OCC_TH):
                    car1_seen = True
                    msg = "SEARCH: CAR1 Seen!"

                # 2. GAP 감지 (빠른 판단: 1000mm만 넘으면 바로 진입)
                if car1_seen and rs_ok and (rs_right > RIGHT_EMPTY_TH):
                    msg = "SEARCH -> GAP (Detected!)"

                    # 상태 전환 준비
                    state = STATE_GAP
                    gap_t0 = time.time()

                    # [중요] 상태 바뀔 때 LiDAR 버퍼 비우기 (과거 데이터 삭제)
                    flush_lidar(scan_iter)

                    # ======================================================
            # [S1] GAP (CAR2 탐색)
            # ======================================================
            elif state == STATE_GAP:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = "GAP: Moving..."

                elapsed = time.time() - gap_t0

                # 1. 진입 초기 (GAP_MIN_TIME) 동안은 무조건 직진 (판단 금지)
                if elapsed < GAP_MIN_TIME:
                    msg = f"GAP: Force Move ({elapsed:.2f}s)"

                else:
                    # 2. 일정 시간 지났으면 CAR2 탐색
                    # 만약 오른쪽이 다시 막히면 SEARCH로 리셋 (노이즈 방지)
                    if rs_ok and (rs_right < RIGHT_OCC_TH):
                        state = STATE_SEARCH
                        # car1_seen = False # 필요시 리셋
                        msg = "GAP -> SEARCH (False Alarm)"

                    # CAR2(우전방)가 나타나면 정지
                    elif rf_ok and (rf_front < CAR2_RF_TH):
                        msg = "CAR2 Detected -> STOP"

                        smooth_brake(SPEED_SEARCH)
                        send_cmd(SERVO_CENTER, 0)
                        time.sleep(0.4)  # 흔들림 안정화

                        flush_lidar(scan_iter)  # 멈춘 동안 쌓인 데이터 삭제

                        state = STATE_SETUP
                        setup_t0 = time.time()

            # ======================================================
            # [S2] SETUP (각 만들기)
            # ======================================================
            elif state == STATE_SETUP:
                cmd_servo = SERVO_LEFT_MAX
                elapsed = time.time() - setup_t0

                # 1. 서보 딜레이
                if elapsed < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "SETUP: Steer Wait"

                # 2. 최소 전진 시간 (Skipping 방지)
                elif elapsed < (STEER_WAIT_TIME + SETUP_MIN_TIME):
                    cmd_speed = SPEED_SETUP
                    msg = "SETUP: Force Forward"

                # 3. 센서 기반 정지 조건 확인
                else:
                    cmd_speed = SPEED_SETUP
                    msg = "SETUP: Sensing..."

                    # 오른쪽 공간이 충분히 확보되면 정지
                    if rs > SETUP_RS_TH:
                        send_cmd(SERVO_CENTER, 0)
                        time.sleep(0.2)

                        flush_lidar(scan_iter)  # 데이터 삭제

                        state = STATE_REVERSE
                        rev_t0 = time.time()
                        msg = "SETUP -> REVERSE"

            # ======================================================
            # [S3] REVERSE (주차)
            # ======================================================
            elif state == STATE_REVERSE:
                msg = "REVERSE: Parking"

                # P-Control
                err = RS_TARGET - rs
                servo = SERVO_CENTER - (KP_RS * err)

                # 충돌 방지 (우측 후방이 닿으려 하면 풂)
                if rr < RR_MIN_SAFE:
                    servo += RR_RELEASE

                cmd_servo = int(clamp(servo, SERVO_RIGHT_MAX, SERVO_LEFT_MAX))
                cmd_speed = -SPEED_PARK

                # 정지 조건 (후방 Clear)
                # 후진 시작 후 최소시간(REV_MIN_TIME) 지났을 때만 검사
                if (time.time() - rev_t0) >= REV_MIN_TIME:
                    if rear_all > STOP_CLEAR_TH:
                        send_cmd(SERVO_CENTER, 0)
                        state = STATE_WAIT
                        wait_t0 = time.time()
                        msg = "REVERSE -> WAIT"

            # ======================================================
            # [S4] WAIT
            # ======================================================
            elif state == STATE_WAIT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = "WAIT: 4.0s"

                if (time.time() - wait_t0) >= 4.0:
                    state = STATE_EXIT
                    exit_open_t0 = None

                    flush_lidar(scan_iter)
                    msg = "WAIT -> EXIT"

            # ======================================================
            # [S5] EXIT
            # ======================================================
            elif state == STATE_EXIT:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = SPEED_EXIT
                msg = "EXIT: Go Out"

                # 전방 안전
                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT: SAFETY STOP"

                # 완전 탈출 판단
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
                # 종료하고 싶으면 break, 아니면 대기
                # break

            # 초음파 비상 정지 (옵션)
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                msg = "US SAFETY STOP"

            # 명령 전송
            send_cmd(cmd_servo, cmd_speed)

            # 화면 표시
            sub = (
                f"State:{state} | rsR:{int(rs_right)} rfF:{int(rf_front)} | "
                f"RF:{int(rf)} RS:{int(rs)} RR:{int(rr)} | Rear:{int(rear_all)}"
            )
            img = np.zeros((300, 1000, 3), dtype=np.uint8)
            cv2.putText(img, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(img, sub, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Parking Monitor", img)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        send_cmd(SERVO_CENTER, 0)
        if ser: ser.close()
        if lidar:
            lidar.stop()
            lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()