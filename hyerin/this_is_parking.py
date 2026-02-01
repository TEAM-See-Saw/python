import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque

# ==========================================
# [1] 포트/통신 (Arduino와 일치)
# ==========================================
PORT = "COM4"
LIDAR_PORT = "COM3"
SER_BAUD = 115200
SER_TIMEOUT = 0.05

# ==========================================
# [2] 차량/주차공간 치수 (mm)
# ==========================================
CAR_W = 750
CAR_L = 1000
SLOT_W = 950
SLOT_L = 1500
SIDE_CLEAR_EACH = (SLOT_W - CAR_W) / 2

# ==========================================
# [3] 속도/서보
# ==========================================
SPEED_SEARCH = 70
SPEED_SETUP = 80
SPEED_REVERSE = 75

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555

STEER_WAIT_TIME = 0.7

# ==========================================
# [4] 후방주차 시간 파라미터
# ==========================================
TIME_SETUP_MOVE = 1.8
TIME_REVERSE_TURN = 2.2
TIME_REVERSE_STRAIGHT_MAX = 4.5
TIME_DELAY_STOP = 1.5

# ✅ 주차 완료 후 3초 정지
TIME_HOLD_AFTER_PARK = 3.0

# ==========================================
# [5] 라이다 처리(튐 완화)
# ==========================================
LIDAR_TEMPORAL_N = 5
LIDAR_PCTL = 10  # 10~30 튜닝 가능

buf_side = deque(maxlen=LIDAR_TEMPORAL_N)
buf_left = deque(maxlen=LIDAR_TEMPORAL_N)
buf_right = deque(maxlen=LIDAR_TEMPORAL_N)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def percentile_in_sector_with_count(scan, start_angle, end_angle, percentile=10):
    """
    섹터 내 dist 리스트의 분위수 + 포인트 개수 반환.
    포인트가 없으면 (9999.0, 0) 반환.
    """
    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if start_angle <= angle <= end_angle:
            dists.append(dist)

    n = len(dists)
    if n == 0:
        return 9999.0, 0

    v = np.percentile(np.array(dists, dtype=np.float32), percentile)
    return float(np.asarray(v).item()), n

def temporal_median(buf: deque, v: float) -> float:
    buf.append(v)
    m = np.median(np.array(buf, dtype=np.float32))
    return float(np.asarray(m).item())

def min_in_sector(scan, start_angle, end_angle) -> float:
    dists = []
    for (_, angle, dist) in scan:
        if dist > 0 and start_angle <= angle <= end_angle:
            dists.append(dist)
    return float(np.min(dists)) if dists else 9999.0

# ==========================================
# [6] 탐색(차/갭) 임계값 (오른쪽 옆면 기준)
# ==========================================
CAR_EXIST_DIST   = 2200
EMPTY_SPACE_DIST = 2800
GAP_STREAK       = 3

# ==========================================
# [7] 후진 센터링(라이다)
#   - 사용자 환경: 시계방향 / 오른쪽=90 / 왼쪽=270
# ==========================================
REVERSE_CENTER_KP = 0.035
REVERSE_CENTER_LIMIT = 22
REVERSE_CENTER_DEADBAND = 40

# ==========================================
# [8] ✅ 라이다로 "양쪽 동시 감지" 시 정지 로직
# ==========================================
# 섹터(너 환경: 오른쪽=90, 왼쪽=270)
RIGHT_CENTER = 90
LEFT_CENTER  = 270

# 센터링/측정 섹터 폭(좁으면 포인트가 0이 되어 9999가 잦아짐)
SECTOR_HALF_WIDTH = 20  # 10~30 튜닝

RIGHT_SECTOR = (RIGHT_CENTER - SECTOR_HALF_WIDTH, RIGHT_CENTER + SECTOR_HALF_WIDTH)
LEFT_SECTOR  = (LEFT_CENTER  - SECTOR_HALF_WIDTH, LEFT_CENTER  + SECTOR_HALF_WIDTH)

# ✅ 양쪽이 동시에 "가까움"으로 판단하는 거리
BOTH_SIDE_STOP_DIST = 700  # mm (환경에 맞춰 600~900 튜닝)

# ✅ 바로 멈추지 말고, 정지 상태에서 2초간 '양쪽 동시'가 유지되는지 확인
LIDAR_CONFIRM_DURATION = 2.0
LIDAR_CONFIRM_HZ = 10
LIDAR_CONFIRM_PASS_RATIO = 0.7

def lidar_confirm_2s(dist_left_func, dist_right_func):
    """
    2초 동안 정지 유지하면서
    '왼쪽/오른쪽 모두 BOTH_SIDE_STOP_DIST 미만' 비율이 일정 이상이면 통과.
    dist_left_func / dist_right_func: 최신 dist를 읽는 함수(클로저)
    """
    duration = LIDAR_CONFIRM_DURATION
    dt = 1.0 / LIDAR_CONFIRM_HZ
    ok = 0
    total = 0

    t0 = time.time()
    while time.time() - t0 < duration:
        dl = dist_left_func()
        dr = dist_right_func()
        if (dl < BOTH_SIDE_STOP_DIST) and (dr < BOTH_SIDE_STOP_DIST):
            ok += 1
        total += 1
        time.sleep(dt)

    passed = (ok / total) >= LIDAR_CONFIRM_PASS_RATIO if total > 0 else False
    return passed, (ok, total)

# ==========================================
# [9] 상태 정의 (후방주차 FSM)
# ==========================================
STATE_SEARCH = 0
STATE_PAUSE = 99

STATE_SETUP_FORWARD = 10
STATE_REVERSE_TURN = 11
STATE_REVERSE_STRAIGHT = 12

STATE_LIDAR_CONFIRM = 13   # ✅ 초음파 대신 라이다 확인
STATE_HOLD_3S = 14
STATE_PARKED = 15

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP  = 2

# ==========================================
# [10] 메인
# ==========================================
def main():
    cv2.namedWindow("Parking Monitor")

    ser = None
    lidar = None

    # ✅ 오른쪽 옆면 탐색 섹터(SEARCH용): 오른쪽=90 기준
    RIGHT_SEARCH_START = 60
    RIGHT_SEARCH_END   = 120

    try:
        ser = serial.Serial(PORT, SER_BAUD, timeout=SER_TIMEOUT)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 (Serial 115200 + RPLidar)")
        time.sleep(1.0)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return

    # 초기 정렬/정지
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    ser.write(b"D,0\n")
    time.sleep(1.0)

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = time.time()

    valid_gap_count = 0

    pause_start = 0.0
    pause_duration = 0.0
    next_state = STATE_SEARCH
    pause_msg = ""

    side_detect_time = 0.0
    running = True

    # 최신 좌/우 거리(라이다 confirm에서 접근용)
    latest_left = 9999.0
    latest_right = 9999.0

    def get_left():
        return latest_left

    def get_right():
        return latest_right

    try:
        while running:
            for scan in lidar.iter_scans():
                curr_time = time.time()

                # ----------------------------------------
                # (A) SEARCH용: 오른쪽 옆면 거리(60~120)
                # ----------------------------------------
                side_raw, side_n = percentile_in_sector_with_count(scan, RIGHT_SEARCH_START, RIGHT_SEARCH_END, LIDAR_PCTL)
                side_dist = temporal_median(buf_side, side_raw)

                # ----------------------------------------
                # (B) 센터링/양쪽감지용: 오른쪽(90±W), 왼쪽(270±W)
                # ----------------------------------------
                right_raw, rn = percentile_in_sector_with_count(scan, RIGHT_SECTOR[0], RIGHT_SECTOR[1], LIDAR_PCTL)
                left_raw,  ln = percentile_in_sector_with_count(scan, LEFT_SECTOR[0], LEFT_SECTOR[1], LIDAR_PCTL)

                dist_right = temporal_median(buf_right, right_raw)  # right(90)
                dist_left  = temporal_median(buf_left, left_raw)    # left(270)

                latest_right = dist_right
                latest_left  = dist_left

                # 전체 최소거리(라이다 살아있는지)
                overall_min = min_in_sector(scan, 0, 359)

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                msg = ""

                # =========================
                # PAUSE
                # =========================
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = f"WAIT... ({pause_msg})"
                    if curr_time - pause_start > pause_duration:
                        state = next_state
                        state_timer = curr_time
                        valid_gap_count = 0
                        side_detect_time = 0.0
                        lidar.clean_input()

                # =========================
                # [1] SEARCH: 차1 -> 갭 -> 차2 찾기 (오른쪽=90 기준)
                # =========================
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    cmd_servo = SERVO_CENTER

                    if search_step == STEP_FIND_CAR1:
                        msg = f"FIND CAR1 | right(60~120)={int(side_dist)} n={side_n}"
                        if side_dist < CAR_EXIST_DIST and side_n > 5:
                            print(f"🚗 1번 차 감지(오른쪽) {int(side_dist)}mm")
                            search_step = STEP_PASS_CAR1

                    elif search_step == STEP_PASS_CAR1:
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"PASS CAR1 | right(60~120)={int(side_dist)} n={side_n}"

                        # ✅ 빈공간 판정: 거리 OR 포인트 부족(벽/차가 없어짐)
                        is_gap = (side_dist > EMPTY_SPACE_DIST) or (side_n <= 2)

                        if is_gap:
                            valid_gap_count += 1
                            if valid_gap_count >= GAP_STREAK:
                                print(f"👀 빈공간(갭) 진입 {int(side_dist)}mm n={side_n}")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"FIND CAR2 | right(60~120)={int(side_dist)} n={side_n}"

                        # ✅ 다시 차가 보이면(거리 감소 + 포인트 충분)
                        if (side_dist < CAR_EXIST_DIST) and (side_n > 5):
                            print(f"🛑 2번 차 감지 -> 후방주차 시작 {int(side_dist)}mm")
                            ser.write(b"D,-150\n"); time.sleep(0.08); ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state = STATE_SETUP_FORWARD
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Setup(Forward)"

                # =========================
                # [2] SETUP: 전진 공간 확보
                # =========================
                elif state == STATE_SETUP_FORWARD:
                    cmd_servo = SERVO_LEFT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align LEFT..."
                    else:
                        cmd_speed = SPEED_SETUP
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"SETUP FWD: {driving_time:.1f}/{TIME_SETUP_MOVE}s"
                        if driving_time >= TIME_SETUP_MOVE:
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE
                            next_state = STATE_REVERSE_TURN
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Reverse Turn(Right)"

                # =========================
                # [3] REVERSE TURN: 꺾고 후진 진입
                # =========================
                elif state == STATE_REVERSE_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align RIGHT..."
                    else:
                        cmd_speed = -SPEED_REVERSE
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"REV TURN: {driving_time:.1f}/{TIME_REVERSE_TURN}s"
                        if driving_time >= TIME_REVERSE_TURN:
                            ser.write(b"D,0\n")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            side_detect_time = 0.0

                # =========================
                # [4] REVERSE STRAIGHT: 중앙 + 후진(센터링) + ✅양쪽동시감지
                # =========================
                elif state == STATE_REVERSE_STRAIGHT:
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = SERVO_CENTER
                        msg = "Align CENTER..."
                    else:
                        cmd_speed = -SPEED_REVERSE

                        # ✅ 센터링: err = left(270) - right(90)
                        err = dist_left - dist_right
                        if abs(err) < REVERSE_CENTER_DEADBAND:
                            delta = 0
                        else:
                            delta = int(clamp(REVERSE_CENTER_KP * err, -REVERSE_CENTER_LIMIT, REVERSE_CENTER_LIMIT))

                        # delta>0 이면 왼쪽이 더 멀다(차가 오른쪽 붙음) -> 오른쪽 조향 필요
                        # 일반적으로 오른쪽 조향은 SERVO 값 감소 방향인 경우가 많아
                        cmd_servo = SERVO_CENTER - delta

                        msg = (f"REV STRAIGHT | L(270)={int(dist_left)} n={ln} "
                               f"R(90)={int(dist_right)} n={rn} err={int(err)} d={delta} servo={int(cmd_servo)}")

                    # ✅ 핵심 변경: 양쪽이 동시에 가까우면(옆차 동시 감지) 정지 준비
                    both_close = (dist_left < BOTH_SIDE_STOP_DIST) and (dist_right < BOTH_SIDE_STOP_DIST) and (ln > 3) and (rn > 3)

                    if both_close and dt > STEER_WAIT_TIME:
                        if side_detect_time == 0.0:
                            print("✨ (라이다) 좌/우 동시 근접 감지 -> 1.5초 더 후진 후 정지")
                            side_detect_time = curr_time
                        elif (curr_time - side_detect_time) >= TIME_DELAY_STOP:
                            print("✅ 후진 주차 정지(좌/우 동시 근접 기반)")
                            ser.write(b"D,0\n")
                            state_timer = curr_time
                            state = STATE_LIDAR_CONFIRM

                    # 안전 타임아웃
                    if dt >= (STEER_WAIT_TIME + TIME_REVERSE_STRAIGHT_MAX):
                        print("✅ 후진 주차 정지(타임아웃)")
                        ser.write(b"D,0\n")
                        state_timer = curr_time
                        state = STATE_LIDAR_CONFIRM

                # =========================
                # [5] ✅ 라이다 2초 확인(정지 상태에서 좌/우 동시 감지 유지되는지)
                # =========================
                elif state == STATE_LIDAR_CONFIRM:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "LIDAR CONFIRM 2s (both sides close)"

                    ser.write(b"D,0\n")  # 정지 유지
                    passed, (ok, total) = lidar_confirm_2s(get_left, get_right)
                    print(f"📌 LIDAR confirm: passed={passed} ok/total={ok}/{total} th={BOTH_SIDE_STOP_DIST}mm")

                    # 확인 끝나면 3초 정지로
                    state = STATE_HOLD_3S
                    state_timer = curr_time

                # =========================
                # [6] 주차 완료 후 3초 정지
                # =========================
                elif state == STATE_HOLD_3S:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    hold_t = curr_time - state_timer
                    msg = f"HOLD AFTER PARK: {hold_t:.1f}/{TIME_HOLD_AFTER_PARK:.1f}s"
                    ser.write(b"D,0\n")
                    if hold_t >= TIME_HOLD_AFTER_PARK:
                        state = STATE_PARKED
                        state_timer = curr_time

                # =========================
                # [7] PARKED
                # =========================
                elif state == STATE_PARKED:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "PARKED ✅ (press q to quit)"
                    ser.write(b"D,0\n")

                # ---- 명령 송신 ----
                ser.write(f"S,{int(cmd_servo)}\n".encode())
                ser.write(f"D,{int(cmd_speed)}\n".encode())

                # ---- 디버그 UI ----
                debug_img = np.zeros((560, 1200, 3), dtype=np.uint8)

                cv2.putText(debug_img, f"State: {state} | {msg}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.putText(debug_img,
                            f"SEARCH right(60~120)={int(side_dist)} n={side_n} | "
                            f"R(90±{SECTOR_HALF_WIDTH})={int(dist_right)} n={rn} | "
                            f"L(270±{SECTOR_HALF_WIDTH})={int(dist_left)} n={ln}",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.putText(debug_img,
                            f"both_stop_th={BOTH_SIDE_STOP_DIST}mm  delay_stop={TIME_DELAY_STOP}s  "
                            f"confirm={LIDAR_CONFIRM_DURATION}s pass_ratio={LIDAR_CONFIRM_PASS_RATIO}",
                            (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(debug_img,
                            f"overall_min={int(overall_min)}  (9999 always -> sector empty / lidar issue)",
                            (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(debug_img,
                            f"Angle mapping (your env): 0=front, 90=right, 270=left (clockwise)",
                            (10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.putText(debug_img, "q: quit", (10, 520),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", debug_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                    break

    except RPLidarException as e:
        print(f"⚠️ 라이다 오류: {e}")
    except KeyboardInterrupt:
        print("종료(KeyboardInterrupt)")
    except Exception as e:
        print(f"시스템 오류: {e}")
    finally:
        try:
            if ser:
                ser.write(b"D,0\n")
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
