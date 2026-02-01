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
# [2] 차량/주차공간 치수 (mm) - 참고용 표시
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

# ✅ 주차 완료 후 정지 시간 (원하면 4초)
TIME_HOLD_AFTER_PARK = 4.0

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

def temporal_median(buf: deque, v: float) -> float:
    buf.append(v)
    m = np.median(np.array(buf, dtype=np.float32))
    return float(np.asarray(m).item())

def in_sector(angle, start, end):
    """
    섹터 포함 판정(0~359), start<=end면 일반구간,
    start>end면 360 wrap 구간으로 처리.
    """
    if start <= end:
        return start <= angle <= end
    else:
        return (angle >= start) or (angle <= end)

def percentile_in_sector_with_count(scan, start_angle, end_angle, percentile=10):
    """
    섹터 내 dist 분위수 + 포인트 개수(n) 반환.
    포인트가 없으면 (9999.0, 0)
    """
    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if in_sector(angle, start_angle, end_angle):
            dists.append(dist)

    n = len(dists)
    if n == 0:
        return 9999.0, 0

    v = np.percentile(np.array(dists, dtype=np.float32), percentile)
    return float(np.asarray(v).item()), n

def min_in_sector(scan, start_angle, end_angle) -> float:
    dists = []
    for (_, angle, dist) in scan:
        if dist > 0 and in_sector(angle, start_angle, end_angle):
            dists.append(dist)
    return float(np.min(dists)) if dists else 9999.0

# ==========================================
# [6] 탐색(차/갭) 임계값 (오른쪽 옆면 기준: right=90)
# ==========================================
CAR_EXIST_DIST   = 2200    # 이하면 차가 옆에 붙어있다고 봄
EMPTY_SPACE_DIST = 2800    # 이상이면 빈 공간으로 봄(포인트가 있을 때)
GAP_STREAK       = 3       # 갭 연속 프레임
CAR2_STREAK      = 2       # 2번차 재등장 연속 프레임

# “포인트 개수” 기준: 너무 좁으면 n이 0~2가 자주 나옴 → 섹터를 넓히거나 n 기준 완화
N_CAR_MIN = 5              # 차로 보려면 최소 포인트 수
N_GAP_MAX = 2              # 갭으로 보려면 포인트가 이 이하로 떨어져도 갭 인정(9999 포함)

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
RIGHT_CENTER = 90
LEFT_CENTER  = 270

SECTOR_HALF_WIDTH = 20  # 10~30 튜닝 (0이 자주 뜨면 25~35로)
RIGHT_SECTOR = (RIGHT_CENTER - SECTOR_HALF_WIDTH, RIGHT_CENTER + SECTOR_HALF_WIDTH)
LEFT_SECTOR  = (LEFT_CENTER  - SECTOR_HALF_WIDTH, LEFT_CENTER  + SECTOR_HALF_WIDTH)

BOTH_SIDE_STOP_DIST = 700

# 확인(정지 상태에서 좌/우 모두 가까움이 유지되는지)
LIDAR_CONFIRM_DURATION = 2.0
LIDAR_CONFIRM_HZ = 10
LIDAR_CONFIRM_PASS_RATIO = 0.7

def lidar_confirm_2s(get_left, get_right):
    duration = LIDAR_CONFIRM_DURATION
    dt = 1.0 / LIDAR_CONFIRM_HZ
    ok = 0
    total = 0
    t0 = time.time()

    while time.time() - t0 < duration:
        dl = get_left()
        dr = get_right()
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
STATE_LIDAR_CONFIRM = 13
STATE_HOLD = 14
STATE_PARKED = 15

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_CAR2 = 2   # (기존 FIND_GAP 의미를 명확히)

# ==========================================
# [10] 유저 모드 UI 도우미
# ==========================================
def draw_panel(img, x, y, w, h, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def put_line(img, text, x, y, scale=0.62, color=(255,255,255), thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# ==========================================
# [11] 메인
# ==========================================
def main():
    cv2.namedWindow("Parking Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parking Monitor", 1280, 720)

    ser = None
    lidar = None

    # ✅ SEARCH에서 오른쪽 옆면을 보는 섹터(90도 기준)
    # 너무 좁으면 포인트가 0이 잦아져 9999가 자주 뜹니다.
    RIGHT_SEARCH_START = 55
    RIGHT_SEARCH_END   = 125

    # 최신 좌/우 거리(Confirm에서 사용)
    latest_left = 9999.0
    latest_right = 9999.0

    def get_left():
        return latest_left

    def get_right():
        return latest_right

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

    gap_streak = 0
    car2_streak = 0

    pause_start = 0.0
    pause_duration = 0.0
    next_state = STATE_SEARCH
    pause_msg = ""

    both_close_start = 0.0  # 좌/우 동시 근접 감지 후 지연 정지용
    running = True

    def reset_search_buffers():
        buf_side.clear()
        buf_left.clear()
        buf_right.clear()

    # ====== 실행 루프 ======
    try:
        for scan in lidar.iter_scans():
            if not running:
                break

            curr_time = time.time()

            # -----------------------------
            # (A) SEARCH용: 오른쪽 옆면(55~125) 거리 + 포인트 수
            # -----------------------------
            side_raw, side_n = percentile_in_sector_with_count(
                scan, RIGHT_SEARCH_START, RIGHT_SEARCH_END, LIDAR_PCTL
            )
            side_dist = temporal_median(buf_side, side_raw)

            # -----------------------------
            # (B) 센터링/양쪽감지용: 오른쪽(90±W), 왼쪽(270±W)
            # -----------------------------
            right_raw, rn = percentile_in_sector_with_count(
                scan, RIGHT_SECTOR[0], RIGHT_SECTOR[1], LIDAR_PCTL
            )
            left_raw, ln = percentile_in_sector_with_count(
                scan, LEFT_SECTOR[0], LEFT_SECTOR[1], LIDAR_PCTL
            )

            dist_right = temporal_median(buf_right, right_raw)  # right(90)
            dist_left  = temporal_median(buf_left, left_raw)    # left(270)

            latest_right = dist_right
            latest_left  = dist_left

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
                    gap_streak = 0
                    car2_streak = 0
                    both_close_start = 0.0
                    reset_search_buffers()
                    lidar.clean_input()

            # =========================
            # [1] SEARCH: CAR1 -> GAP -> CAR2 (right=90 기준)
            # =========================
            elif state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                # 탐지 판단은 "RAW + n" 위주로, 표시/안정화는 side_dist도 함께 사용
                is_car = (side_raw < CAR_EXIST_DIST) and (side_n >= N_CAR_MIN)
                is_gap = (side_raw > EMPTY_SPACE_DIST) or (side_n <= N_GAP_MAX)  # 9999/0 포함

                if search_step == STEP_FIND_CAR1:
                    msg = f"SEARCH: FIND CAR1 | right={int(side_raw)}(raw) n={side_n}"
                    if is_car:
                        print(f"🚗 CAR1 detected (right) raw={int(side_raw)} n={side_n}")
                        search_step = STEP_PASS_CAR1
                        gap_streak = 0
                        car2_streak = 0

                elif search_step == STEP_PASS_CAR1:
                    cmd_servo = SERVO_SLIGHT_RIGHT
                    msg = f"SEARCH: PASS CAR1 | right={int(side_raw)}(raw) n={side_n} gap_streak={gap_streak}/{GAP_STREAK}"

                    if is_gap:
                        gap_streak += 1
                        if gap_streak >= GAP_STREAK:
                            print(f"👀 GAP entered raw={int(side_raw)} n={side_n}")
                            search_step = STEP_FIND_CAR2
                            # ✅ 갭 진입 순간: 버퍼 리셋(9999가 median에 남아 2번차 재등장 감지를 방해하는 문제 완화)
                            buf_side.clear()
                            car2_streak = 0
                    else:
                        gap_streak = 0

                elif search_step == STEP_FIND_CAR2:
                    cmd_servo = SERVO_SLIGHT_RIGHT
                    msg = f"SEARCH: FIND CAR2 | right={int(side_raw)}(raw) n={side_n} car2_streak={car2_streak}/{CAR2_STREAK}"

                    # ✅ 2번차는 "갭 이후 다시 car 조건이 연속으로 들어올 때" 확정
                    if is_car:
                        car2_streak += 1
                        if car2_streak >= CAR2_STREAK:
                            print(f"🛑 CAR2 detected -> start reverse parking raw={int(side_raw)} n={side_n}")

                            # 살짝 브레이크 느낌
                            ser.write(b"D,-150\n"); time.sleep(0.08); ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state = STATE_SETUP_FORWARD
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Setup(Forward)"
                            state_timer = curr_time
                    else:
                        car2_streak = 0

            # =========================
            # [2] SETUP: 전진 공간 확보
            # =========================
            elif state == STATE_SETUP_FORWARD:
                cmd_servo = SERVO_LEFT_MAX

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "SETUP: Align LEFT..."
                else:
                    cmd_speed = SPEED_SETUP
                    driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                    msg = f"SETUP: Forward {driving_time:.1f}/{TIME_SETUP_MOVE:.1f}s"

                    if driving_time >= TIME_SETUP_MOVE:
                        ser.write(b"D,0\n")
                        state = STATE_PAUSE
                        next_state = STATE_REVERSE_TURN
                        pause_duration = 0.7
                        pause_start = curr_time
                        pause_msg = "Ready for Reverse Turn(Right)"
                        state_timer = curr_time

            # =========================
            # [3] REVERSE TURN: 꺾고 후진 진입
            # =========================
            elif state == STATE_REVERSE_TURN:
                cmd_servo = SERVO_RIGHT_MAX

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "REVERSE_TURN: Align RIGHT..."
                else:
                    cmd_speed = -SPEED_REVERSE
                    driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                    msg = f"REVERSE_TURN: Back {driving_time:.1f}/{TIME_REVERSE_TURN:.1f}s"

                    if driving_time >= TIME_REVERSE_TURN:
                        ser.write(b"D,0\n")
                        state = STATE_REVERSE_STRAIGHT
                        state_timer = curr_time
                        both_close_start = 0.0

            # =========================
            # [4] REVERSE STRAIGHT: 센터링 + 후진 + 좌/우 동시 근접으로 정지
            # =========================
            elif state == STATE_REVERSE_STRAIGHT:
                dt = curr_time - state_timer

                if dt < STEER_WAIT_TIME:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "REVERSE: Align CENTER..."
                else:
                    cmd_speed = -SPEED_REVERSE

                    # 센터링: err = left(270) - right(90)
                    err = dist_left - dist_right

                    if abs(err) < REVERSE_CENTER_DEADBAND:
                        delta = 0
                    else:
                        delta = int(clamp(REVERSE_CENTER_KP * err, -REVERSE_CENTER_LIMIT, REVERSE_CENTER_LIMIT))

                    # 오른쪽 조향이 SERVO 감소 방향이라는 가정 유지
                    cmd_servo = SERVO_CENTER - delta

                    msg = (f"REVERSE: L(270)={int(dist_left)} n={ln} | "
                           f"R(90)={int(dist_right)} n={rn} | err={int(err)} d={delta} servo={int(cmd_servo)}")

                # ✅ 좌/우 동시 근접 감지
                both_close = (dist_left < BOTH_SIDE_STOP_DIST) and (dist_right < BOTH_SIDE_STOP_DIST) and (ln > 3) and (rn > 3)

                if both_close and dt > STEER_WAIT_TIME:
                    if both_close_start == 0.0:
                        print("✨ both sides close detected -> delay then stop")
                        both_close_start = curr_time
                    elif (curr_time - both_close_start) >= TIME_DELAY_STOP:
                        print("✅ stop by both-sides-close")
                        ser.write(b"D,0\n")
                        state = STATE_LIDAR_CONFIRM
                        state_timer = curr_time
                else:
                    both_close_start = 0.0

                # 안전 타임아웃
                if dt >= (STEER_WAIT_TIME + TIME_REVERSE_STRAIGHT_MAX):
                    print("✅ stop by timeout")
                    ser.write(b"D,0\n")
                    state = STATE_LIDAR_CONFIRM
                    state_timer = curr_time

            # =========================
            # [5] 정지 상태에서 라이다 확인(2초)
            # =========================
            elif state == STATE_LIDAR_CONFIRM:
                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                msg = "CONFIRM: LIDAR 2s (both close 유지 확인)"

                ser.write(b"D,0\n")
                passed, (ok, total) = lidar_confirm_2s(get_left, get_right)
                print(f"📌 confirm: passed={passed} ok/total={ok}/{total} th={BOTH_SIDE_STOP_DIST}mm")

                state = STATE_HOLD
                state_timer = curr_time

            # =========================
            # [6] 주차 완료 후 정지 유지
            # =========================
            elif state == STATE_HOLD:
                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                hold_t = curr_time - state_timer
                msg = f"HOLD: {hold_t:.1f}/{TIME_HOLD_AFTER_PARK:.1f}s"

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
                msg = "PARKED ✅ (q to quit)"
                ser.write(b"D,0\n")

            # ---- 명령 송신 ----
            ser.write(f"S,{int(cmd_servo)}\n".encode())
            ser.write(f"D,{int(cmd_speed)}\n".encode())

            # ==========================================
            # [유저 모드 화면 출력]
            # ==========================================
            img = np.zeros((720, 1280, 3), dtype=np.uint8)

            # 상단 패널
            draw_panel(img, 20, 20, 1240, 220, alpha=0.40)
            put_line(img, f"STATE: {state}   STEP: {search_step}   |   {msg}", 40, 65, scale=0.80, color=(0,255,0), thickness=2)
            put_line(img, "Env: clockwise | 0=front | 90=RIGHT | 270=LEFT", 40, 105, scale=0.65, color=(0,255,255), thickness=2)
            put_line(img, f"SEARCH sector(right): {RIGHT_SEARCH_START}~{RIGHT_SEARCH_END} deg   PCTL={LIDAR_PCTL}", 40, 145, scale=0.65, color=(255,255,255), thickness=2)
            put_line(img, f"RIGHT(90±{SECTOR_HALF_WIDTH})={RIGHT_SECTOR}  LEFT(270±{SECTOR_HALF_WIDTH})={LEFT_SECTOR}", 40, 185, scale=0.65, color=(255,255,255), thickness=2)

            # 중간 패널(측정값)
            draw_panel(img, 20, 270, 1240, 260, alpha=0.35)
            put_line(img, f"RIGHT_SEARCH raw={int(side_raw)} mm   n={side_n}   | filtered(median)={int(side_dist)} mm", 40, 320, scale=0.72, color=(255,255,255), thickness=2)

            put_line(img, f"RIGHT(90) raw={int(right_raw)} mm n={rn}  | filtered={int(dist_right)} mm", 40, 370, scale=0.72, color=(255,255,255), thickness=2)
            put_line(img, f"LEFT(270)  raw={int(left_raw)}  mm n={ln}  | filtered={int(dist_left)}  mm", 40, 420, scale=0.72, color=(255,255,255), thickness=2)

            put_line(img, f"overall_min(any angle)={int(overall_min)} mm  (9999 지속이면 라이다/각도/섹터 문제)", 40, 470, scale=0.65, color=(0,255,255), thickness=2)

            # 하단 패널(임계값/판정)
            draw_panel(img, 20, 560, 1240, 140, alpha=0.35)
            put_line(img, f"CAR_EXIST<{CAR_EXIST_DIST} & n>={N_CAR_MIN} | GAP if dist>{EMPTY_SPACE_DIST} OR n<={N_GAP_MAX}", 40, 610, scale=0.65, color=(255,255,255), thickness=2)
            put_line(img, f"GAP_STREAK={GAP_STREAK}  CAR2_STREAK={CAR2_STREAK}  BOTH_STOP<{BOTH_SIDE_STOP_DIST}  DELAY_STOP={TIME_DELAY_STOP}s", 40, 650, scale=0.65, color=(255,255,255), thickness=2)
            put_line(img, "Keys: q=quit", 40, 690, scale=0.70, color=(0,255,255), thickness=2)

            cv2.imshow("Parking Monitor", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False

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
