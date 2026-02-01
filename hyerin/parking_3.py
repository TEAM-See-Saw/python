import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque

# ==========================================
# [1] 포트/통신
# ==========================================
PORT = "COM4"
LIDAR_PORT = "COM3"
SER_BAUD = 115200  # ✅ Arduino Mega와 반드시 동일

# ==========================================
# [2] 차량/주차공간 치수 (mm)
# ==========================================
CAR_W = 750      # 차량 폭(짧은쪽) 75cm
CAR_L = 1000     # 차량 길이(긴쪽) 100cm
SLOT_W = 950     # 주차면 폭(짧은쪽) 95cm
SLOT_L = 1500    # 주차면 길이(긴쪽) 150cm

SIDE_CLEAR_EACH = (SLOT_W - CAR_W) / 2  # 100mm (양쪽 10cm)
# 폭이 매우 타이트하므로 센터링 제어가 유리함

# ==========================================
# [3] 속도/서보
# ==========================================
SPEED_SEARCH = 70
SPEED_SETUP  = 80
SPEED_REVERSE = 75
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX  = 680
SERVO_SLIGHT_RIGHT = 555  # 벽타기/안정용(필요하면)

STEER_WAIT_TIME = 0.7

# ==========================================
# [4] 후방주차 시간 파라미터 (튜닝 포인트)
# ==========================================
TIME_SETUP_MOVE   = 1.8   # 전진으로 공간 확보
TIME_REVERSE_TURN = 2.2   # 핸들 꺾고 후진 진입
TIME_REVERSE_STRAIGHT_MAX = 4.5  # 후진 직진(최대)

# 감지 후 약간 더 움직였다가 정지(마무리 감각)
TIME_DELAY_STOP = 1.5

# ==========================================
# [5] 라이다 처리(튐 완화)
# ==========================================
LIDAR_TEMPORAL_N = 5
LIDAR_PCTL = 10  # 하위 퍼센타일(최소값보다 안정적)

buf_side = deque(maxlen=LIDAR_TEMPORAL_N)
buf_left = deque(maxlen=LIDAR_TEMPORAL_N)
buf_right = deque(maxlen=LIDAR_TEMPORAL_N)

def percentile_in_sector(scan, start_angle, end_angle, percentile=10):
    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if start_angle <= angle <= end_angle:
            dists.append(dist)
    if not dists:
        return 9999.0
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))

def temporal_median(buf, v):
    buf.append(v)
    return float(np.median(np.array(buf, dtype=np.float32))) if buf else 9999.0

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ==========================================
# [6] 탐색(차/갭) 임계값 (튜닝 포인트)
# ==========================================
# 옆(주차면 방향)에서 "차가 있다" / "빈공간이다" 판단 기준
CAR_EXIST_DIST = 2500     # 이하면 '차 있음'
EMPTY_SPACE_DIST = 3500   # 이상이면 '빈공간(갭)'

GAP_STREAK = 2            # 연속 N번이면 확정(센서 튐 대비)

# ==========================================
# [7] 후진 주차 마무리 감지(라이다 기반)
# ==========================================
SIDE_STOP_DIST = 700  # 좌/우(90/270 근처) 어느 쪽이든 70cm 내면 “가까움”
REVERSE_CENTER_KP = 0.035
REVERSE_CENTER_LIMIT = 22
REVERSE_CENTER_DEADBAND = 40

# ==========================================
# [8] 초음파 확인(2초 구간에서만 폴링)
# ==========================================
US_CONFIRM_ENABLE = True
US_CONFIRM_DURATION = 2.0
US_CONFIRM_HZ = 10
US_SIDE_TH_MM = 300         # 옆차 인지 거리(250~450 튜닝)
US_CONFIRM_PASS_RATIO = 0.7 # 2초 샘플 중 70% 이상 동시 인지면 통과

def poll_ultrasonic_once(ser: serial.Serial, timeout_s=0.03):
    """
    Arduino에 'U\\n' 보내고, 'US:LF,LR,RF,RR' 한 줄만 읽는다.
    상시 수신 금지. 이 함수는 2초 확인 구간에서만 호출.
    """
    try:
        ser.reset_input_buffer()  # ✅ 찌꺼기 제거(중요)
    except Exception:
        pass

    ser.write(b"U\n")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("US:"):
                parts = line[3:].split(",")
                if len(parts) == 4:
                    try:
                        return [int(x) for x in parts]  # [LF, LR, RF, RR]
                    except:
                        return None
    return None

def ultrasonic_confirm_2s(ser: serial.Serial):
    """
    2초 동안 정지 유지하면서,
    LF,LR,RF,RR 4개 초음파가 동시에(th 이하) 옆차를 인지하는지 확인.
    """
    duration = US_CONFIRM_DURATION
    dt = 1.0 / US_CONFIRM_HZ
    ok = 0
    total = 0

    t0 = time.time()
    while time.time() - t0 < duration:
        ser.write(b"D,0\n")  # 정지 유지

        vals = poll_ultrasonic_once(ser, timeout_s=0.03)
        if vals is not None:
            lf, lr, rf, rr = vals
            if (lf < US_SIDE_TH_MM and lr < US_SIDE_TH_MM and
                rf < US_SIDE_TH_MM and rr < US_SIDE_TH_MM):
                ok += 1
            total += 1

        time.sleep(dt)

    if total < max(3, int(duration * US_CONFIRM_HZ * 0.3)):
        return False, (ok, total)
    passed = (ok / total) >= US_CONFIRM_PASS_RATIO
    return passed, (ok, total)

# ==========================================
# [9] 상태 정의 (후방주차 FSM)
# ==========================================
STATE_SEARCH = 0
STATE_PAUSE = 99

STATE_SETUP_FORWARD = 10      # 전진 공간 확보
STATE_REVERSE_TURN = 11       # 핸들 꺾고 후진 진입
STATE_REVERSE_STRAIGHT = 12   # 후진 직진(센터링 포함)
STATE_US_CONFIRM = 13         # ✅ 2초 초음파 동시 인지 확인
STATE_PARKED = 14

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP  = 2
STEP_FIND_CAR2 = 3

# ==========================================
# [10] 메인
# ==========================================
def main():
    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, SER_BAUD, timeout=0.05)
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

    side_detect_time = 0.0  # 후진 마무리 감지용

    running = True

    try:
        while running:
            for scan in lidar.iter_scans():
                curr_time = time.time()

                # [A] 탐색용 “옆(주차면 방향)” 거리: 30~110 (너가 기존에 쓰던 범위 기반)
                side_raw = percentile_in_sector(scan, 30, 110, LIDAR_PCTL)
                side_dist = temporal_median(buf_side, side_raw)

                # [B] 후진 센터링/마무리용: 좌(90) 우(270) 근처
                left_raw  = percentile_in_sector(scan, 80, 100, LIDAR_PCTL)   # 약 90±10
                right_raw = percentile_in_sector(scan, 260, 280, LIDAR_PCTL)  # 약 270±10
                dist_90   = temporal_median(buf_left, left_raw)
                dist_270  = temporal_median(buf_right, right_raw)

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                msg = ""

                # =========================
                # PAUSE
                # =========================
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    msg = f"WAIT... ({pause_msg})"
                    if curr_time - pause_start > pause_duration:
                        state = next_state
                        state_timer = curr_time
                        valid_gap_count = 0
                        side_detect_time = 0.0
                        lidar.clean_input()

                # =========================
                # [1] SEARCH: 차1 -> 갭 -> 차2 찾기
                # =========================
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    cmd_servo = SERVO_CENTER

                    if search_step == STEP_FIND_CAR1:
                        msg = f"FIND CAR1 | side={int(side_dist)}"
                        if side_dist < CAR_EXIST_DIST:
                            print(f"🚗 1번 차 감지 ({int(side_dist)}mm)")
                            search_step = STEP_PASS_CAR1

                    elif search_step == STEP_PASS_CAR1:
                        # 벽타기 안정이 필요하면 SERVO_SLIGHT_RIGHT 사용
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"PASS CAR1 | side={int(side_dist)}"
                        if side_dist > EMPTY_SPACE_DIST:
                            valid_gap_count += 1
                            if valid_gap_count >= GAP_STREAK:
                                print(f"👀 빈공간(갭) 진입 ({int(side_dist)}mm)")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"FIND CAR2 | side={int(side_dist)}"
                        if side_dist < CAR_EXIST_DIST:
                            print(f"🛑 2번 차 감지 -> 후방주차 시퀀스 시작 ({int(side_dist)}mm)")
                            # 살짝 브레이크 느낌(선택)
                            ser.write(b"D,-150\n")
                            time.sleep(0.08)
                            ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state = STATE_SETUP_FORWARD
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Setup(Forward)"

                # =========================
                # [2] SETUP: 전진하며 공간 확보 (핸들 왼쪽 최대)
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
                # [3] REVERSE TURN: 핸들 오른쪽 최대 + 후진
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
                # [4] REVERSE STRAIGHT: 핸들 중앙 + 후진 (센터링 포함)
                # =========================
                elif state == STATE_REVERSE_STRAIGHT:
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = SERVO_CENTER
                        msg = "Align CENTER..."
                    else:
                        cmd_speed = -SPEED_REVERSE

                        # ✅ 폭이 타이트하므로 후진 중 미세 센터링
                        # dist_90(좌) - dist_270(우) 차이를 0으로 만들기
                        err = dist_90 - dist_270
                        if abs(err) < REVERSE_CENTER_DEADBAND:
                            delta = 0
                        else:
                            delta = int(clamp(REVERSE_CENTER_KP * err, -REVERSE_CENTER_LIMIT, REVERSE_CENTER_LIMIT))
                        cmd_servo = SERVO_CENTER + delta

                        msg = f"REV STRAIGHT | L={int(dist_90)} R={int(dist_270)} err={int(err)} dS={delta}"

                    # ---- 마무리 감지: 좌/우가 충분히 가까워짐(슬롯 안으로 들어가면 양옆 차가 가까워짐) ----
                    # 여기 기준은 트랙에 따라 달라서 튜닝 필요.
                    detected = (dist_90 < SIDE_STOP_DIST) or (dist_270 < SIDE_STOP_DIST)

                    if detected and (curr_time - state_timer) > STEER_WAIT_TIME:
                        if side_detect_time == 0.0:
                            print("✨ (라이다) 근접 감지 -> 1.5초 더 후진 후 정지")
                            side_detect_time = curr_time
                        elif (curr_time - side_detect_time) >= TIME_DELAY_STOP:
                            print("✅ 후진 주차 정지(지연 정지)")
                            ser.write(b"D,0\n")
                            state = STATE_US_CONFIRM if US_CONFIRM_ENABLE else STATE_PARKED
                            state_timer = curr_time

                    # 안전 타임아웃
                    if dt >= (STEER_WAIT_TIME + TIME_REVERSE_STRAIGHT_MAX):
                        print("✅ 후진 주차 정지(타임아웃)")
                        ser.write(b"D,0\n")
                        state = STATE_US_CONFIRM if US_CONFIRM_ENABLE else STATE_PARKED
                        state_timer = curr_time

                # =========================
                # [5] ✅ 2초 초음파 동시 인지 확인 단계 (LF,LR,RF,RR)
                # =========================
                elif state == STATE_US_CONFIRM:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "US CONFIRM 2s (LF,LR,RF,RR)"

                    print("⏱️ 초음파 2초 확인 시작 (LF,LR,RF,RR 동시 인지)")
                    passed, (ok, total) = ultrasonic_confirm_2s(ser)
                    print(f"📌 US confirm: passed={passed} ok/total={ok}/{total} th={US_SIDE_TH_MM}mm")

                    ser.write(b"D,0\n")
                    state = STATE_PARKED
                    state_timer = curr_time

                # =========================
                # [6] PARKED
                # =========================
                elif state == STATE_PARKED:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "PARKED ✅ (press q to quit)"
                    ser.write(b"D,0\n")

                # ---- 명령 송신 ----
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                # ---- 디버그 UI ----
                debug_img = np.zeros((380, 980, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | {msg}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f"side(30~110)={int(side_dist)}  L(90)={int(dist_90)}  R(270)={int(dist_270)}",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f"Car {CAR_W}x{CAR_L}  Slot {SLOT_W}x{SLOT_L}  SideClearEach {int(SIDE_CLEAR_EACH)}mm",
                            (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(debug_img, "q: quit", (10, 255),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
            ser.write(b"D,0\n")
            ser.close()
        except:
            pass
        try:
            lidar.stop()
            lidar.disconnect()
        except:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()