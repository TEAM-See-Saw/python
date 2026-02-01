import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque
import math

# ==========================================
# [1] PORT / BAUD
# ==========================================
PORT = "COM4"
LIDAR_PORT = "COM3"
SER_BAUD = 115200
SER_TIMEOUT = 0.05

# ==========================================
# [2] SPEED / SERVO
# ==========================================
SPEED_FORWARD = 70
SPEED_REVERSE = 70  # 전/후진 속도 같게 두는게 "시간=거리" 근사에 유리

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680

STEER_WAIT_TIME = 0.7

# 요청: 좌회전 1.5초
TIME_TURN_LEFT = 1.5

# 요청: 잠깐 1초 멈춤
TIME_STOP_1S = 1.0

# ==========================================
# [3] LIDAR ANGLE MAP (당신 환경)
#   - clockwise
#   - 0 front, 90 right, 270 left
# ==========================================
RIGHT_SEARCH_START = 55
RIGHT_SEARCH_END   = 125

# 차 폭(앞면) 60cm = 600mm (검증 필터)
CAR_FACE_WIDTH_MM = 600
CAR_WIDTH_TOL = 250   # 600 ± 250mm 정도로 넓게 허용

# 포인트 너무 적으면 무시(9999 오염 방지)
MIN_POINTS_VALID = 8

# 거리 추출 분위수: 낮을수록 가까운 물체에 민감
LIDAR_PCTL = 10

# baseline 학습(벽/기본 거리)
BASELINE_ALPHA = 0.03
CAR_DROP_MM = 650  # baseline보다 이만큼 가까우면 car 후보

# car_present 히스테리시스
CAR_STREAK_ON = 3
CAR_STREAK_OFF = 3

# 필터(temporal median)
class RobustMedian:
    def __init__(self, maxlen=5):
        self.buf = deque(maxlen=maxlen)
        self.last = 9999.0

    def update(self, v, n, n_min=8):
        if n >= n_min and v < 9000:
            self.buf.append(v)
            self.last = float(np.median(np.array(self.buf, dtype=np.float32)))
        return self.last

def in_sector(angle, start, end):
    angle = angle % 360
    start = start % 360
    end = end % 360
    if start <= end:
        return start <= angle <= end
    else:
        return (angle >= start) or (angle <= end)

def sector_points(scan, start_angle, end_angle):
    pts = []
    for (_, ang, dist) in scan:
        if dist <= 0:
            continue
        if in_sector(ang, start_angle, end_angle):
            pts.append((ang % 360, dist))
    return pts

def percentile_dist(pts, pctl=10):
    if not pts:
        return 9999.0
    arr = np.array([d for (_, d) in pts], dtype=np.float32)
    return float(np.percentile(arr, pctl))

def estimate_object_width_mm(pts, dist_ref):
    """
    오른쪽 섹터에서 '가까운 점들' 각도 범위를 이용해 물체 폭을 대략 추정.
    width ≈ 2 * dist_ref * sin(theta_span/2)
    """
    if len(pts) < MIN_POINTS_VALID:
        return None, None

    # 가까운 점들만 추리기: dist_ref보다 조금 큰 것까지 포함
    near = [(a, d) for (a, d) in pts if d < dist_ref + 250]
    if len(near) < MIN_POINTS_VALID:
        return None, None

    angles = [a for (a, _) in near]
    a_min = min(angles)
    a_max = max(angles)
    span_deg = a_max - a_min
    if span_deg <= 0:
        return None, None

    span_rad = math.radians(span_deg)
    width = 2.0 * float(dist_ref) * math.sin(span_rad / 2.0)
    return width, span_deg

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ==========================================
# [4] FSM STATES (시나리오 그대로)
# ==========================================
S_START_FORWARD      = 0
S_FIND_CAR1_FRONT    = 1
S_PASS_CAR1          = 2
S_FIND_CAR2_FRONT    = 3
S_REVERSE_HALF_GAP   = 4
S_STOP_1S            = 5
S_TURN_LEFT_1P5S     = 6
S_ALIGN_RIGHT_WAIT   = 7
S_REVERSE_RIGHT      = 8
S_DONE               = 9

STATE_NAME = {
    0: "START_FORWARD",
    1: "FIND_CAR1_FRONT",
    2: "PASS_CAR1 (find end edge)",
    3: "FIND_CAR2_FRONT (find start edge)",
    4: "REVERSE_HALF_GAP",
    5: "STOP_1S",
    6: "TURN_LEFT_1P5S",
    7: "ALIGN_RIGHT_WAIT",
    8: "REVERSE_RIGHT",
    9: "DONE",
}

def send_cmd(ser, servo, speed):
    ser.write(f"S,{int(servo)}\n".encode())
    ser.write(f"D,{int(speed)}\n".encode())

def main():
    cv2.namedWindow("Parking Monitor")

    ser = None
    lidar = None

    try:
        ser = serial.Serial(PORT, SER_BAUD, timeout=SER_TIMEOUT)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 연결 완료 (Serial + RPLidar)")
        time.sleep(1.0)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return

    # 초기 정지/센터
    send_cmd(ser, SERVO_CENTER, 0)
    time.sleep(0.5)

    # 오른쪽 섹터 필터
    f_right = RobustMedian(maxlen=5)

    baseline = None
    car_present = False
    on_cnt = 0
    off_cnt = 0

    # 이벤트 시간 기록
    t_car1_end = None   # (CAR1 RF로 간주) car_present True -> False
    t_car2_start = None # (CAR2 LF로 간주) car_present False -> True

    # 후진해야 할 시간
    rev_duration = 0.0
    rev_start = 0.0

    state = S_START_FORWARD
    state_t0 = time.time()

    running = True

    try:
        while running:
            for scan in lidar.iter_scans():
                now = time.time()

                # -------------------------
                # 오른쪽 섹터 데이터
                # -------------------------
                pts = sector_points(scan, RIGHT_SEARCH_START, RIGHT_SEARCH_END)
                n = len(pts)

                raw = percentile_dist(pts, LIDAR_PCTL)
                dist = f_right.update(raw, n, n_min=MIN_POINTS_VALID)

                # baseline 업데이트(차로 의심되는 너무 가까운 값에는 baseline을 안 따라가게)
                if n >= MIN_POINTS_VALID and raw < 9000:
                    if baseline is None:
                        baseline = raw
                    else:
                        # raw가 너무 가까우면(차) baseline 업데이트 보류
                        if raw > 1200:
                            baseline = (1 - BASELINE_ALPHA) * baseline + BASELINE_ALPHA * raw

                # car 후보 판단
                is_car_frame = False
                width_est = None
                span_deg = None

                if baseline is not None and n >= MIN_POINTS_VALID:
                    if raw < (baseline - CAR_DROP_MM):
                        # 폭(60cm) 검증 필터(선택)
                        width_est, span_deg = estimate_object_width_mm(pts, raw)
                        if width_est is None:
                            # 폭 계산 불가면 일단 car 후보로 둠(환경에 따라)
                            is_car_frame = True
                        else:
                            if abs(width_est - CAR_FACE_WIDTH_MM) <= CAR_WIDTH_TOL:
                                is_car_frame = True
                            else:
                                # 폭이 너무 다르면 벽/기타일 수 있음
                                is_car_frame = False

                # 히스테리시스(연속 프레임)
                prev_car = car_present
                if is_car_frame:
                    on_cnt += 1
                    off_cnt = 0
                else:
                    off_cnt += 1
                    on_cnt = 0

                if (not car_present) and (on_cnt >= CAR_STREAK_ON):
                    car_present = True
                if car_present and (off_cnt >= CAR_STREAK_OFF):
                    car_present = False

                # 이벤트 에지 검출
                edge_on = (not prev_car) and car_present   # False -> True
                edge_off = prev_car and (not car_present)  # True -> False

                # -------------------------
                # FSM: 시나리오 그대로
                # -------------------------
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = ""

                if state == S_START_FORWARD:
                    # 1. 시작하고 앞으로 직진
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_FORWARD
                    msg = "Go straight (start)"
                    # 바로 다음 상태로
                    state = S_FIND_CAR1_FRONT
                    state_t0 = now

                elif state == S_FIND_CAR1_FRONT:
                    # 2~3. 직진하면서 오른쪽 차(1번) 확인 (폭 60cm 근처)
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_FORWARD
                    msg = "Find CAR1 front on right"

                    if edge_on:
                        print("🚗 CAR1 발견(edge ON)")
                        state = S_PASS_CAR1
                        state_t0 = now

                elif state == S_PASS_CAR1:
                    # 3. CAR1을 확인하고 지나감 => CAR1 끝 경계(edge OFF) 찾기
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_FORWARD
                    msg = "Passing CAR1, wait for end edge (OFF)"

                    if edge_off:
                        t_car1_end = now
                        print(f"✅ CAR1 끝 경계 감지 (t={t_car1_end:.2f})")
                        state = S_FIND_CAR2_FRONT
                        state_t0 = now

                elif state == S_FIND_CAR2_FRONT:
                    # 4. 계속 직진하며 CAR2 시작(edge ON) 찾기
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_FORWARD
                    msg = "Find CAR2 front edge (ON)"

                    if edge_on and t_car1_end is not None:
                        t_car2_start = now
                        dt_gap = max(0.0, t_car2_start - t_car1_end)
                        rev_duration = dt_gap / 2.0

                        print(f"🛑 CAR2 시작 경계 감지 (t={t_car2_start:.2f}) dt_gap={dt_gap:.2f}s => reverse {rev_duration:.2f}s")

                        # 5. 그 거리의 반만큼 뒤로 후진
                        state = S_REVERSE_HALF_GAP
                        rev_start = now
                        state_t0 = now

                elif state == S_REVERSE_HALF_GAP:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = -SPEED_REVERSE
                    t = now - rev_start
                    msg = f"Reverse half gap {t:.2f}/{rev_duration:.2f}s"

                    if t >= rev_duration:
                        cmd_speed = 0
                        send_cmd(ser, SERVO_CENTER, 0)
                        # 5. 후진 후 1초 정지
                        state = S_STOP_1S
                        state_t0 = now

                elif state == S_STOP_1S:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = 0
                    msg = f"Stop {now - state_t0:.2f}/{TIME_STOP_1S:.2f}s"
                    if (now - state_t0) >= TIME_STOP_1S:
                        # 6. 왼쪽으로 꺾어서 좌회전 1.5초
                        state = S_TURN_LEFT_1P5S
                        state_t0 = now

                elif state == S_TURN_LEFT_1P5S:
                    cmd_servo = SERVO_LEFT_MAX
                    cmd_speed = SPEED_FORWARD
                    msg = f"Turn LEFT forward {now - state_t0:.2f}/{TIME_TURN_LEFT:.2f}s"
                    if (now - state_t0) >= TIME_TURN_LEFT:
                        # 7. 오른쪽으로 다 꺾고(정렬시간) 후진
                        state = S_ALIGN_RIGHT_WAIT
                        state_t0 = now

                elif state == S_ALIGN_RIGHT_WAIT:
                    cmd_servo = SERVO_RIGHT_MAX
                    cmd_speed = 0
                    msg = f"Align RIGHT wait {now - state_t0:.2f}/{STEER_WAIT_TIME:.2f}s"
                    if (now - state_t0) >= STEER_WAIT_TIME:
                        state = S_REVERSE_RIGHT
                        state_t0 = now

                elif state == S_REVERSE_RIGHT:
                    cmd_servo = SERVO_RIGHT_MAX
                    cmd_speed = -SPEED_REVERSE
                    msg = "Reverse with RIGHT max (parking entry)"
                    # 여기서부터는 “후진 주차 마무리 로직”을 추가로 붙이면 됨
                    # 일단 데모로 3초만 후진하고 종료
                    if (now - state_t0) >= 3.0:
                        state = S_DONE
                        state_t0 = now

                elif state == S_DONE:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = 0
                    msg = "DONE (press q)"
                    # 정지 유지

                # 명령 송신
                send_cmd(ser, cmd_servo, cmd_speed)

                # -------------------------
                # 유저모드 UI
                # -------------------------
                ui = np.zeros((520, 1280, 3), dtype=np.uint8)

                b = -1 if baseline is None else int(baseline)
                cv2.putText(ui, f"STATE: {STATE_NAME[state]}", (20, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

                cv2.putText(ui, msg, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                w_txt = "None" if width_est is None else f"{int(width_est)}mm (span {span_deg:.1f}deg)"
                cv2.putText(ui,
                            f"RIGHT sector({RIGHT_SEARCH_START}~{RIGHT_SEARCH_END}) raw={int(raw)} filt={int(dist)} n={n} baseline={b}",
                            (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                cv2.putText(ui,
                            f"car_present={car_present} is_car_frame={is_car_frame} width_est={w_txt}",
                            (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                t1 = -1 if t_car1_end is None else t_car1_end
                t2 = -1 if t_car2_start is None else t_car2_start
                cv2.putText(ui,
                            f"t_car1_end={t1:.2f}  t_car2_start={t2:.2f}  rev_duration={rev_duration:.2f}s",
                            (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                cv2.putText(ui,
                            "Angle map: clockwise | 0 front | 90 right | 270 left",
                            (20, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

                cv2.putText(ui, "q: quit", (20, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", ui)
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
                send_cmd(ser, SERVO_CENTER, 0)
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
