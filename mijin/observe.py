import cv2
import numpy as np
import math
import serial
import time

from rplidar import RPLidar  # pip install rplidar

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = True

# ---- 포트 설정(요청 반영) ----
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# ---- Camera (COM1 -> 보통 OpenCV index=1) ----
CAM_INDEX = 1   # ✅ 요청 반영
WIDTH, HEIGHT = 640, 480

# ---- Speed ----
SPEED_NORMAL = 120
SPEED_SLOW = 80
SPEED_STOP = 0

# ---- Servo PWM (✅ 왼쪽 480, 오른쪽 680) ----
SERVO_CENTER = 570
SERVO_LEFT_MAX = 480
SERVO_RIGHT_MAX = 680

# ---- ROI ----
ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

# ---- Auto Tuning ----
last_target_x = WIDTH // 2
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

if IS_SUNNY:
    print("☀️ 모드: SUNNY")
    current_l_min = 200
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX_VAL = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    print("🌙 모드: NORMAL")
    current_l_min = 140
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX_VAL = 80
    MORPH_SIZE = (3, 3)
    BLUR_K = 5


# ==========================================
# [2] 장애물/회피(정지 후) 설정 (단위: mm)
# ==========================================
ROAD_W_MM = 850
CAR_W_MM = 650
SAFETY_MM = 50
REQ_GAP_MM = CAR_W_MM + 2 * SAFETY_MM  # 통과에 필요한 최소 통로 폭

# ✅ 거리 보장(정지 후 회피가 안전거리 확보되도록)
STOP_TRIGGER_MM = 1400   # 이 거리 안이면 "정지 상태로 진입"
TURN_SAFE_MM = 1100      # 정지 후, 이 거리 이상일 때만 출발/차선변경
CRASH_GUARD_MM = 700     # 이 이하로 가까우면 절대 출발 금지(정지 유지)
CLEAR_MM = 1600          # 장애물 충분히 멀어지면 복귀 판단

STOP_BEFORE_AVOID_SEC = 0.6
LANECHANGE_RAMP_SEC = 0.7
SWITCH_COOLDOWN = 1.0

# 라인트레이싱 기반 차선 변경: target_x에 픽셀 오프셋을 걸어 "라인을 타면서" 옆차선으로 이동
DEFAULT_LANE_W_PX = 220
SHIFT_RATIO = 0.55  # lane width * 0.55 만큼 좌/우 이동(너무 크면 침범 위험)


# ==========================================
# [3] Arduino 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=0.1)
    print(f"✅ Arduino {ARDUINO_PORT} 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ Arduino 연결 실패: {e}")
    ser = None


# ==========================================
# [4] LiDAR 연결
# ==========================================
lidar = None
scan_gen = None
try:
    lidar = RPLidar(LIDAR_PORT)
    scan_gen = lidar.iter_scans()
    print(f"✅ LiDAR {LIDAR_PORT} 연결 성공!")
except Exception as e:
    print(f"⚠️ LiDAR 연결 실패(장애물 회피 비활성): {e}")
    lidar = None
    scan_gen = None


# ==========================================
# [5] 유틸/라인 함수
# ==========================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * ROI_HEIGHT_RATIO)
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    """기존 라인트레이싱 로직 그대로"""
    global last_target_x
    h, w = image.shape[:2]
    car_x = w / 2
    target_y = int(h * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (w * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (w * 0.25)
    else:
        target_x = last_target_x

    last_target_x = target_x
    dx = target_x - car_x
    dy = (h - target_y)
    angle = math.degrees(math.atan2(dx, abs(dy)))
    return angle, int(target_x)

def map_servo_from_angle(angle_deg):
    """
    angle_deg: 음수=왼쪽, 양수=오른쪽 기준
    """
    angle_deg = max(-45.0, min(45.0, angle_deg))
    if angle_deg < 0:
        return int(SERVO_CENTER + (SERVO_LEFT_MAX - SERVO_CENTER) * (abs(angle_deg) / 45.0))
    else:
        return int(SERVO_CENTER + (SERVO_RIGHT_MAX - SERVO_CENTER) * (angle_deg / 45.0))

def line_x_at_y(line, y):
    if line is None:
        return None
    x1, y1, x2, y2 = line[0]
    if y2 == y1:
        return None
    t = (y - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)

def estimate_lane_width_px(left_line, right_line, y_ref):
    xl = line_x_at_y(left_line, y_ref)
    xr = line_x_at_y(right_line, y_ref)
    if xl is None or xr is None:
        return DEFAULT_LANE_W_PX
    w = abs(xr - xl)
    return int(clamp(w, 140, 360))


# ==========================================
# [6] LiDAR 장애물 추정
# ==========================================
def extract_obstacle_lateral_range(scan, fwd_y_max=1800, ang_limit=35):
    """
    전방 포인트를 차량좌표로 변환:
    x = dist*sin(theta) (좌(-)/우(+))
    y = dist*cos(theta) (전방(+))
    """
    xs = []
    y_min = 1e9

    for (_, ang, dist) in scan:
        if not (200 < dist < 3000):
            continue

        theta = ang
        if theta > 180:
            theta -= 360
        if abs(theta) > ang_limit:
            continue

        th = math.radians(theta)
        x = dist * math.sin(th)
        y = dist * math.cos(th)

        if 0 < y < fwd_y_max:
            xs.append(x)
            y_min = min(y_min, y)

    if len(xs) < 10:
        return False, 0.0, 0.0, 9999.0

    xs.sort()
    k = max(1, int(len(xs) * 0.1))
    core = xs[k:len(xs)-k] if len(xs) > 2*k else xs
    return True, float(min(core)), float(max(core)), float(y_min)

def compute_gaps(obs_xmin, obs_xmax):
    road_left = -ROAD_W_MM / 2.0
    road_right = ROAD_W_MM / 2.0
    gap_left = obs_xmin - road_left
    gap_right = road_right - obs_xmax
    left_ok = gap_left >= REQ_GAP_MM
    right_ok = gap_right >= REQ_GAP_MM
    return gap_left, gap_right, left_ok, right_ok

def choose_avoid_lane(left_ok, right_ok):
    """
    장애물이 차선 중앙 가정:
    - 양쪽 가능: 왼쪽(1차선) 우선 (우측 실선 침범 리스크 ↓)
    - 한쪽만 가능: 가능한 쪽
    - 둘 다 불가: None
    """
    if left_ok and right_ok:
        return 1
    if left_ok:
        return 1
    if right_ok:
        return 2
    return None


# ==========================================
# [7] 장애물 회피 상태 머신
# ==========================================
FOLLOW = 0
STOP = 1
SHIFT = 2
PASS = 3
RETURN_STOP = 4
RETURN_SHIFT = 5

default_lane = 2   # 2차선 시작
current_lane = 2   # 2: 기본, 1: 왼쪽 회피

avoid_state = FOLLOW
state_ts = 0.0
last_switch_ts = 0.0

# 램프용(픽셀 오프셋)
last_shift_px = 0


# ==========================================
# [8] 메인
# ==========================================
def main():
    global current_l_min, avoid_state, state_ts, last_switch_ts, current_lane, last_shift_px

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(15, -6)

    if not cap.isOpened():
        print("❌ 카메라 오류 (인덱스 1이 아니면 0으로 바꿔보세요)")
        return

    print("\n🚀 3초 후 출발!")
    for i in range(3, 0, -1):
        print(f"{i}..")
        time.sleep(1)

    if ser:
        ser.write(f"D,{SPEED_NORMAL}\n".encode())

    last_serial_time = 0.0
    last_speed_time = 0.0

    # 최신 장애물 정보
    obs_found = False
    obs_xmin = obs_xmax = 0.0
    obs_ymin = 9999.0
    gap_left = gap_right = 0.0
    left_ok = right_ok = True

    try:
        while True:
            now = time.time()

            # ---------------------------
            # (0) Arduino RX 버퍼 비우기
            # ---------------------------
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            # ---------------------------
            # (1) LiDAR: scan 1개 업데이트
            # ---------------------------
            if scan_gen is not None:
                try:
                    scan = next(scan_gen)
                    obs_found, obs_xmin, obs_xmax, obs_ymin = extract_obstacle_lateral_range(scan)
                    if obs_found:
                        gap_left, gap_right, left_ok, right_ok = compute_gaps(obs_xmin, obs_xmax)
                    else:
                        left_ok = right_ok = True
                except StopIteration:
                    pass
                except Exception:
                    # 라이다 일시 오류 시 이전 값 유지
                    pass

            # ---------------------------
            # (2) Camera frame
            # ---------------------------
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != WIDTH:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

            h, w = frame.shape[:2]

            # ---------------------------
            # (3) 전처리 + 마스크
            # ---------------------------
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            lower_white = np.array([0, current_l_min, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            # ---------------------------
            # (4) ROI & Auto-tuning
            # ---------------------------
            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            roi_mask_poly = np.zeros_like(mask)
            cv2.fillPoly(roi_mask_poly, [roi_points], 255)
            roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

            white_count = cv2.countNonZero(roi_pixels)
            total_area = cv2.contourArea(roi_points) or 1
            ratio = white_count / total_area

            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)

            # ---------------------------
            # (5) 라인트레이싱(기본) 계산
            # ---------------------------
            edges = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges, roi_points)
            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

            left_line, right_line = average_slope_intercept(frame, lines)
            base_angle, base_target_x = calculate_steering_angle(frame, left_line, right_line)

            # 차선 폭 추정(픽셀)
            y_ref = int(h * ROI_HEIGHT_RATIO)
            lane_w_px = estimate_lane_width_px(left_line, right_line, y_ref)
            shift_px_goal = int(lane_w_px * SHIFT_RATIO)  # 왼쪽 차선으로 갈 때 목표 오프셋

            # ---------------------------
            # (6) 장애물 상태 머신(정지 후 회피)
            # ---------------------------
            status = "FOLLOW"
            final_speed = SPEED_NORMAL

            obstacle_logic_on = (scan_gen is not None)  # 라이다 연결되었을 때만

            obstacle_near = obstacle_logic_on and obs_found and (obs_ymin < STOP_TRIGGER_MM)
            obstacle_clear = (not obstacle_logic_on) or (not obs_found) or (obs_ymin > CLEAR_MM)

            if obstacle_logic_on:
                if avoid_state == FOLLOW:
                    if obstacle_near:
                        avoid_state = STOP
                        state_ts = now
                        status = "OBSTACLE -> STOP"

                elif avoid_state == STOP:
                    final_speed = SPEED_STOP
                    status = "STOP (BEFORE AVOID)"

                    too_close = obs_found and (obs_ymin < CRASH_GUARD_MM)
                    can_start_turn = (not obs_found) or (obs_ymin > TURN_SAFE_MM)

                    if too_close:
                        status = "TOO CLOSE -> HOLD STOP"
                    elif (now - state_ts >= STOP_BEFORE_AVOID_SEC) and can_start_turn:
                        next_lane = choose_avoid_lane(left_ok, right_ok)
                        if next_lane is None:
                            status = "NO GAP -> HOLD STOP"
                        else:
                            if (now - last_switch_ts > SWITCH_COOLDOWN):
                                current_lane = next_lane
                                avoid_state = SHIFT
                                state_ts = now
                                last_switch_ts = now
                                status = f"SHIFT -> LANE {current_lane}"

                elif avoid_state == SHIFT:
                    final_speed = SPEED_SLOW
                    status = "LANE CHANGE (RAMP)"
                    if now - state_ts >= LANECHANGE_RAMP_SEC:
                        avoid_state = PASS
                        state_ts = now
                        status = "PASS"

                elif avoid_state == PASS:
                    final_speed = SPEED_SLOW
                    status = "PASSING"
                    # 다음 장애물도 동일하게 정지-회피
                    if obstacle_near:
                        avoid_state = STOP
                        state_ts = now
                        status = "NEXT OBSTACLE -> STOP"
                    elif obstacle_clear and current_lane != default_lane:
                        avoid_state = RETURN_STOP
                        state_ts = now
                        status = "RETURN -> STOP"

                elif avoid_state == RETURN_STOP:
                    final_speed = SPEED_STOP
                    status = "STOP (BEFORE RETURN)"
                    if now - state_ts >= STOP_BEFORE_AVOID_SEC:
                        if now - last_switch_ts > SWITCH_COOLDOWN:
                            current_lane = default_lane
                            avoid_state = RETURN_SHIFT
                            state_ts = now
                            last_switch_ts = now
                            status = "RETURN SHIFT"

                elif avoid_state == RETURN_SHIFT:
                    final_speed = SPEED_SLOW
                    status = "RETURN (RAMP)"
                    if now - state_ts >= LANECHANGE_RAMP_SEC:
                        avoid_state = FOLLOW
                        state_ts = now
                        status = "FOLLOW"

            # ---------------------------
            # (7) 라인트레이싱 기반 목표점 보정(차선 변경 = target_x shift)
            # ---------------------------
            # current_lane=2(기본): shift=0
            # current_lane=1(왼쪽 회피): shift=-shift_px_goal
            desired_shift = 0
            if current_lane == 1:
                desired_shift = -shift_px_goal

            # 램프 상태에서만 부드럽게 shift 변화
            if avoid_state in (SHIFT, RETURN_SHIFT):
                alpha = clamp((now - state_ts) / max(1e-3, LANECHANGE_RAMP_SEC), 0.0, 1.0)
                shift_px = int(last_shift_px + (desired_shift - last_shift_px) * alpha)
            else:
                shift_px = desired_shift

            # shift 적용
            target_x = int(base_target_x + shift_px)
            target_x = clamp(target_x, 0, w - 1)

            # shift 갱신
            last_shift_px = shift_px

            # 새 angle 계산(기존 angle을 그대로 쓰면 shift 반영이 약함)
            # target_x 기준으로 새 steering angle 계산
            car_x = w / 2
            dy = (h - y_ref)
            dx = (target_x - car_x)
            final_angle = math.degrees(math.atan2(dx, abs(dy)))

            # 조향 큰 구간 감속(트랙 이탈 방지)
            steer_abs = abs(final_angle)
            if steer_abs > 20:
                final_speed = min(final_speed, 65)
            elif steer_abs > 12:
                final_speed = min(final_speed, 85)

            # ---------------------------
            # (8) Servo / Speed 송신
            # ---------------------------
            servo_val = map_servo_from_angle(final_angle)

            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # ---------------------------
            # (9) 디스플레이
            # ---------------------------
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)

            # base target / shifted target 표시
            cv2.circle(frame, (int(base_target_x), y_ref), 6, (0, 255, 0), -1)   # 기본 라인트레이싱 타겟(초록)
            cv2.circle(frame, (int(target_x), y_ref), 10, (0, 0, 255), -1)      # 실제 타겟(빨강)

            combined = np.hstack((frame, mask_bgr))

            cv2.putText(combined, f"L-Min:{current_l_min} Ratio:{ratio*100:.1f}%",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, f"State:{status} Lane:{current_lane} Shift:{shift_px}px",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if obstacle_logic_on:
                if obs_found:
                    cv2.putText(combined,
                                f"OBS y:{obs_ymin:.0f} gapL:{gap_left:.0f} gapR:{gap_right:.0f} okL:{left_ok} okR:{right_ok}",
                                (20, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                else:
                    cv2.putText(combined, "OBS: none", (20, HEIGHT - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            else:
                cv2.putText(combined, "LiDAR OFF (line-only)", (20, HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("LineTracing + ObstacleAvoid (Stop-Then-Shift)", combined)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지")
        try:
            if ser:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(f"S,{SERVO_CENTER}\n".encode())
                    time.sleep(0.05)
                ser.close()
        except:
            pass

        try:
            if lidar:
                lidar.stop()
                lidar.disconnect()
        except:
            pass

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()