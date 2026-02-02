import serial
from rplidar import RPLidar
import time
import cv2
import numpy as np
import math
from Function_Library import libCAMERA

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = True
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

# --- 통신 설정 ---
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# --- 속도 & 모터 설정 ---
SPEED_NORMAL = 100
SPEED_SLOW = 80
SPEED_STOP = 0

# ✅ 사용자 측정값 반영: 왼쪽 480, 오른쪽 680, 센터 570
SERVO_CENTER = 570
SERVO_LEFT_MAX = 480
SERVO_RIGHT_MAX = 680

# --- 장애물 회피/정지 설정 ---
OBST_FWD_Y_MAX = 1400          # 장애물 범위 추정 전방(mm)
OBST_DETECT_Y_MM = 900         # 이 거리 안이면 회피 동작 후보
OBST_CLEAR_Y_MM = 1300         # 이 거리 밖이면 클리어(히스테리시스)
STOP_BEFORE_AVOID_SEC = 0.6    # ✅ 회피 전 정지 시간
LANECHANGE_RAMP_SEC = 0.7      # 차선 변경 램프
SWITCH_COOLDOWN = 1.0          # 토글 쿨다운

# --- 치수(단위: mm) ---
ROAD_W_MM = 850    # 도로폭 85cm
CAR_W_MM  = 650    # 차체폭 65cm
OBS_W_MM  = 550    # 장애물 폭 55cm
SAFETY_MM = 50     # 안전여유(30~80 튜닝)
REQ_GAP_MM = CAR_W_MM + 2 * SAFETY_MM  # 통과에 필요한 최소 통로폭

# --- ROI 설정 ---
ROI_LANE_HEIGHT_RATIO = 0.6
ROI_TRAFFIC_HEIGHT = ROI_LANE_HEIGHT_RATIO - 0.05
ROI_LANE_X_LEFT = 0.3125
ROI_LANE_X_RIGHT = 0.6875

# --- 횡단보도(정지선) 설정 ---
CROSSWALK_RATIO_MIN = 0.30
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# --- 자동 튜닝 설정 (Auto-Tuning) ---
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

if IS_SUNNY:
    print("☀️ 모드: SUNNY (Auto-Tuning ON)")
    current_l_min = 200
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    print("🌙 모드: NORMAL (Auto-Tuning ON)")
    current_l_min = 140
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX = 80
    MORPH_SIZE = (3, 3)
    BLUR_K = 5


# ==========================================
# [2] 공통 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def angle_from_target_px(frame, target_px):
    h, w = frame.shape[:2]
    car_x = w / 2
    target_y = int(h * ROI_LANE_HEIGHT_RATIO)
    dx = target_px - car_x
    dy = (h - target_y)
    return math.degrees(math.atan2(dx, abs(dy)))

def map_servo(angle_deg):
    """
    angle_deg: 음수=왼쪽, 양수=오른쪽 기준
    (실차에서 반대면 angle_deg = -angle_deg 한 줄 추가)
    """
    angle_deg = max(-45.0, min(45.0, angle_deg))
    if angle_deg < 0:
        return int(SERVO_CENTER + (SERVO_LEFT_MAX - SERVO_CENTER) * (abs(angle_deg) / 45.0))
    else:
        return int(SERVO_CENTER + (SERVO_RIGHT_MAX - SERVO_CENTER) * (angle_deg / 45.0))

def extract_obstacle_lateral_range(scan, fwd_y_max=1400, ang_limit=35):
    """
    전방 포인트를 차량 좌표계로 변환해 장애물의 횡방향 x범위를 추정
    x<0: 좌, x>0: 우 (라이다 장착 방향이 반대면 부호 반대될 수 있음)
    """
    xs = []
    y_min = 1e9

    for (_, ang, dist) in scan:
        if not (200 < dist < 2500):
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
        return False, 0, 0, 9999

    xs.sort()
    k = max(1, int(len(xs) * 0.1))
    core = xs[k:len(xs) - k] if len(xs) > 2 * k else xs
    return True, min(core), max(core), y_min

def compute_gaps_from_obstacle(obs_xmin, obs_xmax):
    road_left_mm  = -ROAD_W_MM / 2.0
    road_right_mm =  ROAD_W_MM / 2.0
    gap_left  = obs_xmin - road_left_mm
    gap_right = road_right_mm - obs_xmax
    left_ok  = gap_left  >= REQ_GAP_MM
    right_ok = gap_right >= REQ_GAP_MM
    return gap_left, gap_right, left_ok, right_ok

def detect_stop_line(frame, roi_ratio=0.6):
    h, w = frame.shape[:2]
    roi_h = int(h * roi_ratio)
    roi = frame[roi_h:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=60, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            ang = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(ang) < 15:
                cv2.line(frame, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 255, 0), 3)
                return True
    return False

def pick_center_and_right_x(lines, w, y_ref):
    """
    중앙 점선(x_center) + 우측 실선(x_right) 선택
    """
    if lines is None:
        return None, None

    xs = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if y2 == y1:
            continue
        t = (y_ref - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        if 0 <= x <= w:
            if abs(x2 - x1) < 120:
                xs.append(float(x))

    if len(xs) < 2:
        return None, None

    x_center = min(xs, key=lambda v: abs(v - w * 0.5))
    right_pool = [v for v in xs if v > w * 0.62]
    if len(right_pool) == 0:
        return int(x_center), None
    x_right = max(right_pool)
    return int(x_center), int(x_right)

def compute_lane_targets_and_clamp(x_center, x_right):
    """
    x_center(중앙 점선), x_right(우측 실선)만으로 좌측 경계를 대칭 추정.
    우측 실선 침범 방지 위해 차폭+여유를 픽셀로 환산해 clamp 구성.
    """
    half_road_px = float(x_right - x_center)
    if half_road_px < 60:
        return None

    x_left = int(x_center - half_road_px)  # 대칭 가정

    mm_per_px = (ROAD_W_MM / 2.0) / half_road_px
    margin_px = int((CAR_W_MM / 2.0 + SAFETY_MM) / max(1e-6, mm_per_px))

    min_x = x_left + margin_px
    max_x = x_right - margin_px
    if min_x > max_x:
        mid = int((x_left + x_right) / 2)
        min_x = max_x = mid

    # "차선 중심"이라기보다, 중앙선 기준 좌/우로 살짝 치우친 안전 목표점
    lane2_target = int(x_center + half_road_px * 0.5)  # 2차선(오른쪽)
    lane1_target = int(x_center - half_road_px * 0.5)  # 1차선(왼쪽)

    lane1_target = max(min_x, min(max_x, lane1_target))
    lane2_target = max(min_x, min(max_x, lane2_target))

    return {
        "x_left": x_left, "x_center": x_center, "x_right": x_right,
        "min_x": min_x, "max_x": max_x,
        "lane1": lane1_target, "lane2": lane2_target,
        "half_road_px": half_road_px,
    }

def choose_avoid_lane(current_lane, left_ok, right_ok):
    """
    ✅ 장애물이 차선 중앙에 있다고 가정할 때의 최적 선택:
    - 양쪽 다 가능: 우측 실선 침범 리스크 때문에 '왼쪽(1차선)' 우선
    - 한쪽만 가능: 가능한 쪽
    - 둘 다 불가능: None
    """
    if left_ok and right_ok:
        return 1  # 왼쪽 우선
    if left_ok and (not right_ok):
        return 1
    if right_ok and (not left_ok):
        return 2
    return None


# ==========================================
# [3] 초기화
# ==========================================
lidar = None
ser = None
cam0 = None
video_writer = None

try:
    ser = serial.Serial(ARDUINO_PORT, 115200, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)

    cam0.set(3, 640)
    cam0.set(4, 480)
    cam0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cam0.isOpened():
        raise Exception("카메라 에러")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('mission_tuning.avi', fourcc, 20.0, (640, 480))

    print("✅ 준비 완료 (3초 대기)")
    time.sleep(3)
    if ser:
        ser.write(f"D,{SPEED_NORMAL}\n".encode())

except Exception as e:
    print(f"❌ 초기화 오류: {e}")
    exit()

# ==========================================
# [4] 상태 변수
# ==========================================
# 신호/횡단보도
is_crosswalk_stop = False
crosswalk_start_time = 0
crosswalk_cooldown_timer = 0
crosswalk_detect_timer = 0

# 차선/장애물 상태
LANE_1 = 1
LANE_2 = 2
default_lane = LANE_2
current_lane = default_lane

# 장애물 상태 머신(정지 후 회피)
AV_FOLLOW = 0
AV_STOP = 1
AV_SHIFT = 2
AV_PASS = 3
AV_RETURN_STOP = 4
AV_RETURN_SHIFT = 5

avoid_state = AV_FOLLOW
avoid_state_ts = 0.0
last_switch_time = 0.0

last_target_px = None

# 통신 타이밍
last_valid_angle = 0.0
last_serial_time = 0
last_speed_time = 0


# ==========================================
# [5] 메인 루프
# ==========================================
try:
    print("🚀 주행 시작 (장애물=차선 중앙, 정지 후 회피)")

    for scan in lidar.iter_scans():
        current_time = time.time()

        # ---- 라이다 장애물 범위 ----
        obs_found, obs_xmin, obs_xmax, obs_ymin = extract_obstacle_lateral_range(
            scan, fwd_y_max=OBST_FWD_Y_MAX, ang_limit=35
        )

        # ---- 카메라 ----
        ret, frame = cam0.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        blurred = cv2.medianBlur(frame, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        mask = cv2.inRange(
            hls,
            np.array([0, current_l_min, 0]),
            np.array([179, 255, S_MAX])
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

        # ---- 신호등 ROI ----
        traffic_roi_h = int(h * ROI_TRAFFIC_HEIGHT)
        traffic_frame = frame[0:traffic_roi_h, :]
        traffic_light = camera.object_detection(traffic_frame, sample=3, print_enable=False)

        # ---- 차선 ROI ----
        lane_roi_points = np.array([[
            (0, h), (w, h),
            (int(w * ROI_LANE_X_RIGHT), int(h * ROI_LANE_HEIGHT_RATIO)),
            (int(w * ROI_LANE_X_LEFT), int(h * ROI_LANE_HEIGHT_RATIO))
        ]], dtype=np.int32)

        # ---- 흰색 비율(오토튜닝/횡단보도) ----
        roi_mask_poly = np.zeros_like(mask)
        cv2.fillPoly(roi_mask_poly, [lane_roi_points], 255)
        roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

        white_count = cv2.countNonZero(roi_pixels)
        total_area = cv2.contourArea(lane_roi_points) or 1
        ratio = white_count / total_area

        # Auto-Tuning (횡단보도 아닐 때만)
        if ratio < CROSSWALK_RATIO_MIN:
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)

        # ==========================================================
        # (A) 신호/횡단보도 우선 로직
        # ==========================================================
        status_msg = "NORMAL"
        status_color = (0, 255, 0)
        final_speed = SPEED_NORMAL

        if is_crosswalk_stop:
            final_speed = SPEED_STOP
            elapsed = current_time - crosswalk_start_time

            if traffic_light == "GREEN":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "🟢 GREEN -> GO"
                status_color = (0, 255, 0)

            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "⚠️ TIMEOUT -> GO"
                status_color = (0, 255, 255)

            else:
                status_msg = f"WAIT GREEN.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        elif (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
            if detect_stop_line(frame, ROI_LANE_HEIGHT_RATIO):
                if crosswalk_detect_timer == 0:
                    crosswalk_detect_timer = current_time
                    status_msg = "Checking Line..."
                    status_color = (0, 255, 255)
                elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                    is_crosswalk_stop = True
                    crosswalk_start_time = current_time
                    final_speed = SPEED_STOP
                    status_msg = "STOP LINE DETECTED!"
                    status_color = (0, 0, 255)
                    crosswalk_detect_timer = 0
                else:
                    status_msg = "Checking Line..."
                    status_color = (0, 255, 255)
            else:
                crosswalk_detect_timer = 0
                if ratio > 0.35:
                    status_msg = "Glare Ignored"
                    status_color = (0, 255, 255)

        else:
            crosswalk_detect_timer = 0
            if traffic_light == "RED":
                final_speed = SPEED_STOP
                status_msg = "TRAFFIC RED"
                status_color = (0, 0, 255)

        # ==========================================================
        # (B) 라인 검출(센터 점선 + 우측 실선)
        # ==========================================================
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=120)

        y_ref = int(h * ROI_LANE_HEIGHT_RATIO)
        x_center, x_right = pick_center_and_right_x(lines, w, y_ref)
        lane_info = None
        if x_center is not None and x_right is not None:
            lane_info = compute_lane_targets_and_clamp(x_center, x_right)

        # 기본 target
        target_px = int(w / 2)
        if lane_info is not None:
            target_px = lane_info["lane2"] if current_lane == LANE_2 else lane_info["lane1"]
            target_px = max(lane_info["min_x"], min(lane_info["max_x"], target_px))

        # ==========================================================
        # (C) 장애물 회피(정지 후 회피) - 신호/횡단보도 다음 우선순위
        # ==========================================================
        obstacle_logic_allowed = (
            (final_speed != SPEED_STOP) and
            (not is_crosswalk_stop) and
            (traffic_light != "RED") and
            (lane_info is not None)
        )

        gap_left = gap_right = None
        left_ok = right_ok = True
        if obs_found:
            gap_left, gap_right, left_ok, right_ok = compute_gaps_from_obstacle(obs_xmin, obs_xmax)

        if lane_info is not None:
            obstacle_near = obs_found and (obs_ymin < OBST_DETECT_Y_MM)
            obstacle_clear = (not obs_found) or (obs_ymin > OBST_CLEAR_Y_MM)

            if obstacle_logic_allowed:
                if avoid_state == AV_FOLLOW:
                    # 내 진행경로 중앙에 장애물이 있다고 했으므로,
                    # "가까워지면 무조건 정지 진입" (단, 양쪽 통로가 둘 다 없으면 계속 정지)
                    if obstacle_near:
                        avoid_state = AV_STOP
                        avoid_state_ts = current_time
                        status_msg = "OBSTACLE -> STOP"
                        status_color = (0, 0, 255)

                elif avoid_state == AV_STOP:
                    final_speed = SPEED_STOP
                    status_msg = "STOP BEFORE AVOID"
                    status_color = (0, 0, 255)

                    if current_time - avoid_state_ts >= STOP_BEFORE_AVOID_SEC:
                        # ✅ 통로 가능하면: 왼쪽 우선으로 회피 차선 선택
                        next_lane = choose_avoid_lane(current_lane, left_ok, right_ok)

                        if (next_lane is not None) and (current_time - last_switch_time > SWITCH_COOLDOWN):
                            # 회피 차선으로 전환
                            current_lane = next_lane
                            last_switch_time = current_time
                            avoid_state = AV_SHIFT
                            avoid_state_ts = current_time
                            status_msg = f"SWITCH -> LANE {current_lane}"
                            status_color = (255, 255, 0)
                        else:
                            # 통로가 없으면 계속 정지 유지
                            status_msg = "NO GAP -> HOLD STOP"
                            status_color = (0, 0, 255)

                elif avoid_state == AV_SHIFT:
                    final_speed = min(final_speed, SPEED_SLOW)
                    status_msg = "LANE CHANGE (RAMP)"
                    status_color = (255, 255, 0)

                    if current_time - avoid_state_ts >= LANECHANGE_RAMP_SEC:
                        avoid_state = AV_PASS
                        avoid_state_ts = current_time

                elif avoid_state == AV_PASS:
                    final_speed = min(final_speed, SPEED_SLOW)
                    status_msg = "PASSING OBSTACLE"
                    status_color = (0, 255, 255)

                    # 장애물 사라지면 2차선 복귀를 시도하되,
                    # 복귀 시점에 다시 obstacle_near이면 복귀 보류(안정성)
                    if obstacle_clear:
                        if current_lane != default_lane:
                            # 복귀 전에 2차선 쪽이 가능한지(우측 통로) 한 번 더 체크
                            # (obs_found가 False면 당연히 가능)
                            can_return = (not obs_found) or right_ok
                            if can_return:
                                avoid_state = AV_RETURN_STOP
                                avoid_state_ts = current_time
                            else:
                                # 복귀 위험: 그대로 유지
                                status_msg = "RETURN BLOCKED -> HOLD"
                                status_color = (0, 255, 255)
                        else:
                            avoid_state = AV_FOLLOW
                            avoid_state_ts = current_time

                    # 주행 중 다시 obstacle_near가 뜨면(다른 장애물) 즉시 STOP로 재진입
                    if obstacle_near:
                        avoid_state = AV_STOP
                        avoid_state_ts = current_time
                        status_msg = "NEXT OBST -> STOP"
                        status_color = (0, 0, 255)

                elif avoid_state == AV_RETURN_STOP:
                    final_speed = SPEED_STOP
                    status_msg = "STOP BEFORE RETURN"
                    status_color = (0, 0, 255)

                    if current_time - avoid_state_ts >= STOP_BEFORE_AVOID_SEC:
                        if current_time - last_switch_time > SWITCH_COOLDOWN:
                            current_lane = default_lane
                            last_switch_time = current_time
                            avoid_state = AV_RETURN_SHIFT
                            avoid_state_ts = current_time

                elif avoid_state == AV_RETURN_SHIFT:
                    final_speed = min(final_speed, SPEED_SLOW)
                    status_msg = "RETURN TO LANE 2 (RAMP)"
                    status_color = (255, 255, 0)

                    if current_time - avoid_state_ts >= LANECHANGE_RAMP_SEC:
                        avoid_state = AV_FOLLOW
                        avoid_state_ts = current_time

            # ---- 목표점 램프 + clamp ----
            lane_target = lane_info["lane2"] if current_lane == LANE_2 else lane_info["lane1"]

            if last_target_px is None:
                last_target_px = lane_target

            if avoid_state in (AV_SHIFT, AV_RETURN_SHIFT):
                alpha = min(1.0, (current_time - avoid_state_ts) / max(1e-3, LANECHANGE_RAMP_SEC))
                target_px = int(last_target_px + (lane_target - last_target_px) * alpha)
            else:
                target_px = lane_target

            target_px = max(lane_info["min_x"], min(lane_info["max_x"], target_px))
            last_target_px = target_px

        # ==========================================================
        # (D) 조향 + 속도 안정화(큰 조향 감속)
        # ==========================================================
        final_angle = last_valid_angle
        if lane_info is not None:
            final_angle = angle_from_target_px(frame, target_px)
            last_valid_angle = final_angle

        steer_abs = abs(final_angle)
        if steer_abs > 20:
            final_speed = min(final_speed, 65)
        elif steer_abs > 12:
            final_speed = min(final_speed, 85)

        # ==========================================================
        # (E) 송신
        # ==========================================================
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                pwm_cmd = map_servo(final_angle)
                ser.write(f"S,{pwm_cmd}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # ==========================================================
        # (F) 디스플레이
        # ==========================================================
        cv2.polylines(frame, [lane_roi_points], True, (255, 0, 0), 2)
        cv2.line(frame, (0, traffic_roi_h), (w, traffic_roi_h), (100, 100, 100), 1)
        cv2.circle(frame, (int(target_px), y_ref), 10, (0, 0, 255), -1)

        tune_info = f"L-Min:{current_l_min} | Ratio:{ratio*100:.1f}%"
        cv2.putText(frame, tune_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Light:{traffic_light}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_msg, (20, traffic_roi_h + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if x_center is not None:
            cv2.circle(frame, (x_center, y_ref), 6, (255, 0, 255), -1)
        if x_right is not None:
            cv2.circle(frame, (x_right, y_ref), 6, (0, 255, 255), -1)

        if lane_info is not None:
            cv2.line(frame, (lane_info["min_x"], y_ref), (lane_info["min_x"], h), (0, 0, 255), 2)
            cv2.line(frame, (lane_info["max_x"], y_ref), (lane_info["max_x"], h), (0, 0, 255), 2)
            cv2.circle(frame, (lane_info["lane1"], y_ref), 6, (200, 50, 255), -1)
            cv2.circle(frame, (lane_info["lane2"], y_ref), 6, (50, 200, 255), -1)

        if obs_found:
            txt = f"OBS y:{obs_ymin:.0f} x:[{obs_xmin:.0f},{obs_xmax:.0f}] gapL:{gap_left:.0f} gapR:{gap_right:.0f}"
            cv2.putText(frame, txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Mission Auto-Tuning", frame)
        if video_writer is not None:
            video_writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    try:
        if lidar:
            lidar.stop()
            lidar.disconnect()
    except:
        pass

    try:
        if ser:
            ser.write(b"D,0\n")
            ser.write(f"S,{SERVO_CENTER}\n".encode())
            ser.close()
    except:
        pass

    try:
        if video_writer is not None:
            video_writer.release()
    except:
        pass

    try:
        if cam0:
            cam0.release()
    except:
        pass

    cv2.destroyAllWindows()