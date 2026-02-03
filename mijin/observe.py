import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = True

PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 0.4  # (속도 반영 지연 줄이고 싶으면 0.25~0.4로)

CAM_INDEX = 1
CAM_INDEX_TRAFFIC = 0

# ✅ 기본(크루즈) 속도 = 120
MAX_SPEED = 120

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# ✅ ROI는 그대로 유지 (절대 변경 X)
ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

last_target_x = 320
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
# ✅ [어두움 보완] (ROI 불변)
# ==========================================
DARK_L_MEAN_TH = 85
DARK_S_MAX_VAL = 90
DARK_MIN_L_VAL = 70

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

dark_state = False
dark_enter_cnt = 0
dark_exit_cnt = 0
DARK_ENTER_FRAMES = 3
DARK_EXIT_FRAMES = 6

# ==========================================
# [라이다/장애물 회피]
# ==========================================
LIDAR_PORT = 'COM3'

OBSTACLE_PREP_DIST  = 1600  # mm
OBSTACLE_START_DIST = 1500  # mm
OBSTACLE_CLEAR_TIME = 1.2

LIDAR_FRONT_DEG = 60
SHIFT_GAIN = 2.1

# ==========================================
# ✅ [장애물 감속 정책]  (기본 120 기준으로 재조정)
# ==========================================
# PREP 구간(<=160cm): 120보다 살짝 낮게
SPEED_CAP_PREP_MAX = 105
# AVOID 구간(<=150cm): 더 낮게
SPEED_CAP_AVOID_MAX = 90
# CRIT 구간(<=90cm): 안전 최저
SPEED_CAP_CRIT_MAX = 75

# ✅ 곡선/조향각 기반 감속 (120 기준)
ANGLE_SLOW_1 = 14
ANGLE_SLOW_2 = 20
ANGLE_SLOW_3 = 28
CAP_A1 = 110
CAP_A2 = 95
CAP_A3 = 80

# ✅ 곡선 진입(각도 변화량) 감속 (120 기준)
ANGLE_DDELTA_1 = 6.0
ANGLE_DDELTA_2 = 10.0
CAP_DA1 = 95
CAP_DA2 = 80

# ✅ 차선 신뢰도 낮을 때 감속 (120 기준)
CAP_LOWCONF = 90
LOWCONF_HOLD_FRAMES = 6

# ✅ 서보 레이트 리미트
SERVO_RATE_LIMIT = 16

# ==========================================
# [횡단보도/정지선]
# ==========================================
CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# ==========================================
# [신호등 ROI]
# ==========================================
TRAFFIC_BOX_X1 = 0.22
TRAFFIC_BOX_X2 = 0.62
TRAFFIC_BOX_Y1 = 0.35
TRAFFIC_BOX_Y2 = 0.62
TRAFFIC_HIGHLIGHT_PCTL = 98
TRAFFIC_TH_MIN = 200

# ==========================================
# [2] 시리얼/라이다/카메라 연결
# ==========================================
ser = None
lidar = None
cap_lane = None
cap_traffic = None

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (1초 대기)")
    time.sleep(1)
except Exception as e:
    print(f"❌ 시리얼 연결 실패: {e}")
    ser = None

try:
    lidar = RPLidar(LIDAR_PORT)
except Exception as e:
    print(f"❌ 라이다 초기화 실패: {e}")
    lidar = None

cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)

width = 640
height = 480

cap_lane.set(3, width)
cap_lane.set(4, height)
cap_lane.set(15, -6)

cap_traffic.set(3, width)
cap_traffic.set(4, height)
cap_traffic.set(15, -6)

if not cap_lane.isOpened():
    print("❌ 차선 카메라 오류 (CAM_INDEX 확인)")
if not cap_traffic.isOpened():
    print("❌ 신호등 카메라 오류 (CAM_INDEX_TRAFFIC 확인)")

# ==========================================
# [3] 영상 처리 함수들
# ==========================================
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

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 60:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                if length < 90:
                    continue
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def map_value(x, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return out_min
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def detect_stop_line(mask, frame_to_draw, roi_ratio=0.6):
    h, w = mask.shape[:2]
    roi_h = int(h * roi_ratio)
    roi = mask[roi_h:h, 0:w]

    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=40, maxLineGap=20)

    detected = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            ang = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(ang) < 30:
                cv2.line(frame_to_draw, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 255, 255), 3)
                detected = True
    return detected

# ==========================================
# ✅ 신호등 인식(원본 유지)
# ==========================================
def detect_traffic_lr_robust(frame_bgr):
    H, W = frame_bgr.shape[:2]

    x1 = int(W * TRAFFIC_BOX_X1)
    x2 = int(W * TRAFFIC_BOX_X2)
    y1 = int(H * TRAFFIC_BOX_Y1)
    y2 = int(H * TRAFFIC_BOX_Y2)

    x1 = max(0, min(W - 2, x1))
    x2 = max(x1 + 1, min(W - 1, x2))
    y1 = max(0, min(H - 2, y1))
    y2 = max(y1 + 1, min(H - 1, y2))

    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 5 or rw < 5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "ROI_TOO_SMALL"}

    third = max(1, rw // 3)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    S_GATE = 60
    V_GATE = 80
    COLOR_TH = 0.003

    Ss = hsv[:, :, 1]
    sat_mask = (Ss >= S_GATE).astype(np.uint8) * 255

    red1 = cv2.inRange(hsv, (0,   S_GATE, V_GATE), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, S_GATE, V_GATE), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, (35, S_GATE, V_GATE), (85, 255, 255))

    red_mask = cv2.bitwise_and(red_mask, sat_mask)
    green_mask = cv2.bitwise_and(green_mask, sat_mask)

    k = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)

    red_left = red_mask[:, 0:third]
    green_right = green_mask[:, 2*third:rw] if (2*third) < rw else green_mask[:, third:rw]

    red_left_ratio = cv2.countNonZero(red_left) / float(red_left.size)
    green_right_ratio = cv2.countNonZero(green_right) / float(green_right.size)

    if red_left_ratio > COLOR_TH and green_right_ratio < COLOR_TH:
        return "LEFT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR",
                        "red_left": red_left_ratio, "green_right": green_right_ratio}
    if green_right_ratio > COLOR_TH and red_left_ratio < COLOR_TH:
        return "RIGHT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR",
                         "red_left": red_left_ratio, "green_right": green_right_ratio}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    p = np.percentile(gray, TRAFFIC_HIGHLIGHT_PCTL)
    thr = int(max(TRAFFIC_TH_MIN, p))
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_NONE", "thr": thr,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 30:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_SMALL", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    roi_area = float(rh * rw)
    if area > roi_area * 0.25:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_TOO_BIG", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    x, y, ww, hh = cv2.boundingRect(c)
    if ww > hh * 2.5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_WIDE", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    M = cv2.moments(c)
    if M["m00"] == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_BADMOM", "thr": thr,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    cx = int(M["m10"] / M["m00"])
    if cx < third:
        state = "LEFT"
    elif cx > 2 * third:
        state = "RIGHT"
    else:
        state = "NONE"

    return state, {"box": (x1, y1, x2, y2), "mode": "BRIGHT_BLOB", "thr": thr,
                   "cx": cx, "area": area, "red_left": red_left_ratio, "green_right": green_right_ratio}

# ==========================================
# ✅ 라이다: 좌/우 전방 섹터 최소거리
# ==========================================
def get_front_lr_min_dist(scan, front_deg=60, dist_min=150, dist_max=2500):
    if scan is None:
        return 2000, 2000

    left_min = 2000
    right_min = 2000
    lo = 360 - front_deg
    hi = front_deg

    for (_, ang, dist) in scan:
        if dist_min < dist < dist_max:
            if 0 <= ang <= hi:
                left_min = min(left_min, dist)
            elif lo <= ang <= 360:
                right_min = min(right_min, dist)

    return left_min, right_min

# ==========================================
# ✅ 우측 실선 품질 평가(right_q)
# ==========================================
def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

prev_right_x_for_q = None

def right_line_quality(right_line, img_w):
    global prev_right_x_for_q

    if right_line is None:
        prev_right_x_for_q = None
        return 0.0, None, 0.0, 0.0, 999.0

    x1, y1, x2, y2 = right_line[0]
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = math.hypot(dx, dy)

    if abs(dx) < 1e-3:
        slope = 999.0
    else:
        slope = dy / dx

    xR = float(x2)

    q_len = clamp01((length - 70.0) / (110.0))
    s = abs(slope)
    q_slope = clamp01((s - 0.6) / (1.4))

    if prev_right_x_for_q is None:
        xjump = 0.0
        q_jump = 1.0
    else:
        xjump = abs(xR - prev_right_x_for_q)
        q_jump = clamp01((40.0 - xjump) / 28.0)

    prev_right_x_for_q = xR

    edge_pen = 1.0
    if xR < 40 or xR > (img_w - 40):
        edge_pen = 0.6

    q = (0.45 * q_len + 0.35 * q_slope + 0.20 * q_jump) * edge_pen
    return float(clamp01(q)), xR, float(slope), float(length), float(xjump)

# ==========================================
# ✅ 2차선 유지: 우측 실선 앵커 방식(중앙선 점선 무시)
# ==========================================
lane_width_px = 260.0
LANE_W_MIN = 160.0
LANE_W_MAX = 420.0
LANE_W_EMA = 0.08

last_right_x = None
lost_right_count = 0

RIGHT_MARGIN_PX = 18
INNER_MARGIN_PX = 18
TARGET_SMOOTH_A = 0.82
TARGET_MAX_STEP = 28

def compute_lane2_target_right_anchor(img_w, left_line, right_line, prev_target_x, right_q):
    global lane_width_px, last_right_x, lost_right_count

    xR = right_line[0][2] if right_line is not None else None
    xL = left_line[0][2] if left_line is not None else None

    right_valid = False
    if xR is not None and 0 <= xR <= img_w:
        last_right_x = float(xR)
        lost_right_count = 0
        right_valid = True
    else:
        lost_right_count += 1
        if last_right_x is not None:
            xR = last_right_x
        else:
            return int(prev_target_x), None, int(lane_width_px), 0, img_w - 1, 0.0, False

    if (xL is not None) and (xR is not None):
        w_est = float(xR - xL)
        if LANE_W_MIN <= w_est <= LANE_W_MAX:
            lane_width_px = (1 - LANE_W_EMA) * lane_width_px + (LANE_W_EMA) * w_est

    lane2_center = float(xR) - (lane_width_px / 2.0)

    max_x = float(xR) - RIGHT_MARGIN_PX
    min_x = max_x - lane_width_px + INNER_MARGIN_PX

    min_x = max(0.0, min(float(img_w - 1), min_x))
    max_x = max(0.0, min(float(img_w - 1), max_x))
    if min_x > max_x:
        min_x, max_x = max_x, min_x

    lane2_center = max(min_x, min(max_x, lane2_center))

    a = TARGET_SMOOTH_A
    smooth = a * float(prev_target_x) + (1 - a) * lane2_center

    step = smooth - float(prev_target_x)
    if abs(step) > TARGET_MAX_STEP:
        smooth = float(prev_target_x) + (TARGET_MAX_STEP if step > 0 else -TARGET_MAX_STEP)

    if right_valid:
        base_conf = 1.0
    else:
        base_conf = max(0.0, 1.0 - (lost_right_count / float(LOWCONF_HOLD_FRAMES)))

    lane_conf = clamp01(base_conf * (0.35 + 0.65 * right_q))

    return int(smooth), int(xR), int(lane_width_px), int(min_x), int(max_x), float(lane_conf), right_valid

def calculate_angle_from_target(image, target_x):
    height, width = image.shape[:2]
    car_x = width / 2.0
    target_y = int(height * ROI_HEIGHT_RATIO)
    dx = target_x - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy)))

def apply_obstacle_shift(target_x, obstacle_dist, avoid_direction):
    if avoid_direction == 0:
        return target_x, 0
    calc_dist = min(obstacle_dist, OBSTACLE_START_DIST)
    d = max(0.0, float(OBSTACLE_START_DIST - calc_dist))
    shift_amount = (d ** 1.25) * (SHIFT_GAIN / (OBSTACLE_START_DIST ** 0.25))
    shifted = target_x - shift_amount
    return int(shifted), int(shift_amount)

def speed_cap_by_angle(angle_abs):
    if angle_abs > ANGLE_SLOW_3:
        return CAP_A3
    if angle_abs > ANGLE_SLOW_2:
        return CAP_A2
    if angle_abs > ANGLE_SLOW_1:
        return CAP_A1
    return MAX_SPEED

def speed_cap_by_obstacle(dist_min):
    if dist_min <= 900:
        return SPEED_CAP_CRIT_MAX
    if dist_min <= OBSTACLE_START_DIST:
        return SPEED_CAP_AVOID_MAX
    if dist_min <= OBSTACLE_PREP_DIST:
        return SPEED_CAP_PREP_MAX
    return MAX_SPEED

def speed_cap_by_angle_delta(dang):
    if dang > ANGLE_DDELTA_2:
        return CAP_DA2
    if dang > ANGLE_DDELTA_1:
        return CAP_DA1
    return MAX_SPEED

def clamp_servo_rate(new_servo, prev_servo, limit):
    if prev_servo is None:
        return new_servo
    d = new_servo - prev_servo
    if abs(d) <= limit:
        return new_servo
    return int(prev_servo + (limit if d > 0 else -limit))

# ==========================================
# [4] 메인 루프 변수
# ==========================================
is_crosswalk_stop = False
crosswalk_start_time = 0.0
crosswalk_cooldown_timer = 0.0
crosswalk_detect_timer = 0.0

obstacle_count = 0
avoid_active = False
avoid_hold_until = 0.0
AVOID_HOLD_SEC = 0.6

obstacle_last_seen_time = 0.0
obs_clear_finished_time = 0.0

last_serial_time = 0.0
last_speed_time = 0.0

prev_angle = 0.0
prev_servo = None

try:
    if ser:
        # ✅ 시작 기본 속도도 120으로
        ser.write(f"D,{MAX_SPEED}\n".encode())

    print("🚀 주행 시작 (기본속도=120 / 감속 정책 120 기준 재조정)")

    scan_iter = lidar.iter_scans() if lidar is not None else [None] * 10**9

    for scan in scan_iter:
        if ser:
            try:
                if ser.in_waiting > 0:
                    ser.read(ser.in_waiting)
            except:
                pass

        current_time = time.time()

        # (1) 라이다 거리
        left_min, right_min = get_front_lr_min_dist(scan, front_deg=LIDAR_FRONT_DEG)
        raw_dist = min(left_min, right_min)

        # (2) 카메라
        ret_l, frame_lane = cap_lane.read()
        ret_t, frame_traffic = cap_traffic.read()
        if not ret_l or not ret_t:
            print("❌ 카메라 신호 끊김")
            break

        if frame_lane.shape[1] != width:
            frame_lane = cv2.resize(frame_lane, (width, height))
        if frame_traffic.shape[1] != width:
            frame_traffic = cv2.resize(frame_traffic, (width, height))

        h, w = frame_lane.shape[:2]

        # (3) 신호등
        traffic_state, tdbg = detect_traffic_lr_robust(frame_traffic)

        # (4) 차선 마스크 (어두움 보완, ROI 불변)
        blurred = cv2.medianBlur(frame_lane, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        roi_points = np.array([[
            (0, h), (w, h),
            (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
            (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
        ]], dtype=np.int32)

        roi_mask_poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask_poly, [roi_points], 255)

        L = hls[:, :, 1]
        roi_L = L[roi_mask_poly == 255]
        mean_L = float(np.mean(roi_L)) if roi_L.size > 0 else 255.0

        if mean_L < DARK_L_MEAN_TH:
            dark_enter_cnt += 1
            dark_exit_cnt = 0
        else:
            dark_exit_cnt += 1
            dark_enter_cnt = 0

        if (not dark_state) and (dark_enter_cnt >= DARK_ENTER_FRAMES):
            dark_state = True
        if dark_state and (dark_exit_cnt >= DARK_EXIT_FRAMES):
            dark_state = False

        if dark_state:
            L_eq = clahe.apply(L)
            hls[:, :, 1] = L_eq
            effective_min_l = max(DARK_MIN_L_VAL, MIN_L_VAL)
            effective_s_max = DARK_S_MAX_VAL
        else:
            effective_min_l = MIN_L_VAL
            effective_s_max = S_MAX_VAL

        lower_white = np.array([0, current_l_min, 0])
        upper_white = np.array([179, 255, effective_s_max])
        mask = cv2.inRange(hls, lower_white, upper_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)
        white_count = cv2.countNonZero(roi_pixels)
        total_area = cv2.contourArea(roi_points) or 1
        ratio = white_count / total_area

        if ratio > TARGET_RATIO_MAX:
            current_l_min = min(current_l_min + 2, MAX_L_VAL)
        elif ratio < TARGET_RATIO_MIN:
            if dark_state or ratio < 0.010:
                current_l_min = max(current_l_min - 8, effective_min_l)
            else:
                current_l_min = max(current_l_min - 4, effective_min_l)

        current_l_min = int(max(effective_min_l, min(MAX_L_VAL, current_l_min)))

        # (5) 라인 검출
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame_lane, lines)

        right_q, rq_xR, rq_slope, rq_len, rq_jump = right_line_quality(right, w)

        # (6) 2차선 목표점
        lane2_target, right_x, road_width_px, min_x, max_x, lane_conf, right_valid = compute_lane2_target_right_anchor(
            w, left, right, last_target_x, right_q
        )
        last_target_x = lane2_target

        # (7) 장애물 판단(2차선 장애물만 회피)
        status_msg = "LANE2 HOLD"
        status_color = (0, 255, 0)

        obstacle_seen_start = raw_dist < OBSTACLE_START_DIST

        right_obstacle = (
            (right_min < OBSTACLE_START_DIST) and (
                (left_min - right_min > 200) or (left_min > OBSTACLE_PREP_DIST)
            )
        )

        avoid_direction = 0
        if right_obstacle and (current_time - obs_clear_finished_time > 0.8):
            avoid_active = True
            avoid_hold_until = current_time + AVOID_HOLD_SEC
            obstacle_last_seen_time = current_time
            obstacle_count += 1

        if avoid_active:
            if current_time <= avoid_hold_until or obstacle_seen_start:
                avoid_direction = -1
                obstacle_last_seen_time = current_time
                status_msg = f"AVOID LEFT (lane2 obs) #{obstacle_count}"
                status_color = (0, 255, 255)
            else:
                if current_time - obstacle_last_seen_time > OBSTACLE_CLEAR_TIME:
                    avoid_active = False
                    obs_clear_finished_time = current_time
                    status_msg = "OBS END -> LANE2 HOLD"
                    status_color = (0, 255, 0)

        # (8) 최종 목표 + 클램프
        final_target, shift_px = apply_obstacle_shift(lane2_target, raw_dist, avoid_direction)
        final_target = max(min_x, min(max_x, final_target))

        angle = calculate_angle_from_target(frame_lane, final_target)
        angle_abs = abs(angle)
        dang = abs(angle - prev_angle)
        prev_angle = angle

        raw_servo = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
        servo_val = clamp_servo_rate(raw_servo, prev_servo, SERVO_RATE_LIMIT)
        prev_servo = servo_val

        # (9) ✅ 속도 정책(기본 120에서 cap로만 줄임)
        final_speed = MAX_SPEED
        final_speed = min(final_speed, speed_cap_by_angle(angle_abs))
        final_speed = min(final_speed, speed_cap_by_angle_delta(dang))
        final_speed = min(final_speed, speed_cap_by_obstacle(raw_dist))

        if lane_conf < 0.6:
            final_speed = min(final_speed, CAP_LOWCONF)

        if avoid_direction != 0:
            final_speed = min(final_speed, SPEED_CAP_AVOID_MAX)

        # (10) 정지선 + 신호등
        if is_crosswalk_stop:
            final_speed = 0
            elapsed = current_time - crosswalk_start_time

            if traffic_state == "RIGHT":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (RIGHT ON)"
                status_color = (0, 255, 0)

            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (TIMEOUT)"
                status_color = (0, 255, 0)

            else:
                status_msg = f"WAIT RIGHT.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        else:
            if (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                if detect_stop_line(mask, mask_bgr, ROI_HEIGHT_RATIO):
                    if traffic_state == "LEFT":
                        if crosswalk_detect_timer == 0:
                            crosswalk_detect_timer = current_time
                        elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                            is_crosswalk_stop = True
                            crosswalk_start_time = current_time
                            final_speed = 0
                            status_msg = "STOP! (Line+LEFT ON)"
                            status_color = (0, 0, 255)
                            crosswalk_detect_timer = 0
                        else:
                            status_msg = "Checking Line+LEFT..."
                            status_color = (0, 255, 255)
                    else:
                        crosswalk_detect_timer = 0
                        status_msg = f"Line Detected ({traffic_state})"
                        status_color = (0, 255, 255)
                else:
                    crosswalk_detect_timer = 0
            else:
                crosswalk_detect_timer = 0

        # (11) 통신
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # (12) 디버그 표시
        box = tdbg.get("box", None)
        mode = tdbg.get("mode", "-")
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_traffic, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)

        y_guide = int(h * ROI_HEIGHT_RATIO)
        if right_x is not None:
            cv2.line(mask_bgr, (right_x, y_guide - 25), (right_x, y_guide + 25), (255, 255, 255), 2)

        cv2.circle(mask_bgr, (lane2_target, y_guide), 9, (255, 0, 0), -1)
        cv2.circle(mask_bgr, (final_target, y_guide), 10, (0, 0, 255), -1)

        cv2.putText(mask_bgr,
                    f"Conf:{lane_conf:.2f} rightQ:{right_q:.2f} rLen:{rq_len:.0f} rJump:{rq_jump:.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 2)
        cv2.putText(mask_bgr,
                    f"Dist:{raw_dist} L:{left_min} R:{right_min} | Angle:{angle:.1f} Speed:{final_speed} | {status_msg}",
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2)

        combined = np.hstack((frame_traffic, mask_bgr))
        cv2.imshow("Dual View (Traffic + LaneMask)", combined)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("사용자 종료")

except Exception as e:
    print(f"❌ 오류 발생: {e}")

finally:
    print("\n🛑 안전 정지")
    if ser:
        try:
            for _ in range(3):
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")
                time.sleep(0.05)
            ser.close()
        except:
            pass

    if lidar is not None:
        try:
            lidar.stop()
            lidar.disconnect()
        except:
            pass

    if cap_lane is not None:
        cap_lane.release()
    if cap_traffic is not None:
        cap_traffic.release()

    cv2.destroyAllWindows()