import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar

# =========================================================
# [1] 포트/카메라 설정
# =========================================================
ARDUINO_PORT = "COM4"
LIDAR_PORT = "COM3"
BAUDRATE = 115200

CAM_INDEX_TRAFFIC = 0  # 신호등 카메라
CAM_INDEX_LANE = 1     # 라인트레이싱 카메라

width, height = 640, 480

SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 0.15   # 정지/출발 반응 빠르게(0.15~0.3 권장)

# =========================================================
# [2] 차량 제어 설정
# =========================================================
MAX_SPEED = 255
SPEED_STOP = 0
SPEED_OBS_SLOW = 120

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# =========================================================
# [3] 라인트레이싱(HLS) 설정
# =========================================================
IS_SUNNY = True

ROI_HEIGHT_RATIO = 0.60
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

if IS_SUNNY:
    current_l_min = 200
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX_VAL = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    current_l_min = 140
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX_VAL = 80
    MORPH_SIZE = (3, 3)
    BLUR_K = 5

# =========================================================
# [4] 장애물(LiDAR) 회피 설정
# =========================================================
OBSTACLE_START_DIST = 1000
OBSTACLE_SLOW_DIST = 800
SHIFT_GAIN = 1.2
OBSTACLE_CLEAR_TIME = 1.5

obstacle_count = 0
is_obstacle_detected = False
obs_clear_finished_time = 0.0
obstacle_last_seen_time = 0.0

# =========================================================
# [5] 정지선+신호 상태머신 설정
# =========================================================
STOPLINE_CONFIRM_TIME = 0.20   # 정지선 연속 확인 시간
LIGHT_CONFIRM_TIME = 0.15      # 신호 연속 확인 시간

is_waiting_at_stopline = False
stopline_seen_timer = 0.0
light_red_timer = 0.0
light_green_timer = 0.0

# =========================================================
# [6] 신호등(카메라0) “흰색 빛 위치” 판정 설정
# =========================================================
TRAFFIC_ROI_Y1 = 0
TRAFFIC_ROI_Y2 = int(height * 0.55)
TRAFFIC_ROI_X1 = 0
TRAFFIC_ROI_X2 = width

TRAFFIC_BRIGHT_PERCENTILE = 99.3
TRAFFIC_MIN_BRIGHT_PIXELS = 60
TRAFFIC_DOMINANCE_RATIO = 1.35  # 1등 영역이 2등보다 이 비율 이상 커야 확정

# =========================================================
# [7] 횡단보도(지브라) 억제/검출 설정
# =========================================================
# 횡단보도는 "수평 선분이 다수"가 반복되는 패턴이므로, 아래 밴드에서 수평 선분 개수가 많으면 횡단보도
CROSSWALK_BAND_RATIO = 0.32      # ROI 하단에서 몇 %를 검사할지
CROSSWALK_HLINE_MIN_COUNT = 6    # 수평 선분이 이 개수 이상이면 "횡단보도"
CROSSWALK_HLINE_ANGLE_DEG = 12   # abs(angle) < 12° -> 수평으로 간주
CROSSWALK_HLINE_MIN_LEN_RATIO = 0.10  # 폭 대비 최소 길이(짧은 노이즈 제거)
CROSSWALK_HLINE_MAX_LEN_RATIO = 0.80  # 너무 긴 건(정지선)과 구분

# 수평 성분 제거용 커널(횡단보도 줄무늬 억제)
HORIZ_SUPPRESS_KERNEL = (25, 3)  # (가로, 세로)  가로가 길수록 수평 성분 제거 강함
VERT_ENHANCE_KERNEL = (3, 15)    # 세로 성분(차선 방향 성분) 유지/강조

# =========================================================
# [8] 유틸 함수
# =========================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min + 1e-6) + out_min

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

last_target_x = 320

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
    left_fit, right_fit = [], []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit[0], fit[1]
            # 수평에 가까운(횡단보도) 라인은 애초에 제외(차선으로 쓰지 않음)
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    global last_target_x
    h, w = image.shape[:2]
    car_x = w / 2.0
    target_y = int(h * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2.0
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

# =========================================================
# [9] 신호등(카메라0) : 흰색 빛 위치 기반 RED/GREEN
# =========================================================
def detect_traffic_light_by_position(frame_traffic):
    roi = frame_traffic[TRAFFIC_ROI_Y1:TRAFFIC_ROI_Y2, TRAFFIC_ROI_X1:TRAFFIC_ROI_X2]
    if roi.size == 0:
        return "UNKNOWN", None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thr = int(np.percentile(gray, TRAFFIC_BRIGHT_PERCENTILE))
    thr = clamp(thr, 200, 250)
    bw = (gray >= thr).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)

    bright_pixels = int(cv2.countNonZero(bw))
    if bright_pixels < TRAFFIC_MIN_BRIGHT_PIXELS:
        return "UNKNOWN", bw

    h, w = bw.shape[:2]
    third = w // 3
    left_sum = int(cv2.countNonZero(bw[:, :third]))
    mid_sum = int(cv2.countNonZero(bw[:, third:2*third]))
    right_sum = int(cv2.countNonZero(bw[:, 2*third:]))

    sums = sorted([left_sum, mid_sum, right_sum], reverse=True)
    if sums[0] < TRAFFIC_MIN_BRIGHT_PIXELS:
        return "UNKNOWN", bw
    if sums[0] < sums[1] * TRAFFIC_DOMINANCE_RATIO:
        return "UNKNOWN", bw

    if left_sum == sums[0]:
        return "RED", bw
    if right_sum == sums[0]:
        return "GREEN", bw
    return "UNKNOWN", bw

# =========================================================
# [10] 횡단보도 패턴 검출 (수평 선분 다수)
#       - “흰색” 자체가 아니라, “수평 선분 개수”로 판단
# =========================================================
def detect_crosswalk_pattern(frame_lane, roi_points):
    h, w = frame_lane.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], 255)

    y_top = int(h * ROI_HEIGHT_RATIO)
    band_h = int((h - y_top) * CROSSWALK_BAND_RATIO)
    band_y1 = h - band_h

    # ROI 하단 밴드만
    gray = cv2.cvtColor(frame_lane, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    band = gray[band_y1:h, :]
    band_mask = roi_mask[band_y1:h, :]

    # 에지 기반
    edges = cv2.Canny(band, 50, 150)
    edges = cv2.bitwise_and(edges, band_mask)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=25,
        minLineLength=int(w * CROSSWALK_HLINE_MIN_LEN_RATIO),
        maxLineGap=15
    )

    if lines is None:
        return False, edges, band_y1

    # 수평 선분 개수 카운트
    hline_count = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang = np.degrees(np.arctan2(y2 - y1, (x2 - x1) + 1e-6))
        length = abs(x2 - x1)

        if abs(ang) < CROSSWALK_HLINE_ANGLE_DEG:
            # 너무 긴 건(정지선 후보) 제외하고 "여러 개"면 횡단보도
            if (length > int(w * CROSSWALK_HLINE_MIN_LEN_RATIO)) and (length < int(w * CROSSWALK_HLINE_MAX_LEN_RATIO)):
                hline_count += 1

    return (hline_count >= CROSSWALK_HLINE_MIN_COUNT), edges, band_y1

# =========================================================
# [11] 정지선 검출 (형태 기반) + “단일 피크” 검증
#       - 횡단보도(다중 가로줄)면 정지선으로 인정하지 않음
# =========================================================
def detect_stop_line_shape_only(frame_lane, roi_points, band_ratio=0.25,
                                sobel_percentile=92, row_thresh_ratio=0.28):
    h, w = frame_lane.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], 255)

    y_top = int(h * ROI_HEIGHT_RATIO)
    band_h = int((h - y_top) * band_ratio)
    band_y1 = h - band_h
    band_mask = np.zeros_like(roi_mask)
    band_mask[band_y1:h, :] = 255
    mask = cv2.bitwise_and(roi_mask, band_mask)

    gray = cv2.cvtColor(frame_lane, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gy = np.abs(gy)
    gy = cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gy = cv2.bitwise_and(gy, mask)

    vals = gy[mask == 255]
    if vals.size < 80:
        return False, None, band_y1

    thr = int(np.percentile(vals, sobel_percentile))
    bw = (gy >= thr).astype(np.uint8) * 255

    row_sum = bw.sum(axis=1)
    y_peak = int(np.argmax(row_sum))
    peak = row_sum[y_peak]

    # (1) 길이가 충분히 긴 수평 구조인가?
    if peak < (w * 255 * row_thresh_ratio):
        return False, None, band_y1

    # (2) “단일 피크”인지 검증 (횡단보도는 피크가 여러 개)
    # peak의 70% 이상인 행이 여러 개면(예: 3개 이상) => 횡단보도 가능성 높음 => 정지선으로 안 봄
    strong_rows = np.sum(row_sum >= (peak * 0.70))
    if strong_rows >= 3:
        return False, None, band_y1

    # ROI 밴드 내부 y -> 원본 y
    return True, y_peak, band_y1

# =========================================================
# [12] 횡단보도 억제 마스크 (차선용)
#       - 수평 성분 제거 + 세로 성분 유지
# =========================================================
def suppress_horizontal_components(lane_mask, stronger=False):
    """
    lane_mask: 흰색 마스크(0/255)
    stronger: 횡단보도 감지 시 더 강하게 수평 제거
    """
    # 1) 기본 노이즈 오픈
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

    # 2) 수평 성분(횡단보도) 제거: horizontal opening으로 수평 성분만 뽑아낸 뒤 빼기
    kx = HORIZ_SUPPRESS_KERNEL[0] + (15 if stronger else 0)
    ky = HORIZ_SUPPRESS_KERNEL[1]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    horiz = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, horiz_kernel)

    lane_mask2 = cv2.subtract(lane_mask, horiz)

    # 3) 세로 성분 강화(차선 방향 성분 유지)
    vx = VERT_ENHANCE_KERNEL[0]
    vy = VERT_ENHANCE_KERNEL[1] + (6 if stronger else 0)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (vx, vy))
    lane_mask2 = cv2.morphologyEx(lane_mask2, cv2.MORPH_OPEN, vert_kernel)

    return lane_mask2, horiz

# =========================================================
# [13] 초기화
# =========================================================
ser = None
lidar = None

cap_lane = None
cap_traffic = None

try:
    ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=0.1)
    print(f"✅ Arduino 연결: {ARDUINO_PORT}")
    time.sleep(2)
except Exception as e:
    print(f"❌ Arduino 연결 실패: {e}")
    ser = None

try:
    lidar = RPLidar(LIDAR_PORT)
    print(f"✅ LiDAR 연결: {LIDAR_PORT}")
except Exception as e:
    print(f"❌ LiDAR 연결 실패: {e}")
    lidar = None

cap_lane = cv2.VideoCapture(CAM_INDEX_LANE, cv2.CAP_DSHOW)
cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)
cap_lane.set(3, width); cap_lane.set(4, height); cap_lane.set(15, -6)
cap_traffic.set(3, width); cap_traffic.set(4, height); cap_traffic.set(15, -6)

try:
    cap_lane.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap_traffic.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except:
    pass

if not cap_lane.isOpened():
    raise RuntimeError("❌ 차선 카메라(1) 오픈 실패")
if not cap_traffic.isOpened():
    raise RuntimeError("❌ 신호등 카메라(0) 오픈 실패")

# =========================================================
# [14] 메인 루프
# =========================================================
last_serial_time = 0.0
last_speed_time = 0.0

print("\n🚀 3초 후 출발!")
for i in range(3, 0, -1):
    print(f"{i}..")
    time.sleep(1)

if ser:
    ser.write(f"D,{MAX_SPEED}\n".encode())
print("🚀 주행 시작")

try:
    scan_iter = lidar.iter_scans() if lidar is not None else [None] * (10**9)

    for scan in scan_iter:
        # RX flush
        if ser:
            try:
                if ser.in_waiting > 0:
                    ser.read(ser.in_waiting)
            except:
                pass

        current_time = time.time()

        # -----------------------------------------
        # (1) LiDAR 전방 거리
        # -----------------------------------------
        raw_dist = 2000
        if scan is not None:
            for (_, ang, dist) in scan:
                if 200 < dist < 1500:
                    if ang >= 330 or ang <= 30:
                        if dist < raw_dist:
                            raw_dist = dist

        # -----------------------------------------
        # (2) 카메라 프레임 읽기
        # -----------------------------------------
        ret_l, frame_lane = cap_lane.read()
        ret_t, frame_traffic = cap_traffic.read()
        if not ret_l or not ret_t:
            print("❌ 카메라 프레임 끊김")
            break

        if frame_lane.shape[1] != width:
            frame_lane = cv2.resize(frame_lane, (width, height))
        if frame_traffic.shape[1] != width:
            frame_traffic = cv2.resize(frame_traffic, (width, height))

        h, w = frame_lane.shape[:2]

        # -----------------------------------------
        # (3) ROI 폴리곤(차선 영역)
        # -----------------------------------------
        roi_points = np.array([[
            (0, h), (w, h),
            (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
            (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
        ]], dtype=np.int32)

        # -----------------------------------------
        # (4) 신호등 판정(흰색 빛 위치)
        # -----------------------------------------
        traffic_light, traffic_bw = detect_traffic_light_by_position(frame_traffic)

        # -----------------------------------------
        # (5) 차선 마스크(HLS)
        # -----------------------------------------
        blurred = cv2.medianBlur(frame_lane, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, current_l_min, 0])
        upper_white = np.array([179, 255, S_MAX_VAL])
        lane_mask_raw = cv2.inRange(hls, lower_white, upper_white)

        # Auto-tuning ratio는 "원래 방식" 유지(너가 이미 잘 됐다고 한 단독 코드랑 동일 계열)
        roi_mask_poly = np.zeros_like(lane_mask_raw)
        cv2.fillPoly(roi_mask_poly, [roi_points], 255)
        roi_pixels = cv2.bitwise_and(lane_mask_raw, roi_mask_poly)
        white_count = cv2.countNonZero(roi_pixels)
        total_area = cv2.contourArea(roi_points) or 1
        ratio = white_count / total_area

        if ratio > TARGET_RATIO_MAX:
            current_l_min = min(current_l_min + 2, MAX_L_VAL)
        elif ratio < TARGET_RATIO_MIN:
            current_l_min = max(current_l_min - 2, MIN_L_VAL)

        # -----------------------------------------
        # (6) 횡단보도 패턴 감지(수평 선분 다수)
        # -----------------------------------------
        crosswalk_detected, cross_edges, cross_band_y1 = detect_crosswalk_pattern(frame_lane, roi_points)

        # -----------------------------------------
        # (7) 라인트레이싱용 마스크에서 “횡단보도(수평줄)” 제거
        # -----------------------------------------
        lane_mask, horiz_only = suppress_horizontal_components(lane_mask_raw, stronger=crosswalk_detected)

        # -----------------------------------------
        # (8) 라인트레이싱 계산
        # -----------------------------------------
        edges = cv2.Canny(lane_mask, 50, 150)
        cropped = region_of_interest(edges, roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

        left, right = average_slope_intercept(frame_lane, lines)
        base_angle, base_target = calculate_steering_angle(frame_lane, left, right)

        # -----------------------------------------
        # (9) 장애물 회피 shift
        # -----------------------------------------
        avoid_direction = 0
        if raw_dist < OBSTACLE_START_DIST:
            obstacle_last_seen_time = current_time
            if (not is_obstacle_detected) and (current_time - obs_clear_finished_time > 1.5):
                obstacle_count += 1
                is_obstacle_detected = True
                print(f"⚠️ 장애물 #{obstacle_count} 감지!")

            # 예시: 1번째 장애물은 좌로(2->1), 2번째는 우로(1->2)
            if obstacle_count == 1:
                avoid_direction = -1
            elif obstacle_count == 2:
                avoid_direction = 1
            else:
                avoid_direction = 0
        else:
            if is_obstacle_detected:
                if current_time - obstacle_last_seen_time < OBSTACLE_CLEAR_TIME:
                    if obstacle_count == 1:
                        avoid_direction = -1
                    elif obstacle_count == 2:
                        avoid_direction = 1
                else:
                    is_obstacle_detected = False
                    obs_clear_finished_time = current_time
                    avoid_direction = 0
                    print("✅ 장애물 회피 종료, 복귀")

        target_x = float(base_target)
        shift_px = 0.0
        if avoid_direction != 0:
            calc_dist = min(raw_dist, OBSTACLE_START_DIST)
            shift_px = (OBSTACLE_START_DIST - calc_dist) * SHIFT_GAIN
            target_x = target_x + (avoid_direction * shift_px)

        # 타겟 기반 각도 재계산
        car_x = w / 2.0
        target_y = int(h * ROI_HEIGHT_RATIO)
        dx = target_x - car_x
        dy = (h - target_y)
        angle = math.degrees(math.atan2(dx, abs(dy)))
        target = int(clamp(target_x, 0, w - 1))

        servo_val = int(map_value(clamp(angle, -45, 45), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # -----------------------------------------
        # (10) 정지선(단일 가로선) 검출 (횡단보도는 제외됨)
        # -----------------------------------------
        stopline_detected, stopline_y_in_band, stopline_band_y1 = detect_stop_line_shape_only(
            frame_lane, roi_points,
            band_ratio=0.25,
            sobel_percentile=92,
            row_thresh_ratio=0.28
        )
        stopline_y = (stopline_band_y1 + stopline_y_in_band) if (stopline_detected and stopline_y_in_band is not None) else None

        # -----------------------------------------
        # (11) 정지/출발 상태머신
        # 조건:
        # - "가로선(정지선)이 보일 때" + "RED 위치에 빛" => 정지
        # - 정지 중에는 GREEN 위치 빛 => 출발
        # - 가로선이 안보일 때는 절대 멈추지 않음
        # - 가로선을 지나가면 안됨 => stopline+RED면 속도 0 유지
        # -----------------------------------------
        final_speed = MAX_SPEED
        status_msg = "DRIVE"
        status_color = (0, 255, 0)

        if is_waiting_at_stopline:
            final_speed = SPEED_STOP
            status_msg = "STOP (WAIT GREEN)"
            status_color = (0, 0, 255)

            if traffic_light == "GREEN":
                if light_green_timer == 0.0:
                    light_green_timer = current_time
                elif current_time - light_green_timer >= LIGHT_CONFIRM_TIME:
                    # 출발 → 기존 장애물 회피 + 라인트레이싱 로직 그대로 계속
                    is_waiting_at_stopline = False
                    light_red_timer = 0.0
                    light_green_timer = 0.0
                    stopline_seen_timer = 0.0
                    status_msg = "GO (GREEN)"
                    status_color = (0, 255, 0)
            else:
                light_green_timer = 0.0

        else:
            # 정지선이 연속 확인되면(stopline_confirmed)
            if stopline_detected:
                if stopline_seen_timer == 0.0:
                    stopline_seen_timer = current_time
                stopline_confirmed = (current_time - stopline_seen_timer >= STOPLINE_CONFIRM_TIME)
            else:
                stopline_seen_timer = 0.0
                stopline_confirmed = False

            if stopline_confirmed:
                # 정지선이 보일 때만 RED면 정지
                if traffic_light == "RED":
                    if light_red_timer == 0.0:
                        light_red_timer = current_time
                    elif current_time - light_red_timer >= LIGHT_CONFIRM_TIME:
                        is_waiting_at_stopline = True
                        final_speed = SPEED_STOP
                        status_msg = "STOP (LINE+RED)"
                        status_color = (0, 0, 255)
                        light_green_timer = 0.0
                else:
                    light_red_timer = 0.0
            else:
                # 정지선이 안보이면 신호등만으로는 절대 멈추지 않음
                light_red_timer = 0.0

                if raw_dist < OBSTACLE_SLOW_DIST:
                    final_speed = min(final_speed, SPEED_OBS_SLOW)
                    status_msg = f"OBSTACLE {raw_dist:.0f}mm"
                    status_color = (0, 255, 255)

        # -----------------------------------------
        # (12) 통신
        # -----------------------------------------
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # -----------------------------------------
        # (13) 디버그 출력
        # -----------------------------------------
        # Traffic ROI 표시 + 분할
        cv2.rectangle(frame_traffic, (TRAFFIC_ROI_X1, TRAFFIC_ROI_Y1), (TRAFFIC_ROI_X2, TRAFFIC_ROI_Y2), (0, 255, 255), 2)
        cv2.putText(frame_traffic, f"Traffic:{traffic_light}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if traffic_bw is not None:
            traffic_bw_bgr = cv2.cvtColor(traffic_bw, cv2.COLOR_GRAY2BGR)
        else:
            traffic_bw_bgr = np.zeros((TRAFFIC_ROI_Y2-TRAFFIC_ROI_Y1, TRAFFIC_ROI_X2-TRAFFIC_ROI_X1, 3), dtype=np.uint8)

        # Lane mask 디버그
        lane_dbg = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        cv2.polylines(lane_dbg, [roi_points], True, (0, 255, 255), 2)
        cv2.circle(lane_dbg, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        if crosswalk_detected:
            cv2.putText(lane_dbg, "CROSSWALK: YES (suppressed)", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        else:
            cv2.putText(lane_dbg, "CROSSWALK: NO", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        if stopline_detected and stopline_y is not None:
            cv2.line(lane_dbg, (0, stopline_y), (w-1, stopline_y), (255, 0, 255), 2)
            cv2.putText(lane_dbg, "STOPLINE: DETECT", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        else:
            cv2.putText(lane_dbg, "STOPLINE: ---", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        cv2.putText(lane_dbg, f"L-Min:{current_l_min} Ratio:{ratio*100:.1f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(lane_dbg, f"Angle:{angle:.1f} Target:{target} Shift:{shift_px:.0f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        cv2.putText(lane_dbg, status_msg, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # 화면 구성: [traffic | lane]
        traffic_show = frame_traffic.copy()
        cv2.imshow("Traffic Camera (0)", traffic_show)
        cv2.imshow("Lane Camera (1) - LaneMask (Crosswalk Suppressed)", lane_dbg)

        if cv2.waitKey(1) == ord("q"):
            break

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    print("\n🛑 안전 정지")
    if ser:
        try:
            for _ in range(3):
                ser.write(b"D,0\n")
                ser.write(f"S,{SERVO_CENTER}\n".encode())
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