# common_lane_base.py
import cv2
import numpy as np
import math
import time

# ==========================================
# [공통 설정] 
# ==========================================
PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
CAM_INDEX_TRAFFIC = 0

MAX_SPEED = 120

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

L_THRESHOLD = 170
BLUR_K = 5
MORPH_SIZE = (3, 3)

# [횡단보도/정지선]
CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# [신호등 ROI 박스] (traffic 파일에서만 실사용)
TRAFFIC_BOX_X1 = 0.22
TRAFFIC_BOX_X2 = 0.62
TRAFFIC_BOX_Y1 = 0.35
TRAFFIC_BOX_Y2 = 0.62
TRAFFIC_HIGHLIGHT_PCTL = 98
TRAFFIC_TH_MIN = 200


# ==========================================
# 카메라 AUTO 설정
# ==========================================
def configure_camera_auto(cap):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


# ==========================================
# 공통 유틸
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters, roi_height_ratio):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * roi_height_ratio)
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


# ✅ 점선 제거 로직 (왼쪽 60, 오른쪽 90)
def average_slope_intercept(image, lines, roi_height_ratio):
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

    left_line = make_points(image, np.mean(left_fit, axis=0), roi_height_ratio) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0), roi_height_ratio) if len(right_fit) > 0 else None
    return left_line, right_line


def map_value(x, in_min, in_max, out_min, out_max):
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
# [라인트레이싱 코어] 마스크/ROI/라인검출/기본타겟 산출
# - obstacle/traffic 두 모드가 공통으로 사용
# ==========================================
def compute_lane(frame_lane):
    h, w = frame_lane.shape[:2]

    blurred = cv2.medianBlur(frame_lane, BLUR_K)
    hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(
        hls,
        np.array([0, L_THRESHOLD, 0]),
        np.array([179, 255, 255])
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    roi_points = np.array([[
        (0, h), (w, h),
        (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
        (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
    ]], dtype=np.int32)

    roi_mask_poly = np.zeros_like(mask)
    cv2.fillPoly(roi_mask_poly, [roi_points], 255)
    roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

    white_count = cv2.countNonZero(roi_pixels)
    total_area = cv2.contourArea(roi_points)
    if total_area == 0:
        total_area = 1
    ratio = white_count / total_area

    edges = cv2.Canny(mask, 50, 150)
    cropped = region_of_interest(edges, roi_points)
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    left, right = average_slope_intercept(frame_lane, lines, ROI_HEIGHT_RATIO)

    return {
        "h": h, "w": w,
        "mask": mask,
        "mask_bgr": mask_bgr,
        "roi_points": roi_points,
        "ratio": ratio,
        "left": left,
        "right": right,
    }


def base_target_from_lines(w, left, right):
    if left is not None and right is not None:
        return (left[0][2] + right[0][2]) / 2
    elif left is not None:
        return left[0][2] + (w * 0.25)
    elif right is not None:
        return right[0][2] - (w * 0.25)
    else:
        return w / 2


def angle_from_target(w, h, target_x):
    car_x = w / 2
    target_y = int(h * ROI_HEIGHT_RATIO)
    dx = target_x - car_x
    dy = (h - target_y)
    return math.degrees(math.atan2(dx, abs(dy)))


def servo_from_angle(angle_deg):
    angle_deg = max(-45, min(45, angle_deg))
    return int(map_value(angle_deg, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
