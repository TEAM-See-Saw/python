# [linetracing.py]
import cv2
import numpy as np
import math

# 설정값 (메인 코드와 공유)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 핸들 떨림 방지용 전역 변수
prev_servo_value = SERVO_CENTER 

def apply_clahe_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(img):
    height, width = img.shape
    polygons = np.array([
        [(0, height), (width, height), 
         (width//2 + 60, int(height * 0.55)), 
         (width//2 - 60, int(height * 0.55))]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_params):
    if line_params is None: return None
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None: return None, None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5: left_fit.append((slope, intercept))
            elif slope > 0.5: right_fit.append((slope, intercept))
            
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering(image, left, right):
    h, w = image.shape[:2]
    car_x = w / 2
    
    if left is not None and right is not None:
        target_x = (left[0][2] + right[0][2]) / 2
    elif left is not None:
        target_x = left[0][2] + (w * 0.22)
    elif right is not None:
        target_x = right[0][2] - (w * 0.22)
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = h * 0.6
    angle = math.degrees(math.atan2(dx, dy))
    return angle

def smooth_servo(new_val):
    global prev_servo_value
    alpha = 0.5 
    smoothed = int(prev_servo_value * (1 - alpha) + new_val * alpha)
    prev_servo_value = smoothed
    return smoothed

# ★ 메인 코드에서 호출할 핵심 함수
def get_servo_value(frame):
    # 1. 해상도 리사이즈
    frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    
    # 2. 영상 처리
    edges = apply_clahe_edge(frame_resized)
    roi_edges = region_of_interest(edges)

    # 3. 라인 검출
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)
    left_line, right_line = average_slope_intercept(frame_resized, lines)

    # 4. 각도 및 서보값 계산
    angle = calculate_steering(frame_resized, left_line, right_line)
    
    clamped_angle = max(-45, min(45, angle))
    raw_servo = np.interp(clamped_angle, [-45, 45], [SERVO_LEFT_MAX, SERVO_RIGHT_MAX])
    final_servo = smooth_servo(raw_servo)
    
    return final_servo