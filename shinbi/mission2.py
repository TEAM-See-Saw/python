import cv2
import numpy as np
import math
import serial
import time
import datetime
import os
import threading

# ==========================================
# [1] 환경 설정 (사용자 환경에 맞게 수정)
# ==========================================
IS_SUNNY = True  # True: 햇빛 쨍쨍 / False: 실내, 흐림

# ⚙️ 포트 설정 (장치 관리자 확인 필수!)
PORT_ARDUINO = 'COM4'   # 아두이노
PORT_LIDAR   = 'COM5'   # 라이다 (없으면 None으로 두면 무시됨)
BAUDRATE     = 9600
CAM_INDEX    = 0

# ⚙️ 주행 값 설정
MAX_SPEED      = 120   # 평소 주행 속도 (안전을 위해 조금 줄임)
SERVO_CENTER   = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX= 480

# ⚙️ 미션 임계값
LIDAR_STOP_DIST = 600  # 600mm(60cm) 이내 장애물 감지 시 정지
TRAFFIC_ROI_H   = 0.4  # 화면 상단 40%만 신호등으로 인식

# ==========================================
# [2] 라이브러리 로드 (RPLidar)
# ==========================================
try:
    from rplidar import RPLidar
    HAS_LIDAR = True
except ImportError:
    print("⚠️ 'rplidar' 라이브러리가 없습니다. (pip install rplidar-types)")
    HAS_LIDAR = False

# ==========================================
# [3] 모드별 자동 튜닝값
# ==========================================
if IS_SUNNY:
    print(f"☀️ [모드: SUNNY] 강력한 햇빛 대응 설정")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print(f"🌙 [모드: NORMAL] 실내/저녁 설정")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)

# ==========================================
# [4] 공유 변수 및 스레드 (라이다, 신호등)
# ==========================================
shared_data = {
    "front_dist": 9999,  # 전방 장애물 거리 (mm)
    "traffic": "NONE",   # 신호등 상태: NONE, RED, GREEN
    "running": True      # 프로그램 종료 플래그
}

def thread_lidar():
    """ 라이다 센서 값을 백그라운드에서 계속 읽어옴 """
    if not HAS_LIDAR: return
    
    lidar = RPLidar(PORT_LIDAR)
    print(f"📡 라이다 연결 성공: {PORT_LIDAR}")
    
    try:
        for scan in lidar.iter_scans():
            if not shared_data["running"]: break
            
            min_dist = 9999
            for (_, angle, dist) in scan:
                # 전방 30도 부채꼴 (0~15도, 345~360도)
                if (angle < 15 or angle > 345) and dist > 0:
                    if dist < min_dist:
                        min_dist = dist
            
            shared_data["front_dist"] = min_dist
    except Exception as e:
        print(f"❌ 라이다 오류: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()

def detect_traffic_light(frame):
    """ 화면 상단에서 신호등 색상 인식 """
    h, w = frame.shape[:2]
    # 화면 상단 40%만 잘라서 확인 (신호등은 위에 있으므로)
    roi = frame[0:int(h * TRAFFIC_ROI_H), :]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 (두 개로 나뉨)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 초록색 범위
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 픽셀 수 카운트 (노이즈 제거를 위해 일정 개수 이상이어야 인식)
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    threshold = 200 # 인식 최소 픽셀 수
    
    status = "NONE"
    if red_pixels > threshold and red_pixels > green_pixels:
        status = "RED"
    elif green_pixels > threshold and green_pixels > red_pixels:
        status = "GREEN"
        
    return status, roi

# ==========================================
# [5] 기존 영상 처리 함수 (라인트레이싱)
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
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
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    car_position_x = width / 2

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = car_position_x

    dx = target_x - car