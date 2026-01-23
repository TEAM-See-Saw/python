# [MJPG 고속 입력] + [640x480 리사이즈] + [CLAHE 자연광 대응] + [이동평균필터]
# CLAHE + Canny

import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 사용자 설정 (내 차에 맞게 수정)
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

# 서보 모터 값 (사용자 환경)
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 80     # 테스트 속도

# 연산 해상도 (640x480 고정 -> 속도 핵심)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# 핸들 떨림 방지용 변수
prev_servo_value = SERVO_CENTER 

# ==========================================
# [2] 하드웨어 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"✅ 아두이노 연결 성공 ({PORT})")
except Exception as e:
    print(f"⚠️ 아두이노 연결 실패 (테스트 모드): {e}")

# ==========================================
# [3] 카메라 설정 (MJPG 강제 주입 - 속도 핵심)
# ==========================================
params = [
    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
    cv2.CAP_PROP_FRAME_WIDTH, 1920, 
    cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    cv2.CAP_PROP_FPS, 30
]
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW, params)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("🚀 주행 시작 [Mode: CLAHE + Canny]")

# ==========================================
# [4] 알고리즘 함수
# ==========================================
def apply_clahe_edge(image):
    """자연광 대응: 밝기 평탄화 후 에지 검출"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # CLAHE: 그림자 속 차선은 밝게, 햇빛 받은 도로는 어둡게 눌러줌
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # 블러 & 캐니 에지
    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    # 햇빛 반사가 너무 심해서 자잘한 점이 많으면 50 -> 70으로 올리세요
    edges = cv2.Canny(blur, 50, 150) 
    return edges

def region_of_interest(img):
    """관심 영역 설정 (640x480 해상도 기준)"""
    height, width = img.shape
    # 사다리꼴 모양으로 도로 바닥만 봄
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
        target_x = left[0][2] + (w * 0.22) # 한쪽만 보일 때 오프셋
    elif right is not None:
        target_x = right[0][2] - (w * 0.22)
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = h * 0.6
    angle = math.degrees(math.atan2(dx, dy))
    return angle, target_x

def smooth_servo(new_val):
    """이동 평균 필터: 핸들 떨림 방지"""
    global prev_servo_value
    alpha = 0.5 
    smoothed = int(prev_servo_value * (1 - alpha) + new_val * alpha)
    prev_servo_value = smoothed
    return smoothed

# ==========================================
# [5] 메인 루프
# ==========================================
def main():
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 해상도 리사이즈 (속도 최적화)
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # 2. 영상 처리 (CLAHE)
        edges = apply_clahe_edge(frame_resized)
        roi_edges = region_of_interest(edges)

        # 3. 라인 검출
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=100)
        left_line, right_line = average_slope_intercept(frame_resized, lines)

        # 4. 조향 계산
        angle, target_x = calculate_steering(frame_resized, left_line, right_line)
        
        # 5. 서보 값 매핑 및 전송
        clamped_angle = max(-45, min(45, angle))
        raw_servo = np.interp(clamped_angle, [-45, 45], [SERVO_LEFT_MAX, SERVO_RIGHT_MAX])
        final_servo = smooth_servo(raw_servo)
        
        if ser: ser.write(f"S,{final_servo}\n".encode())

        # 6. 디버깅 화면 (FPS 표시)
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time)
        prev_time = cur_time

        debug_img = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
        if left_line is not None:
             cv2.line(debug_img, (left_line[0][0], left_line[0][1]), (left_line[0][2], left_line[0][3]), (255,0,0), 5)
        if right_line is not None:
             cv2.line(debug_img, (right_line[0][0], right_line[0][1]), (right_line[0][2], right_line[0][3]), (0,0,255), 5)
        
        cv2.putText(debug_img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        result = cv2.hconcat([frame_resized, debug_img])
        cv2.imshow('Final Lane Tracing', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if ser:
        ser.write(b"D,0\n")
        ser.write(f"S,{SERVO_CENTER}\n".encode())
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()