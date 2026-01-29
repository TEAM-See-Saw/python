import cv2
import numpy as np
import math
import serial
import time
import datetime
import os

# ==========================================
# [1] 환경 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX = 0

MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# [추가] 마지막 조향각 기억 (차선을 놓쳤을 때 대비)
last_servo_val = SERVO_CENTER

# ==========================================
# [2] 시리얼 연결 (기존과 동일)
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 연결 성공")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}")

# ==========================================
# [3] 영상 처리 보조 함수 (기존 로직 유지)
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except: return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / (slope if slope != 0 else 0.001))
    x2 = int((y2 - intercept) / (slope if slope != 0 else 0.001))
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    if lines is None: return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit[0], fit[1]
            if slope < -0.5: left_fit.append((slope, intercept))
            elif slope > 0.5: right_fit.append((slope, intercept))
    
    left_line = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    car_pos_x = width / 2
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.2) # 조금 더 안쪽으로 보정
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.2)
    else:
        return None, int(car_pos_x) # 차선 못 찾음 표시
    
    dx = target_x - car_pos_x
    dy = (height * 0.6) - height
    return math.degrees(math.atan2(dx, abs(dy))), int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    global last_servo_val
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 조명 변화 대응을 위해 노출은 자동 혹은 약한 고정값 추천
    cap.set(cv2.CAP_PROP_EXPOSURE, -5) 

    # CLAHE 객체 생성 (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    if not cap.isOpened(): return

    # 녹화 설정 등은 동일하게 유지 (생략)

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- [STEP 1] 대비 최적화 (CLAHE) ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l) # 어두운 곳 밝히고 밝은 곳 누름
        enhanced_frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        # --- [STEP 2] 적응형 이진화 (Adaptive Threshold) ---
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        # 주변 31픽셀을 보고 상대적으로 밝은 차선 추출
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 31, -15)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # --- [STEP 3] 차선 검출 ---
        edges = cv2.Canny(mask, 50, 150)
        roi_vertices = [(0, height), (width//2-80, int(height*0.6)), 
                        (width//2+80, int(height*0.6)), (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 40, minLineLength=30, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)

        # --- [STEP 4] 조향 결정 (Fail-safe 추가) ---
        angle, target = calculate_steering_angle(frame, left, right)
        
        if angle is not None:
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
            last_servo_val = servo_val # 차선을 찾았을 때만 마지막 값 갱신
        else:
            # 차선을 놓쳤다면? 마지막 조향을 유지하거나 천천히 중앙으로
            servo_val = last_servo_val 
            cv2.putText(frame, "LOST LANE!", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if ser:
            ser.write(f"S,{servo_val}\n".encode())

        # 디스플레이
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))
        cv2.imshow("Adaptive Driving", combined)

        if cv2.waitKey(1) == ord('q'): break

    if ser:
        ser.write(b"D,0\n")
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()