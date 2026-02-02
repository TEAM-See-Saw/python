import numpy as np
import math
import serial
import time
import cv2  # cv2 임포트 누락 방지

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.5
ROI_X_LEFT_RATIO = 0.0
ROI_X_RIGHT_RATIO = 1.0

last_target_x = 320

TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

MIN_L_VAL = 80       
MAX_L_VAL = 220      
S_MAX_VAL = 80       

MORPH_SIZE = (3, 3)
BLUR_K = 3

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공!")
    time.sleep(1)
except Exception as e:
    print(f"❌ 연결 실패: {e}")


# ==========================================
# [3] 영상 처리 및 설정 함수
# ==========================================

def setup_camera_exposure(cap):
    """
    [추가] 카메라 노출값을 자동으로 계산하여 고정하는 함수
    """
    print("📸 카메라 노출 최적화 중...")
    
    # 1. 일단 자동 노출을 켭니다 (주변 밝기 파악용)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # 3: Auto mode
    time.sleep(1.5) # 카메라가 빛에 적응할 시간 필요

    # 2. 현재 평균 밝기 확인
    ret, frame = cap.read()
    if not ret: return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # 3. 노출을 수동(Manual)으로 전환하여 고정 (값이 변하지 않게 함)
    # 1: Manual mode (카메라 하드웨어/드라이버마다 0 또는 1이 매뉴얼임)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    
    print(f"✅ 노출 고정 완료 (현재 평균 밝기: {avg_brightness:.1f})")


def find_optimal_l_min(cap):
    print("🔎 HLS 임계값(L_min) 분석 중...")
    try:
        # 워밍업
        for _ in range(10): cap.read()

        ret, frame = cap.read()
        if not ret or frame is None: return 140

        frame = cv2.resize(frame, (640, 480))
        hls = cv2.cvtColor(cv2.medianBlur(frame, BLUR_K), cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        h, w = frame.shape[:2]
        roi_l = l_channel[int(h * ROI_HEIGHT_RATIO):h, :]

        pixels = np.sort(roi_l.flatten())
        target_idx = int(len(pixels) * 0.90)
        detected_l = pixels[target_idx]

        optimal_l = detected_l - 30
        final_l = max(MIN_L_VAL, min(optimal_l, MAX_L_VAL))
        
        print(f"✅ HLS 설정 완료! (감지:{detected_l} -> 설정:{final_l})")
        return int(final_l)
    except:
        return 140

# ... (기타 함수: region_of_interest, average_slope_intercept 등은 기존과 동일) ...
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    if line_parameters is None: return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * ROI_HEIGHT_RATIO)
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
            slope, intercept = fit[0], fit[1]
            if abs(slope) < 0.7: continue
            if slope < -0.7: left_fit.append((slope, intercept))
            elif slope > 0.7: right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else: target_x = last_target_x
    last_target_x = target_x
    dx, dy = target_x - car_x, (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy))), int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    global current_l_min

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ 카메라 오류")
        return

    # 1단계: 카메라 노출 설정 (가장 먼저 실행)
    setup_camera_exposure(cap)
    
    # 2단계: 최적의 HLS 밝기 임계값 찾기
    current_l_min = find_optimal_l_min(cap)

    print("\n🚀 3초 후 출발!")
    for i in range(3, 0, -1):
        print(f"{i}..")
        time.sleep(1)

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0
    last_speed_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            # 필터링 및 ROI 계산
            lower_white = np.array([0, current_l_min, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            roi_points = np.array([[(0, h), (w, h), (w, int(h*ROI_HEIGHT_RATIO)), (0, int(h*ROI_HEIGHT_RATIO))]], dtype=np.int32)
            
            # 자동 HLS 튜닝 로직
            white_count = cv2.countNonZero(cv2.bitwise_and(mask, cv2.bitwise_and(mask, mask, mask=cv2.fillPoly(np.zeros_like(mask), [roi_points], 255))))
            total_area = (w * h * (1 - ROI_HEIGHT_RATIO))
            ratio = white_count / total_area

            if ratio > TARGET_RATIO_MAX: current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN: current_l_min = max(current_l_min - 2, MIN_L_VAL)

            # 라인 인식 및 주행 제어
            edges = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges, roi_points)
            lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)
            angle, target = calculate_steering_angle(frame, left, right)
            
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            if ser:
                curr = time.time()
                if curr - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = curr
                if curr - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = curr

            # 화면 출력
            cv2.putText(frame, f"L:{current_l_min} R:{ratio*100:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Result", np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

            if cv2.waitKey(1) == ord('q'): break

    finally:
        if ser:
            ser.write(b"D,0\n")
            ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()