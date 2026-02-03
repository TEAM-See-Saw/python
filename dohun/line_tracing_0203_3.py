import numpy as np
import math
import serial
import time
import cv2

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05       # 0.05초 = 20Hz 전송 (반응 속도 최적화)
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

# 자동 튜닝 목표 비율 (흰색이 차지하는 비율)
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

# HLS 임계값 범위
MIN_L_VAL = 80       
MAX_L_VAL = 220      
S_MAX_VAL = 80       

MORPH_SIZE = (3, 3)
BLUR_K = 3

# ==========================================
# [2] 시리얼 연결 (Flush 초기화 추가)
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    # [수정 1] 연결 즉시 잔여 데이터 제거
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(f"✅ {PORT} 포트 연결 성공!")
    time.sleep(1)
except Exception as e:
    print(f"❌ 연결 실패: {e}")

# ==========================================
# [3] 핵심 설정 함수 (노출 & HLS)
# ==========================================

def setup_roi_exposure(cap):
    """
    [신규] 전체 화면이 아닌 바닥(ROI) 밝기 기준으로 카메라 노출을 맞추고 고정함
    """
    print("📸 ROI(바닥) 밝기 분석 및 노출 최적화 중...")
    
    # 자동 노출을 켜서 초기값 확보
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
    time.sleep(1.5) 

    target_brightness = 125  # 우리가 원하는 ROI의 평균 밝기
    
    for i in range(20): 
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        roi = frame[int(h * ROI_HEIGHT_RATIO):h, :]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        current_avg = np.mean(gray_roi)

        curr_exp = cap.get(cv2.CAP_PROP_EXPOSURE)

        if current_avg < target_brightness - 10:
            cap.set(cv2.CAP_PROP_EXPOSURE, curr_exp + 1)
        elif current_avg > target_brightness + 10:
            cap.set(cv2.CAP_PROP_EXPOSURE, curr_exp - 1)
        else:
            break
        time.sleep(0.05)

    # 설정 완료 후 수동(Manual) 모드로 고정
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
    print(f"✅ 노출 고정 완료 (ROI 평균 밝기: {current_avg:.1f})")


def find_optimal_l_min(cap):
    """
    고정된 노출 상태에서 최적의 HLS L_min 초기값을 계산
    """
    print("🔎 HLS 초기값 분석 중...")
    try:
        for _ in range(10): cap.read()
        ret, frame = cap.read()
        if not ret: return 140

        frame = cv2.resize(frame, (640, 480))
        hls = cv2.cvtColor(cv2.medianBlur(frame, BLUR_K), cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        h, w = frame.shape[:2]
        roi_l = l_channel[int(h * ROI_HEIGHT_RATIO):h, :]
        
        pixels = np.sort(roi_l.flatten())
        detected_l = pixels[int(len(pixels) * 0.90)] 

        final_l = max(MIN_L_VAL, min(detected_l - 30, MAX_L_VAL))
        print(f"✅ HLS 분석 완료: L_min = {final_l}")
        return int(final_l)
    except:
        return 140

# ==========================================
# [4] 주행 보조 함수
# ==========================================

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    if line_parameters is None: return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * ROI_HEIGHT_RATIO)
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
            if abs(slope) < 0.7: continue
            if slope < -0.7: left_fit.append((slope, intercept))
            elif slope > 0.7: right_fit.append((slope, intercept))
    
    left = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left, right

def calculate_steering_angle(image, left, right):
    global last_target_x
    h, w = image.shape[:2]
    car_x, target_y = w / 2, int(h * ROI_HEIGHT_RATIO)

    if left and right: target_x = (left[0][2] + right[0][2]) / 2
    elif left: target_x = left[0][2] + (w * 0.25)
    elif right: target_x = right[0][2] - (w * 0.25)
    else: target_x = last_target_x

    last_target_x = target_x
    return math.degrees(math.atan2(target_x - car_x, h - target_y)), int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [5] 메인 루프 (통신 최적화 적용)
# ==========================================
def main():
    global current_l_min
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, 640); cap.set(4, 480)

    if not cap.isOpened(): print("❌ 카메라 오류"); return

    # [중요] 시작 전 하드웨어 노출 최적화 및 HLS 초기값 설정
    setup_roi_exposure(cap)
    current_l_min = find_optimal_l_min(cap)

    print("\n🚀 주행 시작!"); time.sleep(1)
    if ser: 
        ser.write(f"D,{MAX_SPEED}\n".encode())
        ser.flush() # 초기 속도 즉시 전송

    last_serial_time = last_speed_time = 0

    try:
        while True:
            # [수정 2] 루프 시작 시 입력 버퍼 비우기 (딜레이 방지)
            if ser: ser.reset_input_buffer()

            ret, frame = cap.read()
            if not ret: break
            h, w = frame.shape[:2]

            # 1. 전처리 (HLS 필터링)
            hls = cv2.cvtColor(cv2.medianBlur(frame, BLUR_K), cv2.COLOR_BGR2HLS)
            lower_white = np.array([0, current_l_min, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            # 2. ROI 및 자동 HLS 미세 조정
            roi_v = np.array([[(0, h), (w, h), (w, int(h*ROI_HEIGHT_RATIO)), (0, int(h*ROI_HEIGHT_RATIO))]], dtype=np.int32)
            roi_mask = np.zeros_like(mask)
            cv2.fillPoly(roi_mask, [roi_v], 255)
            white_pixels = cv2.countNonZero(cv2.bitwise_and(mask, roi_mask))
            ratio = white_pixels / (w * h * ROI_HEIGHT_RATIO)

            if ratio > TARGET_RATIO_MAX: current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN: current_l_min = max(current_l_min - 2, MIN_L_VAL)

            # 3. 라인 인식 및 조향
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(region_of_interest(edges, roi_v), 1, np.pi/180, 50, 40, 100)
            left, right = average_slope_intercept(frame, lines)
            angle, target_x = calculate_steering_angle(frame, left, right)
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            # 4. 통신 (플러시 기능 추가)
            curr = time.time()
            if ser:
                # 조향 제어
                if curr - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    ser.flush() # [수정 3] 즉시 전송
                    last_serial_time = curr
                
                # 속도 제어
                if curr - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    ser.flush() # [수정 3] 즉시 전송
                    last_speed_time = curr

            # 5. 시각화
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(frame, (target_x, int(h*ROI_HEIGHT_RATIO)), 10, (0,0,255), -1)
            cv2.putText(frame, f"L:{current_l_min} R:{ratio*100:.1f}%", (20,40), 1, 1.5, (0,255,0), 2)
            cv2.imshow("Lane Tracking", np.hstack([frame, mask_bgr]))

            if cv2.waitKey(1) == ord('q'): break
    finally:
        print("🛑 안전 정지")
        if ser: 
            for _ in range(3):
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")
                ser.flush() # [수정 4] 종료 신호도 즉시 전송
                time.sleep(0.05)
            ser.close()
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()