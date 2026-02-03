import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = False

PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

# 차선 폭 설정 (화면에 꽉 차는 환경이므로 0.45로 상향 조정)
LANE_OFFSET_RATIO = 0.45 

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
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}")


# ==========================================
# [3] 영상 처리 함수들 (2차 함수 적용)
# ==========================================

def fit_polynomial(mask, roi_points):
    """흰색 픽셀들을 추출하여 좌/우 차선별 2차 함수 계수를 구합니다."""
    # ROI 영역 추출
    roi_mask = np.zeros_like(mask)
    cv2.fillPoly(roi_mask, [roi_points], 255)
    lane_pixels = cv2.bitwise_and(mask, roi_mask)
    
    # 픽셀 좌표 추출
    nonzero = lane_pixels.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if len(nonzerox) < 50:
        return None, None

    # 화면 중앙 기준으로 좌/우 분리
    mid_x = mask.shape[1] // 2
    left_lane_inds = (nonzerox < mid_x)
    right_lane_inds = (nonzerox >= mid_x)

    left_fit, right_fit = None, None
    
    # 최소 50픽셀 이상일 때만 다항식 근사 실행
    if np.sum(left_lane_inds) > 50:
        left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
    if np.sum(right_lane_inds) > 50:
        right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
        
    return left_fit, right_fit

def calculate_steering_angle_poly(image, left_fit, right_fit):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)
    
    # x = ay^2 + by + c 계산
    if left_fit is not None and right_fit is not None:
        lx = left_fit[0]*target_y**2 + left_fit[1]*target_y + left_fit[2]
        rx = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
        target_x = (lx + rx) / 2
    elif left_fit is not None:
        lx = left_fit[0]*target_y**2 + left_fit[1]*target_y + left_fit[2]
        target_x = lx + (width * LANE_OFFSET_RATIO)
    elif right_fit is not None:
        rx = right_fit[0]*target_y**2 + right_fit[1]*target_y + right_fit[2]
        target_x = rx - (width * LANE_OFFSET_RATIO)
    else:
        target_x = last_target_x
        
    last_target_x = target_x
    dx = target_x - car_x
    dy = (height - target_y)
    
    return math.degrees(math.atan2(dx, abs(dy))), int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    global current_l_min
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap.set(3, width)
    cap.set(4, height)
    cap.set(15, -10)

    if not cap.isOpened(): print("❌ 카메라 오류"); return
    print("\n🚀 3초 후 출발!");
    for i in range(3, 0, -1): print(f"{i}.."); time.sleep(1)
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0
    last_speed_time = 0

    try:
        while True:
            if ser and ser.in_waiting > 0:
                try: ser.read(ser.in_waiting)
                except: pass

            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            # 1. 전처리
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            lower_white = np.array([0, current_l_min, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            # 2. ROI & Auto Tuning
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
            ratio = white_count / max(total_area, 1)

            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)

            # 3. 주행 계산 (2차 함수 기반)
            left_fit, right_fit = fit_polynomial(mask, roi_points)
            angle, target = calculate_steering_angle_poly(frame, left_fit, right_fit)
            
            # 각도 매핑 및 서보 값 계산
            clipped_angle = max(-45, min(45, angle))
            servo_val = int(map_value(clipped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            # 4. 통신 (Heartbeat)
            if ser:
                curr_time = time.time()
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = curr_time
                if curr_time - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = curr_time

            # 5. 시각화
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(frame, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            
            combined = np.hstack((frame, mask_bgr))
            cv2.putText(combined, f"Angle: {angle:.1f} | Servo: {servo_val}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow("Curve Lane Detection (Polynomial)", combined)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지")
        if ser:
            for _ in range(3): 
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")
                time.sleep(0.05)
            ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()