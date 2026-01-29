import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600
SERIAL_DELAY = 0.05

CAM_INDEX = 0
MAX_SPEED = 200
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# ⚙️ [핵심] 자동 튜닝 초기값 및 범위 설정
current_threshold = 165  # 시작값
MIN_THRESH = 120  # 이 밑으론 안 내려감 (너무 어두움 방지)
MAX_THRESH = 230  # 이 위론 안 올라감 (차선까지 지워짐 방지)

# 흰색 픽셀 비율 목표 (전체 화면 중 차선이 차지하는 비중)
# 보통 차선은 화면의 3% ~ 8% 정도를 차지함
TARGET_RATIO_MIN = 0.03  # 3% 미만이면 -> 너무 어둡다 (기준 낮춰!)
TARGET_RATIO_MAX = 0.08  # 8% 초과면 -> 너무 밝다 (기준 높여!)

ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

last_target_x = 320

# ==========================================
# [2] 초기화
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    time.sleep(2)
except:
    pass


# ==========================================
# [3] 함수 정의
# ==========================================
def preprocess_sunlight_lane(frame, threshold_val):
    """
    threshold_val을 인자로 받아서 동적으로 적용하는 함수
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE (이건 고정해도 됨, 임계값이 더 중요함)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # ★ 변수로 받은 threshold_val 적용
    _, binary_img = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    return binary_img


# ... (ROI, make_points, average_slope 등 기존 함수 동일 - 생략 가능하면 생략하겠지만 전체 코드 요청시 유지) ...
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
    left_fit = [];
    right_fit = []
    if lines is None: return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0];
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
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
    else:
        target_x = last_target_x

    last_target_x = target_x
    dx = target_x - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy))), int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ==========================================
# [4] 메인 실행 (자동 튜닝 로직 포함)
# ==========================================
def main():
    global current_threshold  # 전역 변수 사용

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width = 640;
    height = 480
    cap.set(3, width);
    cap.set(4, height);
    cap.set(15, -6)

    if not cap.isOpened(): return

    print("🚀 Auto-Tuning Drive Start!")
    time.sleep(3)
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())
    last_serial_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            # ------------------------------------------------
            # 1. 자동 튜닝 전처리 (Auto-Tuning)
            # ------------------------------------------------
            # 현재 Threshold 값으로 이미지 처리
            binary_mask = preprocess_sunlight_lane(frame, current_threshold)

            # ★ 흰색 픽셀 비율 계산 (ROI 영역만 보는게 정확함)
            # ROI 영역 마스크 생성
            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            # 전체 화면 말고 ROI 안쪽만 잘라서 비율 계산 (하늘이나 배경 노이즈 무시)
            roi_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(roi_mask, [roi_points], 255)
            roi_area_pixels = cv2.bitwise_and(binary_mask, roi_mask)

            # 흰색 점 개수 세기
            white_pixels = cv2.countNonZero(roi_area_pixels)
            total_pixels = cv2.contourArea(roi_points)  # ROI 면적 (대략적)
            if total_pixels == 0: total_pixels = 1

            ratio = white_pixels / total_pixels

            # ★ [핵심] 차선 굵기에 따른 Threshold 자동 조절
            # 3% ~ 8% 사이를 유지하려고 노력함
            if ratio > TARGET_RATIO_MAX:
                current_threshold = min(current_threshold + 2, MAX_THRESH)  # 너무 밝으면 기준 올림
            elif ratio < TARGET_RATIO_MIN:
                current_threshold = max(current_threshold - 2, MIN_THRESH)  # 너무 어두우면 기준 낮춤

            # ------------------------------------------------
            # 2. 이후 로직은 동일
            # ------------------------------------------------
            edges = cv2.Canny(binary_mask, 50, 150)
            cropped = region_of_interest(edges, roi_points)
            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            angle, target = calculate_steering_angle(frame, left, right)
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            if ser and (time.time() - last_serial_time > SERIAL_DELAY):
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = time.time()

            # ------------------------------------------------
            # 3. 디스플레이 (튜닝 상태 표시)
            # ------------------------------------------------
            mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
            cv2.circle(frame, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            combined = np.hstack((frame, mask_bgr))

            # ★ 현재 튜닝 상태 출력 (이걸 보면 됨)
            info_text = f"Thresh: {current_threshold} | White Ratio: {ratio * 100:.1f}%"
            status_color = (0, 255, 0)  # 정상 (초록)
            if ratio > TARGET_RATIO_MAX:
                status_color = (0, 0, 255)  # 너무 밝음 (빨강)
            elif ratio < TARGET_RATIO_MIN:
                status_color = (0, 255, 255)  # 너무 어두움 (노랑)

            cv2.putText(combined, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("Auto-Tuning Lane Keep", combined)

            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(e)
    finally:
        if ser: ser.write(b"D,0\n"); ser.write(b"S,570\n"); ser.close()
        cap.release();
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()