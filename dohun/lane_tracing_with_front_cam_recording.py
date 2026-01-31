import cv2
import numpy as np
import math
import serial
import time
import datetime

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = False

PORT = 'COM4'
BAUDRATE = 9600
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 0
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

last_target_x = 320

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

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    time.sleep(2)
except:
    ser = None

# ==========================================
# [3] 영상 처리 함수
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
    if slope == 0: slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit, right_fit = [], []
    if lines is None: return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left, right

def calculate_steering_angle(image, left, right):
    global last_target_x
    h, w = image.shape[:2]
    car_x = w / 2
    target_y = int(h * ROI_HEIGHT_RATIO)

    if left and right:
        target_x = (left[0][2] + right[0][2]) / 2
    elif left:
        target_x = left[0][2] + (w * 0.25)
    elif right:
        target_x = right[0][2] - (w * 0.25)
    else:
        target_x = last_target_x

    last_target_x = target_x
    dx = target_x - car_x
    dy = h - target_y
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

    if not cap.isOpened():
        return

    # ===============================
    # VIDEO RECORDING SETUP
    # ===============================
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"record_{timestamp}.avi"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:
        fps = 30

    out = cv2.VideoWriter(video_path, fourcc, fps, (width * 2, height))
    print(f"🎥 녹화 시작: {video_path}")

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0
    last_speed_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            mask = cv2.inRange(
                hls,
                np.array([0, current_l_min, 0]),
                np.array([179, 255, S_MAX_VAL])
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            roi_points = np.array([[
                (0, height), (width, height),
                (int(width * ROI_X_RIGHT_RATIO), int(height * ROI_HEIGHT_RATIO)),
                (int(width * ROI_X_LEFT_RATIO), int(height * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            edges = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges, roi_points)
            lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, 40, 100)

            left, right = average_slope_intercept(frame, lines)
            angle, target = calculate_steering_angle(frame, left, right)

            servo = int(map_value(
                max(-45, min(45, angle)),
                -45, 45,
                SERVO_LEFT_MAX, SERVO_RIGHT_MAX
            ))

            if ser:
                t = time.time()
                if t - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo}\n".encode())
                    last_serial_time = t
                if t - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = t

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_bgr))

            out.write(combined)

            cv2.imshow("HLS + ROI Visualized", combined)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        if ser:
            ser.write(b"D,0\n")
            ser.close()
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 녹화 종료")

if __name__ == "__main__":
    main()
