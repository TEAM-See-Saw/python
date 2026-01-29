import serial
from rplidar import RPLidar
import time
import cv2
import numpy as np
import math
from Function_Library import libCAMERA

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = False
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'
SERIAL_DELAY = 0.05

# --- 속도 & 모터 설정 ---
SPEED_NORMAL = 200
SPEED_SLOW = 100
SPEED_STOP = 0
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2

# ========================================================
# ★ [핵심] ROI 자동 연동 설정 (여기만 고치세요!)
# ========================================================
# 1. 차선 ROI 비율 (Master)
# 카메라를 들수록 이 값을 키우세요. (0.6 -> 0.7 -> 0.8)
ROI_LANE_HEIGHT_RATIO = 0.6

# 2. 신호등 ROI 비율 (Slave - 자동 계산)
# 차선 ROI보다 5%(0.05) 위쪽까지만 봅니다. (겹침 방지 버퍼)
ROI_TRAFFIC_HEIGHT = ROI_LANE_HEIGHT_RATIO - 0.05

# 3. 차선 사다리꼴 폭 (황금 비율 고정)
ROI_LANE_X_LEFT = 0.3125
ROI_LANE_X_RIGHT = 0.6875

# 영상 처리 필터
if IS_SUNNY:
    L_MIN = 160;
    S_MAX = 50;
    EXPOSURE = -9;
    MORPH_SIZE = (5, 5)
else:
    L_MIN = 80;
    S_MAX = 120;
    EXPOSURE = -4;
    MORPH_SIZE = (3, 3)


# ==========================================
# [2] 함수 정의
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
    y1 = image.shape[0];
    y2 = int(y1 * ROI_LANE_HEIGHT_RATIO)  # ★ 변수 사용
    if slope == 0: slope = 0.001
    x1 = int((y1 - intercept) / slope);
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = [];
    right_fit = []
    if lines is None: return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: continue
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


def calculate_avoid_angle(image, left_line, right_line, obstacle_dist, last_angle):
    height, width, _ = image.shape
    car_x = width / 2

    target_y = int(height * ROI_LANE_HEIGHT_RATIO)  # ★ 변수 사용

    if left_line is not None and right_line is not None:
        base_target = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        base_target = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        base_target = right_line[0][2] - (width * 0.25)
    else:
        return last_angle, int(car_x + (last_angle * 5)), 0

    final_target = base_target
    shift_amount = 0

    if 0 < obstacle_dist < OBSTACLE_START_DIST:
        shift_amount = (OBSTACLE_START_DIST - obstacle_dist) * SHIFT_GAIN
        final_target = base_target - shift_amount

    dx = final_target - car_x
    dy = (height - target_y)
    angle = math.degrees(math.atan2(dx, abs(dy)))

    return angle, int(final_target), int(shift_amount)


def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


# ==========================================
# [3] 초기화
# ==========================================
video_writer = None
ser = None

try:
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)
    cam0.set(3, 640);
    cam0.set(4, 480);
    cam0.set(15, EXPOSURE)

    if not cam0.isOpened(): raise Exception("카메라 에러")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('mission_auto_roi.avi', fourcc, 20.0, (640, 480))

    print(f"✅ 설정 완료 | 차선ROI: {ROI_LANE_HEIGHT_RATIO * 100:.1f}% | 신호등ROI: {ROI_TRAFFIC_HEIGHT * 100:.1f}%")
    time.sleep(2)

    if ser: ser.write(f"D,{SPEED_NORMAL}\n".encode())

except Exception as e:
    print(f"❌ 초기화 오류: {e}");
    exit()

# ==========================================
# [4] 메인 루프
# ==========================================
last_valid_angle = 0
last_serial_time = 0

try:
    for scan in lidar.iter_scans():
        raw_dist = 2000
        for (_, angle, dist) in scan:
            if 200 < dist < 1500:
                if angle >= 330 or angle <= 30:
                    if dist < raw_dist: raw_dist = dist

        ret, frame = cam0.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        # [A] 신호등 ROI (자동 계산된 비율 적용)
        traffic_roi_h = int(h * ROI_TRAFFIC_HEIGHT)
        traffic_frame = frame[0:traffic_roi_h, :]
        traffic_light = camera.object_detection(traffic_frame, sample=3, print_enable=False)

        # [B] 차선 ROI (설정된 비율 적용)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, L_MIN, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        edges = cv2.Canny(mask, 50, 150)

        lane_roi_points = np.array([[
            (0, h),
            (w, h),
            (int(w * ROI_LANE_X_RIGHT), int(h * ROI_LANE_HEIGHT_RATIO)),
            (int(w * ROI_LANE_X_LEFT), int(h * ROI_LANE_HEIGHT_RATIO))
        ]], dtype=np.int32)

        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        l_line, r_line = average_slope_intercept(frame, lines)

        # 제어
        final_angle, target_px, shift_val = calculate_avoid_angle(frame, l_line, r_line, raw_dist, last_valid_angle)
        last_valid_angle = final_angle

        final_speed = SPEED_NORMAL
        status_msg = "Run: Normal"

        if raw_dist < 800:
            final_speed = SPEED_SLOW
            status_msg = "Run: AVOID"

        if traffic_light == "RED":
            final_speed = SPEED_STOP
            status_msg = "XXX RED STOP XXX"

        # 전송 (딜레이 적용)
        current_time = time.time()
        if ser and (current_time - last_serial_time > SERIAL_DELAY):
            pwm_cmd = map_servo(final_angle)
            ser.write(f"S,{pwm_cmd}\n".encode())
            ser.write(f"D,{final_speed}\n".encode())
            last_serial_time = current_time

        # 디스플레이
        # 1. 신호등 영역 표시 (빨간 박스)
        cv2.rectangle(frame, (0, 0), (w, traffic_roi_h), (0, 0, 255), 2)

        # 2. 차선 영역 표시 (파란 사다리꼴)
        cv2.polylines(frame, [lane_roi_points], True, (255, 0, 0), 2)

        # 3. 사이 공간 (버퍼존) 표시 - 회색 빗금 느낌 (선 하나 긋기)
        # 신호등 끝선과 차선 시작선 사이의 빈 공간
        cv2.line(frame, (0, traffic_roi_h), (w, traffic_roi_h), (100, 100, 100), 1)

        # 목표점
        cv2.circle(frame, (target_px, int(h * ROI_LANE_HEIGHT_RATIO)), 15, (0, 255, 255), -1)
        if l_line is not None:
            for x1, y1, x2, y2 in l_line: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if r_line is not None:
            for x1, y1, x2, y2 in r_line: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.putText(frame, f"Lidar: {int(raw_dist)}mm", (20, traffic_roi_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)
        cv2.putText(frame, status_msg, (20, traffic_roi_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Mission Auto ROI", frame)
        if video_writer is not None: video_writer.write(frame)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("종료")
finally:
    if lidar: lidar.stop(); lidar.disconnect()
    if ser: ser.write(b"D,0\n"); ser.close()
    if video_writer is not None: video_writer.release()
    cam0.release();
    cv2.destroyAllWindows()