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
IS_SUNNY = True
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

# --- 통신 설정 ---
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# --- 속도 & 모터 설정 ---
SPEED_NORMAL = 200
SPEED_SLOW = 100
SPEED_STOP = 0
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 장애물 회피 설정 ---
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2

# --- ROI 설정 ---
ROI_LANE_HEIGHT_RATIO = 0.6
ROI_TRAFFIC_HEIGHT = ROI_LANE_HEIGHT_RATIO - 0.05
ROI_LANE_X_LEFT = 0.3125
ROI_LANE_X_RIGHT = 0.6875

# --- 횡단보도 설정 ---
CROSSWALK_RATIO_MIN = 0.30  # 30% 이상이면 횡단보도
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0

# --- ★ [추가] 자동 튜닝 설정 (Auto-Tuning) ---
# 목표: 흰색 비율을 3% ~ 10% 사이로 유지
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

if IS_SUNNY:
    print("☀️ 모드: SUNNY (Auto-Tuning ON)")
    current_l_min = 200  # 시작값 (높게)
    MIN_L_VAL = 150  # 너무 어두워지지 않게 하한선 방어
    MAX_L_VAL = 240  # 상한선
    S_MAX = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    print("🌙 모드: NORMAL (Auto-Tuning ON)")
    current_l_min = 140  # 시작값 (평범하게)
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX = 80
    MORPH_SIZE = (3, 3)
    BLUR_K = 5


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
    y1 = image.shape[0]
    y2 = int(y1 * ROI_LANE_HEIGHT_RATIO)
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
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_LANE_HEIGHT_RATIO)

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
lidar = None;
ser = None;
cam0 = None;
video_writer = None

try:
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)

    cam0.set(3, 640);
    cam0.set(4, 480)
    cam0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cam0.isOpened(): raise Exception("카메라 에러")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('mission_tuning.avi', fourcc, 20.0, (640, 480))

    print("✅ 준비 완료 (3초 대기)")
    time.sleep(3)
    if ser: ser.write(f"D,{SPEED_NORMAL}\n".encode())

except Exception as e:
    print(f"❌ 초기화 오류: {e}")
    exit()

# ==========================================
# [4] 메인 루프
# ==========================================
last_valid_angle = 0
last_serial_time = 0
last_speed_time = 0

is_crosswalk_stop = False
crosswalk_start_time = 0
crosswalk_cooldown_timer = 0

try:
    print("🚀 Auto-Tuning 주행 시작!")
    for scan in lidar.iter_scans():
        # 1. 라이다 거리 측정
        raw_dist = 2000
        for (_, angle, dist) in scan:
            if 200 < dist < 1500:
                if angle >= 330 or angle <= 30:
                    if dist < raw_dist: raw_dist = dist

        # 2. 영상 처리
        ret, frame = cam0.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        blurred = cv2.medianBlur(frame, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        # ★ [핵심] 자동 튜닝된 current_l_min 값 적용
        mask = cv2.inRange(hls, np.array([0, current_l_min, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

        # 3. 객체 인식 (신호등)
        traffic_roi_h = int(h * ROI_TRAFFIC_HEIGHT)
        traffic_frame = frame[0:traffic_roi_h, :]
        traffic_light = camera.object_detection(traffic_frame, sample=3, print_enable=False)

        # 4. 흰색 비율 계산 (차선 ROI 기준)
        lane_roi_points = np.array([[
            (0, h), (w, h),
            (int(w * ROI_LANE_X_RIGHT), int(h * ROI_LANE_HEIGHT_RATIO)),
            (int(w * ROI_LANE_X_LEFT), int(h * ROI_LANE_HEIGHT_RATIO))
        ]], dtype=np.int32)

        roi_mask_poly = np.zeros_like(mask)
        cv2.fillPoly(roi_mask_poly, [lane_roi_points], 255)
        roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

        white_count = cv2.countNonZero(roi_pixels)
        total_area = cv2.contourArea(lane_roi_points)
        if total_area == 0: total_area = 1
        ratio = white_count / total_area

        # ★ [핵심] Auto-Tuning 로직 (횡단보도 아닐 때만 작동)
        # 횡단보도(비율 > 30%)일 때 튜닝하면 L_MIN이 폭주하므로 막아야 함
        if ratio < CROSSWALK_RATIO_MIN:
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)  # 너무 밝으면 기준 올림
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)  # 너무 어두우면 기준 낮춤

        # ==========================================================
        # 미션 & 주행 로직
        # ==========================================================
        status_msg = "NORMAL"
        status_color = (0, 255, 0)
        final_speed = SPEED_NORMAL
        current_time = time.time()

        # (1) 횡단보도 정지
        if is_crosswalk_stop:
            final_speed = SPEED_STOP
            elapsed = current_time - crosswalk_start_time

            if traffic_light == "GREEN":  # 초록불 출발
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                print(f"🟢 Green Light! Go!")
            elif elapsed > CROSSWALK_MAX_WAIT:  # 7초 타임아웃 출발
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                print(f"⚠️ Timeout! Go!")
            else:
                status_msg = f"WAIT GREEN.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        # (2) 횡단보도 감지
        elif (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
            is_crosswalk_stop = True
            crosswalk_start_time = current_time
            final_speed = SPEED_STOP
            status_msg = "CROSSWALK STOP"
            status_color = (0, 0, 255)

        # (3) 신호등 정지
        elif traffic_light == "RED":
            final_speed = SPEED_STOP
            status_msg = "TRAFFIC RED"
            status_color = (0, 0, 255)

        # (4) 장애물 감속
        elif raw_dist < 800:
            final_speed = SPEED_SLOW
            status_msg = f"OBSTACLE ({int(raw_dist)}mm)"
            status_color = (0, 255, 255)

        # ----------------------------------------
        # 5. 조향 & 통신
        # ----------------------------------------
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        l_line, r_line = average_slope_intercept(frame, lines)

        final_angle, target_px, shift_val = calculate_avoid_angle(frame, l_line, r_line, raw_dist, last_valid_angle)
        last_valid_angle = final_angle

        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                pwm_cmd = map_servo(final_angle)
                ser.write(f"S,{pwm_cmd}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # ----------------------------------------
        # 6. 디스플레이
        # ----------------------------------------
        cv2.polylines(frame, [lane_roi_points], True, (255, 0, 0), 2)
        cv2.line(frame, (0, traffic_roi_h), (w, traffic_roi_h), (100, 100, 100), 1)
        cv2.circle(frame, (target_px, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        # 튜닝 정보 출력 (왼쪽 상단)
        tune_info = f"L-Min: {current_l_min} | Ratio: {ratio * 100:.1f}%"
        cv2.putText(frame, tune_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, f"Light: {traffic_light}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, status_msg, (20, traffic_roi_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.imshow("Mission Auto-Tuning", frame)
        if video_writer is not None: video_writer.write(frame)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("사용자 종료")
finally:
    if lidar: lidar.stop(); lidar.disconnect()
    if ser: ser.write(b"D,0\n"); ser.write(b"S,570\n"); ser.close()
    if video_writer is not None: video_writer.release()
    cam0.release();
    cv2.destroyAllWindows()