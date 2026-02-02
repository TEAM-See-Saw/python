import serial
from rplidar import RPLidar
import time
import cv2
import numpy as np
import math

from Function_Library import libCAMERA

# ==========================================
# [1] 통합 환경 및 튜닝 설정
# ==========================================
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

CAM_IDX_TRAFFIC = 0
CAM_IDX_LANE = 1

# --- 통신 설정 ---
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# --- 속도 & 모터 설정 ---
SPEED_NORMAL = 150
SPEED_SLOW = 100
SPEED_STOP = 0
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 장애물 회피 설정 ---
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2
OBSTACLE_CLEAR_TIME = 1.5

# --- ROI 설정 ---
ROI_LANE_HEIGHT_RATIO = 0.5
ROI_LANE_X_LEFT = 0.0
ROI_LANE_X_RIGHT = 1.0

# --- 횡단보도/정지선 설정 ---
CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# --- 신호등 ROI 설정 (3구 하우징 타이트하게) ---
TRAFFIC_BOX_X1 = 0.22
TRAFFIC_BOX_X2 = 0.62
TRAFFIC_BOX_Y1 = 0.35
TRAFFIC_BOX_Y2 = 0.62
TRAFFIC_HIGHLIGHT_PCTL = 98
TRAFFIC_TH_MIN = 200

# --- 자동 튜닝 설정 ---
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10
last_target_x = 320

# 초기 안전장치 범위
MIN_L_VAL = 80;
MAX_L_VAL = 220;
S_MAX = 80
MORPH_SIZE = (3, 3);
BLUR_K = 3
current_l_min = 140  # 초기값 (함수로 덮어씌워짐)

print("\n" + "=" * 50)
print(" 🚀 [최종 통합] 적응형 주행 + 강력한 신호등/정지선 인식")
print("=" * 50 + "\n")

# ==========================================
# [2] 핵심 함수 정의
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    if ser is not None:
        while ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("US:"):
                    parts = line.replace("US:", "").split(",")
                    if len(parts) == 6: sonar_data = [int(p) for p in parts]
            except:
                pass


# ★ [1] 출발 전 밝기 분석 함수
def find_optimal_l_min(cap):
    print("🔎 출발 전 밝기 분석 중...")
    try:
        for i in range(20):  # 워밍업
            ret, _ = cap.read()
            if not ret: time.sleep(0.05)

        ret, frame = cap.read()
        if not ret or frame is None: return 140

        frame = cv2.resize(frame, (640, 480))
        blurred = cv2.medianBlur(frame, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        h, w = frame.shape[:2]
        roi_l = l_channel[int(h * ROI_LANE_HEIGHT_RATIO):h, :]
        pixels = np.sort(roi_l.flatten())

        target_idx = int(len(pixels) * 0.90)
        detected_l = pixels[target_idx]
        final_l = max(MIN_L_VAL, min(detected_l - 30, MAX_L_VAL))

        print(f"✅ 분석 완료! (감지:{detected_l} -> 설정:{final_l})")
        return int(final_l)
    except Exception as e:
        print(f"⚠️ 에러({e}) -> 기본값(140)")
        return 140


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


# ★ [2] 가로선 무시하고 주행 차선만 추출
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
            if abs(slope) < 0.7: continue  # 가로선 무시
            if slope < -0.7:
                left_fit.append((slope, intercept))
            elif slope > 0.7:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


# ★ [3] 고급 신호등 인식 (HSV + 밝기 Fallback)
def detect_traffic_lr_robust(frame_bgr):
    H, W = frame_bgr.shape[:2]
    x1 = int(W * TRAFFIC_BOX_X1);
    x2 = int(W * TRAFFIC_BOX_X2)
    y1 = int(H * TRAFFIC_BOX_Y1);
    y2 = int(H * TRAFFIC_BOX_Y2)
    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 5 or rw < 5: return "NONE", {}

    third = max(1, rw // 3)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)

    # 1. HSV Color Check
    S_GATE = 60;
    V_GATE = 80;
    COLOR_TH = 0.003
    sat_mask = (Ss >= S_GATE).astype(np.uint8) * 255
    red1 = cv2.inRange(hsv, (0, S_GATE, V_GATE), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, S_GATE, V_GATE), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (35, S_GATE, V_GATE), (85, 255, 255))

    red_mask = cv2.bitwise_and(cv2.bitwise_or(red1, red2), sat_mask)
    green_mask = cv2.bitwise_and(green_mask, sat_mask)

    red_left = red_mask[:, 0:third]
    green_right = green_mask[:, 2 * third:rw] if (2 * third) < rw else green_mask[:, third:rw]

    rl_ratio = cv2.countNonZero(red_left) / float(red_left.size)
    gr_ratio = cv2.countNonZero(green_right) / float(green_right.size)

    if rl_ratio > COLOR_TH: return "LEFT", {"box": (x1, y1, x2, y2), "mode": "HSV"}
    if gr_ratio > COLOR_TH: return "RIGHT", {"box": (x1, y1, x2, y2), "mode": "HSV"}

    # 2. Brightness Fallback
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    p = np.percentile(gray, TRAFFIC_HIGHLIGHT_PCTL)
    thr = int(max(TRAFFIC_TH_MIN, p))
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0 and cv2.contourArea(c) > 30:
            cx = int(M["m10"] / M["m00"])
            if cx < third:
                return "LEFT", {"box": (x1, y1, x2, y2), "mode": "BLOB"}
            elif cx > 2 * third:
                return "RIGHT", {"box": (x1, y1, x2, y2), "mode": "BLOB"}

    return "NONE", {"box": (x1, y1, x2, y2), "mode": "NONE"}


# ★ [4] 정지선 감지
def detect_stop_line(mask, frame_to_draw, roi_ratio=0.6):
    h, w = mask.shape[:2]
    roi_h = int(h * roi_ratio)
    roi = mask[roi_h:h, 0:w]
    edges = cv2.Canny(roi, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=40, maxLineGap=20)

    detected = False
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 30:  # 가로선이면
                cv2.line(frame_to_draw, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 0, 255), 3)
                detected = True
    return detected


# ★ [5] 통합 조향 (라인트레이싱 + 회피)
def calculate_steering_and_avoid(image, left_line, right_line, obstacle_dist, direction):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_LANE_HEIGHT_RATIO)

    # 기본 타겟 (안정화 적용)
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = last_target_x
    last_target_x = target_x

    # 장애물 회피 (Shift)
    final_target = target_x
    shift_amount = 0
    if direction != 0:
        calc_dist = min(obstacle_dist, OBSTACLE_START_DIST)
        shift_amount = (OBSTACLE_START_DIST - calc_dist) * SHIFT_GAIN
        if direction == -1:
            final_target = target_x - shift_amount
        elif direction == 1:
            final_target = target_x + shift_amount

    dx = final_target - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy))), int(final_target), int(shift_amount)


def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


# ==========================================
# [3] 초기화
# ==========================================
lidar = None;
ser = None
cap_traffic = None;
cap_lane = None
camera_lib = None;
video_writer = None

try:
    ser = serial.Serial(ARDUINO_PORT, 115200, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    camera_lib = libCAMERA()  # 레거시 호환용

    # Cam 0 (신호등)
    cap_traffic = cv2.VideoCapture(CAM_IDX_TRAFFIC, cv2.CAP_DSHOW)
    cap_traffic.set(3, 640);
    cap_traffic.set(4, 480);
    cap_traffic.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap_traffic.set(cv2.CAP_PROP_EXPOSURE, -6)  # 신호등은 어둡게

    # Cam 1 (차선)
    cap_lane = cv2.VideoCapture(CAM_IDX_LANE, cv2.CAP_DSHOW)
    cap_lane.set(3, 640);
    cap_lane.set(4, 480);
    cap_lane.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap_lane.set(cv2.CAP_PROP_EXPOSURE, -4)

    if not cap_traffic.isOpened() or not cap_lane.isOpened():
        raise Exception("❌ 카메라 연결 실패")

    # ★ 초기 밝기 분석
    current_l_min = find_optimal_l_min(cap_lane)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('final_run.avi', fourcc, 20.0, (1280, 480))

    print("✅ 준비 완료 (3초 대기)")
    time.sleep(3)
    if ser: ser.write(f"D,{SPEED_NORMAL}\n".encode())

except Exception as e:
    print(f"❌ 초기화 오류: {e}")
    exit()

# ==========================================
# [4] 메인 루프
# ==========================================
last_serial_time = 0
last_speed_time = 0

is_crosswalk_stop = False
crosswalk_start_time = 0
crosswalk_cooldown_timer = 0
crosswalk_detect_timer = 0

obstacle_count = 0
is_obstacle_detected = False
obs_clear_finished_time = 0
obstacle_last_seen_time = 0

try:
    print("🚀 주행 시작")
    for scan in lidar.iter_scans():
        read_sensors()

        # 1. 라이다
        raw_dist = 2000
        for (_, angle, dist) in scan:
            if 200 < dist < 1500:
                if angle >= 330 or angle <= 30:
                    if dist < raw_dist: raw_dist = dist

        # 2. 영상 읽기
        ret_t, frame_traffic = cap_traffic.read()
        ret_l, frame_lane = cap_lane.read()
        if not ret_t or not ret_l: break

        frame_traffic = cv2.resize(frame_traffic, (640, 480))
        frame_lane = cv2.resize(frame_lane, (640, 480))
        h, w = frame_lane.shape[:2]

        # 3. 신호등 인식 (새로운 Robust 함수 사용)
        traffic_state, tdbg = detect_traffic_lr_robust(frame_traffic)

        # 4. 차선 처리
        blurred = cv2.medianBlur(frame_lane, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, current_l_min, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        lane_roi_points = np.array([[
            (0, h), (w, h),
            (int(w * ROI_LANE_X_RIGHT), int(h * ROI_LANE_HEIGHT_RATIO)),
            (int(w * ROI_LANE_X_LEFT), int(h * ROI_LANE_HEIGHT_RATIO))
        ]], dtype=np.int32)

        roi_mask_poly = np.zeros_like(mask)
        cv2.fillPoly(roi_mask_poly, [lane_roi_points], 255)
        roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

        # 자동 튜닝 (횡단보도 아닐 때만)
        white_count = cv2.countNonZero(roi_pixels)
        ratio = white_count / (cv2.contourArea(lane_roi_points) + 1)
        tuning_msg = "Hold"
        if ratio < CROSSWALK_RATIO_MIN:
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL); tuning_msg = "Up"
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL); tuning_msg = "Down"

        # 5. 주행 로직
        status_msg = "NORMAL"
        status_color = (0, 255, 0)
        final_speed = SPEED_NORMAL
        current_time = time.time()
        avoid_direction = 0

        # (A) 장애물 감지
        if raw_dist < OBSTACLE_START_DIST:
            obstacle_last_seen_time = current_time
            if not is_obstacle_detected:
                if current_time - obs_clear_finished_time > 1.5:
                    obstacle_count += 1
                    is_obstacle_detected = True
                    print(f"⚠️ 장애물 #{obstacle_count} 감지!")

            if obstacle_count == 1:
                avoid_direction = -1; status_msg = "AVOID LEFT"
            elif obstacle_count == 2:
                avoid_direction = 1; status_msg = "AVOID RIGHT"
            else:
                status_msg = f"OBSTACLE #{obstacle_count}"; final_speed = SPEED_SLOW

        # (B) 장애물 복귀
        else:
            if is_obstacle_detected:
                if current_time - obstacle_last_seen_time < OBSTACLE_CLEAR_TIME:
                    if obstacle_count == 1:
                        avoid_direction = -1
                    elif obstacle_count == 2:
                        avoid_direction = 1
                    status_msg = "CLEARING..."
                else:
                    is_obstacle_detected = False;
                    obs_clear_finished_time = current_time
                    avoid_direction = 0
            else:
                avoid_direction = 0

        # (C) 정지선 + 신호등
        if is_crosswalk_stop:
            final_speed = SPEED_STOP
            elapsed = current_time - crosswalk_start_time

            if traffic_state == "RIGHT":  # 초록불(오른쪽) 인식
                is_crosswalk_stop = False;
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0;
                status_msg = "GO (RIGHT ON)"
            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False;
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0;
                status_msg = "GO (TIMEOUT)"
            else:
                status_msg = f"WAIT RIGHT.. ({elapsed:.1f}s)";
                status_color = (0, 0, 255)

        elif (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
            if detect_stop_line(mask, mask_bgr, ROI_LANE_HEIGHT_RATIO):
                if traffic_state == "LEFT":  # 빨간불(왼쪽)일 때만
                    if crosswalk_detect_timer == 0:
                        crosswalk_detect_timer = current_time
                    elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                        is_crosswalk_stop = True;
                        crosswalk_start_time = current_time
                        final_speed = SPEED_STOP;
                        status_msg = "STOP! (Line+LEFT ON)";
                        status_color = (0, 0, 255)
                        crosswalk_detect_timer = 0
                    else:
                        status_msg = "Checking..."
                else:
                    crosswalk_detect_timer = 0; status_msg = f"Line Detected ({traffic_state})"
            else:
                crosswalk_detect_timer = 0
        else:
            crosswalk_detect_timer = 0
            if raw_dist < 800 and avoid_direction == 0: final_speed = SPEED_SLOW

        # 6. 조향 계산
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        l_line, r_line = average_slope_intercept(frame_lane, lines)

        eff_dist = raw_dist
        if avoid_direction != 0 and raw_dist > OBSTACLE_START_DIST: eff_dist = OBSTACLE_START_DIST / 2

        angle, target_px, shift_val = calculate_steering_and_avoid(
            frame_lane, l_line, r_line, eff_dist, avoid_direction
        )

        # 7. 전송
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                pwm = map_servo(angle)
                ser.write(f"S,{pwm}\n".encode())
                last_serial_time = current_time
            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # 8. 디스플레이
        box = tdbg.get("box");
        mode = tdbg.get("mode", "-")
        if box: cv2.rectangle(frame_traffic, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
        cv2.putText(frame_traffic, f"Light: {traffic_state} ({mode})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        cv2.polylines(mask_bgr, [lane_roi_points], True, (0, 100, 255), 2)
        cv2.circle(mask_bgr, (target_px, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
        cv2.putText(mask_bgr, f"L:{current_l_min} {tuning_msg} | {status_msg}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    status_color, 2)

        combined = np.hstack((frame_traffic, mask_bgr))
        cv2.imshow("Main View", combined)
        if video_writer: video_writer.write(combined)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("종료")
finally:
    if ser: ser.write(b"D,0\n"); ser.write(b"S,570\n"); ser.close()
    if lidar: lidar.stop(); lidar.disconnect()
    cv2.destroyAllWindows()