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
SPEED_NORMAL = 100
SPEED_SLOW = 70          # 실 감속 속도 (NORMAL보다 낮게)
SPEED_ULTRA_SLOW = 60    # 매우 근거리 장애물에서 추가 감속
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 장애물 회피 설정 ---
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2

# --- 회피/복귀 상태 머신 ---
OBSTACLE_CLEAR_DIST = 1200      # 회피 종료 판정(START보다 크게: 히스테리시스)
RECOVER_TIME = 0.8              # 회피 직후 복귀 유지 시간(초)
SHIFT_DECAY_PER_SEC = 2200.0    # shift를 0으로 되돌리는 속도(px/s)
MAX_SHIFT_PX = 260              # shift 상한(px)

# --- 라인 중앙 추정 안정화 ---
LANE_WIDTH_ALPHA = 0.2
DEFAULT_LANE_WIDTH_PX = 280

# --- ROI 설정 ---
ROI_LANE_HEIGHT_RATIO = 0.6
ROI_TRAFFIC_HEIGHT = ROI_LANE_HEIGHT_RATIO - 0.05
ROI_LANE_X_LEFT = 0.3125
ROI_LANE_X_RIGHT = 0.6875

# --- 횡단보도(정지선) 설정 ---
CROSSWALK_RATIO_MIN = 0.30
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# --- 자동 튜닝 설정 (Auto-Tuning) ---
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

if IS_SUNNY:
    print("☀️ 모드: SUNNY (Auto-Tuning ON)")
    current_l_min = 200
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    print("🌙 모드: NORMAL (Auto-Tuning ON)")
    current_l_min = 140
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
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = fit[0], fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line  # ✅ 누락돼 있던 return 추가


def update_and_fill_lane_lines(image, left_line, right_line, lane_width_px):
    """
    - 양쪽 라인이 있으면 lane_width_px(차선폭)를 EMA로 업데이트
    - 한쪽 라인만 있으면 lane_width_px로 반대편 라인을 가상 생성
    """
    h, w = image.shape[:2]

    # lane_width 업데이트(양쪽 라인이 있을 때)
    if left_line is not None and right_line is not None:
        lx2 = left_line[0][2]
        rx2 = right_line[0][2]
        cur_width = abs(rx2 - lx2)

        if cur_width > 50:
            if lane_width_px is None:
                lane_width_px = float(cur_width)
            else:
                lane_width_px = (1 - LANE_WIDTH_ALPHA) * lane_width_px + LANE_WIDTH_ALPHA * float(cur_width)

    if lane_width_px is None:
        lane_width_px = float(DEFAULT_LANE_WIDTH_PX)

    # 한쪽 라인만 있을 때 가상 라인 생성
    if left_line is None and right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        left_line = [[int(x1 - lane_width_px), y1, int(x2 - lane_width_px), y2]]
    elif right_line is None and left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        right_line = [[int(x1 + lane_width_px), y1, int(x2 + lane_width_px), y2]]

    return left_line, right_line, lane_width_px


def calculate_avoid_angle(image, left_line, right_line, shift_px, last_angle):
    """shift_px: 바깥(상태머신)에서 계산/감쇠한 회피량(px)"""
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
        # 라인 완전 유실: 마지막 각 유지
        return last_angle, int(car_x + (last_angle * 5)), 0

    shift_px = max(0.0, min(float(shift_px), float(MAX_SHIFT_PX)))
    final_target = base_target - shift_px

    dx = final_target - car_x
    dy = (height - target_y)
    angle = math.degrees(math.atan2(dx, abs(dy)))
    return angle, int(final_target), int(shift_px)


def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


def detect_stop_line(frame, roi_ratio=0.6):
    h, w = frame.shape[:2]
    roi_h = int(h * roi_ratio)
    roi = frame[roi_h:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=60, maxLineGap=20)
    if lines is None:
        return False

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        ang = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        length = abs(x2 - x1)

        if abs(ang) < 15 and length > (w * 0.30):
            cv2.line(frame, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 255, 0), 3)
            return True

    return False


# ==========================================
# [3] 초기화
# ==========================================
lidar = None
ser = None
cam0 = None
video_writer = None

try:
    ser = serial.Serial(ARDUINO_PORT, 115200, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)

    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)

    cam0.set(3, 640)
    cam0.set(4, 480)
    cam0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    if not cam0.isOpened():
        raise Exception("카메라 에러")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('mission_tuning.avi', fourcc, 20.0, (640, 480))

    print("✅ 준비 완료 (3초 대기)")
    time.sleep(3)
    if ser:
        ser.write(f"D,{SPEED_NORMAL}\n".encode())

except Exception as e:
    print(f"❌ 초기화 오류: {e}")
    raise SystemExit


# ==========================================
# [4] 메인 루프
# ==========================================
last_valid_angle = 0.0
last_serial_time = 0.0
last_speed_time = 0.0

# 회피/복귀 상태
avoid_state = "NORMAL"   # "NORMAL" | "AVOID" | "RECOVER"
recover_start_time = 0.0
shift_px_hold = 0.0

# 차선폭 기억
lane_width_px = None

# stop-line 상태
is_crosswalk_stop = False
crosswalk_start_time = 0.0
crosswalk_cooldown_timer = 0.0
crosswalk_detect_timer = 0.0

prev_time = time.time()

try:
    print("🚀 Auto-Tuning 주행 시작!")
    for scan in lidar.iter_scans():
        current_time = time.time()
        dt = max(0.001, current_time - prev_time)
        prev_time = current_time

        # 1) 라이다 거리(전방 최소)
        raw_dist = 2000
        for (_, ang, dist) in scan:
            if 200 < dist < 1500:
                if ang >= 330 or ang <= 30:
                    if dist < raw_dist:
                        raw_dist = dist

        # 2) 카메라 프레임
        ret, frame = cam0.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        blurred = cv2.medianBlur(frame, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        mask = cv2.inRange(hls, np.array([0, current_l_min, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

        # 3) 신호등
        traffic_roi_h = int(h * ROI_TRAFFIC_HEIGHT)
        traffic_frame = frame[0:traffic_roi_h, :]
        traffic_light = camera.object_detection(traffic_frame, sample=3, print_enable=False)

        # 4) 흰색 비율
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
        if total_area == 0:
            total_area = 1
        ratio = white_count / total_area

        # Auto-tuning (횡단보도 아닐 때만)
        if ratio < CROSSWALK_RATIO_MIN:
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)

        # ==========================================
        # (A) 미션 로직(정지/감속)
        # ==========================================
        status_msg = "NORMAL"
        status_color = (0, 255, 0)
        final_speed = SPEED_NORMAL

        # 1) 횡단보도 정지 중
        if is_crosswalk_stop:
            final_speed = SPEED_STOP
            elapsed = current_time - crosswalk_start_time

            if traffic_light == "GREEN":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GREEN -> GO"
                status_color = (0, 255, 0)
            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "TIMEOUT -> GO"
                status_color = (0, 255, 0)
            else:
                status_msg = f"WAIT GREEN.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        # 2) 횡단보도/정지선 감지
        elif (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
            if detect_stop_line(frame, ROI_LANE_HEIGHT_RATIO):
                if crosswalk_detect_timer == 0:
                    crosswalk_detect_timer = current_time
                elif (current_time - crosswalk_detect_timer) > CROSSWALK_CONFIRM_TIME:
                    is_crosswalk_stop = True
                    crosswalk_start_time = current_time
                    final_speed = SPEED_STOP
                    status_msg = "STOP LINE DETECTED!"
                    status_color = (0, 0, 255)
                    crosswalk_detect_timer = 0
                else:
                    status_msg = "Checking Line..."
            else:
                crosswalk_detect_timer = 0
                if ratio > 0.35:
                    status_msg = "Glare Ignored"

        # 3) 일반 주행
        else:
            crosswalk_detect_timer = 0

            if traffic_light == "RED":
                final_speed = SPEED_STOP
                status_msg = "TRAFFIC RED"
                status_color = (0, 0, 255)
            elif raw_dist < 800:
                final_speed = SPEED_ULTRA_SLOW if raw_dist < 500 else SPEED_SLOW
                status_msg = f"OBSTACLE ({int(raw_dist)}mm)"
                status_color = (0, 255, 255)

        # ==========================================
        # (B) 차선 검출
        # ==========================================
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

        l_line, r_line = average_slope_intercept(frame, lines)
        l_line, r_line, lane_width_px = update_and_fill_lane_lines(frame, l_line, r_line, lane_width_px)

        # ==========================================
        # (C) 장애물 회피/복귀 상태 머신(핵심)
        # ==========================================
        # 회피 진입
        if avoid_state == "NORMAL":
            if 0 < raw_dist < OBSTACLE_START_DIST:
                avoid_state = "AVOID"

        # 회피 종료 -> 복귀
        if avoid_state == "AVOID":
            if raw_dist >= OBSTACLE_CLEAR_DIST:
                avoid_state = "RECOVER"
                recover_start_time = current_time

        # shift 계산
        if avoid_state == "AVOID":
            target_shift = (OBSTACLE_START_DIST - raw_dist) * SHIFT_GAIN
            target_shift = max(0.0, min(float(target_shift), float(MAX_SHIFT_PX)))
            shift_px_hold = target_shift
        else:
            # 부드러운 복귀(shift 감소)
            shift_px_hold = max(0.0, shift_px_hold - SHIFT_DECAY_PER_SEC * dt)

        # RECOVER 종료
        if avoid_state == "RECOVER":
            if (current_time - recover_start_time) > RECOVER_TIME and shift_px_hold <= 1.0:
                avoid_state = "NORMAL"

        # 상태 표시(디버그)
        if avoid_state == "AVOID":
            status_msg = f"{status_msg} | AVOID"
        elif avoid_state == "RECOVER":
            status_msg = f"{status_msg} | RECOVER"

        # ==========================================
        # (D) 조향 계산
        # ==========================================
        final_angle, target_px, shift_val = calculate_avoid_angle(frame, l_line, r_line, shift_px_hold, last_valid_angle)
        last_valid_angle = final_angle

        # ==========================================
        # (E) 시리얼 전송
        # ==========================================
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                pwm_cmd = map_servo(final_angle)
                ser.write(f"S,{pwm_cmd}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # ==========================================
        # (F) 디스플레이
        # ==========================================
        cv2.polylines(frame, [lane_roi_points], True, (255, 0, 0), 2)
        cv2.line(frame, (0, traffic_roi_h), (w, traffic_roi_h), (100, 100, 100), 1)
        cv2.circle(frame, (target_px, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        tune_info = f"L-Min: {current_l_min} | Ratio: {ratio * 100:.1f}% | Dist:{int(raw_dist)}"
        cv2.putText(frame, tune_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Light: {traffic_light}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(frame, status_msg, (20, traffic_roi_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
        cv2.putText(frame, f"Shift:{int(shift_val)}px | laneW:{int(lane_width_px)}", (20, traffic_roi_h + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        cv2.imshow("Mission Auto-Tuning", frame)
        if video_writer is not None:
            video_writer.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    if lidar:
        try:
            lidar.stop()
            lidar.disconnect()
        except Exception:
            pass

    if ser:
        try:
            ser.write(b"D,0\n")
            ser.write(b"S,570\n")
            ser.close()
        except Exception:
            pass

    if video_writer is not None:
        video_writer.release()

    if cam0 is not None:
        cam0.release()

    cv2.destroyAllWindows()
