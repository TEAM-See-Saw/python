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
IS_SUNNY = True  # ★ 환경에 맞춰 설정

ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

# ✅ 네 단독 라인트레이싱 코드와 "카메라 인덱스/오픈 방식" 동일
CAM_IDX_TRAFFIC = 0   # 신호등 카메라
CAM_IDX_LANE = 1      # 차선 카메라

# --- 통신 설정 ---
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# --- 속도 & 모터 설정 ---
SPEED_NORMAL = 120
SPEED_SLOW = 50        # ✅ 감속이 되게 조정(원하면 100으로 되돌려도 됨)
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 장애물 회피 설정 ---
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2
OBSTACLE_CLEAR_TIME = 1.5

# --- ROI 설정 ---
ROI_LANE_HEIGHT_RATIO = 0.6
ROI_LANE_X_LEFT = 0.3125
ROI_LANE_X_RIGHT = 0.6875

# --- 횡단보도 설정 ---
CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# --- 자동 튜닝 설정 ---
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

print("\n" + "=" * 40)
if IS_SUNNY:
    print("   ☀️  현재 모드: SUNNY (햇빛 강함) ☀️")
    current_l_min = 200
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX = 50
    MORPH_SIZE = (5, 5)
    BLUR_K = 7
else:
    print("   🌙  현재 모드: NORMAL (실내/흐림) 🌙")
    current_l_min = 140
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX = 80
    MORPH_SIZE = (3, 3)
    BLUR_K = 5
print("=" * 40 + "\n")

# ==========================================
# [2] 함수 정의
# ==========================================
sonar_data = [999] * 6
ser = None  # read_sensors에서 사용


def read_sensors():
    global sonar_data, ser
    if ser is not None:
        while ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("US:"):
                    parts = line.replace("US:", "").split(",")
                    if len(parts) == 6:
                        sonar_data = [int(p) for p in parts]
            except:
                pass


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
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


def calculate_avoid_angle(image, left_line, right_line, obstacle_dist, last_angle, direction):
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

    if direction != 0:
        calc_dist = min(obstacle_dist, OBSTACLE_START_DIST)
        shift_amount = (OBSTACLE_START_DIST - calc_dist) * SHIFT_GAIN
        if direction == -1:
            final_target = base_target - shift_amount
        elif direction == 1:
            final_target = base_target + shift_amount

    dx = final_target - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy))), int(final_target), int(shift_amount)


def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


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
            if x2 - x1 == 0:
                continue
            ang = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(ang) < 30:
                cv2.line(frame_to_draw, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 255, 255), 3)
                detected = True
    return detected


# ==========================================
# [3] 초기화
# ==========================================
lidar = None
cap_traffic = None
cap_lane = None
camera_lib = None
video_writer = None

try:
    # 시리얼 / 라이다 / 신호등 라이브러리
    ser = serial.Serial(ARDUINO_PORT, 115200, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    camera_lib = libCAMERA()

    # ✅ 카메라 오픈: 네 단독 코드와 "정확히 동일"
    cap_traffic = cv2.VideoCapture(CAM_IDX_TRAFFIC, cv2.CAP_DSHOW)
    cap_lane = cv2.VideoCapture(CAM_IDX_LANE, cv2.CAP_DSHOW)

    width, height = 640, 480

    cap_traffic.set(3, width)
    cap_traffic.set(4, height)
    cap_traffic.set(15, -6)  # ✅ 단독 코드와 동일

    cap_lane.set(3, width)
    cap_lane.set(4, height)
    cap_lane.set(15, -6)     # ✅ 단독 코드와 동일

    if not cap_traffic.isOpened() or not cap_lane.isOpened():
        raise Exception("❌ 카메라 연결 실패 (인덱스 확인 필요)")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('final_run.avi', fourcc, 20.0, (1280, 480))

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

is_crosswalk_stop = False
crosswalk_start_time = 0.0
crosswalk_cooldown_timer = 0.0
crosswalk_detect_timer = 0.0

obstacle_count = 0
is_obstacle_detected = False
obs_clear_finished_time = 0.0
obstacle_last_seen_time = 0.0

try:
    print("🚀 주행 시작")
    for scan in lidar.iter_scans():
        read_sensors()

        # 1) 라이다 전방 최소거리
        raw_dist = 2000
        for (_, ang, dist) in scan:
            if 200 < dist < 1500:
                if ang >= 330 or ang <= 30:
                    if dist < raw_dist:
                        raw_dist = dist

        # 2) 영상 읽기
        ret_t, frame_traffic = cap_traffic.read()
        ret_l, frame_lane = cap_lane.read()
        if not ret_t or not ret_l:
            print("❌ 카메라 신호 끊김")
            break

        frame_traffic = cv2.resize(frame_traffic, (640, 480))
        frame_lane = cv2.resize(frame_lane, (640, 480))
        h, w = frame_lane.shape[:2]

        # 3) [Cam 0] 신호등
        traffic_light = camera_lib.object_detection(frame_traffic, sample=3, print_enable=False)

        # 4) [Cam 1] 차선 처리 & 바이너리
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

        white_count = cv2.countNonZero(roi_pixels)
        total_area = cv2.contourArea(lane_roi_points)
        if total_area == 0:
            total_area = 1
        ratio = white_count / total_area

        # 자동 튜닝
        tuning_msg = "Hold"
        if ratio < CROSSWALK_RATIO_MIN:
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
                tuning_msg = "Up (Dark)"
            elif ratio < TARGET_RATIO_MIN:
                if ratio < 0.005:
                    current_l_min = max(current_l_min - 10, MIN_L_VAL)
                    tuning_msg = "!! DOWN FAST !!"
                else:
                    current_l_min = max(current_l_min - 2, MIN_L_VAL)
                    tuning_msg = "Down (Bright)"

        # 5) 주행 로직
        status_msg = "NORMAL"
        status_color = (0, 255, 0)
        final_speed = SPEED_NORMAL
        current_time = time.time()
        avoid_direction = 0

        # (A) 장애물
        if raw_dist < OBSTACLE_START_DIST:
            obstacle_last_seen_time = current_time
            if not is_obstacle_detected:
                if current_time - obs_clear_finished_time > 1.5:
                    obstacle_count += 1
                    is_obstacle_detected = True
                    print(f"⚠️ 장애물 #{obstacle_count} 감지!")

            if obstacle_count == 1:
                avoid_direction = -1
                status_msg = "AVOID LEFT (#1)"
                status_color = (0, 255, 255)
            elif obstacle_count == 2:
                avoid_direction = 1
                status_msg = "AVOID RIGHT (#2)"
                status_color = (255, 0, 255)
            else:
                status_msg = f"OBSTACLE #{obstacle_count}"
                final_speed = SPEED_SLOW

        # (B) 장애물 없음 (복귀)
        else:
            if is_obstacle_detected:
                if current_time - obstacle_last_seen_time < OBSTACLE_CLEAR_TIME:
                    if obstacle_count == 1:
                        avoid_direction = -1
                    elif obstacle_count == 2:
                        avoid_direction = 1
                    status_msg = f"CLEARING... ({current_time - obstacle_last_seen_time:.1f}s)"
                    status_color = (0, 100, 255)
                else:
                    is_obstacle_detected = False
                    obs_clear_finished_time = current_time
                    print("✅ 복귀")
                    avoid_direction = 0
            else:
                avoid_direction = 0

        # (C) 횡단보도 및 신호등
        if is_crosswalk_stop:
            final_speed = SPEED_STOP
            elapsed = current_time - crosswalk_start_time
            if traffic_light == "GREEN":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                print("🟢 GO!")
            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                print("⚠️ Timeout GO!")
            else:
                status_msg = f"WAIT GREEN.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        elif (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
            if detect_stop_line(mask, mask_bgr, ROI_LANE_HEIGHT_RATIO):
                if traffic_light == "RED":
                    if crosswalk_detect_timer == 0:
                        crosswalk_detect_timer = current_time
                    elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                        is_crosswalk_stop = True
                        crosswalk_start_time = current_time
                        final_speed = SPEED_STOP
                        status_msg = "STOP! (Line+RED)"
                        status_color = (0, 0, 255)
                        crosswalk_detect_timer = 0
                    else:
                        status_msg = "Checking Line+RED..."
                else:
                    crosswalk_detect_timer = 0
                    status_msg = f"Line Detected ({traffic_light})"
            else:
                crosswalk_detect_timer = 0
                if ratio > 0.3:
                    status_msg = "Glare Ignored"
        else:
            crosswalk_detect_timer = 0
            if raw_dist < 800:
                final_speed = SPEED_SLOW

        # 6) 조향 (라인 기반 + 회피 shift)
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, lane_roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        l_line, r_line = average_slope_intercept(frame_lane, lines)

        eff_dist = raw_dist
        if avoid_direction != 0 and raw_dist > OBSTACLE_START_DIST:
            eff_dist = OBSTACLE_START_DIST / 2

        final_angle, target_px, shift_val = calculate_avoid_angle(
            frame_lane, l_line, r_line, eff_dist, last_valid_angle, avoid_direction
        )
        last_valid_angle = final_angle

        # 7) 통신
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                pwm_cmd = map_servo(final_angle)
                ser.write(f"S,{pwm_cmd}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # 8) 디스플레이
        cv2.putText(frame_traffic, f"Light: {traffic_light}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if traffic_light == "RED":
            cv2.circle(frame_traffic, (50, 100), 20, (0, 0, 255), -1)
        elif traffic_light == "GREEN":
            cv2.circle(frame_traffic, (50, 100), 20, (0, 255, 0), -1)

        cv2.polylines(mask_bgr, [lane_roi_points], True, (0, 100, 255), 2)
        cv2.circle(mask_bgr, (target_px, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        cv2.putText(mask_bgr, f"L-Min:{current_l_min} ({tuning_msg})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(mask_bgr, f"Ratio: {ratio * 100:.1f}%", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(mask_bgr, status_msg, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        combined_view = np.hstack((frame_traffic, mask_bgr))
        cv2.imshow("Dual View", combined_view)

        if video_writer is not None:
            video_writer.write(combined_view)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("사용자 종료")

finally:
    if lidar:
        try:
            lidar.stop()
            lidar.disconnect()
        except:
            pass
    if ser:
        try:
            ser.write(b"D,0\n")
            ser.write(b"S,570\n")
            ser.close()
        except:
            pass
    if video_writer is not None:
        video_writer.release()
    if cap_traffic is not None:
        cap_traffic.release()
    if cap_lane is not None:
        cap_lane.release()
    cv2.destroyAllWindows()
