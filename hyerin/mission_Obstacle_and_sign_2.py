import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar
from Function_Library import libCAMERA

# ==========================================
# [1] 환경 및 튜닝 설정  (✅ 위 단독 코드와 동일)
# ==========================================
IS_SUNNY = True

PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# ✅ 카메라 인덱스: 위 코드와 동일하게 "차선 카메라 = 1"
CAM_INDEX = 1
# 추가로 신호등 카메라만 별도 사용
CAM_INDEX_TRAFFIC = 0

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
# [추가] 미션/회피/정지선 관련 (라이다/신호등/횡단보도)
# ==========================================
LIDAR_PORT = 'COM3'

OBSTACLE_START_DIST = 1000   # mm
SHIFT_GAIN = 1.2             # px per (mm 부족분) 스케일
OBSTACLE_CLEAR_TIME = 1.5

CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# ==========================================
# [2] 시리얼/라이다/카메라 연결
# ==========================================
ser = None
lidar = None
camera_lib = None
cap_lane = None
cap_traffic = None

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 시리얼 연결 실패: {e}")

try:
    lidar = RPLidar(LIDAR_PORT)
    camera_lib = libCAMERA()
except Exception as e:
    print(f"❌ 라이다/카메라라이브러리 초기화 실패: {e}")

# ✅ 위 코드와 "완전 동일"한 카메라 오픈/세팅
cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)

width = 640
height = 480

cap_lane.set(3, width)
cap_lane.set(4, height)
cap_lane.set(15, -6)

cap_traffic.set(3, width)
cap_traffic.set(4, height)
cap_traffic.set(15, -6)

if not cap_lane.isOpened():
    print("❌ 차선 카메라 오류")
if not cap_traffic.isOpened():
    print("❌ 신호등 카메라 오류")

# ==========================================
# [3] 영상 처리 함수들 (✅ 위 단독 코드 그대로)
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * ROI_HEIGHT_RATIO)
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
            # 수직선 방지
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
# [추가] 정지선 감지 (mask 기반)
# ==========================================
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
# [4] 메인 루프
# ==========================================
is_crosswalk_stop = False
crosswalk_start_time = 0.0
crosswalk_cooldown_timer = 0.0
crosswalk_detect_timer = 0.0

obstacle_count = 0
is_obstacle_detected = False
obs_clear_finished_time = 0.0
obstacle_last_seen_time = 0.0

last_serial_time = 0.0
last_speed_time = 0.0

try:
    print("\n🚀 3초 후 출발!")
    for i in range(3, 0, -1):
        print(f"{i}..")
        time.sleep(1)
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    print("🚀 주행 시작")

    # 라이다가 없으면 카메라만 돌리도록(디버깅 편의)
    scan_iter = lidar.iter_scans() if lidar is not None else [None] * 10**9

    for scan in scan_iter:
        # -----------------------------
        # (0) 아두이노 버퍼 비우기 (위 코드와 동일)
        # -----------------------------
        if ser:
            try:
                if ser.in_waiting > 0:
                    ser.read(ser.in_waiting)
            except:
                pass

        current_time = time.time()

        # -----------------------------
        # (1) 라이다 전방 거리
        # -----------------------------
        raw_dist = 2000
        if scan is not None:
            for (_, ang, dist) in scan:
                if 200 < dist < 1500:
                    if ang >= 330 or ang <= 30:
                        if dist < raw_dist:
                            raw_dist = dist

        # -----------------------------
        # (2) 카메라 읽기
        # -----------------------------
        ret_l, frame_lane = cap_lane.read()
        ret_t, frame_traffic = cap_traffic.read()

        if not ret_l or not ret_t:
            print("❌ 카메라 신호 끊김")
            break

        if frame_lane.shape[1] != width:
            frame_lane = cv2.resize(frame_lane, (width, height))
        if frame_traffic.shape[1] != width:
            frame_traffic = cv2.resize(frame_traffic, (width, height))

        h, w = frame_lane.shape[:2]

        # -----------------------------
        # (3) 신호등
        # -----------------------------
        traffic_light = "UNKNOWN"
        if camera_lib is not None:
            traffic_light = camera_lib.object_detection(frame_traffic, sample=3, print_enable=False)

        # -----------------------------
        # (4) 차선 마스크 (✅ 위 코드 그대로)
        # -----------------------------
        blurred = cv2.medianBlur(frame_lane, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, current_l_min, 0])
        upper_white = np.array([179, 255, S_MAX_VAL])
        mask = cv2.inRange(hls, lower_white, upper_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ROI
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
        if total_area == 0:
            total_area = 1
        ratio = white_count / total_area

        # Auto Tuning (✅ 위 코드 그대로)
        if ratio > TARGET_RATIO_MAX:
            current_l_min = min(current_l_min + 2, MAX_L_VAL)
        elif ratio < TARGET_RATIO_MIN:
            current_l_min = max(current_l_min - 2, MIN_L_VAL)

        # -----------------------------
        # (5) 라인 트레이싱 (✅ 위 코드 그대로)
        # -----------------------------
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame_lane, lines)

        base_angle, base_target = calculate_steering_angle(frame_lane, left, right)

        # -----------------------------
        # (6) 장애물 회피: "타겟 x"에 shift를 더/빼서 복귀되게 구성
        # -----------------------------
        avoid_direction = 0
        if raw_dist < OBSTACLE_START_DIST:
            obstacle_last_seen_time = current_time
            if not is_obstacle_detected and (current_time - obs_clear_finished_time > 1.5):
                obstacle_count += 1
                is_obstacle_detected = True
                print(f"⚠️ 장애물 #{obstacle_count} 감지!")

            # 1번은 좌, 2번은 우
            if obstacle_count == 1:
                avoid_direction = -1
            elif obstacle_count == 2:
                avoid_direction = 1
            else:
                avoid_direction = 0

        else:
            if is_obstacle_detected:
                if current_time - obstacle_last_seen_time < OBSTACLE_CLEAR_TIME:
                    if obstacle_count == 1:
                        avoid_direction = -1
                    elif obstacle_count == 2:
                        avoid_direction = 1
                else:
                    is_obstacle_detected = False
                    obs_clear_finished_time = current_time
                    avoid_direction = 0
                    print("✅ 복귀")

        # 회피 shift 적용(타겟 기반)
        target_x = float(base_target)
        shift_px = 0.0
        if avoid_direction != 0:
            calc_dist = min(raw_dist, OBSTACLE_START_DIST)
            shift_px = (OBSTACLE_START_DIST - calc_dist) * SHIFT_GAIN
            target_x = target_x + (avoid_direction * shift_px)

        # 타겟 기반으로 다시 각도 계산(라인 트레이싱 수식 동일)
        car_x = w / 2.0
        target_y = int(h * ROI_HEIGHT_RATIO)
        dx = target_x - car_x
        dy = (h - target_y)
        angle = math.degrees(math.atan2(dx, abs(dy)))
        target = int(max(0, min(w - 1, target_x)))

        # 서보 매핑(✅ 위 코드 그대로)
        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # -----------------------------
        # (7) 정지선/신호등 로직 (원래 코드 유지)
        # -----------------------------
        final_speed = MAX_SPEED
        status_msg = "NORMAL"
        status_color = (0, 255, 0)

        if is_crosswalk_stop:
            final_speed = 0
            elapsed = current_time - crosswalk_start_time
            if traffic_light == "GREEN":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (GREEN)"
                status_color = (0, 255, 0)
            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (TIMEOUT)"
                status_color = (0, 255, 0)
            else:
                status_msg = f"WAIT GREEN.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)
        else:
            # 정지선 감지(ROI 조건)
            if (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                if detect_stop_line(mask, mask_bgr, ROI_HEIGHT_RATIO):
                    if traffic_light == "RED":
                        if crosswalk_detect_timer == 0:
                            crosswalk_detect_timer = current_time
                        elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                            is_crosswalk_stop = True
                            crosswalk_start_time = current_time
                            final_speed = 0
                            status_msg = "STOP! (Line+RED)"
                            status_color = (0, 0, 255)
                            crosswalk_detect_timer = 0
                        else:
                            status_msg = "Checking Line+RED..."
                            status_color = (0, 255, 255)
                    else:
                        crosswalk_detect_timer = 0
                        status_msg = f"Line Detected ({traffic_light})"
                        status_color = (0, 255, 255)
                else:
                    crosswalk_detect_timer = 0
            else:
                crosswalk_detect_timer = 0

        # 장애물 가까우면 감속(원래 너 코드 스타일 유지)
        if raw_dist < 800 and not is_crosswalk_stop:
            final_speed = min(final_speed, 120)  # 필요하면 값 조정
            status_msg = f"OBSTACLE {raw_dist:.0f}mm"
            status_color = (0, 255, 255)

        # -----------------------------
        # (8) 통신 (✅ 위 코드 Heartbeat 스타일)
        # -----------------------------
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # -----------------------------
        # (9) 디스플레이 (위 코드 스타일 + 듀얼)
        # -----------------------------
        # traffic overlay
        cv2.putText(frame_traffic, f"Light: {traffic_light}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # mask overlay
        cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
        cv2.circle(mask_bgr, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        cv2.putText(mask_bgr, f"L-Min: {current_l_min} | Ratio: {ratio * 100:.1f}%", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(mask_bgr, f"Angle:{angle:.1f}  Target:{target}  Shift:{shift_px:.0f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(mask_bgr, status_msg, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        combined = np.hstack((frame_traffic, mask_bgr))
        cv2.imshow("Dual View (Traffic + LaneMask)", combined)

        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"❌ 오류 발생: {e}")

finally:
    print("\n🛑 안전 정지")
    if ser:
        try:
            for _ in range(3):
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")
                time.sleep(0.05)
            ser.close()
        except:
            pass

    if lidar is not None:
        try:
            lidar.stop()
            lidar.disconnect()
        except:
            pass

    if cap_lane is not None:
        cap_lane.release()
    if cap_traffic is not None:
        cap_traffic.release()

    cv2.destroyAllWindows()
