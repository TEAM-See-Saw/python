import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar
from Function_Library import libCAMERA  # (사용 안 해도 됨, 기존 코드 유지)

# ==========================================
# [1] 환경 및 튜닝 설정  (✅ 단독 라인트레이싱 코드와 동일)
# ==========================================
IS_SUNNY = True

PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

# ✅ 카메라 인덱스: 단독 코드와 동일하게 "차선 카메라 = 1"
CAM_INDEX = 1
# 신호등 카메라(별도)
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
# [추가] 미션/회피/정지선 관련 (라이다/횡단보도)
# ==========================================
LIDAR_PORT = 'COM3'

OBSTACLE_START_DIST = 1000   # mm
SHIFT_GAIN = 1.2             # px per (mm 부족분)
OBSTACLE_CLEAR_TIME = 1.5

CROSSWALK_RATIO_MIN = 0.12
CROSSWALK_MAX_WAIT = 7.0
CROSSWALK_COOLDOWN = 5.0
CROSSWALK_CONFIRM_TIME = 0.2

# ==========================================
# [추가] 신호등(좌/우 밝기) 판정 파라미터
# ==========================================
TRAFFIC_Y_MAX_RATIO = 0.60          # 신호등이 상단에 있으니 상단 60%만 검사
TRAFFIC_BRIGHT_RATIO_TH = 0.015     # "하얀 빛" 픽셀 비율 임계값 (0.01~0.05 튜닝)
TRAFFIC_DOMINANCE_K = 1.25          # 좌/우 둘 다 밝게 잡힐 때 우세한 쪽만 선택

# ==========================================
# [2] 시리얼/라이다/카메라 연결
# ==========================================
ser = None
lidar = None
cap_lane = None
cap_traffic = None

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 시리얼 연결 실패: {e}")
    ser = None

try:
    lidar = RPLidar(LIDAR_PORT)
except Exception as e:
    print(f"❌ 라이다 초기화 실패: {e}")
    lidar = None

# ✅ 단독 코드와 동일한 방식으로 카메라 오픈/세팅
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
    print("❌ 차선 카메라 오류 (CAM_INDEX 확인)")
if not cap_traffic.isOpened():
    print("❌ 신호등 카메라 오류 (CAM_INDEX_TRAFFIC 확인)")

# ==========================================
# [3] 영상 처리 함수들 (✅ 단독 코드 그대로)
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
# [추가] Cam0 신호등 "좌/우 밝기(하얀 빛)" 판정
# - 왼쪽이 밝으면 "LEFT"  (정지 신호)
# - 오른쪽이 밝으면 "RIGHT" (출발 신호)
# - 애매하면 "NONE"
# ==========================================
def detect_traffic_lr(frame_bgr):
    h, w = frame_bgr.shape[:2]
    y2 = int(h * TRAFFIC_Y_MAX_RATIO)
    roi = frame_bgr[0:y2, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu로 밝은 영역 자동 분리
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    third = w // 3
    left = th[:, 0:third]
    right = th[:, 2 * third:w]

    left_ratio = cv2.countNonZero(left) / float(left.size)
    right_ratio = cv2.countNonZero(right) / float(right.size)

    left_on = left_ratio > TRAFFIC_BRIGHT_RATIO_TH
    right_on = right_ratio > TRAFFIC_BRIGHT_RATIO_TH

    # 둘 다 켜져 보이면 우세한 쪽만
    if left_on and (left_ratio > right_ratio * TRAFFIC_DOMINANCE_K):
        return "LEFT", left_ratio, right_ratio
    if right_on and (right_ratio > left_ratio * TRAFFIC_DOMINANCE_K):
        return "RIGHT", left_ratio, right_ratio

    # 한쪽만 충분히 크면 인정
    if left_on and not right_on:
        return "LEFT", left_ratio, right_ratio
    if right_on and not left_on:
        return "RIGHT", left_ratio, right_ratio

    return "NONE", left_ratio, right_ratio


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

    # 라이다 없으면 카메라만 계속(디버깅)
    scan_iter = lidar.iter_scans() if lidar is not None else [None] * 10**9

    for scan in scan_iter:
        # -----------------------------
        # (0) 아두이노 버퍼 비우기 (단독 코드와 동일)
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
        # (3) Cam0 신호등: 좌/우 밝기 판정
        # -----------------------------
        traffic_state, lratio, rratio = detect_traffic_lr(frame_traffic)

        # -----------------------------
        # (4) 차선 마스크 (단독 코드 그대로)
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

        # Auto Tuning (단독 코드 그대로)
        if ratio > TARGET_RATIO_MAX:
            current_l_min = min(current_l_min + 2, MAX_L_VAL)
        elif ratio < TARGET_RATIO_MIN:
            current_l_min = max(current_l_min - 2, MIN_L_VAL)

        # -----------------------------
        # (5) 라인 트레이싱 (단독 코드 그대로)
        # -----------------------------
        edges = cv2.Canny(mask, 50, 150)
        cropped = region_of_interest(edges, roi_points)
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame_lane, lines)

        base_angle, base_target = calculate_steering_angle(frame_lane, left, right)

        # -----------------------------
        # (6) 장애물 회피: 타겟 x에 shift 적용
        # -----------------------------
        avoid_direction = 0
        if raw_dist < OBSTACLE_START_DIST:
            obstacle_last_seen_time = current_time
            if not is_obstacle_detected and (current_time - obs_clear_finished_time > 1.5):
                obstacle_count += 1
                is_obstacle_detected = True
                print(f"⚠️ 장애물 #{obstacle_count} 감지!")

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

        target_x = float(base_target)
        shift_px = 0.0
        if avoid_direction != 0:
            calc_dist = min(raw_dist, OBSTACLE_START_DIST)
            shift_px = (OBSTACLE_START_DIST - calc_dist) * SHIFT_GAIN
            target_x = target_x + (avoid_direction * shift_px)

        # 다시 각도 계산
        car_x = w / 2.0
        target_y = int(h * ROI_HEIGHT_RATIO)
        dx = target_x - car_x
        dy = (h - target_y)
        angle = math.degrees(math.atan2(dx, abs(dy)))
        target = int(max(0, min(w - 1, target_x)))

        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # -----------------------------
        # (7) 정지선/신호등 로직 (✅ 요청사항 반영)
        # - 정지선 + LEFT(왼쪽 빛) => 정지
        # - 정지 중 RIGHT(오른쪽 빛) => 출발
        # -----------------------------
        final_speed = MAX_SPEED
        status_msg = "NORMAL"
        status_color = (0, 255, 0)

        if is_crosswalk_stop:
            final_speed = 0
            elapsed = current_time - crosswalk_start_time

            # ✅ 오른쪽 빛 ON이면 출발
            if traffic_state == "RIGHT":
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (RIGHT ON)"
                status_color = (0, 255, 0)

            elif elapsed > CROSSWALK_MAX_WAIT:
                is_crosswalk_stop = False
                crosswalk_cooldown_timer = current_time
                crosswalk_detect_timer = 0
                status_msg = "GO (TIMEOUT)"
                status_color = (0, 255, 0)

            else:
                status_msg = f"WAIT RIGHT.. ({elapsed:.1f}s)"
                status_color = (0, 0, 255)

        else:
            # 정지선 감지(ROI 조건)
            if (ratio > CROSSWALK_RATIO_MIN) and (current_time - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                if detect_stop_line(mask, mask_bgr, ROI_HEIGHT_RATIO):

                    # ✅ 정지선 + 왼쪽 빛 ON이면 정지 진입
                    if traffic_state == "LEFT":
                        if crosswalk_detect_timer == 0:
                            crosswalk_detect_timer = current_time
                        elif current_time - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                            is_crosswalk_stop = True
                            crosswalk_start_time = current_time
                            final_speed = 0
                            status_msg = "STOP! (Line+LEFT ON)"
                            status_color = (0, 0, 255)
                            crosswalk_detect_timer = 0
                        else:
                            status_msg = "Checking Line+LEFT..."
                            status_color = (0, 255, 255)

                    else:
                        # 정지선은 있는데 왼쪽 빛이 아니면 정지하지 않음
                        crosswalk_detect_timer = 0
                        status_msg = f"Line Detected ({traffic_state})"
                        status_color = (0, 255, 255)
                else:
                    crosswalk_detect_timer = 0
            else:
                crosswalk_detect_timer = 0

        # 장애물 가까우면 감속(정지 중 아닐 때)
        if raw_dist < 800 and not is_crosswalk_stop:
            final_speed = min(final_speed, 120)  # 필요시 튜닝
            status_msg = f"OBSTACLE {raw_dist:.0f}mm"
            status_color = (0, 255, 255)

        # -----------------------------
        # (8) 통신 (단독 코드 Heartbeat 스타일)
        # -----------------------------
        if ser:
            if current_time - last_serial_time > SERIAL_DELAY:
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            if current_time - last_speed_time > SPEED_REFRESH_DELAY:
                ser.write(f"D,{final_speed}\n".encode())
                last_speed_time = current_time

        # -----------------------------
        # (9) 디스플레이 (듀얼)
        # -----------------------------
        # traffic overlay
        cv2.putText(frame_traffic, f"Traffic: {traffic_state}  L:{lratio:.3f} R:{rratio:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # (디버깅용) 좌/우 검사 영역 표시
        th = width // 3
        y2 = int(height * TRAFFIC_Y_MAX_RATIO)
        cv2.rectangle(frame_traffic, (0, 0), (th, y2), (255, 255, 0), 2)           # left box
        cv2.rectangle(frame_traffic, (2*th, 0), (width-1, y2), (255, 255, 0), 2)  # right box

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

except KeyboardInterrupt:
    print("사용자 종료")

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
