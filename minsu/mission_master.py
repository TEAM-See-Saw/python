import serial
from rplidar import RPLidar, RPLidarException
import time
import cv2
import numpy as np
import math
import sys
from Function_Library import libCAMERA

# ==========================================
# [1] 통합 환경 및 튜닝 설정 (튜닝 참고용 상수 유지)
# ==========================================
IS_SUNNY = True  # 초기 참고값

ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'
BAUDRATE = 115200

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

# --- 신호등 ROI 설정 ---
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
current_l_min = 140

print("\n" + "=" * 50)
print(" 🚀 [최종 통합] 적응형 주행 + 버퍼링 방지 패치 완료")
print("=" * 50 + "\n")

# ==========================================
# [2] 핵심 함수 정의
# ==========================================
sonar_data = [999] * 6
ser = None
lidar = None


def init_hardware():
    global ser, lidar
    print(f"🔌 하드웨어 연결 시도: Arduino({ARDUINO_PORT}), Lidar({LIDAR_PORT})...")
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        lidar.clean_input()
        print("✅ 하드웨어 연결 성공!")
        return True
    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR] 하드웨어 연결 실패: {e}")
        print("👉 케이블 연결 상태와 포트 번호를 확인하세요.")
        return False


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


# ★ [3] 고급 신호등 인식
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
            if abs(angle) < 30:
                cv2.line(frame_to_draw, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 0, 255), 3)
                detected = True
    return detected


# ★ [5] 통합 조향
def calculate_steering_and_avoid(image, left_line, right_line, obstacle_dist, direction):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_LANE_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = last_target_x
    last_target_x = target_x

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
# [모드 1] 고속 라인 트레이싱
# ==========================================
def run_mode_1_lane_tracing():
    print("\n🏎️ [모드 1] 라인 트레이싱 시작")
    global last_target_x

    # 설정
    SPEED = 150
    cap = cv2.VideoCapture(CAM_IDX_LANE, cv2.CAP_DSHOW)
    cap.set(3, 640);
    cap.set(4, 480);
    cap.set(15, -6)

    if not cap.isOpened():
        print("❌ 카메라 연결 실패");
        return

    last_serial_time = 0

    try:
        l_min = find_optimal_l_min(cap)
        if ser: ser.write(f"D,{SPEED}\n".encode())

        while True:
            # ★ [수정] 아두이노 데이터 비우기 추가 (모드 1 필수)
            read_sensors()

            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]

            # 전처리
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            mask = cv2.inRange(hls, (0, l_min, 0), (179, 255, 80))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            # ROI
            roi_poly = np.array([[(0, h), (w, h), (int(w * 1.0), int(h * ROI_LANE_HEIGHT_RATIO)),
                                  (int(w * 0.0), int(h * ROI_LANE_HEIGHT_RATIO))]], dtype=np.int32)
            mask_roi = np.zeros_like(mask);
            cv2.fillPoly(mask_roi, [roi_poly], 255)
            mask = cv2.bitwise_and(mask, mask_roi)

            # 라인
            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            # 조향 (회피 없음)
            angle, target, _ = calculate_steering_and_avoid(frame, left, right, 2000, 0)

            # 전송
            if time.time() - last_serial_time > 0.05:
                pwm = map_servo(angle)
                if ser: ser.write(f"S,{pwm}\n".encode()); ser.write(f"D,{SPEED}\n".encode())
                last_serial_time = time.time()

            # 디스플레이
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.circle(mask_bgr, (target, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            cv2.imshow("Mode 1", mask_bgr)
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release();
        cv2.destroyAllWindows()


# ==========================================
# [모드 2] 장애물 + 신호등 + 정지선
# ==========================================
def run_mode_2_obstacle_traffic():
    print("\n🚦 [모드 2] 통합 주행 모드 시작")
    global last_target_x, current_l_min

    cap_L = cv2.VideoCapture(CAM_IDX_LANE, cv2.CAP_DSHOW)
    cap_T = cv2.VideoCapture(CAM_IDX_TRAFFIC, cv2.CAP_DSHOW)
    cap_L.set(3, 640);
    cap_L.set(4, 480);
    cap_L.set(15, -6)
    cap_T.set(3, 640);
    cap_T.set(4, 480);
    cap_T.set(15, -6)

    if not cap_L.isOpened() or not cap_T.isOpened():
        print("❌ 카메라 오류");
        return

    lib = libCAMERA()
    current_l_min = find_optimal_l_min(cap_L)

    is_stop = False;
    stop_timer = 0;
    detect_timer = 0
    obs_detected = False;
    obs_clear_time = 0;
    obs_seen_time = 0
    avoid_dir = 0;
    last_serial = 0

    try:
        if ser: ser.write(f"D,{SPEED_NORMAL}\n".encode())

        for scan in lidar.iter_scans():
            read_sensors()  # 버퍼 비우기

            raw_dist = 2000
            for (_, ang, dist) in scan:
                if 200 < dist < 1500 and (ang >= 330 or ang <= 30):
                    if dist < raw_dist: raw_dist = dist

            ret1, frame_L = cap_L.read();
            ret2, frame_T = cap_T.read()
            if not ret1 or not ret2: break
            frame_L = cv2.resize(frame_L, (640, 480))
            frame_T = cv2.resize(frame_T, (640, 480))
            h, w = frame_L.shape[:2]

            # 신호등 & 차선
            t_state, tdbg = detect_traffic_lr_robust(frame_T)
            blurred = cv2.medianBlur(frame_L, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            mask = cv2.inRange(hls, (0, current_l_min, 0), (179, 255, S_MAX))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # ROI
            roi_poly = np.array(
                [[(0, h), (w, h), (w, int(h * ROI_LANE_HEIGHT_RATIO)), (0, int(h * ROI_LANE_HEIGHT_RATIO))]],
                dtype=np.int32)
            mask_roi = np.zeros_like(mask);
            cv2.fillPoly(mask_roi, [roi_poly], 255)
            mask = cv2.bitwise_and(mask, mask_roi)

            # Auto Tuning
            white_cnt = cv2.countNonZero(mask)
            ratio = white_cnt / (cv2.contourArea(roi_poly) + 1)
            if ratio < CROSSWALK_RATIO_MIN:
                if ratio > TARGET_RATIO_MAX:
                    current_l_min = min(current_l_min + 2, MAX_L_VAL)
                elif ratio < TARGET_RATIO_MIN:
                    current_l_min = max(current_l_min - 2, MIN_L_VAL)

            edges = cv2.Canny(mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame_L, lines)

            # 주행 판단
            speed = SPEED_NORMAL
            msg = "NORMAL"
            curr_time = time.time()

            # 1. 장애물
            if raw_dist < OBSTACLE_START_DIST:
                obs_seen_time = curr_time
                if not obs_detected:
                    if curr_time - obs_clear_time > 1.5:
                        obs_detected = True;
                        avoid_dir = -1  # 왼쪽 회피
                if obs_detected: msg = "AVOID L"; speed = SPEED_SLOW
            else:
                if obs_detected:
                    if curr_time - obs_seen_time < 1.5:
                        msg = "CLEARING..."
                    else:
                        obs_detected = False; obs_clear_time = curr_time; avoid_dir = 0

            # 2. 정지선 + 신호등
            if is_stop:
                speed = 0
                if t_state == "RIGHT" or (curr_time - stop_timer > 7.0):
                    is_stop = False;
                    detect_timer = 0;
                    msg = "GO!"
                else:
                    msg = "WAIT GREEN..."
            elif (ratio > CROSSWALK_RATIO_MIN) and (curr_time - stop_timer > 5.0):
                if detect_stop_line(mask, mask_bgr, ROI_LANE_HEIGHT_RATIO):
                    if t_state == "LEFT":
                        if detect_timer == 0:
                            detect_timer = curr_time
                        elif curr_time - detect_timer > 0.2:
                            is_stop = True;
                            stop_timer = curr_time;
                            speed = 0;
                            msg = "STOP (RED)"
                    else:
                        detect_timer = 0; msg = "LINE DETECTED"
            else:
                detect_timer = 0

            if raw_dist < 800 and avoid_dir == 0: speed = SPEED_SLOW

            # 조향
            eff_dist = raw_dist if avoid_dir != 0 else 2000
            angle, target, _ = calculate_steering_and_avoid(frame_L, left, right, eff_dist, avoid_dir)

            if curr_time - last_serial > 0.05:
                if ser:
                    ser.write(f"S,{map_servo(angle)}\n".encode())
                    ser.write(f"D,{speed}\n".encode())
                last_serial = curr_time

            # 디스플레이
            cv2.circle(mask_bgr, (target, int(h * ROI_LANE_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            cv2.putText(frame_L, f"Light: {t_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_L, f"Msg: {msg}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Mode 2", np.hstack((frame_L, mask_bgr)))
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        cap_L.release();
        cap_T.release();
        cv2.destroyAllWindows()


# ==========================================
# [모드 3] 자동 주차
# ==========================================
def run_mode_3_parking():
    print("\n🅿️ [모드 3] 자동 주차 시작")
    SPEED_PARK = 70
    STATE_SEARCH = 0;
    STATE_SETUP = 1;
    STATE_PARK = 2;
    STATE_EXIT = 3;
    STATE_FINISH = 4
    state = STATE_SEARCH;
    timer = 0

    try:
        lidar.clean_input()
        if ser: ser.write(f"S,570\n".encode()); ser.write(b"D,0\n"); time.sleep(1)
        if ser: ser.write(f"D,{SPEED_PARK}\n".encode())

        for scan in lidar.iter_scans():
            read_sensors()
            side_dist = 2000
            for (_, ang, dist) in scan:
                if 80 < ang < 100 and dist > 0:
                    if dist < side_dist: side_dist = dist

            curr_time = time.time();
            cmd_speed = SPEED_PARK;
            cmd_steer = 570

            if state == STATE_SEARCH:
                if side_dist > 1500:
                    print("✅ 빈 공간 발견!");
                    state = STATE_SETUP;
                    timer = curr_time
                    if ser: ser.write(b"D,0\n"); time.sleep(1)
            elif state == STATE_SETUP:
                if curr_time - timer > 2.0:
                    state = STATE_PARK;
                    timer = curr_time;
                    print("✅ 후진 주차")
            elif state == STATE_PARK:
                cmd_speed = -SPEED_PARK;
                cmd_steer = 440
                if curr_time - timer > 4.0:
                    state = STATE_EXIT;
                    timer = curr_time;
                    print("✅ 출차 대기")
                    if ser: ser.write(b"D,0\n"); time.sleep(2)
            elif state == STATE_EXIT:
                cmd_speed = SPEED_PARK;
                cmd_steer = 680
                if curr_time - timer > 3.0: state = STATE_FINISH
            elif state == STATE_FINISH:
                print("🏁 종료");
                break

            if ser:
                ser.write(f"S,{cmd_steer}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

    except KeyboardInterrupt:
        pass


# ==========================================
# [메인] 메뉴 선택
# ==========================================
def main():
    if not init_hardware(): return

    while True:
        print("\n" + "=" * 40)
        print(" 🚗 자율주행 통합 시스템 🚗")
        print(" 1. 라인 트레이싱 (고속)")
        print(" 2. 장애물 회피 & 신호등")
        print(" 3. 자동 주차 모드")
        print(" Q. 종료")
        print("=" * 40)

        choice = input("👉 모드 선택: ").strip().upper()

        if choice == '1':
            run_mode_1_lane_tracing()
        elif choice == '2':
            run_mode_2_obstacle_traffic()
        elif choice == '3':
            run_mode_3_parking()
        elif choice == 'Q':
            break
        else:
            print("❌ 잘못된 입력")

        if ser: ser.write(b"D,0\n"); ser.write(b"S,570\n"); time.sleep(0.5)

    if ser: ser.close()
    if lidar: lidar.stop(); lidar.disconnect()


if __name__ == "__main__":
    main()