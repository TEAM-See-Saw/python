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

# --- 속도 설정 ---
SPEED_NORMAL = 200
SPEED_SLOW = 100
SPEED_STOP = 0

# --- 서보 모터 PWM 설정 ---
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 회피 기동 튜닝 ---
OBSTACLE_START_DIST = 1000 # 라이다 감지 거리 - 박으면 늘리기
SHIFT_GAIN = 1.2 # 타겟 오프셋 밀림 정도 - 박으면 늘리기

# --- 영상 처리 설정 ---
if IS_SUNNY:
    L_MIN = 160; S_MAX = 50; EXPOSURE = -9; MORPH_SIZE = (5, 5)
else:
    L_MIN = 80; S_MAX = 120; EXPOSURE = -4; MORPH_SIZE = (3, 3)


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
    y2 = int(y1 * 0.6)
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

    # 1. 기본 타겟 계산
    if left_line is not None and right_line is not None:
        base_target = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        base_target = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        base_target = right_line[0][2] - (width * 0.25)
    else:
        return last_angle, int(car_x + (last_angle * 5)), 0

    # 2. 장애물 회피 (Raw 거리 사용)
    final_target = base_target
    shift_amount = 0

    if 0 < obstacle_dist < OBSTACLE_START_DIST:
        shift_amount = (OBSTACLE_START_DIST - obstacle_dist) * SHIFT_GAIN
        final_target = base_target - shift_amount

    # 3. 조향 각도 산출
    dx = final_target - car_x
    dy = (height * 0.6) - height
    angle = math.degrees(math.atan2(dx, abs(dy)))

    return angle, int(final_target), int(shift_amount)


def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


# ==========================================
# [3] 초기화
# ==========================================
video_writer = None

try:
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    lidar = RPLidar(LIDAR_PORT)
    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)
    cam0.set(3, 640);
    cam0.set(4, 480);
    cam0.set(15, EXPOSURE)

    if not cam0.isOpened(): raise Exception("카메라 에러")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('mission_record_nofilter.avi', fourcc, 20.0, (640, 480))

    print("✅ 시스템 준비 완료: 라이다 필터 OFF (즉각 반응)")
    time.sleep(2)

except Exception as e:
    print(f"❌ 초기화 오류: {e}");
    exit()

# ==========================================
# [4] 메인 루프
# ==========================================
last_valid_angle = 0

try:
    for scan in lidar.iter_scans():

        # 1. 라이다 (Raw Data 수집)
        raw_dist = 2000
        for (_, angle, dist) in scan:
            if 200 < dist < 1500:
                if angle >= 330 or angle <= 30:
                    if dist < raw_dist: raw_dist = dist

        # [필터 로직 제거됨] -> raw_dist를 그대로 사용합니다.

        # 2. 영상
        ret, frame = cam0.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))

        traffic_light = camera.object_detection(frame, sample=5, print_enable=False)

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, L_MIN, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        edges = cv2.Canny(mask, 50, 150)

        # ROI (좁은 시야 유지)
        roi = [(0, 480), (200, 280), (440, 280), (640, 480)]
        cropped = region_of_interest(edges, np.array([roi], np.int32))
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

        l_line, r_line = average_slope_intercept(frame, lines)

        # 3. 제어 (raw_dist 사용)
        final_angle, target_px, shift_val = calculate_avoid_angle(frame, l_line, r_line, raw_dist,
                                                                  last_valid_angle)

        last_valid_angle = final_angle

        final_speed = SPEED_NORMAL
        status_msg = "Run: Normal"

        # 속도 제어도 Raw 거리 기준
        if raw_dist < 800:
            final_speed = SPEED_SLOW
            status_msg = "Run: AVOID"

        if traffic_light == "RED":
            final_speed = SPEED_STOP
            status_msg = "RED STOP"

        # 4. 전송
        pwm_cmd = map_servo(final_angle)
        ser.write(f"S,{pwm_cmd}\n".encode())
        ser.write(f"D,{final_speed}\n".encode())

        # 5. 디스플레이
        cv2.circle(frame, (target_px, 300), 15, (0, 0, 255), -1)
        cv2.polylines(frame, [np.array(roi, np.int32)], True, (255, 0, 0), 2)

        if l_line is not None:
            for x1, y1, x2, y2 in l_line: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if r_line is not None:
            for x1, y1, x2, y2 in r_line: cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # 필터 없는 거리값 표시
        cv2.putText(frame, f"Lidar: {int(raw_dist)}mm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, status_msg, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if shift_val > 0:
            cv2.putText(frame, f"SHIFT: {int(shift_val)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.imshow("Mission", frame)
        if video_writer is not None: video_writer.write(frame)
        if cv2.waitKey(1) == ord('q'): break

except KeyboardInterrupt:
    print("종료")
finally:
    lidar.stop();
    lidar.disconnect()
    ser.write(b"D,0\n");
    ser.close()
    if video_writer is not None: video_writer.release()
    cam0.release();
    cv2.destroyAllWindows()