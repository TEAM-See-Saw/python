import cv2
import numpy as np
import math
import serial
import time
import datetime
import os
import threading
import Function_Library as LiDAR_Lib

# ==========================================
# [1] 환경 설정
# ==========================================
IS_SUNNY = True

PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX = 0

MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# LiDAR 설정
LIDAR_PORT = 'COM3'

# 회피 시작 기준 거리 (이 거리 안에 뭔가 보이면 피하기 시작)
AVOID_START_DIST = 800   # mm
# 이 거리보다 더 가까우면 회피 각도 최대
AVOID_NEAR_DIST = 400    # mm

# 회피 각도 범위
AVOID_ANGLE_MIN = 10     # 최소 회피 각도
AVOID_ANGLE_MAX = 40     # 최대 회피 각도

AVOID_MARGIN = 100       # 좌/우 비교 여유 (mm)

# LiDAR 섹터 (각도는 LiDAR 기준, 필요하면 나중에 조정)
# 정면: -20 ~ 20, 좌측 전방: 20 ~ 80, 우측 전방: -80 ~ -20

# ==========================================
# [2] 모드별 자동 튜닝값
# ==========================================
if IS_SUNNY:
    print("☀️ [모드: SUNNY] 강한 햇빛 설정")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print("🌙 [모드: NORMAL] 일반/저조도 설정")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)

# ==========================================
# [3] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    print("⚠️ 영상 처리만 진행됩니다.")

# ==========================================
# [LiDAR 공유 변수]
# ==========================================
lidar_stop_event = threading.Event()
obstacle_front = None
obstacle_left = None
obstacle_right = None


def normalize_angle(a):
    return a % 360.0


def in_sector(angles, start_deg, end_deg):
    a = normalize_angle(angles)
    s = normalize_angle(start_deg)
    e = normalize_angle(end_deg)
    if s <= e:
        return (a >= s) & (a <= e)
    else:
        return (a >= s) | (a <= e)


def lidar_worker(stop_event):
    global obstacle_front, obstacle_left, obstacle_right

    print("========================================")
    print(f"   [LiDAR 장애물 감지 스레드 시작] 포트: {LIDAR_PORT}")
    print("========================================")
    try:
        env = LiDAR_Lib.libLIDAR(LIDAR_PORT)
        env.init()
        print("✅ LiDAR 연결 성공! 스캔 시작...")
    except Exception as e:
        print(f"❌ LiDAR 연결 실패: {e}")
        return

    try:
        for scan in env.scanning():
            if stop_event.is_set():
                break
            if scan is None or len(scan) == 0:
                obstacle_front = None
                obstacle_left = None
                obstacle_right = None
                continue

            angles = scan[:, 0]
            dists = scan[:, 1]
            valid = dists > 0

            mask_front = valid & in_sector(angles, -20, 20)
            mask_left = valid & in_sector(angles, 20, 80)
            mask_right = valid & in_sector(angles, -80, -20)

            obstacle_front = float(np.min(dists[mask_front])) if np.any(mask_front) else None
            obstacle_left = float(np.min(dists[mask_left])) if np.any(mask_left) else None
            obstacle_right = float(np.min(dists[mask_right])) if np.any(mask_right) else None

            if stop_event.is_set():
                break

    except Exception as e:
        print(f"❌ LiDAR 스레드 에러: {e}")
    finally:
        try:
            env.stop()
        except:
            pass
        print("LiDAR 스레드 종료")


# ==========================================
# [4] 영상 처리 함수들
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
    y2 = int(y1 * 0.6)
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
    height, width, _ = image.shape
    car_position_x = width / 2

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = car_position_x

    dx = target_x - car_position_x
    dy = (height * 0.6) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ==========================================
# [5] 메인 실행 함수
# ==========================================
def main():
    global obstacle_front, obstacle_left, obstacle_right

    lidar_thread = threading.Thread(target=lidar_worker, args=(lidar_stop_event,), daemon=True)
    lidar_thread.start()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        lidar_stop_event.set()
        return

    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset/autodrive_{now}.mp4"

    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 20.0
    out = cv2.VideoWriter(filename, fourcc, fps, (width * 2, height))

    print(f"🎥 녹화 준비 완료: {filename}")

    print("\n" + "=" * 30)
    print(f"🚀 {MAX_SPEED} 속도로 출발합니다!")
    print("=" * 30)
    for i in range(3, 0, -1):
        print(f"Count: {i}")
        time.sleep(1)
    print("GO!!!")

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())
        time.sleep(0.1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            lower_white = np.array([0, L_MIN, 0])
            upper_white = np.array([179, 255, S_MAX])
            mask = cv2.inRange(hls, lower_white, upper_white)

            kernel = np.ones(MORPH_SIZE, np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            edges = cv2.Canny(mask, 50, 150)

            roi_vertices = [
                (0, height),
                (width // 2 - 50, int(height * 0.6)),
                (width // 2 + 50, int(height * 0.6)),
                (width, height)
            ]
            cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50,
                                    minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            base_angle, target = calculate_steering_angle(frame, left, right)

            # ===== LiDAR 회피 (다이나믹) =====
            avoid_angle = 0.0

            if (obstacle_front is not None) and (obstacle_front < AVOID_START_DIST):
                l = obstacle_left
                r = obstacle_right
                d = max(obstacle_front, 1.0)

                if d <= AVOID_NEAR_DIST:
                    w = 1.0
                else:
                    w = (AVOID_START_DIST - d) / (AVOID_START_DIST - AVOID_NEAR_DIST)
                    w = max(0.0, min(1.0, w))

                dyn_angle = AVOID_ANGLE_MIN + (AVOID_ANGLE_MAX - AVOID_ANGLE_MIN) * w

                if (l is not None) and (r is not None):
                    if l + AVOID_MARGIN < r:
                        avoid_angle = +dyn_angle
                    elif r + AVOID_MARGIN < l:
                        avoid_angle = -dyn_angle
                    else:
                        avoid_angle = +dyn_angle
                elif (l is not None) and (l < AVOID_START_DIST):
                    avoid_angle = +dyn_angle
                elif (r is not None) and (r < AVOID_START_DIST):
                    avoid_angle = -dyn_angle
                else:
                    avoid_angle = +dyn_angle

            total_angle = base_angle + avoid_angle
            total_angle = max(-45, min(45, total_angle))
            servo_val = int(map_value(total_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            if ser:
                ser.write(f"S,{servo_val}\n".encode())

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_bgr))

            if obstacle_front is not None:
                front_text = f"{obstacle_front:.0f}mm"
            else:
                front_text = "---"

            info_text = (
                f"Mode: {'SUNNY' if IS_SUNNY else 'NORMAL'} | "
                f"Angle: {total_angle:.1f} | LiDAR front: {front_text}"
            )

            cv2.putText(combined, info_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            out.write(combined)
            cv2.imshow("Autonomous Driving", combined)

            if cv2.waitKey(1) == ord('q'):
                print("🛑 긴급 정지!")
                break

    finally:
        if ser:
            ser.write(b"D,0\n")
            time.sleep(0.1)
            ser.write(f"S,{SERVO_CENTER}\n".encode())
            ser.close()

        out.release()
        cap.release()
        cv2.destroyAllWindows()

        lidar_stop_event.set()
        time.sleep(0.2)

        print("💾 녹화 완료 및 프로그램 종료")


if __name__ == "__main__":
    main()
