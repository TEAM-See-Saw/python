import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2
import math

# ==========================================
# [1] 물리 엔진 설정 (제원 기반 자동 계산)
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# --- 모터 및 센서 설정 ---
SPEED_SEARCH = 80
SPEED_PARK = 75
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680

# --- 차량 제원 (벤츠 AMG GT) ---
CAR_LENGTH_MM = 990  # 차량 길이 (mm)
WHEELBASE_MM = 700  # 축거 (앞바퀴~뒷바퀴 거리, 추정치)
TURNING_RADIUS_MM = 1400  # 회전 반경 (핸들 다 꺾었을 때, 추정치)

# ★ [핵심] 차량 실제 속도 (튜닝 포인트)
# PWM 80일 때 차가 1초에 몇 cm 가는지 입력하세요.
# (보통 40~60cm 정도 갑니다. 너무 많이 돌면 이 값을 키우고, 덜 돌면 줄이세요)
REAL_SPEED_CM_S = 50.0

# --- 자동 계산된 시간 (건드리지 마세요) ---
speed_mm_s = REAL_SPEED_CM_S * 10

# 1. 출차 직진 시간: 차체 길이만큼 앞으로 나가는 시간
TIME_EXIT_FWD = (CAR_LENGTH_MM / speed_mm_s) * 1.2  # 여유있게 1.2배

# 2. 우회전 90도 시간: 90도 부채꼴의 호의 길이 / 속도
# 호의 길이 = 2 * pi * R * (90/360)
arc_length_mm = 2 * math.pi * TURNING_RADIUS_MM * 0.25
TIME_EXIT_TURN = arc_length_mm / speed_mm_s

print(f"⏱️ 자동 계산된 시간 | 직진: {TIME_EXIT_FWD:.2f}초 | 회전: {TIME_EXIT_TURN:.2f}초")

# --- 기타 주차 설정 ---
LIDAR_X_OFFSET = 1000
CAR_WIDTH = 550
SIDE_CHECK_MIN = 450
SIDE_CHECK_MAX = 700
STOP_OFFSET_X = -500
TIME_WAIT_AFTER_PARK = 2.0

# ==========================================
# [2] 초기화
# ==========================================
lidar = None;
ser = None
try:
    ser = serial.Serial(PORT, 9600, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    print("✅ 주차 시스템 연결됨")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}");
    exit()


def get_local_map(scan):
    points = []
    for (_, angle, dist) in scan:
        if dist > 0:
            theta_rad = math.radians(angle)
            x = dist * math.cos(theta_rad) + LIDAR_X_OFFSET
            y = dist * math.sin(theta_rad)
            if 0 < y < 1500: points.append([x, y])
    return np.array(points)


def detect_neighbor_car_corner(points):
    if len(points) == 0: return None, None
    target_indices = np.where((points[:, 1] > SIDE_CHECK_MIN) & (points[:, 1] < SIDE_CHECK_MAX))
    target_points = points[target_indices]
    if len(target_points) < 5: return None, None
    target_points = target_points[target_points[:, 0].argsort()]
    candidates = target_points[target_points[:, 0] > -1200]
    if len(candidates) > 3: return candidates[0, 0], candidates[0, 1]
    return None, None


# ==========================================
# [3] 메인 루프
# ==========================================
# 상태 정의
STATE_SEARCH = 0
STATE_READY_TO_REVERSE = 1
STATE_PARKING_MOVE = 2
STATE_WAIT = 3
STATE_EXIT_FWD = 4
STATE_EXIT_TURN = 5
STATE_EXIT_STR = 6
STATE_ALL_DONE = 99

current_state = STATE_SEARCH
state_timer = 0


def main():
    global current_state, state_timer, ser, lidar
    if ser: ser.write(f"D,{SPEED_SEARCH}\n".encode())

    cv2.namedWindow("Lidar Parking Map")
    map_scale = 0.2;
    cx, cy = 150, 250

    try:
        for scan in lidar.iter_scans():
            points = get_local_map(scan)
            corner_x, corner_y = detect_neighbor_car_corner(points)

            cmd_speed = 0;
            cmd_servo = SERVO_CENTER;
            status_msg = "INIT"
            curr_time = time.time()

            # -------------------------------------------------
            # [1] 주차 진입
            # -------------------------------------------------
            if current_state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH;
                status_msg = "SEARCHING..."
                if corner_x is not None:
                    dist_to_stop = corner_x - STOP_OFFSET_X
                    status_msg = f"TARGET: {int(dist_to_stop)}mm"
                    if corner_x < STOP_OFFSET_X:
                        print("🛑 정지 및 조향 준비");
                        ser.write(b"D,-150\n");
                        time.sleep(0.1)
                        for _ in range(3): ser.write(b"D,0\n"); time.sleep(0.05)
                        current_state = STATE_READY_TO_REVERSE

            elif current_state == STATE_READY_TO_REVERSE:
                status_msg = "STEERING RIGHT...";
                ser.write(f"S,{SERVO_RIGHT_MAX}\n".encode());
                time.sleep(1.2)
                current_state = STATE_PARKING_MOVE;
                state_timer = time.time()

            elif current_state == STATE_PARKING_MOVE:
                status_msg = "PARKING...";
                cmd_speed = -SPEED_PARK;
                cmd_servo = SERVO_RIGHT_MAX
                if curr_time - state_timer > 3.8:
                    print("✅ 주차 완료");
                    ser.write(b"D,0\n");
                    current_state = STATE_WAIT;
                    state_timer = time.time()

            # -------------------------------------------------
            # [2] 출차 (자동 계산된 시간 적용)
            # -------------------------------------------------
            elif current_state == STATE_WAIT:
                status_msg = f"WAITING... {int(TIME_WAIT_AFTER_PARK - (curr_time - state_timer))}"
                cmd_speed = 0;
                cmd_servo = SERVO_CENTER
                if curr_time - state_timer > TIME_WAIT_AFTER_PARK:
                    print(f"🚀 출차 직진: {TIME_EXIT_FWD:.1f}초 동안")
                    current_state = STATE_EXIT_FWD;
                    state_timer = time.time()

            elif current_state == STATE_EXIT_FWD:
                status_msg = "EXIT: FORWARD";
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_CENTER
                if curr_time - state_timer > TIME_EXIT_FWD:
                    print(f"🔄 출차 우회전: {TIME_EXIT_TURN:.1f}초 동안")
                    current_state = STATE_EXIT_TURN;
                    state_timer = time.time()

            elif current_state == STATE_EXIT_TURN:
                status_msg = "EXIT: TURN RIGHT";
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_RIGHT_MAX
                if curr_time - state_timer > TIME_EXIT_TURN:
                    print("⬆️ 출차 완료: 무한 직진 시작")
                    current_state = STATE_EXIT_STR

            elif current_state == STATE_EXIT_STR:
                status_msg = "FINAL STRAIGHT (INFINITE)";
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_CENTER
                # 시간 제한 없음 (무한 직진)

            # 명령 전송
            if current_state != STATE_READY_TO_REVERSE:
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

            # 시각화
            img = np.zeros((500, 500, 3), np.uint8)
            rear_px, front_px = int(150 * map_scale), int(840 * map_scale);
            w_px = int(CAR_WIDTH * map_scale / 2)
            cv2.rectangle(img, (cx - rear_px, cy - w_px), (cx + front_px, cy + w_px), (0, 0, 255), 2)
            if len(points) > 0:
                for p in points:
                    px, py = int(p[0] * map_scale + cx), int(p[1] * map_scale + cy)
                    if 0 <= px < 500 and 0 <= py < 500: cv2.circle(img, (px, py), 1, (255, 255, 255), -1)
            if corner_x is not None:
                cv2.circle(img, (int(corner_x * map_scale + cx), int(corner_y * map_scale + cy)), 8, (0, 255, 0), -1)
            cv2.putText(img, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Lidar Parking Map", img)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()