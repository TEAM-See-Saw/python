import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2
import math

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# 모터/서보 설정
SPEED_SEARCH = 80
SPEED_PARK = 75
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680

# --- 초음파 센서 인덱스 (사용자 핀 맵) ---
# 순서: [0:LF, 1:LM, 2:LT, 3:RF, 4:RM, 5:RT]
IDX_FL = 0  # 전방 좌측 (LF)
IDX_FR = 3  # 전방 우측 (RF)
IDX_BL = 2  # 후방 좌측 (LT)
IDX_BR = 5  # 후방 우측 (RT)

# ★ [수정 1] 전방 측면 감지 거리 (50cm 이내일 때만 인식)
DETECT_DIST_MAX = 500  # 500mm = 50cm
SONAR_VALID_MAX = 2000  # 2m 이상 무시

# ★ [수정 2] 주차 완료 후 대기 시간 (4초)
TIME_WAIT_AFTER_PARK = 4.0

# 차량 제원 (벤츠 AMG GT)
LIDAR_X_OFFSET = 1000
CAR_WIDTH = 550
STOP_OFFSET_X = -500

# 물리 엔진 시간 계산
REAL_SPEED_CM_S = 50.0
speed_mm_s = REAL_SPEED_CM_S * 10
TIME_EXIT_FWD = (990 / speed_mm_s) * 1.2
TIME_EXIT_TURN = (2 * math.pi * 1400 * 0.25) / speed_mm_s

# ==========================================
# [2] 초기화 및 센서 함수
# ==========================================
sonar_data = [2000] * 6


def read_sensors():
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) >= 6:
                    new_data = []
                    for p in parts:
                        try:
                            val = int(p)
                            if val == 0: val = 2000
                            new_data.append(val)
                        except:
                            new_data.append(2000)
                    sonar_data = new_data
        except:
            pass


lidar = None;
ser = None
try:
    ser = serial.Serial(PORT, 9600, timeout=0.1)
    lidar = RPLidar(LIDAR_PORT)
    print("✅ 주차 시스템 연결됨 (조건 수정 완료)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}"); exit()


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
    SIDE_CHECK_MIN = 450;
    SIDE_CHECK_MAX = 700
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
STATE_SEARCH = 0
STATE_READY_TO_REVERSE = 1
STATE_PARKING_TURN = 2
STATE_PARKING_ALIGN = 3
STATE_PARKING_STRAIGHT = 4
STATE_WAIT = 5
STATE_EXIT_FWD = 6
STATE_EXIT_TURN = 7
STATE_EXIT_STR = 8

current_state = STATE_SEARCH
state_timer = 0


def main():
    global current_state, state_timer, ser, lidar
    if ser: ser.write(f"D,{SPEED_SEARCH}\n".encode())

    cv2.namedWindow("Parking Monitor")
    map_scale = 0.2;
    cx, cy = 150, 250

    try:
        for scan in lidar.iter_scans():
            read_sensors()
            points = get_local_map(scan)
            corner_x, corner_y = detect_neighbor_car_corner(points)

            cmd_speed = 0;
            cmd_servo = SERVO_CENTER;
            status_msg = "INIT"
            curr_time = time.time()

            # -------------------------------------------------
            # 주차 로직
            # -------------------------------------------------
            if current_state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH

                dist_FL = sonar_data[IDX_FL]
                dist_FR = sonar_data[IDX_FR]

                status_msg = f"SEARCH: LF={dist_FL} RF={dist_FR}"

                # 정지 조건: 라이다 OR (전방 양쪽 초음파 < 50cm)
                sonar_stop_condition = (dist_FL < DETECT_DIST_MAX) and (dist_FR < DETECT_DIST_MAX)
                lidar_stop_condition = (corner_x is not None and corner_x < STOP_OFFSET_X)

                if lidar_stop_condition or sonar_stop_condition:
                    stop_reason = "SONAR(LF/RF)" if sonar_stop_condition else "LIDAR(CORNER)"
                    print(f"🛑 정지 조건 만족! ({stop_reason})")

                    ser.write(b"D,-150\n");
                    time.sleep(0.1)
                    for _ in range(3): ser.write(b"D,0\n"); time.sleep(0.05)

                    current_state = STATE_READY_TO_REVERSE

            elif current_state == STATE_READY_TO_REVERSE:
                status_msg = "STEERING RIGHT..."
                ser.write(f"S,{SERVO_RIGHT_MAX}\n".encode());
                time.sleep(1.2)
                current_state = STATE_PARKING_TURN;
                state_timer = time.time()

            elif current_state == STATE_PARKING_TURN:
                status_msg = "PARKING: TURN"
                cmd_speed = -SPEED_PARK;
                cmd_servo = SERVO_RIGHT_MAX
                if curr_time - state_timer > 2.2:
                    ser.write(b"D,0\n");
                    current_state = STATE_PARKING_ALIGN;
                    state_timer = time.time()

            elif current_state == STATE_PARKING_ALIGN:
                status_msg = "ALIGNING..."
                cmd_servo = SERVO_CENTER
                if curr_time - state_timer > 1.0:
                    current_state = STATE_PARKING_STRAIGHT;
                    state_timer = time.time()

            elif current_state == STATE_PARKING_STRAIGHT:
                dist_L = sonar_data[IDX_BL]
                dist_R = sonar_data[IDX_BR]

                if dist_L < SONAR_VALID_MAX and dist_R < SONAR_VALID_MAX:
                    error = dist_L - dist_R
                    correction = int(error * 0.8)
                    correction = max(-50, min(50, correction))
                    cmd_servo = SERVO_CENTER + correction
                    status_msg = f"LT:{dist_L} RT:{dist_R} | Adj:{correction}"
                else:
                    cmd_servo = SERVO_CENTER

                cmd_speed = -SPEED_PARK

                if curr_time - state_timer > 2.5:
                    print(f"✅ 주차 완료. {TIME_WAIT_AFTER_PARK}초 대기")
                    ser.write(b"D,0\n")
                    current_state = STATE_WAIT;
                    state_timer = time.time()

            # -------------------------------------------------
            # 출차 로직 (대기 시간 4초 적용됨)
            # -------------------------------------------------
            elif current_state == STATE_WAIT:
                remaining = int(TIME_WAIT_AFTER_PARK - (curr_time - state_timer))
                status_msg = f"WAITING... {remaining}s"
                cmd_speed = 0
                if curr_time - state_timer > TIME_WAIT_AFTER_PARK:
                    print("🚀 대기 종료, 출차 시작")
                    current_state = STATE_EXIT_FWD;
                    state_timer = time.time()

            elif current_state == STATE_EXIT_FWD:
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_CENTER
                if curr_time - state_timer > TIME_EXIT_FWD:
                    current_state = STATE_EXIT_TURN;
                    state_timer = time.time()

            elif current_state == STATE_EXIT_TURN:
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_RIGHT_MAX
                if curr_time - state_timer > TIME_EXIT_TURN:
                    current_state = STATE_EXIT_STR

            elif current_state == STATE_EXIT_STR:
                cmd_speed = SPEED_SEARCH;
                cmd_servo = SERVO_CENTER
                status_msg = "EXIT STRAIGHT"

            if current_state not in [STATE_READY_TO_REVERSE, STATE_PARKING_ALIGN]:
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

            img = np.zeros((500, 500, 3), np.uint8)
            rear_px, front_px = int(150 * map_scale), int(840 * map_scale);
            w_px = int(CAR_WIDTH * map_scale / 2)
            cv2.rectangle(img, (cx - rear_px, cy - w_px), (cx + front_px, cy + w_px), (0, 0, 255), 2)
            if len(points) > 0:
                for p in points:
                    px, py = int(p[0] * map_scale + cx), int(p[1] * map_scale + cy)
                    if 0 <= px < 500 and 0 <= py < 500: cv2.circle(img, (px, py), 1, (255, 255, 255), -1)
            cv2.putText(img, status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Parking Monitor", img)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()