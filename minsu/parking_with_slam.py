import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

SPEED_SEARCH = 70
SPEED_SETUP = 80
SPEED_PARK = 75
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
STEER_WAIT_TIME = 0.8

# [시간 설정] 하드코딩 유지
TIME_SETUP_MOVE = 4.0  # 5초
TIME_REVERSE_TURN = 4.0  # 5초

TARGET_PARK_X = -400
CAR_EXIST_DIST = 950
EMPTY_SPACE_DIST = 1000

# ★ [신규 설정] 측면 감지 정지 기준
SIDE_STOP_DIST = 700  # 1.2m 이내에 80도/280도 물체 감지 시 정지

REAR_LIMIT = 200
IDX_LT = 2;
IDX_RT = 5

STATE_SEARCH = 0
STATE_SETUP_LEFT = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_DONE = 4
STATE_PAUSE = 99

STEP_FIND_CAR1 = 0;
STEP_PASS_CAR1 = 1
STEP_FIND_GAP = 2;
STEP_FIND_CAR2 = 3

# ==========================================
# [2] 함수 정의
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6: sonar_data = [int(p) for p in parts]
        except:
            pass


def get_lidar_min_dist(scan, start_angle, end_angle):
    dists = []
    for (_, angle, dist) in scan:
        if dist > 0 and (start_angle <= angle <= end_angle):
            dists.append(dist)
    if len(dists) > 0: return np.min(dists)
    return 9999


# ★ [신규 함수] 특정 각도 근처의 거리값 가져오기
def get_dist_at_angle(scan, target_angle, range_pm=2):
    dists = []
    min_a = target_angle - range_pm
    max_a = target_angle + range_pm
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
    if len(dists) > 0: return np.mean(dists)
    return 9999


def get_car2_corner_x(scan):
    points = []
    for (_, angle, dist) in scan:
        if 40 <= angle <= 140 and 0 < dist < CAR_EXIST_DIST:
            theta = np.radians(angle)
            x = dist * np.cos(theta)
            points.append(x)
    if len(points) > 0: return np.max(points)
    return None


# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser, lidar
    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결")
        time.sleep(1)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 오류: {e}");
        return

    # 초기 바퀴 정렬
    ser.write(f"S,{SERVO_CENTER}\n".encode());
    ser.write(b"D,0\n")
    time.sleep(1.0)
    lidar.clean_input()

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0

    valid_gap_count = 0
    valid_car2_count = 0
    last_corner_x = 0

    pause_start_time = 0
    pause_duration = 0
    next_state_after_pause = 0
    pause_msg = ""

    while True:
        try:
            for scan in lidar.iter_scans():
                read_sensors()

                # 센서 값 갱신
                lidar_radar = get_lidar_min_dist(scan, 30, 110)

                # ★ 80도, 280도 거리 측정
                dist_80 = get_dist_at_angle(scan, 80, 2)
                dist_280 = get_dist_at_angle(scan, 280, 2)

                curr_corner_x = get_car2_corner_x(scan)
                if curr_corner_x is not None: last_corner_x = curr_corner_x
                dist_LT = sonar_data[IDX_LT];
                dist_RT = sonar_data[IDX_RT]

                cmd_speed = 0;
                cmd_servo = SERVO_CENTER
                curr_time = time.time()
                msg = "";
                sub_msg = ""

                # [PAUSE 상태]
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    msg = f"WAITING... ({pause_msg})"
                    if curr_time - pause_start_time > pause_duration:
                        print(f"⏰ 대기 종료 -> {next_state_after_pause}")
                        state = next_state_after_pause
                        state_timer = curr_time
                        lidar.clean_input()

                # [1] 탐색
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH

                    if search_step == STEP_FIND_CAR1:
                        msg = "FIND CAR 1"
                        if lidar_radar < CAR_EXIST_DIST:
                            print(f"🚗 1번 차 감지! ({int(lidar_radar)}mm)")
                            search_step = STEP_PASS_CAR1

                    elif search_step == STEP_PASS_CAR1:
                        msg = "PASS CAR 1..."
                        if lidar_radar > EMPTY_SPACE_DIST:
                            valid_gap_count += 1
                            if valid_gap_count >= 2:
                                print(f"👀 빈 공간 진입! ({int(lidar_radar)}mm)")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        msg = "FIND CAR 2"
                        if lidar_radar < CAR_EXIST_DIST:
                            print(f"🛑 2번 차 감지! 정지 (거리: {int(lidar_radar)}mm)")
                            ser.write(b"D,-150\n");
                            time.sleep(0.1)
                            ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state_after_pause = STATE_SETUP_LEFT
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "Ready for Setup"

                # ---------------------------------------------------
                # [2] 공간 확보 (5초 하드코딩)
                # ---------------------------------------------------
                elif state == STATE_SETUP_LEFT:
                    cmd_servo = SERVO_LEFT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Align Left..."
                    else:
                        cmd_speed = SPEED_SETUP
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"SETUP: {driving_time:.1f}s / {TIME_SETUP_MOVE}s"

                        if driving_time > TIME_SETUP_MOVE:
                            print("🛑 공간 확보 완료 (5초). 정지.")
                            ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state_after_pause = STATE_REVERSE_TURN
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "Ready for Reverse"

                # ---------------------------------------------------
                # [3] 꺾어서 후진 (5초 하드코딩)
                # ---------------------------------------------------
                elif state == STATE_REVERSE_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Align Right..."
                    else:
                        cmd_speed = -SPEED_PARK
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"TURN: {driving_time:.1f}s / {TIME_REVERSE_TURN}s"

                        if driving_time > TIME_REVERSE_TURN:
                            print("🛑 1차 후진 완료 (5초). 정지.")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time

                # ---------------------------------------------------
                # [4] 마무리 (★ 80도 or 280도 인식 시 정지)
                # ---------------------------------------------------
                elif state == STATE_REVERSE_STRAIGHT:
                    cmd_servo = SERVO_CENTER

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Align Center..."
                    else:
                        cmd_speed = -SPEED_PARK
                        msg = f"FINISH: 80deg={int(dist_80)} 280deg={int(dist_280)}"

                    # 1. 후방 센서 체크 (안전장치)
                    if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                        print("✅ 주차 완료 (후방 벽 감지)")
                        state = STATE_DONE

                    # 2. ★ [핵심] 80도(우) or 280도(좌) 물체 감지 시 정지
                    elif dist_80 < SIDE_STOP_DIST or dist_280 < SIDE_STOP_DIST:
                        print(f"✅ 주차 완료 (측면 인식: 80도={int(dist_80)}, 280도={int(dist_280)})")
                        state = STATE_DONE

                    # 3. 시간 제한 (3초)
                    elif (curr_time - state_timer) > (STEER_WAIT_TIME + 3.0):
                        print("✅ 주차 완료 (시간 종료)")
                        state = STATE_DONE

                elif state == STATE_DONE:
                    cmd_speed = 0;
                    msg = "DONE"
                    ser.write(b"D,0\n")

                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                debug_img = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(debug_img, f"80deg: {int(dist_80)} | 280deg: {int(dist_280)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(debug_img, msg, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Parking Monitor", debug_img)
                if cv2.waitKey(1) == ord('q'): break

        except RPLidarException as e:
            print(f"⚠️ 라이다 오류: {e}")
            lidar.clean_input()
            continue
        except KeyboardInterrupt:
            print("종료")
            break
        except Exception as e:
            print(f"시스템 오류: {e}")
            break

    if ser: ser.write(b"D,0\n"); ser.close()
    if lidar: lidar.stop(); lidar.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()