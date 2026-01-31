import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
import math

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

SPEED_SEARCH = 70
SPEED_SETUP = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 560  # 벽타기용

STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 5.0  # 공간 확보 (전진)
TIME_REVERSE_TURN = 7.0  # 후진 진입 (회전)
TIME_EXIT_TURN = 5.0  # 출차 회전 시간
TIME_EXIT_ADJUST = 1.0  # 출차 전 보정(직진) 시간
# TIME_EXIT_STRAIGHT = 2.0  <-- (삭제됨: 무한 직진이므로 필요 없음)

TIME_DELAY_STOP = 1.5  # 감지 후 추가 이동 시간

TARGET_PARK_X = -400
CAR_EXIST_DIST = 1000
EMPTY_SPACE_DIST = 2000

SIDE_STOP_DIST = 700  # 70cm 이내 감지 시 정지

REAR_LIMIT = 200
IDX_LT = 2;
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0
STATE_SETUP_LEFT = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_DONE = 4
STATE_PAUSE = 99

# 출차 관련 상태
STATE_WAIT_AFTER_PARK = 10
STATE_EXIT_ADJUST = 11  # 좁을 때 잠깐 직진 보정
STATE_EXIT_TURN = 12  # 우회전으로 나감
STATE_EXIT_STRAIGHT = 13  # 직진 (무한)
STATE_FINISH = 14  # (도달하지 않음)

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


def get_dist_at_angle(scan, target_angle, range_pm=10):
    dists = []
    min_a = target_angle - range_pm
    max_a = target_angle + range_pm
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
    if len(dists) > 0: return np.min(dists)
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

    side_detect_time = 0

    while True:
        try:
            for scan in lidar.iter_scans():
                read_sensors()

                lidar_radar = get_lidar_min_dist(scan, 30, 110)
                dist_90 = get_dist_at_angle(scan, 90, 10)
                dist_270 = get_dist_at_angle(scan, 270, 10)

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

                # ---------------------------------------------------
                # [1] 탐색
                # ---------------------------------------------------
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    print(f"Step: {search_step} | Radar: {int(lidar_radar)}mm")

                    if search_step == STEP_FIND_CAR1:
                        msg = "FIND CAR 1"
                        cmd_servo = SERVO_CENTER
                        if lidar_radar < CAR_EXIST_DIST:
                            print(f"🚗 1번 차 감지! ({int(lidar_radar)}mm)")
                            search_step = STEP_PASS_CAR1

                    elif search_step == STEP_PASS_CAR1:
                        msg = "PASS CAR 1 (Wall Follow)"
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        if lidar_radar > EMPTY_SPACE_DIST:
                            valid_gap_count += 1
                            if valid_gap_count >= 2:
                                print(f"👀 빈 공간 진입! ({int(lidar_radar)}mm)")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        msg = "FIND CAR 2 (Wall Follow)"
                        cmd_servo = SERVO_SLIGHT_RIGHT
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
                # [2] 공간 확보
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
                            print("🛑 공간 확보 완료. 정지.")
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE
                            next_state_after_pause = STATE_REVERSE_TURN
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "Ready for Reverse"

                # ---------------------------------------------------
                # [3] 꺾어서 후진
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
                            print("🛑 1차 후진 완료. 정지.")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            side_detect_time = 0

                            # ---------------------------------------------------
                # [4] 마무리 & 지연 정지
                # ---------------------------------------------------
                elif state == STATE_REVERSE_STRAIGHT:
                    cmd_servo = SERVO_CENTER

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Align Center..."
                    else:
                        cmd_speed = -SPEED_PARK
                        msg = f"FINISH: 90={int(dist_90)} 270={int(dist_270)}"

                    is_detected = False
                    if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                        msg = "DETECT: Rear Wall!"
                        is_detected = True
                    elif dist_90 < SIDE_STOP_DIST or dist_270 < SIDE_STOP_DIST:
                        msg = "DETECT: Side Car!"
                        is_detected = True

                    if is_detected:
                        if side_detect_time == 0:
                            print("✨ 물체 감지! 1.5초 더 이동...")
                            side_detect_time = curr_time
                        elif curr_time - side_detect_time > TIME_DELAY_STOP:
                            print(f"✅ 주차 완료 (지연 정지).")
                            ser.write(b"D,0\n")
                            state = STATE_WAIT_AFTER_PARK
                            state_timer = curr_time

                    elif (curr_time - state_timer) > (STEER_WAIT_TIME + 5.0):
                        print("✅ 주차 완료 (시간 종료).")
                        ser.write(b"D,0\n")
                        state = STATE_WAIT_AFTER_PARK
                        state_timer = curr_time

                # ---------------------------------------------------
                # [5] 출차 (Exit) 로직
                # ---------------------------------------------------
                elif state == STATE_WAIT_AFTER_PARK:
                    cmd_speed = 0
                    msg = "Parking Done. Judging..."
                    if curr_time - state_timer > 2.0:
                        print(f"📏 출차 판단: L={int(dist_270)} R={int(dist_90)}")
                        # 오른쪽이 좁으면 보정, 아니면 바로 회전
                        if dist_90 < 600 or dist_90 < dist_270:
                            state = STATE_EXIT_ADJUST
                        else:
                            state = STATE_EXIT_TURN
                        state_timer = curr_time

                elif state == STATE_EXIT_ADJUST:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_ADJUST:
                            state = STATE_EXIT_TURN
                            state_timer = curr_time

                elif state == STATE_EXIT_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Exit: Align Right..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"EXIT TURN(R): {driving_time:.1f}s"

                        if driving_time > TIME_EXIT_TURN:
                            print("🚀 출차 직진 (무한 주행 시작)")
                            state = STATE_EXIT_STRAIGHT
                            state_timer = curr_time

                elif state == STATE_EXIT_STRAIGHT:
                    # 핸들 풀고 직진 (시간 제한 없음 = 무한)
                    cmd_servo = SERVO_CENTER

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        msg = "EXIT STRAIGHT (Infinite Run...)"
                        # ★ 여기서 STATE_FINISH로 넘어가는 로직을 삭제했습니다.
                        # 계속 직진합니다.

                elif state == STATE_FINISH:
                    cmd_speed = 0;
                    msg = "ALL MISSION COMPLETE"
                    ser.write(b"D,0\n")

                elif state == STATE_DONE:
                    cmd_speed = 0;
                    msg = "DONE"
                    ser.write(b"D,0\n")

                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                debug_img = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | Msg: {msg}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
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