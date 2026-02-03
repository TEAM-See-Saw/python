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
BAUDRATE = 115200

SERIAL_DELAY = 0.05

# 속도 설정
SPEED_SEARCH = 60
SPEED_SETUP = 80
SPEED_PARK = 75
SPEED_STOP = 0
MAX_SPEED = 255  # ★ 최고 속도

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555

# 센터링 보정 강도
CENTERING_POWER = 30
STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 7.0
TIME_REVERSE_TURN = 6.5
TIME_EXIT_TURN = 18.0
TIME_EXIT_ADJUST = 1.0
TIME_DELAY_STOP = 0.5

# [센서 기준값]
JUMP_THRESHOLD = 30
MAX_VALID_DIST = 4000
SAFETY_DIST_CAR2 = 1200
SIDE_STOP_DIST = 700  # ★ 이 거리(70cm)가 유일한 주차 정지 기준임!

IDX_LT = 2
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0
STATE_SETUP_LEFT = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_WAIT_AFTER_PARK = 10
STATE_EXIT_ADJUST = 11
STATE_EXIT_TURN = 12
STATE_EXIT_STRAIGHT = 13
STATE_PAUSE = 99

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP = 2

# 이름 정의
STATE_NAMES = {
    STATE_SEARCH: "SEARCHING",
    STATE_SETUP_LEFT: "SETUP MOVE",
    STATE_REVERSE_TURN: "REVERSE TURN",
    STATE_REVERSE_STRAIGHT: "REVERSE PARKING (SENSING)",
    STATE_WAIT_AFTER_PARK: "PARKED",
    STATE_EXIT_TURN: "EXITING (TURN)",
    STATE_EXIT_STRAIGHT: "EXITING (FULL SPEED)",  # 이름 변경
    STATE_PAUSE: "PAUSED"
}

# ==========================================
# [2] 메인 루프
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6: sonar_data = [int(p) for p in parts]
        except:
            pass


def get_lidar_min_dist(scan, start, end):
    dists = [d for (_, a, d) in scan if d > 0 and start <= a <= end]
    return np.min(dists) if dists else 9999


def get_dist_at_angle(scan, target, pm=10):
    dists = [d for (_, a, d) in scan if d > 0 and target - pm <= a <= target + pm]
    return np.min(dists) if dists else 9999


def main():
    global ser, lidar
    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 완료")
        time.sleep(1)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 오류: {e}"); return

    ser.write(f"S,{SERVO_CENTER}\n".encode());
    ser.write(b"D,0\n");
    time.sleep(1.0)
    lidar.clean_input()

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0
    side_detect_time = 0
    reference_dist = 0
    stable_count = 0
    last_serial_time = 0

    # PAUSE 관련
    pause_start_time = 0;
    pause_duration = 0;
    next_state_after_pause = 0;
    pause_msg = ""

    while True:
        try:
            for scan in lidar.iter_scans():
                read_sensors()

                lidar_radar = get_lidar_min_dist(scan, 30, 110)
                dist_90 = get_dist_at_angle(scan, 90, 10)
                dist_270 = get_dist_at_angle(scan, 270, 10)
                dist_LT = sonar_data[IDX_LT];
                dist_RT = sonar_data[IDX_RT]

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                curr_time = time.time()
                msg = ""

                # [PAUSE]
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    msg = f"WAIT: {pause_msg}"
                    if curr_time - pause_start_time > pause_duration:
                        state = next_state_after_pause
                        state_timer = curr_time
                        lidar.clean_input()

                # [1] 탐색
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    if search_step == STEP_FIND_CAR1:
                        msg = "FIND CAR 1"
                        if 100 < lidar_radar < 2000:
                            stable_count += 1
                            if stable_count > 2:
                                reference_dist = lidar_radar;
                                search_step = STEP_PASS_CAR1;
                                stable_count = 0
                        else:
                            stable_count = 0
                    elif search_step == STEP_PASS_CAR1:
                        msg = "PASS CAR 1"
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        diff = lidar_radar - reference_dist
                        if (diff > JUMP_THRESHOLD and lidar_radar < MAX_VALID_DIST) or (lidar_radar < SAFETY_DIST_CAR2):
                            stable_count += 1
                            if stable_count > 1:
                                reference_dist = lidar_radar;
                                search_step = STEP_FIND_GAP
                                ser.write(b"D,0\n")
                                state = STATE_PAUSE;
                                next_state_after_pause = STATE_SEARCH
                                pause_duration = 2.0;
                                pause_start_time = curr_time;
                                pause_msg = "GAP FOUND"
                                stable_count = 0
                        else:
                            if lidar_radar < reference_dist: reference_dist = (reference_dist * 0.7) + (
                                        lidar_radar * 0.3)
                            stable_count = 0
                    elif search_step == STEP_FIND_GAP:
                        ser.write(b"D,0\n")
                        state = STATE_PAUSE;
                        next_state_after_pause = STATE_SETUP_LEFT
                        pause_duration = 1.0;
                        pause_start_time = curr_time;
                        pause_msg = "CONFIRM GAP"

                # [2] 공간 확보
                elif state == STATE_SETUP_LEFT:
                    cmd_servo = SERVO_LEFT_MAX
                    if curr_time - state_timer > STEER_WAIT_TIME:
                        cmd_speed = SPEED_SETUP
                        msg = "SETUP MOVE"
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_SETUP_MOVE:
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE;
                            next_state_after_pause = STATE_REVERSE_TURN
                            pause_duration = 1.0;
                            pause_start_time = curr_time;
                            pause_msg = "READY REVERSE"

                # [3] 후진 진입
                elif state == STATE_REVERSE_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer > STEER_WAIT_TIME:
                        cmd_speed = -SPEED_PARK
                        msg = "REVERSE TURN"
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_REVERSE_TURN:
                            state = STATE_REVERSE_STRAIGHT;
                            state_timer = curr_time;
                            side_detect_time = 0

                # ---------------------------------------------------
                # [4] 마무리 후진 (★ 수정됨: 초음파/시간제한 삭제)
                # ---------------------------------------------------
                elif state == STATE_REVERSE_STRAIGHT:
                    # 센터링 로직
                    if abs(dist_LT - dist_RT) > 50:
                        target_steer = SERVO_CENTER - CENTERING_POWER if dist_LT < dist_RT else SERVO_CENTER + CENTERING_POWER
                        msg = "CENTERING..."
                    else:
                        target_steer = SERVO_CENTER
                        msg = "CENTERING OK"

                    cmd_servo = target_steer
                    cmd_speed = -SPEED_PARK if curr_time - state_timer > STEER_WAIT_TIME else 0

                    # ★ [삭제됨] 초음파(REAR_LIMIT) 정지 로직 삭제
                    # ★ [삭제됨] 시간제한(5.8초) 정지 로직 삭제

                    # ★ [유일한 정지 조건] 옆 차 감지 (라이다)
                    is_detected = False
                    if dist_90 < SIDE_STOP_DIST or dist_270 < SIDE_STOP_DIST:
                        msg = "STOP: SIDE CAR"
                        is_detected = True

                    if is_detected:
                        if side_detect_time == 0:
                            side_detect_time = curr_time
                        elif curr_time - side_detect_time > TIME_DELAY_STOP:
                            print(f"✅ 주차 완료 (Side Lidar)")
                            ser.write(b"D,0\n")
                            state = STATE_WAIT_AFTER_PARK;
                            state_timer = curr_time

                # [5] 출차 판단
                elif state == STATE_WAIT_AFTER_PARK:
                    cmd_speed = 0;
                    msg = "PARKED"
                    if curr_time - state_timer > 4.0:
                        state = STATE_EXIT_ADJUST if (dist_90 < 600 or dist_90 < dist_270) else STATE_EXIT_TURN
                        state_timer = curr_time

                # [6] 출차 실행
                elif state == STATE_EXIT_ADJUST:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer > STEER_WAIT_TIME:
                        cmd_speed = SPEED_SEARCH
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_ADJUST:
                            state = STATE_EXIT_TURN;
                            state_timer = curr_time

                elif state == STATE_EXIT_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer > STEER_WAIT_TIME:
                        cmd_speed = SPEED_SEARCH
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_TURN:
                            state = STATE_EXIT_STRAIGHT;
                            state_timer = curr_time

                # ---------------------------------------------------
                # [7] 마지막 무한 직진 (★ 수정됨: 속도 255)
                # ---------------------------------------------------
                elif state == STATE_EXIT_STRAIGHT:
                    drive_time = curr_time - state_timer
                    if drive_time < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        cmd_servo = SERVO_CENTER;
                        msg = "ALIGNING..."
                    else:
                        # ★ 여기서 속도를 MAX_SPEED(255)로 변경
                        cmd_speed = MAX_SPEED

                        real_drive_time = drive_time - STEER_WAIT_TIME
                        if real_drive_time < 3.3:
                            cmd_servo = SERVO_CENTER + 15;
                            msg = "KICK LEFT (FULL SPEED)"
                        else:
                            cmd_servo = SERVO_CENTER;
                            msg = "INFINITE RUN (FULL SPEED)"

                # [통신 & 디스플레이]
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{cmd_servo}\n".encode());
                    ser.write(f"D,{cmd_speed}\n".encode())
                    last_serial_time = curr_time

                debug_img = np.zeros((200, 400, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"STATE: {STATE_NAMES.get(state, 'UNKNOWN')}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_img, f"MSG: {msg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Parking Monitor", debug_img)
                if cv2.waitKey(1) == ord('q'): break

        except RPLidarException:
            lidar.clean_input(); continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Err: {e}"); break

    if ser: ser.write(b"D,0\n"); ser.close()
    if lidar: lidar.stop(); lidar.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()