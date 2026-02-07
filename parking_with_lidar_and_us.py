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
CENTERING_POWER = 40
STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 7.0
TIME_REVERSE_TURN = 6.5
TIME_EXIT_TURN = 18.0
TIME_EXIT_ADJUST = 1.0
TIME_DELAY_STOP = 0.5

# [센서 기준값]
# JUMP_THRESHOLD = 30 # 초기값
JUMP_THRESHOLD = 40
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
    STATE_REVERSE_STRAIGHT: "REVERSE PARKING",
    STATE_WAIT_AFTER_PARK: "PARKED",
    STATE_EXIT_TURN: "EXITING (TURN)",
    STATE_EXIT_STRAIGHT: "EXITING (FULL SPEED)",
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


def draw_bar(img, value, max_val, x, y, w, h, color, label):
    # 배경 바
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    # 값 바
    fill_w = int((min(value, max_val) / max_val) * w)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    # 테두리
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    # 텍스트
    cv2.putText(img, f"{label}: {value}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
        print(f"❌ 오류: {e}");
        return

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
                dist_LT = sonar_data[IDX_LT]
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

                # [4] 마무리 후진 (로직 유지: 70cm 옆차 감지 시 정지)
                elif state == STATE_REVERSE_STRAIGHT:
                    # 센터링 로직
                    center_diff = abs(dist_LT - dist_RT)
                    if center_diff > 50:
                        target_steer = SERVO_CENTER - CENTERING_POWER if dist_LT < dist_RT else SERVO_CENTER + CENTERING_POWER
                        msg = f"CENTERING (Diff:{center_diff})"
                    else:
                        target_steer = SERVO_CENTER
                        msg = "CENTERING OK"

                    cmd_servo = target_steer
                    cmd_speed = -SPEED_PARK if curr_time - state_timer > STEER_WAIT_TIME else 0

                    is_detected = False
                    if dist_90 < SIDE_STOP_DIST or dist_270 < SIDE_STOP_DIST:
                        msg = "STOP: SIDE CAR DETECTED"
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
                    msg = "PARKED (WAITING)"
                    if curr_time - state_timer > 3.0: # 3초 후 출차
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

                # [7] 마지막 무한 직진
                elif state == STATE_EXIT_STRAIGHT:
                    drive_time = curr_time - state_timer
                    if drive_time < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        cmd_servo = SERVO_CENTER;
                        msg = "ALIGNING..."
                    else:
                        cmd_speed = MAX_SPEED
                        real_drive_time = drive_time - STEER_WAIT_TIME
                        if real_drive_time < 3.3:
                            cmd_servo = SERVO_CENTER + 15;
                            msg = f"KICK LEFT ({real_drive_time:.1f}s)"
                        else:
                            cmd_servo = SERVO_CENTER;
                            msg = f"FULL SPEED ({real_drive_time:.1f}s)"

                # [통신]
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{cmd_servo}\n".encode());
                    ser.write(f"D,{cmd_speed}\n".encode())
                    last_serial_time = curr_time

                # ----------------------------------------------------------------
                # ★ [시각화] 상태 모니터링 창 (업그레이드됨)
                # ----------------------------------------------------------------
                # 1. 캔버스 생성 (600 x 500)
                monitor = np.zeros((500, 600, 3), dtype=np.uint8)

                # 2. 헤더 (현재 상태)
                cv2.putText(monitor, f"STATE: {STATE_NAMES.get(state, 'UNKNOWN')}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(monitor, f"MSG: {msg}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 3. 기본 센서 정보 (LiDAR)
                cv2.putText(monitor, "--- LIDAR INFO ---", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(monitor, f"Front Radar: {int(lidar_radar)}mm", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
                cv2.putText(monitor, f"Side L(270): {int(dist_270)}mm", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
                cv2.putText(monitor, f"Side R(90) : {int(dist_90)}mm", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)

                # 4. 상태별 추가 정보
                cv2.putText(monitor, "--- STATE DETAIL ---", (300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                            1)
                if state == STATE_SEARCH:
                    cv2.putText(monitor, f"Ref Dist: {int(reference_dist)}", (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 200), 1)
                    diff_val = lidar_radar - reference_dist if search_step == STEP_PASS_CAR1 else reference_dist - lidar_radar
                    cv2.putText(monitor, f"Diff: {int(diff_val)}", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (100, 100, 255), 1)
                elif state == STATE_REVERSE_STRAIGHT:
                    cv2.putText(monitor, f"Stop Limit: {SIDE_STOP_DIST}mm", (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 1)
                    # 센터링 상태 표시
                    center_color = (0, 255, 0) if abs(dist_LT - dist_RT) <= 50 else (0, 0, 255)
                    cv2.putText(monitor, f"Centering: {'GOOD' if abs(dist_LT - dist_RT) <= 50 else 'BAD'}",
                                (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, center_color, 2)

                # 5. 서보/모터 상태
                cv2.rectangle(monitor, (20, 260), (580, 310), (30, 30, 30), -1)
                cv2.putText(monitor, f"Servo: {cmd_servo} | Speed: {cmd_speed}", (40, 295), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 0), 2)

                # 6. ★ 후방 초음파 실시간 시각화 (LT / RT)
                # 게이지 바 그리기 (최대 1000mm 기준)
                cv2.putText(monitor, "--- REAR ULTRASONIC (LT/RT) ---", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)

                # 균형 상태에 따라 색상 변경 (5cm 이상 차이나면 빨간색)
                bar_color = (0, 255, 0)  # 초록색
                if abs(dist_LT - dist_RT) > 50:
                    bar_color = (0, 0, 255)  # 빨간색 (불균형)

                # LT 게이지 (왼쪽)
                draw_bar(monitor, dist_LT, 1000, 50, 370, 200, 30, bar_color, "LT")

                # RT 게이지 (오른쪽)
                draw_bar(monitor, dist_RT, 1000, 350, 370, 200, 30, bar_color, "RT")

                # 중앙 차이값 표시
                diff_text = f"Diff: {abs(dist_LT - dist_RT)}mm"
                cv2.putText(monitor, diff_text, (220, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Parking Monitor", monitor)
                if cv2.waitKey(1) == ord('q'): break

        except RPLidarException:
            lidar.clean_input();
            continue
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Err: {e}");
            break

    if ser: ser.write(b"D,0\n"); ser.close()
    if lidar: lidar.stop(); lidar.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()