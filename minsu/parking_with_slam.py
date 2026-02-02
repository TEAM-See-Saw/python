import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값 (튜닝됨)
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

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555

# 센터링 보정 강도
CENTERING_POWER = 30

STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 8.0
TIME_REVERSE_TURN = 7.5
TIME_EXIT_TURN = 16.0
TIME_EXIT_ADJUST = 1.0
TIME_DELAY_STOP = 0.5

# [센서 기준값]
JUMP_THRESHOLD = 30  # 3cm(30mm)만 차이나도 빈 공간으로 인식
MAX_VALID_DIST = 4000
SAFETY_DIST_CAR2 = 1200  # 이 거리 안으로 들어오면 2번차로 간주하고 강제 정지

SIDE_STOP_DIST = 700
REAR_LIMIT = 200
IDX_LT = 2
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0
STATE_SETUP_LEFT = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_DONE = 4
STATE_PAUSE = 99
STATE_WAIT_AFTER_PARK = 10
STATE_EXIT_ADJUST = 11
STATE_EXIT_TURN = 12
STATE_EXIT_STRAIGHT = 13
STATE_FINISH = 14

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP = 2
STEP_FIND_CAR2 = 3

# 화면 표시용 상태 이름
STATE_NAMES = {
    STATE_SEARCH: "SEARCHING",
    STATE_SETUP_LEFT: "SETUP MOVE (LEFT)",
    STATE_REVERSE_TURN: "REVERSE TURN (PARKING)",
    STATE_REVERSE_STRAIGHT: "REVERSE PARKING (CENTERING)",
    STATE_PAUSE: "PAUSED",
    STATE_WAIT_AFTER_PARK: "PARKED & JUDGING",
    STATE_EXIT_TURN: "EXITING (TURN)",
    STATE_EXIT_STRAIGHT: "EXITING (STRAIGHT)",
    STATE_FINISH: "FINISHED"
}

# ==========================================
# [2] 함수 정의
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
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


# ==========================================
# [3] 메인 루프
# ==========================================
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

    # 초기화
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    ser.write(b"D,0\n")
    time.sleep(1.0)
    lidar.clean_input()

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0
    side_detect_time = 0

    pause_start_time = 0
    pause_duration = 0
    next_state_after_pause = 0
    pause_msg = ""

    reference_dist = 0
    stable_count = 0
    last_serial_time = 0

    while True:
        try:
            for scan in lidar.iter_scans():
                read_sensors()  # Flush

                lidar_radar = get_lidar_min_dist(scan, 30, 110)
                dist_90 = get_dist_at_angle(scan, 90, 10)
                dist_270 = get_dist_at_angle(scan, 270, 10)

                dist_LT = sonar_data[IDX_LT]
                dist_RT = sonar_data[IDX_RT]

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                curr_time = time.time()
                msg = ""
                log_text = ""

                # ---------------------------------------------------
                # [PAUSE]
                # ---------------------------------------------------
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    msg = f"WAIT: {pause_msg}"
                    log_text = f"[대기중] {pause_msg} ({curr_time - pause_start_time:.1f}/{pause_duration}s)"
                    if curr_time - pause_start_time > pause_duration:
                        print(f"⏰ 대기 종료 -> {next_state_after_pause}로 이동")
                        state = next_state_after_pause
                        state_timer = curr_time
                        lidar.clean_input()

                # ---------------------------------------------------
                # [1] 탐색 (SEARCH)
                # ---------------------------------------------------
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH

                    diff = 0
                    if search_step != STEP_FIND_CAR1:
                        diff = lidar_radar - reference_dist if search_step == STEP_PASS_CAR1 else reference_dist - lidar_radar

                    # Step 1: 1번 차 찾기
                    if search_step == STEP_FIND_CAR1:
                        msg = "STEP 1: FIND CAR 1"
                        log_text = f"[탐색] 1번차 찾는중.. 거리:{int(lidar_radar)}"
                        cmd_servo = SERVO_CENTER

                        if 100 < lidar_radar < 2000:
                            stable_count += 1
                            if stable_count > 2:
                                print(f"🚗 1번 차 발견! (거리: {int(lidar_radar)})")
                                reference_dist = lidar_radar
                                search_step = STEP_PASS_CAR1
                                stable_count = 0
                        else:
                            stable_count = 0

                    # Step 2: 1번 차 통과 -> 빈 공간 찾기
                    elif search_step == STEP_PASS_CAR1:
                        msg = "STEP 2: FIND GAP / CAR 2"
                        diff = lidar_radar - reference_dist
                        log_text = f"[탐색] 1번차 통과중.. 현재:{int(lidar_radar)} 기준:{int(reference_dist)} 차이:{int(diff)}"
                        cmd_servo = SERVO_SLIGHT_RIGHT

                        if (diff > JUMP_THRESHOLD and lidar_radar < MAX_VALID_DIST) or (lidar_radar < SAFETY_DIST_CAR2):
                            stable_count += 1
                            if stable_count > 1: # 라이다로 값 2번 튀면 빈 공간으로 인식
                                if lidar_radar < SAFETY_DIST_CAR2:
                                    print(f"🚨 2번 차 바로 앞 도착! (거리: {int(lidar_radar)}) -> 강제 정지")
                                else:
                                    print(f"👀 빈 공간 발견! (차이: {int(diff)}) -> 2초 정지")

                                reference_dist = lidar_radar
                                search_step = STEP_FIND_GAP

                                ser.write(b"D,0\n")
                                state = STATE_PAUSE
                                next_state_after_pause = STATE_SEARCH
                                pause_duration = 2.0
                                pause_start_time = curr_time
                                pause_msg = "GAP/CAR2 FOUND"
                                stable_count = 0
                        else:
                            if lidar_radar < reference_dist:
                                reference_dist = (reference_dist * 0.7) + (lidar_radar * 0.3)
                            stable_count = 0

                    # Step 3: 2번 차 찾기 (정지 시 역회전 삭제)
                    elif search_step == STEP_FIND_GAP:
                        msg = "STEP 3: CONFIRM CAR 2"
                        print("🛑 2번 차 확인 완료")

                        # ★ [수정] 역회전(-150) 삭제 -> 그냥 정지(0)
                        ser.write(b"D,0\n")

                        state = STATE_PAUSE
                        next_state_after_pause = STATE_SETUP_LEFT
                        pause_duration = 1.0
                        pause_start_time = curr_time
                        pause_msg = "DETECT CAR 2"
                        stable_count = 0

                # ---------------------------------------------------
                # [2] 공간 확보
                # ---------------------------------------------------
                elif state == STATE_SETUP_LEFT:
                    cmd_servo = SERVO_LEFT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "ALIGN LEFT"
                    else:
                        cmd_speed = SPEED_SETUP
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"SETUP MOVE: {driving_time:.1f}s"
                        log_text = f"[공간확보] 좌측 전진 중... {driving_time:.1f}초"
                        if driving_time > TIME_SETUP_MOVE:
                            print("🛑 공간 확보 완료")
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE
                            next_state_after_pause = STATE_REVERSE_TURN
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "READY TO REVERSE"

                # ---------------------------------------------------
                # [3] 후진 진입
                # ---------------------------------------------------
                elif state == STATE_REVERSE_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "ALIGN RIGHT"
                    else:
                        cmd_speed = -SPEED_PARK
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"REVERSE TURN: {driving_time:.1f}s"
                        log_text = f"[후진진입] 우측 후진 중... {driving_time:.1f}초"

                        if driving_time > TIME_REVERSE_TURN:
                            print("🛑 1차 후진 완료")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            side_detect_time = 0

                # ---------------------------------------------------
                # [4] 마무리 후진 (센터링)
                # ---------------------------------------------------
                elif state == STATE_REVERSE_STRAIGHT:
                    target_steer = SERVO_CENTER

                    if abs(dist_LT - dist_RT) > 50:
                        if dist_LT < dist_RT:
                            target_steer = SERVO_CENTER - CENTERING_POWER
                            msg = "CENTERING: >> (Left Close)"
                        else:
                            target_steer = SERVO_CENTER + CENTERING_POWER
                            msg = "CENTERING: << (Right Close)"
                    else:
                        msg = "CENTERING: OK"

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = target_steer
                    else:
                        cmd_servo = target_steer
                        cmd_speed = -SPEED_PARK
                        log_text = f"[주차중] L:{dist_LT} R:{dist_RT} 조향:{cmd_servo}"

                    is_detected = False
                    if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                        msg = "STOP: REAR WALL"
                        is_detected = True
                    elif dist_90 < SIDE_STOP_DIST or dist_270 < SIDE_STOP_DIST:
                        msg = "STOP: SIDE CAR"
                        is_detected = True

                    if is_detected:
                        if side_detect_time == 0:
                            print("✨ 감지! 지연 정지 시작")
                            side_detect_time = curr_time
                        elif curr_time - side_detect_time > TIME_DELAY_STOP:
                            print(f"✅ 주차 완료")
                            ser.write(b"D,0\n")
                            state = STATE_WAIT_AFTER_PARK
                            state_timer = curr_time
                    elif (curr_time - state_timer) > (STEER_WAIT_TIME + 5.0):
                        print("✅ 주차 완료 (시간종료)")
                        ser.write(b"D,0\n")
                        state = STATE_WAIT_AFTER_PARK
                        state_timer = curr_time

                # ---------------------------------------------------
                # [5] 출차 판단
                # ---------------------------------------------------
                elif state == STATE_WAIT_AFTER_PARK:
                    cmd_speed = 0
                    msg = "JUDGING EXIT PATH..."
                    log_text = f"[판단] 출차 경로 계산중.. L:{int(dist_270)} R:{int(dist_90)}"
                    if curr_time - state_timer > 4.0:
                        if dist_90 < 600 or dist_90 < dist_270:
                            state = STATE_EXIT_ADJUST
                        else:
                            state = STATE_EXIT_TURN
                        state_timer = curr_time

                # ---------------------------------------------------
                # [6] 출차 실행
                # ---------------------------------------------------
                elif state == STATE_EXIT_ADJUST:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                    else:
                        cmd_speed = SPEED_SEARCH
                        msg = "EXIT: ADJUSTING"
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_ADJUST:
                            state = STATE_EXIT_TURN
                            state_timer = curr_time

                elif state == STATE_EXIT_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                    else:
                        cmd_speed = SPEED_SEARCH
                        msg = "EXIT: TURNING"
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_TURN:
                            state = STATE_EXIT_STRAIGHT
                            state_timer = curr_time

                elif state == STATE_EXIT_STRAIGHT:
                    drive_time = curr_time - state_timer

                    if drive_time < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = SERVO_CENTER
                        msg = "EXIT COMPLETE: ALIGN"
                    else:
                        cmd_speed = SPEED_SEARCH
                        real_drive_time = drive_time - STEER_WAIT_TIME

                        if real_drive_time < 3.0:
                            cmd_servo = SERVO_CENTER
                            msg = f"RUNNING STRAIGHT ({real_drive_time:.1f}s)"
                        elif real_drive_time < 3.3:
                            cmd_servo = SERVO_CENTER + 15
                            msg = "!!! KICK LEFT !!!"
                        else:
                            cmd_servo = SERVO_CENTER
                            msg = "RUNNING STRAIGHT"

                # ---------------------------------------------------
                # [로그 & 화면 표시 & 통신]
                # ---------------------------------------------------
                if log_text: print(log_text)

                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{cmd_servo}\n".encode())
                    ser.write(f"D,{cmd_speed}\n".encode())
                    last_serial_time = curr_time

                debug_img = np.zeros((300, 600, 3), dtype=np.uint8)
                state_str = STATE_NAMES.get(state, "UNKNOWN")
                cv2.putText(debug_img, f"STATE: {state_str}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(debug_img, f"MSG: {msg}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(debug_img, f"Lidar: {int(lidar_radar)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)

                if state == STATE_SEARCH:
                    cv2.putText(debug_img, f"Ref: {int(reference_dist)}", (150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 200), 1)
                    d_val = lidar_radar - reference_dist if search_step == STEP_PASS_CAR1 else reference_dist - lidar_radar
                    cv2.putText(debug_img, f"Diff: {int(d_val)}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (100, 100, 255), 1)

                if state == STATE_REVERSE_STRAIGHT:
                    cv2.putText(debug_img, f"Sonar L:{dist_LT} R:{dist_RT}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 100, 100), 1)

                cv2.imshow("Parking Monitor", debug_img)

                if cv2.waitKey(1) == ord('q'):
                    print("\n🛑 사용자 강제 종료 (q)")
                    break

        except RPLidarException as e:
            lidar.clean_input()
            continue
        except KeyboardInterrupt:
            print("\n🛑 키보드 인터럽트 종료")
            break
        except Exception as e:
            print(f"오류: {e}")
            break

    print("🛑 시스템 종료: 정지 명령 전송")
    if ser:
        for _ in range(5):
            ser.write(b"D,0\n")
            time.sleep(0.02)
        ser.close()
    if lidar:
        lidar.stop()
        lidar.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()