import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
import math
from collections import deque

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
SERVO_SLIGHT_RIGHT = 555  # 벽타기용

STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 7.0       # 공간 확보 (전진)
TIME_REVERSE_TURN = 7.0     # 후진 진입 (회전)
TIME_EXIT_TURN = 7.0        # 출차 회전 시간
TIME_EXIT_ADJUST = 1.0      # 출차 전 보정(직진) 시간

# 감지 후 추가 이동(기존 1.5초) -> 안전하게 조정 가능
TIME_DELAY_STOP = 1.0       # (기존 1.5) 감지 후 "조금만" 더 이동
DELAY_SLOW_RATIO = 0.5      # 감지 후 감속 비율 (0.5 = 절반 속도)

TARGET_PARK_X = -400
CAR_EXIST_DIST = 2500
EMPTY_SPACE_DIST = 3500

SIDE_STOP_DIST = 700        # 70cm 이내 감지 시 정지

# "너무 가까우면 즉시 정지" (안전장치)
REAR_URGENT = 120           # 초음파 뒤쪽이 12cm 이내면 즉시 정지
SIDE_URGENT = 350           # 라이다 측면이 35cm 이내면 즉시 정지

REAR_LIMIT = 200
IDX_LT = 2
IDX_RT = 5

# EXIT(무한 직진) 최소 안전장치: 전방이 너무 가까우면 정지
EXIT_FRONT_STOP_ENABLE = True
EXIT_FRONT_STOP_DIST = 800  # 80cm 이내면 정지

# 라이다 튐 완화 파라미터
LIDAR_TEMPORAL_N = 5         # 최근 N개를 모아서 median
LIDAR_SECTOR_PCTL = 10       # 섹터 거리들 중 하위 10% 지점(=벽/차에 민감, outlier 덜탐)

# 상태 정의
STATE_SEARCH = 0
STATE_SETUP_LEFT = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_DONE = 4
STATE_PAUSE = 99

# 출차 관련 상태
STATE_WAIT_AFTER_PARK = 10
STATE_EXIT_ADJUST = 11
STATE_EXIT_TURN = 12
STATE_EXIT_STRAIGHT = 13
STATE_FINISH = 14  # (도달하지 않음)

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP = 2

# ==========================================
# [2] 함수 정의
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    """아두이노에서 'US:x,x,x,x,x,x' 형태로 초음파 6채널 수신."""
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass


def sector_percentile_dist(scan, start_angle, end_angle, percentile=10):
    """섹터 내 거리들에서 min 대신 하위 퍼센타일 값을 반환 (outlier에 강함)."""
    dists = []
    for (_, angle, dist) in scan:
        if dist > 0 and (start_angle <= angle <= end_angle):
            dists.append(dist)
    if not dists:
        return 9999
    # np.percentile은 float 반환
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))


def angle_percentile_dist(scan, target_angle, range_pm=10, percentile=10):
    """특정 각도 주변(range) 거리들에서 하위 퍼센타일 반환."""
    dists = []
    min_a = target_angle - range_pm
    max_a = target_angle + range_pm
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
    if not dists:
        return 9999
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))


def temporal_median_push(buf: deque, value: float, fallback=9999):
    """최근 N개 median을 반환. 값이 9999(무효)라도 넣되, median이 과도히 망가지지 않게 사용자가 조정 가능."""
    buf.append(value)
    if not buf:
        return fallback
    return float(np.median(np.array(buf, dtype=np.float32)))


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
        print(f"❌ 오류: {e}")
        return

    # 초기 바퀴 정렬 + 정지
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    ser.write(b"D,0\n")
    time.sleep(1.0)
    lidar.clean_input()

    # 라이다 temporal buffer
    buf_front = deque(maxlen=LIDAR_TEMPORAL_N)
    buf_90 = deque(maxlen=LIDAR_TEMPORAL_N)
    buf_270 = deque(maxlen=LIDAR_TEMPORAL_N)

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = time.time()

    valid_gap_count = 0

    pause_start_time = 0.0
    pause_duration = 0.0
    next_state_after_pause = STATE_SEARCH
    pause_msg = ""

    # 감지 후 지연 정지 타이머
    detect_time = 0.0

    running = True

    while running:
        try:
            for scan in lidar.iter_scans():
                if not running:
                    break

                read_sensors()

                # --- 라이다: 퍼센타일 + temporal median ---
                front_raw = sector_percentile_dist(scan, 30, 110, LIDAR_SECTOR_PCTL)
                d90_raw = angle_percentile_dist(scan, 90, 10, LIDAR_SECTOR_PCTL)
                d270_raw = angle_percentile_dist(scan, 270, 10, LIDAR_SECTOR_PCTL)

                lidar_radar = temporal_median_push(buf_front, front_raw)
                dist_90 = temporal_median_push(buf_90, d90_raw)
                dist_270 = temporal_median_push(buf_270, d270_raw)

                dist_LT = sonar_data[IDX_LT]
                dist_RT = sonar_data[IDX_RT]

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                curr_time = time.time()
                msg = ""

                # [PAUSE 상태]
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    msg = f"WAITING... ({pause_msg})"
                    if curr_time - pause_start_time > pause_duration:
                        state = next_state_after_pause
                        state_timer = curr_time
                        detect_time = 0.0
                        lidar.clean_input()

                # ---------------------------------------------------
                # [1] 탐색
                # ---------------------------------------------------
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH

                    if search_step == STEP_FIND_CAR1:
                        msg = "FIND CAR 1"
                        cmd_servo = SERVO_CENTER
                        if lidar_radar < CAR_EXIST_DIST:
                            print(f"🚗 1번 차 감지! ({int(lidar_radar)}mm)")
                            search_step = STEP_PASS_CAR1
                            valid_gap_count = 0

                    elif search_step == STEP_PASS_CAR1:
                        msg = "PASS CAR 1 (Wall Follow)"
                        cmd_servo = SERVO_SLIGHT_RIGHT

                        if lidar_radar > EMPTY_SPACE_DIST:
                            valid_gap_count += 1
                            if valid_gap_count >= 3:  # (기존 2) 살짝 더 안정적으로
                                print(f"👀 빈 공간 진입! ({int(lidar_radar)}mm)")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        msg = "FIND CAR 2 (Wall Follow)"
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        if lidar_radar < CAR_EXIST_DIST:
                            print(f"🛑 2번 차 감지! 정지 (거리: {int(lidar_radar)}mm)")
                            # 브레이크 느낌
                            ser.write(b"D,-150\n")
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
                        cmd_speed = 0
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
                        cmd_speed = 0
                        msg = "Align Right..."
                    else:
                        cmd_speed = -SPEED_PARK
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"TURN: {driving_time:.1f}s / {TIME_REVERSE_TURN}s"
                        if driving_time > TIME_REVERSE_TURN:
                            print("🛑 1차 후진 완료. 정지.")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            detect_time = 0.0

                # ---------------------------------------------------
                # [4] 마무리 후진 & 안전 정지
                # ---------------------------------------------------
                elif state == STATE_REVERSE_STRAIGHT:
                    cmd_servo = SERVO_CENTER

                    # 조향 중앙 정렬 대기
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align Center..."
                    else:
                        # 기본 후진 속도
                        cmd_speed = -SPEED_PARK
                        msg = f"FINISH: 90={int(dist_90)} 270={int(dist_270)} US(L/R)={dist_LT}/{dist_RT}"

                    # --- 감지 로직 ---
                    rear_detect = (dist_LT < REAR_LIMIT) or (dist_RT < REAR_LIMIT)
                    side_detect = (dist_90 < SIDE_STOP_DIST) or (dist_270 < SIDE_STOP_DIST)

                    # urgent 즉시 정지 조건
                    rear_urgent = (dist_LT < REAR_URGENT) or (dist_RT < REAR_URGENT)
                    side_urgent = (dist_90 < SIDE_URGENT) or (dist_270 < SIDE_URGENT)

                    is_detected = rear_detect or side_detect
                    is_urgent = rear_urgent or side_urgent

                    if is_detected and (curr_time - state_timer >= STEER_WAIT_TIME):
                        if is_urgent:
                            # 너무 가까우면 즉시 정지
                            print("🧯 URGENT STOP!")
                            ser.write(b"D,0\n")
                            state = STATE_WAIT_AFTER_PARK
                            state_timer = curr_time
                            detect_time = 0.0
                            msg = "URGENT STOP"
                        else:
                            # 감지되었으나 여유가 있으면 감속 후 짧게 추가 이동
                            if detect_time == 0.0:
                                print("✨ 물체 감지! 감속 후 짧게 이동...")
                                detect_time = curr_time

                            # 감속
                            cmd_speed = int(cmd_speed * DELAY_SLOW_RATIO)

                            if curr_time - detect_time > TIME_DELAY_STOP:
                                print("✅ 주차 완료 (감속 지연 정지).")
                                ser.write(b"D,0\n")
                                state = STATE_WAIT_AFTER_PARK
                                state_timer = curr_time
                                detect_time = 0.0
                                msg = "STOP (delayed)"
                    else:
                        # 감지 없이도 시간 종료로 안전하게 끝내기 (기존 유지)
                        if (curr_time - state_timer) > (STEER_WAIT_TIME + 5.0):
                            print("✅ 주차 완료 (시간 종료).")
                            ser.write(b"D,0\n")
                            state = STATE_WAIT_AFTER_PARK
                            state_timer = curr_time
                            detect_time = 0.0
                            msg = "STOP (timeout)"

                # ---------------------------------------------------
                # [5] 출차 (Exit) 로직
                # ---------------------------------------------------
                elif state == STATE_WAIT_AFTER_PARK:
                    cmd_speed = 0
                    msg = "Parking Done. Judging..."
                    if curr_time - state_timer > 2.0:
                        print(f"📏 출차 판단: L(270)={int(dist_270)} R(90)={int(dist_90)}")
                        # 오른쪽이 좁으면 보정, 아니면 바로 회전
                        if dist_90 < 600 or dist_90 < dist_270:
                            state = STATE_EXIT_ADJUST
                        else:
                            state = STATE_EXIT_TURN
                        state_timer = curr_time

                elif state == STATE_EXIT_ADJUST:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_ADJUST:
                            state = STATE_EXIT_TURN
                            state_timer = curr_time

                elif state == STATE_EXIT_TURN:
                    cmd_servo = SERVO_RIGHT_MAX
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
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
                    # 핸들 풀고 직진 (시간 제한 없음)
                    cmd_servo = SERVO_CENTER

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        msg = "EXIT STRAIGHT (Infinite Run...)"

                        # 최소 안전장치(원하면 끄기)
                        if EXIT_FRONT_STOP_ENABLE and lidar_radar < EXIT_FRONT_STOP_DIST:
                            print(f"🛑 EXIT FRONT STOP! front={int(lidar_radar)}mm")
                            cmd_speed = 0
                            state = STATE_DONE

                elif state == STATE_DONE:
                    cmd_speed = 0
                    msg = "DONE"
                    ser.write(b"D,0\n")

                # --- 명령 송신 ---
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                # --- 디버그 화면 ---
                debug_img = np.zeros((320, 720, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | Step: {search_step}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(debug_img, f"Msg: {msg}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, f"LidarFront={int(lidar_radar)}  90={int(dist_90)}  270={int(dist_270)}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, f"US(LT/RT)={dist_LT}/{dist_RT}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, f"q: quit", (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", debug_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                    break

        except RPLidarException as e:
            print(f"⚠️ 라이다 오류: {e}")
            try:
                lidar.clean_input()
            except:
                pass
            continue
        except KeyboardInterrupt:
            print("종료(KeyboardInterrupt)")
            break
        except Exception as e:
            print(f"시스템 오류: {e}")
            break

    # --- 종료 처리 ---
    try:
        if ser:
            ser.write(b"D,0\n")
            ser.close()
    except:
        pass

    try:
        if lidar:
            lidar.stop()
            lidar.disconnect()
    except:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
