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

# 속도 설정
SPEED_SEARCH = 70
SPEED_SETUP = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555  # [벽타기용] 살짝 우측

STEER_WAIT_TIME = 0.8

# [시간 설정]
TIME_SETUP_MOVE = 7.0  # 공간 확보 전진 (7초)
TIME_REVERSE_TURN = 7.0  # 후진 진입 (7초)
TIME_EXIT_TURN = 7.0  # 출차 회전 (7초)
TIME_EXIT_ADJUST = 1.0  # 좁을 때 보정 전진 (1초)
TIME_DELAY_STOP = 1.5  # 감지 후 추가 주행 시간

# [센서 기준값]
JUMP_THRESHOLD = 600  # 변화량 기준 (60cm 이상 변하면 상태 전환)
MAX_VALID_DIST = 4000  # 노이즈 필터링용 최대 거리

SIDE_STOP_DIST = 700  # 주차 시 측면 정지 거리 (70cm)
REAR_LIMIT = 200  # 후방 정지 거리 (20cm)
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
    # 데이터가 있으면 싹 다 읽어서 마지막 것만 취함 (Flush 개념)
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass


# [탐색용] 30도 ~ 110도 (넓은 범위 최소값)
def get_lidar_min_dist(scan, start_angle, end_angle):
    dists = []
    for (_, angle, dist) in scan:
        if dist > 0 and (start_angle <= angle <= end_angle):
            dists.append(dist)
    if len(dists) > 0: return np.min(dists)
    return 9999


# [주차 확인용] 특정 각도 근처 (좁은 범위)
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
        ser = serial.Serial(PORT, 115200, timeout=0.1)
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
    side_detect_time = 0

    pause_start_time = 0
    pause_duration = 0
    next_state_after_pause = 0
    pause_msg = ""

    # 상대 거리 계산용 변수
    reference_dist = 0  # 기준 거리 저장
    stable_count = 0  # 노이즈 방지 카운터

    while True:
        try:
            for scan in lidar.iter_scans():
                read_sensors()

                # 1. 탐색용 센서 (30~110도) -> Search 단계에서 사용
                lidar_radar = get_lidar_min_dist(scan, 30, 110)

                # 2. 정밀 확인용 센서 (90도/270도) -> Parking/Exit 단계에서 사용
                dist_90 = get_dist_at_angle(scan, 90, 10)
                dist_270 = get_dist_at_angle(scan, 270, 10)

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
                # [1] 탐색 (Relative Jump Algorithm)
                # ---------------------------------------------------
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    print(f"Step:{search_step} | Radar:{int(lidar_radar)} | Ref:{int(reference_dist)}")

                    # Step 1: 1번 차 찾기 (여긴 절대값 사용)
                    if search_step == STEP_FIND_CAR1:
                        msg = "FIND CAR 1 (Looking for < 2.0m)"
                        cmd_servo = SERVO_CENTER
                        if 100 < lidar_radar < 2000:  # 2m 이내 감지
                            stable_count += 1
                            if stable_count > 2:
                                print(f"🚗 1번 차 발견! (거리: {int(lidar_radar)})")
                                reference_dist = lidar_radar  # 기준점 저장
                                search_step = STEP_PASS_CAR1
                                stable_count = 0
                        else:
                            stable_count = 0

                    # Step 2: 1번 차 통과 (거리가 멀어지면 빈 공간)
                    elif search_step == STEP_PASS_CAR1:
                        msg = f"PASS CAR 1 (Wait Jump Up > {JUMP_THRESHOLD})"
                        cmd_servo = SERVO_SLIGHT_RIGHT  # 벽타기

                        diff = lidar_radar - reference_dist
                        # 거리가 60cm 이상 멀어지면(Jump Up)
                        if diff > JUMP_THRESHOLD and lidar_radar < MAX_VALID_DIST:
                            stable_count += 1
                            if stable_count > 2:
                                print(f"👀 빈 공간 진입 (Diff: {int(diff)})")
                                reference_dist = lidar_radar  # 빈 공간 거리 저장
                                search_step = STEP_FIND_GAP
                                stable_count = 0
                        else:
                            # 1번 차를 지나는 중에는 기준값을 살짝씩 업데이트 (필터링)
                            if lidar_radar < reference_dist + 200:
                                reference_dist = (reference_dist * 0.8) + (lidar_radar * 0.2)
                            stable_count = 0

                    # Step 3: 2번 차 찾기 (거리가 가까워지면 2번 차)
                    elif search_step == STEP_FIND_GAP:
                        msg = f"FIND CAR 2 (Wait Jump Down > {JUMP_THRESHOLD})"
                        cmd_servo = SERVO_SLIGHT_RIGHT  # 벽타기

                        diff = reference_dist - lidar_radar
                        # 거리가 60cm 이상 가까워지면(Jump Down)
                        if diff > JUMP_THRESHOLD:
                            print(f"🛑 2번 차 감지! 정지 (Diff: {int(diff)})")
                            ser.write(b"D,-150\n");
                            time.sleep(0.1)  # 급제동
                            ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state_after_pause = STATE_SETUP_LEFT
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "Ready for Setup"
                        else:
                            # 빈 공간을 지나는 중에도 기준값 업데이트
                            if lidar_radar > reference_dist - 200:
                                reference_dist = (reference_dist * 0.8) + (lidar_radar * 0.2)

                # ---------------------------------------------------
                # [2] 공간 확보 (좌측 전진)
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
                            print("🛑 공간 확보 완료.")
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE
                            next_state_after_pause = STATE_REVERSE_TURN
                            pause_duration = 1.0
                            pause_start_time = curr_time
                            pause_msg = "Ready for Reverse"

                # ---------------------------------------------------
                # [3] 꺾어서 후진 (우측 후진)
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
                            print("🛑 1차 후진 완료.")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            side_detect_time = 0

                # ---------------------------------------------------
                # [4] 마무리 후진 (90도/270도 감지 & 지연 정지)
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
                    # 후방 센서 체크
                    if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                        msg = "DETECT: Rear Wall!"
                        is_detected = True
                    # ★ 측면 체크 (90도 또는 270도)
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
                # [5] 출차 판단 (Intelligent Exit)
                # ---------------------------------------------------
                elif state == STATE_WAIT_AFTER_PARK:
                    cmd_speed = 0
                    msg = "Parking Done. Judging..."
                    if curr_time - state_timer > 4.0: # 4초 경과 후 직진
                        # ★ 90도와 270도 비교
                        print(f"📏 출차 판단: L={int(dist_270)} R={int(dist_90)}")

                        # 오른쪽이 60cm보다 좁거나, 왼쪽보다 더 좁으면 -> 보정 후 출차
                        if dist_90 < 600 or dist_90 < dist_270:
                            print("⚠️ 오른쪽 좁음 -> 보정(ADJUST)")
                            state = STATE_EXIT_ADJUST
                        else:
                            print("✅ 공간 충분 -> 바로 회전(TURN)")
                            state = STATE_EXIT_TURN
                        state_timer = curr_time

                # ---------------------------------------------------
                # [6] 출차 실행 & 무한 직진
                # ---------------------------------------------------
                elif state == STATE_EXIT_ADJUST:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        # 1초간 직진하여 공간 확보
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
                        # 7초간 우회전하여 탈출
                        if (curr_time - state_timer) - STEER_WAIT_TIME > TIME_EXIT_TURN:
                            print("🚀 출차 완료 -> 무한 직진 시작")
                            state = STATE_EXIT_STRAIGHT
                            state_timer = curr_time

                elif state == STATE_EXIT_STRAIGHT:
                    cmd_servo = SERVO_CENTER
                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0;
                        msg = "Exit: Align Center..."
                    else:
                        cmd_speed = SPEED_SEARCH
                        msg = "RUNNING FOREVER..."
                        # 종료 조건 없음 (무한 직진)

                elif state == STATE_FINISH:
                    cmd_speed = 0;
                    msg = "ALL MISSION COMPLETE"
                    ser.write(b"D,0\n")

                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                debug_img = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | Msg: {msg}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(debug_img, f"90deg:{int(dist_90)} 270deg:{int(dist_270)}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
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