import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# 속도 설정
SPEED_SEARCH = 80
SPEED_SETUP = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440  # 값 감소 -> 우측
SERVO_LEFT_MAX = 680  # 값 증가 -> 좌측
STEER_WAIT_TIME = 0.8

# ★ [자세 제어 튜닝] 평행 주행 설정
# 라이다 90도 기준, 앞쪽(80도)과 뒤쪽(100도)을 비교
ALIGN_KP = 0.4  # 보정 강도 (크면 확확 꺾고, 작으면 부드럽게)
MAX_ALIGN_ANGLE = 40  # 최대 보정 범위 (중앙 +-40도 까지만 꺾음)

# [시간 튜닝] 7.7Hz 라이다
CNT_GAP_CONFIRM = 4
CNT_CAR2_DETECT = 2
TIME_SETUP_MOVE = 2.5

# 거리 기준 (mm)
CAR_EXIST_DIST = 900  # 옆 차 인식 거리 (좀 더 넉넉하게 900으로 늘림)
EMPTY_SPACE_DIST = 1200
REAR_LIMIT = 200

# 아두이노 센서 인덱스
IDX_LT = 2;
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0;
STATE_SETUP_LEFT = 1;
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3;
STATE_DONE = 4

# 탐색 단계
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


def get_lidar_dist_at_angle(scan, target_angle, angle_range=2):
    dists = []
    min_a = target_angle - angle_range
    max_a = target_angle + angle_range
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a): dists.append(dist)
    if len(dists) > 0: return np.mean(dists)
    return 9999


# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser, lidar
    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 (평행 보정 모드 ON)")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 오류: {e}"); return

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0

    valid_gap_count = 0
    valid_car2_count = 0

    try:
        for scan in lidar.iter_scans():
            read_sensors()

            # --- 평행 보정을 위한 거리 측정 ---
            # 80도(살짝 앞), 100도(살짝 뒤) 측정
            lidar_front_side = get_lidar_dist_at_angle(scan, 80, 3)
            lidar_rear_side = get_lidar_dist_at_angle(scan, 100, 3)
            # 평균 거리 (탐색용)
            lidar_side = (lidar_front_side + lidar_rear_side) / 2

            dist_LT = sonar_data[IDX_LT];
            dist_RT = sonar_data[IDX_RT]

            cmd_speed = 0;
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = "";
            sub_msg = ""

            # ---------------------------------------------------
            # [1] 탐색 (★ 평행 보정 적용)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH

                # ★ [핵심] 평행 주행 로직 (Wall Following)
                # 옆에 차가 확실히 있을 때만 보정 (Gap 구간에서는 직진 유지)
                if lidar_front_side < CAR_EXIST_DIST and lidar_rear_side < CAR_EXIST_DIST:
                    # 오차 계산 (앞 - 뒤)
                    # 앞이 가까우면(작으면) error는 음수 -> 핸들 왼쪽(+)으로 꺾어야 함
                    # 뒤가 가까우면(작으면) error는 양수 -> 핸들 오른쪽(-)으로 꺾어야 함
                    error = lidar_front_side - lidar_rear_side

                    # 서보 값 계산: Center - (Error * Gain)
                    # 예: error = -100 (앞이 벽쪽) -> correction = +40 -> Center + 40 (Left)
                    correction = int(-error * ALIGN_KP)

                    # 과도한 꺾임 방지 (Limit)
                    correction = max(-MAX_ALIGN_ANGLE, min(MAX_ALIGN_ANGLE, correction))

                    cmd_servo = SERVO_CENTER + correction
                    sub_msg = f"Align: {correction} (F:{int(lidar_front_side)} R:{int(lidar_rear_side)})"
                else:
                    # 빈 공간이나 차가 없으면 그냥 직진
                    cmd_servo = SERVO_CENTER
                    sub_msg = "Align: STRAIGHT (No Wall)"

                # --- 단계별 탐색 로직 ---
                if search_step == STEP_FIND_CAR1:
                    msg = "STEP 1: FINDING CAR 1"
                    if lidar_side < CAR_EXIST_DIST:
                        print("🚗 Car 1 감지! (자세 제어 시작)")
                        search_step = STEP_PASS_CAR1

                elif search_step == STEP_PASS_CAR1:
                    msg = "STEP 2: PASSING CAR 1"
                    if lidar_side > EMPTY_SPACE_DIST:
                        valid_gap_count += 1
                        if valid_gap_count >= CNT_GAP_CONFIRM:
                            print(f"👀 빈 공간 확인 ({valid_gap_count})")
                            search_step = STEP_FIND_GAP
                    else:
                        valid_gap_count = 0

                elif search_step == STEP_FIND_GAP:
                    msg = "STEP 3: FINDING CAR 2"
                    if lidar_side < CAR_EXIST_DIST:
                        valid_car2_count += 1
                        if valid_car2_count >= CNT_CAR2_DETECT:
                            print(f"🛑 2번 차 감지! 정지 ({valid_car2_count})")
                            ser.write(b"D,-150\n");
                            time.sleep(0.1)
                            for _ in range(5): ser.write(b"D,0\n"); time.sleep(0.05)
                            time.sleep(0.5)
                            state = STATE_SETUP_LEFT
                            state_timer = time.time()
                    else:
                        valid_car2_count = 0

            # ---------------------------------------------------
            # [2] 공간 확보 (왼쪽 앞으로)
            # ---------------------------------------------------
            elif state == STATE_SETUP_LEFT:
                cmd_servo = SERVO_LEFT_MAX
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 좌측 정렬 중..."
                else:
                    cmd_speed = SPEED_SETUP;
                    msg = "↪ 공간 확보 (전진)"

                if curr_time - state_timer > (STEER_WAIT_TIME + TIME_SETUP_MOVE):
                    print("🛑 공간 확보 끝.")
                    ser.write(b"D,0\n");
                    time.sleep(1.0)
                    state = STATE_REVERSE_TURN
                    state_timer = time.time()

            # ---------------------------------------------------
            # [3] 꺾어서 후진 (우측)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_TURN:
                cmd_servo = SERVO_RIGHT_MAX
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 우측 정렬 중..."
                else:
                    cmd_speed = -SPEED_PARK;
                    msg = "PARKING: 꺾어서 진입"

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE

                if curr_time - state_timer > (STEER_WAIT_TIME + 3.5):
                    state = STATE_REVERSE_STRAIGHT
                    state_timer = time.time()

            # ---------------------------------------------------
            # [4] 풀고 직진 후진
            # ---------------------------------------------------
            elif state == STATE_REVERSE_STRAIGHT:
                cmd_servo = SERVO_CENTER
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 중앙 정렬 중..."
                else:
                    cmd_speed = -SPEED_PARK;
                    msg = "PARKING: 직진 후진"

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("✅ 후방 감지 -> 주차 완료")
                    state = STATE_DONE

                if curr_time - state_timer > (STEER_WAIT_TIME + 2.5):
                    print("✅ 시간 종료 -> 주차 완료")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [5] 종료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0;
                msg = "PARKING COMPLETE"
                ser.write(b"D,0\n")

            # 명령 및 화면 출력
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            debug_img = np.zeros((300, 600, 3), dtype=np.uint8)
            cv2.putText(debug_img, f"State: {state} | {msg}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            cv2.putText(debug_img, sub_msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            cv2.imshow("Parking Monitor", debug_img)
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        print("종료")
    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()