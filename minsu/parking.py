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
STEER_WAIT_TIME = 0.8  # 정지 후 핸들 돌리는 시간

# ★ [자세 제어 튜닝] 평행 주행 설정
ALIGN_KP = 0.4
MAX_ALIGN_ANGLE = 40

# [시간 튜닝] 7.7Hz 라이다
CNT_GAP_CONFIRM = 4
CNT_CAR2_DETECT = 2

# ★ [동작 시간 튜닝] (요청하신 부분)
# 1. 공간 확보 (왼쪽 앞으로 나가는 시간)
TIME_SETUP_MOVE = 4.0  # 2.5 -> 4.0초로 연장 (더 넓게 벌림)

# 2. 후진 진입 (오른쪽 뒤로 꺾는 시간)
TIME_REVERSE_TURN = 5.5  # 3.5 -> 5.5초로 연장 (더 깊게 들어감)

# 거리 기준 (mm)
CAR_EXIST_DIST = 900
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
        print("✅ 시스템 연결 성공")
        time.sleep(1)  # 포트 안정화 대기
    except Exception as e:
        print(f"❌ 오류: {e}"); return

    # ---------------------------------------------------
    # ★ [추가된 로직] 시작 전 바퀴 정렬 (1.5초간)
    # ---------------------------------------------------
    print("🛠️ 초기화: 바퀴 중앙 정렬 중...")
    ser.write(f"S,{SERVO_CENTER}\n".encode())  # 핸들 중앙
    ser.write(b"D,0\n")  # 모터 정지
    time.sleep(1.5)  # 정렬될 때까지 대기
    print("🚀 정렬 완료! 주행 시작")
    # ---------------------------------------------------

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0

    valid_gap_count = 0
    valid_car2_count = 0

    try:
        for scan in lidar.iter_scans():
            read_sensors()

            # 거리 측정
            lidar_front_side = get_lidar_dist_at_angle(scan, 80, 3)
            lidar_rear_side = get_lidar_dist_at_angle(scan, 100, 3)
            lidar_side = (lidar_front_side + lidar_rear_side) / 2

            dist_LT = sonar_data[IDX_LT];
            dist_RT = sonar_data[IDX_RT]

            cmd_speed = 0;
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = "";
            sub_msg = ""

            # ---------------------------------------------------
            # [1] 탐색 (평행 보정)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH

                # 평행 보정 (차가 있을 때만)
                if lidar_front_side < CAR_EXIST_DIST and lidar_rear_side < CAR_EXIST_DIST:
                    error = lidar_front_side - lidar_rear_side
                    correction = int(-error * ALIGN_KP)
                    correction = max(-MAX_ALIGN_ANGLE, min(MAX_ALIGN_ANGLE, correction))
                    cmd_servo = SERVO_CENTER + correction
                    sub_msg = f"Align: {correction}"
                else:
                    cmd_servo = SERVO_CENTER
                    sub_msg = "Align: STRAIGHT"

                # 단계별 탐색
                if search_step == STEP_FIND_CAR1:
                    msg = "STEP 1: 1번 차 찾는 중..."
                    if lidar_side < CAR_EXIST_DIST:
                        print("🚗 1번 차 발견!")
                        search_step = STEP_PASS_CAR1

                elif search_step == STEP_PASS_CAR1:
                    msg = "STEP 2: 1번 차 지나는 중..."
                    if lidar_side > EMPTY_SPACE_DIST:
                        valid_gap_count += 1
                        if valid_gap_count >= CNT_GAP_CONFIRM:
                            print(f"👀 빈 공간 확인 ({valid_gap_count})")
                            search_step = STEP_FIND_GAP
                    else:
                        valid_gap_count = 0

                elif search_step == STEP_FIND_GAP:
                    msg = "STEP 3: 2번 차 찾는 중..."
                    if lidar_side < CAR_EXIST_DIST:
                        valid_car2_count += 1
                        if valid_car2_count >= CNT_CAR2_DETECT:
                            print(f"🛑 2번 차 발견! 정지 ({valid_car2_count})")
                            ser.write(b"D,-150\n");
                            time.sleep(0.1)
                            for _ in range(5): ser.write(b"D,0\n"); time.sleep(0.05)
                            time.sleep(0.5)

                            state = STATE_SETUP_LEFT
                            state_timer = time.time()
                    else:
                        valid_car2_count = 0

            # ---------------------------------------------------
            # [2] 공간 확보 (왼쪽 앞으로) - 시간 늘림
            # ---------------------------------------------------
            elif state == STATE_SETUP_LEFT:
                cmd_servo = SERVO_LEFT_MAX

                # Stop & Steer
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 좌측 정렬..."
                else:
                    cmd_speed = SPEED_SETUP;
                    msg = "↪ 공간 확보 (전진)"

                # ★ 4.0초 동안 이동
                if curr_time - state_timer > (STEER_WAIT_TIME + TIME_SETUP_MOVE):
                    print("🛑 공간 확보 끝. 후진 준비")
                    ser.write(b"D,0\n");
                    time.sleep(1.0)
                    state = STATE_REVERSE_TURN
                    state_timer = time.time()

            # ---------------------------------------------------
            # [3] 꺾어서 후진 (우측) - 시간 늘림
            # ---------------------------------------------------
            elif state == STATE_REVERSE_TURN:
                cmd_servo = SERVO_RIGHT_MAX

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 우측 정렬..."
                else:
                    cmd_speed = -SPEED_PARK;
                    msg = "PARKING: 꺾어서 진입"

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE

                # ★ 5.5초 동안 후진
                if curr_time - state_timer > (STEER_WAIT_TIME + TIME_REVERSE_TURN):
                    state = STATE_REVERSE_STRAIGHT
                    state_timer = time.time()

            # ---------------------------------------------------
            # [4] 풀고 직진 후진
            # ---------------------------------------------------
            elif state == STATE_REVERSE_STRAIGHT:
                cmd_servo = SERVO_CENTER

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0;
                    msg = "⚡ 핸들 중앙 정렬..."
                else:
                    cmd_speed = -SPEED_PARK;
                    msg = "PARKING: 직진 후진"

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("✅ 후방 감지 -> 완료")
                    state = STATE_DONE

                if curr_time - state_timer > (STEER_WAIT_TIME + 2.5):
                    print("✅ 시간 종료 -> 완료")
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
            cv2.putText(debug_img, f"State: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(debug_img, msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, sub_msg, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

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