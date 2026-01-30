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
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
STEER_WAIT_TIME = 0.8  # Stop & Steer 대기 시간

# ★ [시간 튜닝] 7.7Hz 라이다 특성 반영
# 스캔 1회당 0.13초 소요
# 카운트 2회 = 0.26초 지연 (약 13cm 이동)
CNT_GAP_CONFIRM = 4  # 빈 공간 인정 카운트 (4회 = 0.52초)
CNT_CAR2_DETECT = 2  # 2번 차 감지 카운트 (2회 = 0.26초 -> 빠른 반응)

# ★ [동작 튜닝] 공간 확보 시간
# 반응 속도가 빨라져서 더 앞에서 멈추므로, 나가는 시간을 살짝 더 줘야 함
TIME_SETUP_MOVE = 2.5

# 거리 기준 (mm)
CAR_EXIST_DIST = 800
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
        print("✅ 시스템 연결 (LiDAR 7.7Hz 모드)")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 오류: {e}"); return

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0

    # 노이즈 필터 카운터
    valid_gap_count = 0
    valid_car2_count = 0

    try:
        # iter_scans()는 하드웨어 속도(7.7Hz)에 맞춰 루프가 돕니다.
        for scan in lidar.iter_scans():
            read_sensors()
            lidar_side = get_lidar_dist_at_angle(scan, 90, 5)  # 우측 90도
            dist_LT = sonar_data[IDX_LT];
            dist_RT = sonar_data[IDX_RT]

            cmd_speed = 0;
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = "";
            sub_msg = ""

            # ---------------------------------------------------
            # [1] 탐색
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH

                # 1. 첫 번째 차량 감지
                if search_step == STEP_FIND_CAR1:
                    msg = "STEP 1: FINDING CAR 1"
                    if lidar_side < CAR_EXIST_DIST:
                        print("🚗 Car 1 감지! -> 지나가는 중")
                        search_step = STEP_PASS_CAR1

                # 2. 1번 차 지나가기 (빈 공간 나올 때까지)
                elif search_step == STEP_PASS_CAR1:
                    msg = "STEP 2: PASSING CAR 1"
                    if lidar_side > EMPTY_SPACE_DIST:
                        valid_gap_count += 1
                        # 4회 연속(약 0.5초) 비어있어야 인정
                        if valid_gap_count >= CNT_GAP_CONFIRM:
                            print(f"👀 빈 공간 확인 (Count {valid_gap_count})")
                            search_step = STEP_FIND_GAP
                    else:
                        valid_gap_count = 0

                # 3. 빈 공간 주행 -> 2번 차 찾기
                elif search_step == STEP_FIND_GAP:
                    msg = "STEP 3: FINDING CAR 2 (WALL)"
                    if lidar_side < CAR_EXIST_DIST:
                        valid_car2_count += 1
                        # 2회 연속(약 0.26초) 감지 시 즉시 정지 -> 반응 속도 UP
                        if valid_car2_count >= CNT_CAR2_DETECT:
                            print(f"🛑 2번 차 감지! 정지 (Count {valid_car2_count})")

                            ser.write(b"D,-150\n");
                            time.sleep(0.1)  # 급정거 강화
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
                    msg = "↪ 공간 확보 (전진-좌회전)"

                # 2.0초 동안 이동
                if curr_time - state_timer > (STEER_WAIT_TIME + TIME_SETUP_MOVE):
                    print("🛑 공간 확보 끝. 후진 준비")
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

                # 약 3.5초 후 정렬 단계로
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
            cv2.putText(debug_img, f"Lidar Side: {int(lidar_side)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)
            cv2.putText(debug_img, f"Gap Cnt: {valid_gap_count}/{CNT_GAP_CONFIRM}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)
            cv2.putText(debug_img, f"Car2 Cnt: {valid_car2_count}/{CNT_CAR2_DETECT}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

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