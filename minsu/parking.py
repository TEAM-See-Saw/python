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
SPEED_SETUP = 80  # 앞으로 공간 확보하러 나갈 때 속도
SPEED_PARK = 75  # 천천히 후진
SPEED_STOP = 0

# 서보 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440  # 우측 최대
SERVO_LEFT_MAX = 680  # 좌측 최대

# 조향 대기 시간 (정지 상태에서 핸들 돌리는 시간)
STEER_WAIT_TIME = 0.8

# ★ [추가] 공간 확보(왼쪽 앞으로) 나가는 시간
TIME_SETUP_MOVE = 1.8

# 거리 기준 (mm)
CAR_EXIST_DIST = 800  # 이 거리 안쪽이면 '차'가 있다고 판단
EMPTY_SPACE_DIST = 1200  # 이 거리보다 멀면 '빈 공간'이라고 판단
REAR_LIMIT = 200

# 아두이노 센서 인덱스
IDX_LT = 2
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0  # 주차 공간 탐색
STATE_SETUP_LEFT = 1  # ★ 왼쪽 앞으로 나가서 각도 벌리기
STATE_REVERSE_TURN = 2  # 우측으로 꺾어 후진
STATE_REVERSE_STRAIGHT = 3  # 풀고 직진 후진
STATE_DONE = 4

# 탐색 단계 (차량 구분 로직 강화)
STEP_FIND_CAR1 = 0  # 1번 차 찾는 중
STEP_PASS_CAR1 = 1  # 1번 차 지나가는 중 (아직 옆에 차 있음)
STEP_FIND_GAP = 2  # 1번 차 끝남 -> 빈 공간 확인 중
STEP_FIND_CAR2 = 3  # 2번 차(주차 기준점) 찾는 중

# ==========================================
# [2] 데이터 처리 함수
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass


def get_lidar_dist_at_angle(scan, target_angle, angle_range=2):
    dists = []
    min_a = target_angle - angle_range
    max_a = target_angle + angle_range
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
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
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = 0

    # 노이즈 방지용 카운터
    valid_gap_count = 0
    valid_car2_count = 0

    print("🚀 주차 시스템 시작")

    try:
        for scan in lidar.iter_scans():
            read_sensors()
            # 우측 90도 거리 측정
            lidar_side = get_lidar_dist_at_angle(scan, 90, 5)

            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = ""
            sub_msg = ""

            # ---------------------------------------------------
            # [1] 탐색 (직진하며 공간 분석)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                # 1-1. 첫 번째 차량 감지
                if search_step == STEP_FIND_CAR1:
                    msg = "SEARCHING CAR 1..."
                    if lidar_side < CAR_EXIST_DIST:
                        print("🚗 [1단계] 1번 차량 감지됨 -> 지나가는 중")
                        search_step = STEP_PASS_CAR1

                # 1-2. 1번 차가 끝날 때까지 대기
                elif search_step == STEP_PASS_CAR1:
                    msg = "PASSING CAR 1..."
                    # 거리가 멀어지면(빈 공간이면)
                    if lidar_side > EMPTY_SPACE_DIST:
                        valid_gap_count += 1
                        # LIDAR가 5번 연속으로 빈 공간을 찍어야 진짜 빈 공간으로 인정 (노이즈 필터)
                        if valid_gap_count > 5:
                            print("👀 [2단계] 1번 차 지남 -> 빈 공간(GAP) 확인")
                            search_step = STEP_FIND_GAP
                    else:
                        valid_gap_count = 0  # 다시 차가 보이면 리셋

                # 1-3. 빈 공간 주행 중 -> 2번 차 찾기
                elif search_step == STEP_FIND_GAP:
                    msg = "GAP FOUND. FINDING CAR 2..."
                    # 다시 거리가 가까워지면 (2번 차 시작)
                    if lidar_side < CAR_EXIST_DIST:
                        valid_car2_count += 1
                        if valid_car2_count > 3:  # 3번 연속 감지 시 인정
                            print("🛑 [3단계] 2번 차 감지! -> 정지")

                            # 급정거
                            ser.write(b"D,-100\n");
                            time.sleep(0.1)
                            for _ in range(5): ser.write(b"D,0\n"); time.sleep(0.05)
                            time.sleep(1.0)  # 기어 변경 및 판단 대기

                            # ★ 다음 상태: 바로 후진(X) -> 왼쪽 앞으로 나가기(O)
                            state = STATE_SETUP_LEFT
                            state_timer = time.time()

            # ---------------------------------------------------
            # [2] 공간 확보 기동 (왼쪽 앞으로 전진) ★ NEW
            # ---------------------------------------------------
            elif state == STATE_SETUP_LEFT:
                # 1. 정지 상태에서 핸들 왼쪽 끝으로
                cmd_servo = SERVO_LEFT_MAX

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ SETUP: STEERING LEFT..."
                else:
                    # 2. 앞으로 나감 (공간 확보)
                    cmd_speed = SPEED_SETUP
                    msg = "↪ SETUP: MOVING FORWARD-LEFT..."

                # 설정한 시간만큼만 이동 후 정지
                if curr_time - state_timer > (STEER_WAIT_TIME + TIME_SETUP_MOVE):
                    print("🛑 공간 확보 완료. 후진 준비")
                    ser.write(b"D,0\n");
                    time.sleep(1.0)  # 확실히 정지

                    state = STATE_REVERSE_TURN
                    state_timer = time.time()

            # ---------------------------------------------------
            # [3] 꺾어서 후진 (우측 최대)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_TURN:
                cmd_servo = SERVO_RIGHT_MAX  # 이제 오른쪽으로 꺾어서 들어감

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ PARKING: STEERING RIGHT..."
                else:
                    cmd_speed = -SPEED_PARK
                    msg = "PARKING: REVERSING turn..."

                # 안전장치 (후방 감지)
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE

                # 적절한 시간 후 펴기 단계로 (시간 튜닝 필요, 예: 3.0초)
                if curr_time - state_timer > (STEER_WAIT_TIME + 3.0):
                    state = STATE_REVERSE_STRAIGHT
                    state_timer = time.time()

            # ---------------------------------------------------
            # [4] 풀고 직진 후진
            # ---------------------------------------------------
            elif state == STATE_REVERSE_STRAIGHT:
                cmd_servo = SERVO_CENTER

                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ PARKING: ALIGNING CENTER..."
                else:
                    cmd_speed = -SPEED_PARK
                    msg = "PARKING: REVERSING straight..."

                # 완료 조건 (후방 센서)
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("✅ 주차 완료 (후방 감지)")
                    state = STATE_DONE

                # 혹은 시간으로 종료 (후방 센서가 없을 경우 대비)
                if curr_time - state_timer > (STEER_WAIT_TIME + 2.0):
                    print("✅ 주차 완료 (시간 종료)")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [5] 종료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0
                msg = "PARKING COMPLETE"
                ser.write(b"D,0\n")
                # 루프 종료하지 않고 대기 (센서값 확인용)

            # 명령 전송
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            # 모니터링
            debug_img = np.zeros((300, 600, 3), dtype=np.uint8)

            # 상태 표시
            cv2.putText(debug_img, f"State: {state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(debug_img, msg, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 센서값 표시
            cv2.putText(debug_img, f"LIDAR Side: {int(lidar_side)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)
            cv2.putText(debug_img, f"Step: {search_step} (GapCnt: {valid_gap_count})", (10, 160),
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