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
SPEED_PARK = 75  # 천천히 후진
SPEED_STOP = 0

# 서보 설정 (DC 조향 모터 특성 반영)
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440  # 우측 최대 (가장 많이 꺾임)
SERVO_LEFT_MAX = 680

# 조향 대기 시간 (핸들 돌릴 때 모터 멈춰있는 시간)
STEER_WAIT_TIME = 0.8

# 거리 기준
CAR_DIST_MIN = 150
CAR_DIST_MAX = 1200
EMPTY_DIST_MIN = 1200
SIDE_CAR_DIST = 800
REAR_LIMIT = 200
PASS_CAR_WIDTH_DELAY = 1.5

# 아두이노 센서 인덱스
IDX_LT = 2
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0
STATE_REVERSE_TURN = 1
STATE_REVERSE_STRAIGHT = 2
STATE_DONE = 3

# 탐색 단계
STEP_INIT_EMPTY = 0
STEP_PASSING_CAR1 = 1
STEP_FIND_SPOT = 2
STEP_FIND_CAR2 = 3

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
        print("✅ 하드웨어 연결 성공")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_INIT_EMPTY
    state_timer = 0
    car1_detect_time = 0

    print("🚀 주차 시스템 시작: DC 조향 전압 집중 모드")

    try:
        for scan in lidar.iter_scans():
            read_sensors()
            lidar_search = get_lidar_dist_at_angle(scan, 90, 5)
            lidar_align = get_lidar_dist_at_angle(scan, 90, 2)
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = ""

            # ---------------------------------------------------
            # [1] 탐색 (직진)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER  # 직진 유지
                msg = f"STEP: {search_step} | Dist: {int(lidar_search)}mm"

                if search_step == STEP_INIT_EMPTY:
                    if CAR_DIST_MIN <= lidar_search <= CAR_DIST_MAX:
                        print(f"🚗 [1/4] 차량 발견!")
                        car1_detect_time = curr_time
                        search_step = STEP_PASSING_CAR1

                elif search_step == STEP_PASSING_CAR1:
                    msg = "PASSING CAR 1..."
                    if curr_time - car1_detect_time > PASS_CAR_WIDTH_DELAY:
                        print("👀 [2/4] 빈 공간 탐색 시작")
                        search_step = STEP_FIND_SPOT

                elif search_step == STEP_FIND_SPOT:
                    if lidar_search > EMPTY_DIST_MIN:
                        msg = "SPOT FOUND! Looking for Car 2..."

                    if CAR_DIST_MIN <= lidar_search <= CAR_DIST_MAX:
                        print(f"🛑 [3/4] 두 번째 차 감지 -> 정지")

                        # 브레이크
                        ser.write(b"D,-100\n");
                        time.sleep(0.1)
                        for _ in range(5): ser.write(b"D,0\n"); time.sleep(0.05)

                        print("🛑 정차 및 기어 변경 대기...")
                        time.sleep(1.0)

                        state = STATE_REVERSE_TURN
                        state_timer = time.time()

            # ---------------------------------------------------
            # [2] 꺾어서 후진 (Stop & Steer)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_TURN:
                # 1. 목표: 우측 최대 꺾기
                cmd_servo = SERVO_RIGHT_MAX

                # 2. [전압 집중] 0.8초간 정지 후 핸들링
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ TURNING WHEELS (RIGHT)..."
                else:
                    # 0.8초 후 출발
                    cmd_speed = -SPEED_PARK
                    msg = "REVERSING (TURN)..."

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE

                # 3.5초 후 다음 단계 (시간 넉넉히)
                if curr_time - state_timer > 3.5:
                    # ★ 단계 넘어갈 때 타이머 리셋 필수
                    state = STATE_REVERSE_STRAIGHT
                    state_timer = time.time()

                    # ---------------------------------------------------
            # [3] 풀고 직진 후진 (Stop & Steer) - ★ 여기도 적용됨
            # ---------------------------------------------------
            elif state == STATE_REVERSE_STRAIGHT:
                # 1. 목표: 중앙 정렬 (핸들 풀기)
                cmd_servo = SERVO_CENTER

                # 2. [전압 집중] 0.8초간 정지 후 핸들 복귀
                # 우측으로 꺾여있던 핸들을 중앙으로 돌릴 때 전기를 많이 씀 -> 멈춰서 돌림
                if curr_time - state_timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ REALIGNING WHEELS (CENTER)..."
                else:
                    # 핸들 다 풀렸으면 후진
                    cmd_speed = -SPEED_PARK
                    msg = f"ALIGNING... Dist: {int(lidar_align)}"

                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("🛑 후방 벽 감지 -> 완료")
                    state = STATE_DONE

                elif lidar_align < SIDE_CAR_DIST:
                    print(f"✅ 정렬 완료 (옆 차: {int(lidar_align)}mm)")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [4] 주차 완료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0
                msg = "PARKING COMPLETED"
                ser.write(b"D,0\n")
                break

            # 명령 전송
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            # 디버그 화면
            debug_img = np.zeros((200, 600, 3), dtype=np.uint8)
            cv2.putText(debug_img, f"State: {state} Step: {search_step}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.putText(debug_img, msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Parking Monitor", debug_img)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("종료")
    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()