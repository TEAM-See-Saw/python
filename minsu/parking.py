import serial
from rplidar import RPLidar
import time
import numpy as np

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'  # 아두이노
LIDAR_PORT = 'COM3'  # 라이다

# 속도 설정
SPEED_SEARCH = 80
SPEED_PARK = 90
SPEED_STOP = 0

# 서보 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480  # 우측 주차 진입용
SERVO_LEFT_MAX = 680

# 거리 기준 (mm)
PARKING_DEPTH = 1000  # 빈 공간 판단 (1m 이상)
REAR_LIMIT = 200  # 후방 벽 비상 정지 (20cm)

# ★ [핵심] 앞선 맞추기 기준
# 옆 차와의 간격 (이 거리 안으로 들어오면 "옆에 차가 있구나" 판단)
SIDE_CAR_DIST = 800  # 80cm 이내에 물체가 있으면 감지

# 아두이노 센서 인덱스 (LF, LM, LT, RF, RM, RT)
IDX_LT = 2  # 왼쪽 뒤
IDX_RT = 5  # 오른쪽 뒤

# 상태 정의
STATE_SEARCH = 0
STATE_POSITION = 1
STATE_REVERSE_TURN = 2
STATE_REVERSE_STRAIGHT = 3
STATE_DONE = 4

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
    """
    특정 각도(target_angle) 주변의 물체 거리를 반환
    예: 90도라고 하면 88~92도 사이의 평균 거리를 구함
    """
    dists = []
    min_a = target_angle - angle_range
    max_a = target_angle + angle_range

    for (_, angle, dist) in scan:
        if dist > 0:
            if min_a <= angle <= max_a:
                dists.append(dist)

    if len(dists) > 0:
        return np.mean(dists)
    return 9999  # 감지 안됨


# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser, lidar

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 하드웨어 연결 성공")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    state_timer = 0
    spot_found_time = 0

    print("🚀 주차 시스템 시작: 라이다 90도 정렬 모드")

    try:
        for scan in lidar.iter_scans():
            # 1. 데이터 읽기
            read_sensors()

            # 라이다 데이터 추출
            # 우측 주차 공간 탐색용 (넓은 범위 80~100도)
            lidar_search_right = get_lidar_dist_at_angle(scan, 90, 10)

            # ★ 정렬용: 정확히 90도(우측) 핀포인트
            lidar_align_right = get_lidar_dist_at_angle(scan, 90, 2)

            # 후방 초음파
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]

            # 명령 변수
            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = ""

            # ---------------------------------------------------
            # [1] 빈 공간 탐색
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                msg = f"SEARCHING... Dist: {int(lidar_search_right)}mm"

                if lidar_search_right > PARKING_DEPTH:
                    if spot_found_time == 0:
                        spot_found_time = curr_time
                    elif curr_time - spot_found_time > 1.5:
                        print("✅ 빈 주차 공간 발견!")
                        state = STATE_POSITION
                        state_timer = curr_time
                else:
                    spot_found_time = 0

            # ---------------------------------------------------
            # [2] 위치 잡기 (전진)
            # ---------------------------------------------------
            elif state == STATE_POSITION:
                cmd_speed = SPEED_SEARCH
                msg = "POSITIONING..."
                if curr_time - state_timer > 1.2:
                    ser.write(b"D,0\n");
                    time.sleep(1)
                    state = STATE_REVERSE_TURN
                    state_timer = curr_time

            # ---------------------------------------------------
            # [3] 꺾어서 후진
            # ---------------------------------------------------
            elif state == STATE_REVERSE_TURN:
                cmd_speed = -SPEED_PARK
                cmd_servo = SERVO_RIGHT_MAX
                msg = "TURNING BACK..."

                # 비상 정지
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE

                if curr_time - state_timer > 2.5:
                    state = STATE_REVERSE_STRAIGHT

            # ---------------------------------------------------
            # [4] 직진 후진 & ★ 라이다 90도 정렬
            # ---------------------------------------------------
            elif state == STATE_REVERSE_STRAIGHT:
                cmd_speed = -SPEED_PARK
                cmd_servo = SERVO_CENTER
                msg = f"ALIGNING (90deg)... Dist: {int(lidar_align_right)}mm"

                # 1. 후방 벽 충돌 방지
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("🛑 후방 벽 감지 -> 완료")
                    state = STATE_DONE

                # 2. ★ 라이다 90도 감지 시 정지 (앞선 맞추기)
                # 내 차가 뒤로 들어가다가, 90도 방향(옆)에 차가 감지되면(80cm 이내) 정지
                elif lidar_align_right < SIDE_CAR_DIST:
                    print(f"✅ 옆 차 감지 (90도: {int(lidar_align_right)}mm) -> 라인 정렬 완료")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [5] 완료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0
                msg = "PARKING DONE"
                ser.write(b"D,0\n")
                print(msg)
                break

            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())
            print(f"[{state}] {msg}")

    except KeyboardInterrupt:
        print("종료")
    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()


if __name__ == "__main__":
    main()