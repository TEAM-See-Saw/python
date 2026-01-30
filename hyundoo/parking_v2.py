import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값 (팀장님 하드웨어 세팅 유지)
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# 속도 설정
SPEED_SEARCH = 80
SPEED_SWING = 80   # 앞으로 머리 돌릴 때 속도
SPEED_PARK = 75    # 후진 속도
SPEED_STOP = 0

# 서보 설정 (DC 조향 모터 특성 반영)
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440  # 우측 최대 (진입용)
SERVO_LEFT_MAX = 680   # 좌측 최대 (Swing Out용)

# ★ 전압 안정화 대기 시간 (중요!)
STEER_WAIT_TIME = 0.8

# 거리 기준 (mm)
CAR_DIST_MIN = 150
CAR_DIST_MAX = 1200
EMPTY_DIST_MIN = 1200
REAR_LIMIT = 200       # 후방 벽 감지 (IDX_LT, IDX_RT)
SWING_TIME = 1.0       # ★ 왼쪽으로 머리 돌리는 시간 (튜닝 포인트!)
REVERSE_TURN_TIME = 2.5 # ★ 우측으로 꺾어 들어가는 시간 (튜닝 포인트!)

# 아두이노 센서 인덱스 (팀장님 설정)
IDX_LT = 2
IDX_RT = 5

# 상태 정의
STATE_SEARCH = 0        # 1. 탐색
STATE_SWING_OUT = 1     # 2. [NEW] 왼쪽으로 머리 돌리기
STATE_PREP_REVERSE = 3  # 3. 정지 & 우측 핸들링
STATE_REVERSE_ENTRY = 4 # 4. 후진 진입
STATE_PREP_STRAIGHT = 5 # 5. 정지 & 중앙 정렬
STATE_REVERSE_FINISH = 6 # 6. 마무리 직진 후진
STATE_DONE = 7          # 7. 종료

# 탐색 단계 서브 스테이트
STEP_INIT_EMPTY = 0
STEP_PASSING_CAR1 = 1
STEP_FIND_SPOT = 2

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
        except: pass

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
    timer = 0
    car1_detect_time = 0

    print("🚀 [Swing & Cut] 주차 시스템 시작")

    try:
        for scan in lidar.iter_scans():
            read_sensors()
            
            # 센서 값
            lidar_right = get_lidar_dist_at_angle(scan, 90, 5) # 우측 90도
            dist_LT = sonar_data[IDX_LT] # 좌측 후방
            dist_RT = sonar_data[IDX_RT] # 우측 후방

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            msg = ""

            # ---------------------------------------------------
            # [1] 빈 공간 탐색 (팀장님 로직 유지)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER
                msg = f"SEARCH Step: {search_step}"

                # 1. 첫 차 발견
                if search_step == STEP_INIT_EMPTY:
                    if CAR_DIST_MIN <= lidar_right <= CAR_DIST_MAX:
                        print("🚗 첫 번째 차 발견!")
                        car1_detect_time = curr_time
                        search_step = STEP_PASSING_CAR1

                # 2. 첫 차 통과 (시간 지연)
                elif search_step == STEP_PASSING_CAR1:
                    if curr_time - car1_detect_time > 1.5: # 차폭 통과 시간
                        search_step = STEP_FIND_SPOT

                # 3. 빈 공간 및 두 번째 차(벽) 탐색
                elif search_step == STEP_FIND_SPOT:
                    if lidar_right > EMPTY_DIST_MIN:
                        msg = "Gap Found! Looking for Wall..."
                    
                    # 다시 장애물(두 번째 차) 나타나면 정지!
                    if CAR_DIST_MIN <= lidar_right <= CAR_DIST_MAX:
                        print("🛑 위치 확보 -> Swing Out 준비")
                        
                        # 완전 정지
                        ser.write(b"D,-100\n"); time.sleep(0.1)
                        ser.write(b"D,0\n"); time.sleep(1.0)
                        
                        state = STATE_SWING_OUT
                        timer = time.time()

            # ---------------------------------------------------
            # [2] Swing Out (왼쪽으로 머리 돌리기) ★ 핵심
            # ---------------------------------------------------
            elif state == STATE_SWING_OUT:
                # 핸들 왼쪽 끝
                cmd_servo = SERVO_LEFT_MAX 
                
                # 핸들 돌리는 시간 확보 (Stop & Steer)
                if curr_time - timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "⚡ TURNING LEFT..."
                else:
                    # 핸들 다 돌아갔으면 앞으로 전진!
                    cmd_speed = SPEED_SWING
                    msg = "↪️ SWING OUT (Left Forward)"
                    
                    # 1초 정도 앞으로 나가서 엉덩이 각도 만들기
                    if curr_time - timer > (STEER_WAIT_TIME + SWING_TIME):
                        print("🛑 Swing 완료 -> 후진 준비")
                        ser.write(b"D,0\n"); time.sleep(0.5)
                        state = STATE_PREP_REVERSE
                        timer = time.time()

            # ---------------------------------------------------
            # [3] 후진 준비 (오른쪽으로 핸들 꺾고 대기)
            # ---------------------------------------------------
            elif state == STATE_PREP_REVERSE:
                cmd_servo = SERVO_RIGHT_MAX # 우측 최대
                cmd_speed = 0
                msg = "⚡ STEERING RIGHT (Wait)..."
                
                # 전압 안정화 대기
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_ENTRY
                    timer = time.time()

            # ---------------------------------------------------
            # [4] 후진 진입 (Hard Cut)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_ENTRY:
                cmd_servo = SERVO_RIGHT_MAX # 핸들 고정
                cmd_speed = -SPEED_PARK     # 후진
                msg = "🔙 REVERSING ENTRY..."
                
                # 충돌 방지
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    state = STATE_DONE
                
                # 일정 시간 후진 (이미 각도가 좋으므로 짧게 들어가도 됨)
                if curr_time - timer > REVERSE_TURN_TIME:
                    print("🛑 진입 완료 -> 핸들 정렬 준비")
                    ser.write(b"D,0\n"); time.sleep(0.5)
                    state = STATE_PREP_STRAIGHT
                    timer = time.time()

            # ---------------------------------------------------
            # [5] 중앙 정렬 준비 (핸들 풀기)
            # ---------------------------------------------------
            elif state == STATE_PREP_STRAIGHT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = "⚡ CENTERING WHEELS..."
                
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_FINISH
                    timer = time.time()

            # ---------------------------------------------------
            # [6] 마무리 직진 후진 (깊이 조절)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_FINISH:
                cmd_servo = SERVO_CENTER
                cmd_speed = -SPEED_PARK
                msg = "PARKING FINISH (Straight Back)"
                
                # 후방 센서 감지 시 정지
                if dist_LT < REAR_LIMIT or dist_RT < REAR_LIMIT:
                    print("✅ 후방 벽 감지 -> 주차 완료")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [7] 종료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                msg = "MISSION SUCCESS"
                ser.write(b"D,0\n")
                # 루프 탈출 대신 대기 (데이터 확인용)

            # 명령 전송
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            # 시각화
            debug_img = np.zeros((250, 600, 3), dtype=np.uint8)
            cv2.putText(debug_img, f"State: {state}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            cv2.putText(debug_img, msg, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(debug_img, f"LT:{dist_LT} RT:{dist_RT}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(debug_img, f"Lidar R:{int(lidar_right)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
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