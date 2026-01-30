import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값 (Hardware Config)
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# 속도 (PWM)
SPEED_SEARCH = 80
SPEED_SWING = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보 (DC 모터 보호용 Stop & Steer 적용)
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440  # 진입용
SERVO_LEFT_MAX = 680   # Swing용
STEER_WAIT_TIME = 0.8  # 핸들링 대기 시간

# ★ 튜닝 포인트 (Tuning)
ALIGN_THRES = 50       # 평행 판단 기준 (50mm 이내 차이면 평행)
VALID_DIST = 800       # 유효 감지 거리 (800mm 이내 물체만 벽으로 인정)
PARKING_DEPTH_TIME = 1.5 # 정렬 후 진입 시간
FORWARD_OFFSET_TIME = 0.6 # 2번 차 감지 후 더 나가는 시간

# 거리 기준 (LiDAR)
EMPTY_DIST_MIN = 1000     # 1m 이상이면 빈 공간
OBSTACLE_DIST_MAX = 600   # 60cm 이하면 장애물

# 센서 인덱스 (우측 기준)
# 순서: LF(0), LM(1), LT(2), RF(3), RM(4), RT(5)
IDX_RM = 4  # 우측 허리
IDX_RT = 5  # 우측 꼬리

# 상태 정의
STATE_SEARCH = 0
STATE_FORWARD_OFFSET = 1
STATE_SWING_OUT = 2
STATE_PREP_REVERSE = 3
STATE_REVERSE_ENTRY = 4
STATE_PREP_STRAIGHT = 5
STATE_REVERSE_FINISH = 6
STATE_DONE = 7

# 탐색 단계
STEP_CAR1 = 0
STEP_GAP = 1

# ==========================================
# [2] 센서 및 시각화 함수
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

def get_lidar_dist_at_angle(scan, target_angle, angle_range=5):
    dists = []
    min_a = target_angle - angle_range
    max_a = target_angle + angle_range
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
    if len(dists) > 0: return np.mean(dists)
    return 9999

def draw_dashboard(state, rm, rt, lidar_dist, diff):
    """ 센서 값을 시각화하는 대시보드 화면 생성 """
    # 640x480 검은 배경
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. 차량 그리기 (중앙)
    cx, cy = 320, 240
    cw, ch = 100, 180
    cv2.rectangle(img, (cx-cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (200, 200, 200), 2)
    # 앞쪽 표시 (화살표)
    cv2.arrowedLine(img, (cx, cy), (cx, cy-100), (0, 255, 255), 2)
    
    # 2. 초음파 센서 값 표시 (위치 매핑)
    # LF(0), LM(1), LT(2) -> 왼쪽
    # RF(3), RM(4), RT(5) -> 오른쪽
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 우측 센서 시각화 (RM, RT가 핵심)
    # RM (우측 허리)
    color_rm = (0, 255, 0) if rm < VALID_DIST else (0, 0, 255)
    cv2.putText(img, f"RM: {rm}", (cx+60, cy), font, 0.7, color_rm, 2)
    cv2.line(img, (cx+cw//2, cy), (cx+cw//2 + min(100, int(rm/10)), cy), color_rm, 3)

    # RT (우측 꼬리)
    color_rt = (0, 255, 0) if rt < VALID_DIST else (0, 0, 255)
    cv2.putText(img, f"RT: {rt}", (cx+60, cy+80), font, 0.7, color_rt, 2)
    cv2.line(img, (cx+cw//2, cy+80), (cx+cw//2 + min(100, int(rt/10)), cy+80), color_rt, 3)

    # 나머지 센서 (참고용)
    cv2.putText(img, f"Lidar R: {int(lidar_dist)}", (cx+60, cy-80), font, 0.6, (255, 255, 0), 1)
    
    # 3. 상태 정보 표시
    state_str = [
        "SEARCH", "FWD OFFSET", "SWING OUT", "PREP REVERSE", 
        "ENTRY (ALIGN)", "PREP STRAIGHT", "FINISH", "DONE"
    ]
    curr_state_text = state_str[state] if state < len(state_str) else "UNKNOWN"
    
    cv2.putText(img, f"STATE: {curr_state_text}", (20, 50), font, 1.0, (0, 255, 255), 2)
    
    # 4. 평행 정렬 게이지 (Diff)
    # Entry 단계에서만 중요하게 표시
    if state == STATE_REVERSE_ENTRY:
        bar_color = (0, 255, 0) if diff < ALIGN_THRES else (0, 0, 255)
        cv2.putText(img, f"DIFF: {diff}mm", (20, 400), font, 1.2, bar_color, 3)
        cv2.putText(img, f"Target: < {ALIGN_THRES}", (20, 440), font, 0.7, (200, 200, 200), 1)
    
    return img

# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser, lidar
    
    # 초기화
    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 하드웨어 연결 성공")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_CAR1
    timer = 0
    
    # 카운터
    gap_count = 0
    obs_count = 0

    print("🚀 [Visual Dashboard] 주차 시스템 시작")

    try:
        for scan in lidar.iter_scans():
            read_sensors()
            
            # 핵심 변수 추출
            rm_dist = sonar_data[IDX_RM]
            rt_dist = sonar_data[IDX_RT]
            lidar_right = get_lidar_dist_at_angle(scan, 90, 5) # 우측 90도
            diff = abs(rm_dist - rt_dist)

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()
            
            # ---------------------------------------------------
            # [1] 탐색 (LiDAR Search)
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                
                # 1단계: 첫 차 지나 빈 공간 찾기
                if search_step == STEP_CAR1:
                    if lidar_right > EMPTY_DIST_MIN:
                        gap_count += 1
                    else:
                        gap_count = 0
                        
                    if gap_count > 3:
                        print("👀 빈 공간 시작 (Gap Start)")
                        search_step = STEP_GAP
                        gap_count = 0
                
                # 2단계: 빈 공간 지나 2번 차(벽) 찾기
                elif search_step == STEP_GAP:
                    if lidar_right < OBSTACLE_DIST_MAX:
                        obs_count += 1
                    else:
                        obs_count = 0
                        
                    if obs_count > 2:
                        print("🛑 2번 차 감지! (Trigger)")
                        state = STATE_FORWARD_OFFSET
                        timer = curr_time

            # ---------------------------------------------------
            # [NEW] 오프셋 (조금 더 가서 멈추기)
            # ---------------------------------------------------
            elif state == STATE_FORWARD_OFFSET:
                cmd_speed = SPEED_SEARCH
                if curr_time - timer > FORWARD_OFFSET_TIME:
                    print("🛑 정지 & 스윙 준비")
                    ser.write(b"D,-100\n"); time.sleep(0.1) # 브레이크
                    ser.write(b"D,0\n"); time.sleep(1.0)
                    state = STATE_SWING_OUT
                    timer = curr_time

            # ---------------------------------------------------
            # [2] Swing Out (왼쪽 머리 돌리기)
            # ---------------------------------------------------
            elif state == STATE_SWING_OUT:
                cmd_servo = SERVO_LEFT_MAX
                if curr_time - timer < STEER_WAIT_TIME:
                    cmd_speed = 0 # 핸들링 대기
                else:
                    cmd_speed = SPEED_SWING
                    
                if curr_time - timer > (STEER_WAIT_TIME + 1.2):
                    print("🛑 스윙 완료 -> 후진 준비")
                    ser.write(b"D,0\n"); time.sleep(0.5)
                    state = STATE_PREP_REVERSE
                    timer = curr_time

            # ---------------------------------------------------
            # [3] 후진 준비 (Stop & Steer)
            # ---------------------------------------------------
            elif state == STATE_PREP_REVERSE:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = 0
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_ENTRY
                    timer = curr_time

            # ---------------------------------------------------
            # [4] 후진 진입 & 평행 감지 (★ 핵심)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_ENTRY:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = -SPEED_PARK
                
                # 평행 조건 체크
                valid_sensing = (rm_dist < VALID_DIST and rt_dist < VALID_DIST)
                is_parallel = (diff < ALIGN_THRES)
                min_time_ok = (curr_time - timer > 1.0) # 최소 1초 후진 후 체크
                
                if valid_sensing and is_parallel and min_time_ok:
                    print(f"✅ 평행 감지! (Diff: {diff}) -> 정렬")
                    ser.write(b"D,0\n"); time.sleep(0.5) # 급정거
                    state = STATE_PREP_STRAIGHT
                    timer = curr_time
                    
                elif curr_time - timer > 4.0: # 타임아웃 안전장치
                    print("⚠️ 타임아웃 -> 강제 정렬")
                    state = STATE_PREP_STRAIGHT
                    timer = curr_time

            # ---------------------------------------------------
            # [5] 정렬 준비 (Stop & Center)
            # ---------------------------------------------------
            elif state == STATE_PREP_STRAIGHT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_FINISH
                    timer = curr_time

            # ---------------------------------------------------
            # [6] 마무리 진입 (깊이 조절)
            # ---------------------------------------------------
            elif state == STATE_REVERSE_FINISH:
                cmd_servo = SERVO_CENTER
                cmd_speed = -SPEED_PARK
                
                if curr_time - timer > PARKING_DEPTH_TIME:
                    print("🅿️ 주차 완료")
                    state = STATE_DONE

            # ---------------------------------------------------
            # [7] 종료
            # ---------------------------------------------------
            elif state == STATE_DONE:
                cmd_speed = 0
                ser.write(b"D,0\n")
            
            # 명령 전송
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            # ---------------------------------------------------
            # ★ [시각화] 대시보드 그리기
            # ---------------------------------------------------
            dashboard_img = draw_dashboard(state, rm_dist, rt_dist, lidar_right, diff)
            cv2.imshow("Parking Dashboard", dashboard_img)
            
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt: pass
    finally:
        if ser: ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()