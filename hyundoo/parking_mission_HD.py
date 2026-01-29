import cv2
import numpy as np
import serial
import time
import math

# ==========================================
# [1] 환경 설정 (주신 수치 반영)
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX_REAR = 1  # ★ 후방 카메라 확인

# 물리적 수치 (mm)
OBSTACLE_THRES = 400     # 장애물 인식 거리
EMPTY_THRES = 600        # 빈 공간 인식 거리
REAR_STOP_DIST = 150     # 후방 벽 15cm 남으면 정지

# 속도 및 타이밍
SEARCH_SPEED = 100       # 탐색 속도
REVERSE_SPEED = -120     # 후진 속도
EXIT_SPEED = 130         # 출차 속도 (조금 힘차게)
EXIT_TURN_TIME = 2.0     # 출차 시 좌회전 지속 시간
EXIT_STRAIGHT_TIME = 1.5 # 출차 후 직진 지속 시간

# 서보 값
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480    # 진입용 (우측)
SERVO_LEFT_MAX = 680     # 출차용 (좌측)

# 상태 정의
STATE_SEARCH = 0    # 탐색
STATE_READY = 1     # 위치 잡기 (지나치기)
STATE_ENTRY = 2     # 진입 (우측 꺾고 후진)
STATE_ALIGN = 3     # 정렬
STATE_PARKED = 4    # 주차 완료 (대기)
STATE_EXIT_TURN = 5 # 출차 1 (좌회전)
STATE_EXIT_GO = 6   # 출차 2 (직진)
STATE_DONE = 7      # 미션 성공

# ==========================================
# [2] 초기화
# ==========================================
ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
cap = cv2.VideoCapture(CAM_INDEX_REAR, cv2.CAP_DSHOW)

# 6채널 + 후방 센서 데이터
sensors = [0]*6
rear_dist = 0

def read_sensors():
    """ "US:LF,LM,LT,RF,RM,RT,Rear" (7개 데이터 가정) """
    global sensors, rear_dist
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                for i in range(6): sensors[i] = int(parts[i])
                if len(parts) >= 7: rear_dist = int(parts[6])
                else: rear_dist = 9999 
        except: pass

def send_cmd(servo, speed):
    ser.write(f"S,{servo}\n".encode())
    ser.write(f"D,{speed}\n".encode())

# ==========================================
# [3] 후방 카메라 로직
# ==========================================
def process_rear_view(frame):
    """ 주차선 각도 계산 (0:평행, 양수:우측쏠림, 음수:좌측쏠림) """
    if frame is None: return 0
    h, w = frame.shape[:2]
    roi = frame[int(h*0.5):, :] 
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    if lines is None: return 0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > 1: # 수직에 가까운 선만
            angles.append(math.degrees(math.atan(slope)))
    
    if not angles: return 0
    avg_angle = np.mean(angles)
    return (90 - avg_angle) if avg_angle > 0 else (-90 - avg_angle)

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    state = STATE_SEARCH
    timer = 0
    gap_found = False
    
    print("🅿️ 주차 미션 시작 (Parking -> Exit)")

    while True:
        ret, frame = cap.read()
        if not ret: break
        read_sensors()
        
        # 우측 센서들 (RF, RM, RT)
        rm_dist = sensors[4] # 중간
        rt_dist = sensors[5] # 꼬리

        # -------------------------------------
        # [State Machine]
        # -------------------------------------
        
        # 1. 빈 공간 탐색
        if state == STATE_SEARCH:
            send_cmd(SERVO_CENTER, SEARCH_SPEED)
            if rm_dist > EMPTY_THRES: # 빈 공간 발견
                gap_found = True
                state = STATE_READY

        # 2. 위치 잡기 (RT 센서가 벽을 만날 때까지)
        elif state == STATE_READY:
            send_cmd(SERVO_CENTER, SEARCH_SPEED)
            # 빈 공간을 지나쳐서 다시 장애물(벽/차)이 나타나면 정지
            if rt_dist < OBSTACLE_THRES and gap_found:
                print("🛑 위치 확보! 정지")
                send_cmd(SERVO_CENTER, 0)
                time.sleep(1.0)
                state = STATE_ENTRY
                timer = time.time()

        # 3. 진입 (우측 꺾고 후진)
        elif state == STATE_ENTRY:
            send_cmd(SERVO_RIGHT_MAX, REVERSE_SPEED)
            # 2.5초간 진입 (약 45도 회전)
            if time.time() - timer > 2.5: 
                state = STATE_ALIGN

        # 4. 정렬 (카메라 + 미세 후진)
        elif state == STATE_ALIGN:
            error = process_rear_view(frame)
            steer = SERVO_RIGHT_MAX + int(error * 3) # P제어
            steer = max(480, min(680, steer))
            
            send_cmd(steer, REVERSE_SPEED)
            
            # 후방 30cm 이내 진입 시 정렬 종료
            if rear_dist < 300:
                state = STATE_PARKED

        # 5. 주차 완료 (직진 후진 + 3초 대기)
        elif state == STATE_PARKED:
            send_cmd(SERVO_CENTER, REVERSE_SPEED)
            
            if rear_dist < REAR_STOP_DIST: # 15cm 도착
                print("🅿️ 주차 완료! (3초 대기)")
                send_cmd(SERVO_CENTER, 0) # 완전 정지
                time.sleep(3.0)           # 심판 확인용 3초
                print("🚗 출차 시작!")
                state = STATE_EXIT_TURN
                timer = time.time()

        # 6. 출차 1단계: 좌회전으로 빠져나오기
        elif state == STATE_EXIT_TURN:
            # 핸들 왼쪽 끝 + 전진
            send_cmd(SERVO_LEFT_MAX, EXIT_SPEED)
            
            # 차가 주차칸을 빠져나올 때까지 (약 2초)
            if time.time() - timer > EXIT_TURN_TIME:
                state = STATE_EXIT_GO
                timer = time.time()

        # 7. 출차 2단계: 핸들 풀고 직진 (출구 방향)
        elif state == STATE_EXIT_GO:
            # 핸들 중앙 + 전진
            send_cmd(SERVO_CENTER, EXIT_SPEED)
            
            # 라인 복귀할 만큼 직진
            if time.time() - timer > EXIT_EXIT_STRAIGHT_TIME:
                print("🎉 미션 성공! (본선 복귀 완료)")
                state = STATE_DONE

        # 8. 종료
        elif state == STATE_DONE:
            send_cmd(SERVO_CENTER, 0) # 정지 or 라인트레이싱 전환
            break

        # 디버깅 화면
        cv2.putText(frame, f"State: {state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Parking Monitor", frame)
        if cv2.waitKey(1) == ord('q'): break

    ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()