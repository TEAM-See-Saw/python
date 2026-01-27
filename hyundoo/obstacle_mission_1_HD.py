import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar

# ==========================================
# [1] 설정
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'
BAUDRATE = 9600

# 미션 파라미터
OBS_DIST_THRES = 600      # 전방 장애물 감지 (60cm)
SIDE_CLEAR_THRES = 400    # 측면 장애물 여부 판단 (40cm 이상이면 없는 걸로 간주)
OFFSET_VAL = 150          # 차선 변경 오프셋

# 상태 정의
STATE_NORMAL = 0      # 2차선 주행
STATE_DETECTED = 1    # 감지됨 -> 1차선 변경 중
STATE_AVOIDING = 2    # 1차선 주행 (장애물 옆 통과 중)
STATE_RETURNING = 3   # 장애물 통과 -> 2차선 복귀 중
STATE_FINISHED = 4    # 미션 끝

# ==========================================
# [2] 하드웨어 및 통신
# ==========================================
ser = serial.Serial(PORT, BAUDRATE, timeout=1)
lidar = RPLidar(LIDAR_PORT)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ... (카메라 설정 생략) ...

# 초음파 센서 데이터 파싱용 변수
us_left = 9999
us_right = 9999

def read_sensors():
    """ 아두이노에서 보낸 초음파 값(US:L150,R30)을 읽어서 갱신 """
    global us_left, us_right
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                # "US:150,30" -> ["150", "30"]
                parts = line.replace("US:", "").split(",")
                us_left = int(parts[0])
                us_right = int(parts[1])
        except:
            pass # 통신 에러 무시

# ==========================================
# [3] 메인 로직
# ==========================================
def main():
    state = STATE_NORMAL
    state_timer = 0
    
    if lidar: iterator = lidar.iter_scans()
    
    print("🚀 센서 퓨전 장애물 회피 시작")

    while True:
        # 1. 데이터 수집
        ret, frame = cap.read()
        if not ret: break
        read_sensors() # 초음파 값 갱신 (us_left, us_right)
        
        # 라이다 전방 거리 측정
        front_dist = 9999
        try:
            scan = next(iterator)
            for (_, angle, dist) in scan:
                # 전방 20도 부채꼴
                if (angle >= 170 and angle <= 190) and dist > 0:
                    if dist < front_dist: front_dist = dist
        except: pass

        # 2. 상태 머신 (State Machine)
        offset = 0
        speed = 150
        
        # [Step 0] 2차선 직진
        if state == STATE_NORMAL:
            # 전방에 장애물 감지 시 회피 시작
            if front_dist < OBS_DIST_THRES:
                print(f"🚨 장애물 감지({int(front_dist)}mm) -> 좌측 회피 시작")
                state = STATE_DETECTED
                state_timer = time.time()

        # [Step 1] 좌측으로 차선 변경 (2->1)
        elif state == STATE_DETECTED:
            offset = -OFFSET_VAL # 왼쪽으로 붙음
            
            # (옵션) 타이머 대신 "차선 중앙에 왔는지" 판단하면 좋지만, 
            # 일단 안전하게 시간제(2초)로 진입
            if time.time() - state_timer > 2.0:
                print("↩️ 1차선 진입 완료. 장애물 통과 중...")
                state = STATE_AVOIDING

        # [Step 2] 장애물 옆 통과 중 (★ 핵심 변경 구간)
        elif state == STATE_AVOIDING:
            offset = -OFFSET_VAL # 계속 1차선 유지
            
            # ★ 기존에는 시간으로 했지만, 이제는 우측 센서를 믿습니다.
            # 우측 센서 거리가 40cm보다 커지면 -> 장애물이 내 옆을 지나갔다는 뜻!
            # (단, 노이즈 방지를 위해 전방 장애물도 없어야 함)
            if us_right > SIDE_CLEAR_THRES and front_dist > 800:
                # 확실히 하기 위해 0.5초 정도 더 직진 후 복귀
                time.sleep(0.5) 
                print("✅ 장애물 통과 확인(Side Clear) -> 우측 복귀 시작")
                state = STATE_RETURNING
                state_timer = time.time()

        # [Step 3] 우측으로 복귀 (1->2)
        elif state == STATE_RETURNING:
            offset = +OFFSET_VAL # 오른쪽으로 붙음
            
            if time.time() - state_timer > 2.0:
                print("🏁 복귀 완료. 미션 종료")
                state = STATE_FINISHED

        # [Step 4] 일반 주행
        elif state == STATE_FINISHED:
            offset = 0

        # 3. 주행 명령
        # calculate_steering 함수는 팀장님 코드 + offset 적용 버전 사용
        angle = calculate_steering(frame, offset)
        
        # (디버깅) 화면에 센서값 표시         cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, f"F-LiDAR: {int(front_dist)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"R-Sonar: {us_right}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        # ... (이하 아두이노 전송 및 imshow 동일) ...