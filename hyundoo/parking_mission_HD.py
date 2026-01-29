import cv2
import numpy as np
import serial
import time
import math

# ==========================================
# [1] 환경 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX_REAR = 1  # ★ 후방 카메라 번호 확인 (전방이 0이면 보통 1)

# 해상도 설정 (VGA 표준)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# 물리적 수치 (mm)
OBSTACLE_THRES = 400     # 장애물 인식
EMPTY_THRES = 600        # 빈 공간 인식
REAR_STOP_DIST = 150     # 후방 정지 거리 (15cm)

# 속도
SEARCH_SPEED = 100       # 탐색
REVERSE_SPEED = -120     # 후진
EXIT_SPEED = 130         # 출차

# 타이밍
EXIT_TURN_TIME = 2.0     # 출차 좌회전 시간
EXIT_STRAIGHT_TIME = 1.5 # 출차 직진 시간

# 서보 값
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480    # 진입 (우측)
SERVO_LEFT_MAX = 680     # 출차 (좌측)

# 상태 정의
STATE_SEARCH = 0
STATE_READY = 1
STATE_ENTRY = 2
STATE_ALIGN = 3
STATE_PARKED = 4
STATE_EXIT_TURN = 5
STATE_EXIT_GO = 6
STATE_DONE = 7

# ==========================================
# [2] 초기화
# ==========================================
print("🔄 하드웨어 연결 중...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    print("✅ 아두이노 연결 성공")
except:
    ser = None
    print("❌ 아두이노 연결 실패 (테스트 모드)")

# 카메라 설정
cap = cv2.VideoCapture(CAM_INDEX_REAR, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
# 주차장은 어두울 수 있으니 노출 조절 (필요시 주석 해제)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4) 

if not cap.isOpened():
    print("❌ 후방 카메라 연결 실패")
    exit()

# 센서 데이터
sensors = [0]*6
rear_dist = 0

def read_sensors():
    global sensors, rear_dist
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                for i in range(min(6, len(parts))): sensors[i] = int(parts[i])
                if len(parts) >= 7: rear_dist = int(parts[6])
                else: rear_dist = 9999 
        except: pass

def send_cmd(servo, speed):
    if ser:
        ser.write(f"S,{servo}\n".encode())
        ser.write(f"D,{speed}\n".encode())

# ==========================================
# [3] 시각화 및 영상처리 함수 (핵심)
# ==========================================
def process_and_draw(frame):
    """
    1. 주차선 인식 및 각도 계산
    2. 화면에 인식된 선과 정보 그리기
    Return: angle_error (조향용)
    """
    display_img = frame.copy()
    h, w = frame.shape[:2]
    
    # 1. ROI (화면 하단 50%)
    roi_h = int(h * 0.5)
    roi = frame[roi_h:, :] 
    
    # 2. 전처리
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. 선 검출 (파라미터 튜닝 가능)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=80)
    
    angles = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 수평선 제외, 수직에 가까운 선만 추출
            if x2 == x1: continue
            slope = (y2 - y1) / (x2 - x1)
            
            if abs(slope) > 1: # 기울기가 가파른 선만 (주차선 후보)
                # 각도 계산
                angle = math.degrees(math.atan(slope))
                angles.append(angle)
                
                # ★ 시각화: 인식된 선을 녹색으로 그리기 (원본 좌표계로 변환)
                cv2.line(display_img, (x1, y1 + roi_h), (x2, y2 + roi_h), (0, 255, 0), 3)

    # 4. 에러 계산
    error = 0
    if angles:
        avg_angle = np.mean(angles)
        error = (90 - avg_angle) if avg_angle > 0 else (-90 - avg_angle)
        
        # ★ 시각화: 현재 인식된 각도 표시
        cv2.putText(display_img, f"Angle: {avg_angle:.1f}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(display_img, "No Line", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return error, display_img

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    state = STATE_SEARCH
    timer = 0
    gap_found = False
    
    print("🅿️ 주차 시스템 가동 (640x480)")

    while True:
        # [A] 데이터 수집
        ret, frame = cap.read()
        if not ret: break
        read_sensors()
        
        # 센서값 (RF, RM, RT)
        rm_dist = sensors[4] 
        rt_dist = sensors[5] 

        # [B] 상태 머신
        servo_cmd = SERVO_CENTER
        motor_cmd = 0
        
        if state == STATE_SEARCH:
            servo_cmd = SERVO_CENTER
            motor_cmd = SEARCH_SPEED
            if rm_dist > EMPTY_THRES:
                gap_found = True
                state = STATE_READY

        elif state == STATE_READY:
            servo_cmd = SERVO_CENTER
            motor_cmd = SEARCH_SPEED
            if rt_dist < OBSTACLE_THRES and gap_found:
                print("🛑 위치 확보! 정지")
                send_cmd(SERVO_CENTER, 0) # 즉시 정지
                time.sleep(1.0)
                state = STATE_ENTRY
                timer = time.time()
                continue # 명령 중복 방지

        elif state == STATE_ENTRY:
            servo_cmd = SERVO_RIGHT_MAX
            motor_cmd = REVERSE_SPEED
            if time.time() - timer > 2.5: 
                state = STATE_ALIGN

        elif state == STATE_ALIGN:
            # 영상 처리 결과 받아오기
            error, _ = process_and_draw(frame) # 여기서는 계산만
            
            # P제어
            steer = SERVO_RIGHT_MAX + int(error * 3)
            servo_cmd = max(480, min(680, steer))
            motor_cmd = REVERSE_SPEED
            
            if rear_dist < 300:
                state = STATE_PARKED

        elif state == STATE_PARKED:
            servo_cmd = SERVO_CENTER
            motor_cmd = REVERSE_SPEED
            
            if rear_dist < REAR_STOP_DIST:
                print("🅿️ 주차 완료! (3초 대기)")
                send_cmd(SERVO_CENTER, 0)
                time.sleep(3.0)
                state = STATE_EXIT_TURN
                timer = time.time()
                continue

        elif state == STATE_EXIT_TURN:
            servo_cmd = SERVO_LEFT_MAX
            motor_cmd = EXIT_SPEED
            if time.time() - timer > EXIT_TURN_TIME:
                state = STATE_EXIT_GO
                timer = time.time()

        elif state == STATE_EXIT_GO:
            servo_cmd = SERVO_CENTER
            motor_cmd = EXIT_SPEED
            if time.time() - timer > EXIT_STRAIGHT_TIME:
                print("🎉 미션 종료")
                state = STATE_DONE

        elif state == STATE_DONE:
            servo_cmd = SERVO_CENTER
            motor_cmd = 0
            # 종료 시 break
        
        # 명령 전송 (DONE이 아닐 때만)
        if state != STATE_DONE:
            send_cmd(servo_cmd, motor_cmd)
        else:
            send_cmd(SERVO_CENTER, 0)
            break

        # [C] 화면 시각화 (여기가 핵심!)
        # process_and_draw 함수에서 그려진 이미지를 받아옴
        _, display_frame = process_and_draw(frame)
        
        # 상태 정보 텍스트 오버레이
        state_str = ["SEARCH", "READY", "ENTRY", "ALIGN", "PARKED", "EXIT_L", "EXIT_S", "DONE"]
        cv2.putText(display_frame, f"State: {state_str[state]}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 센서 정보 표시 (중요!)
        # RM: 빈공간 찾기용 / RT: 위치 잡기용 / Rear: 주차용
        info = f"RM:{rm_dist} | RT:{rt_dist} | Rear:{rear_dist}"
        cv2.putText(display_frame, info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 주차선 가이드 (화면 중앙 빨간선 - 내 차 중심)
        cv2.line(display_frame, (CAM_WIDTH//2, 0), (CAM_WIDTH//2, CAM_HEIGHT), (0, 0, 255), 1)

        cv2.imshow("Rear Camera View (640x480)", display_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # 종료 처리
    if ser: ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()