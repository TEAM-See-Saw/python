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
CAM_INDEX = 0  # 전방 카메라

# 속도
SPEED_SEARCH = 80
SPEED_SWING = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
STEER_WAIT_TIME = 0.8

# ★ 영상 처리 튜닝 포인트
# 엣지 밀도 기준 (화면 우측 영역 중 흰색 픽셀 비율)
EDGE_THRES_HIGH = 0.15  # 15% 이상이 엣지면 '차'
EDGE_THRES_LOW = 0.05   # 5% 이하면 '빈 공간'

# 상태
STATE_SEARCH = 0
STATE_SWING_OUT = 1
STATE_REVERSE_ENTRY = 2
STATE_DONE = 3

# 탐색 단계
STEP_CAR1 = 0    # 첫 차 보는 중
STEP_GAP = 1     # 빈 공간 보는 중
STEP_CAR2 = 2    # 두 번째 차 발견 (Trigger)

# ==========================================
# [2] 영상 처리 함수 (핵심)
# ==========================================
def detect_parking_slot(frame):
    """
    화면 우측의 엣지 밀도를 분석하여 상태 반환
    Return: 'CAR', 'EMPTY', 'UNKNOWN'
    """
    if frame is None: return 'UNKNOWN', 0
    
    h, w = frame.shape[:2]
    
    # 1. ROI 설정: 화면 우측 1/3, 수직 중앙부
    # 바닥과 천장을 제외하고 차가 있을만한 높이만 봄
    roi_x = int(w * 0.66) # 오른쪽 1/3 지점부터
    roi_y_start = int(h * 0.4)
    roi_y_end = int(h * 0.8)
    
    roi = frame[roi_y_start:roi_y_end, roi_x:w]
    
    # 2. 전처리 (Blur -> Canny)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # 3. 밀도 계산 (전체 픽셀 중 흰색 픽셀 비율)
    total_pixels = edges.size
    white_pixels = cv2.countNonZero(edges)
    density = white_pixels / total_pixels
    
    # 4. 판단
    status = 'UNKNOWN'
    if density > EDGE_THRES_HIGH:
        status = 'CAR'
    elif density < EDGE_THRES_LOW:
        status = 'EMPTY'
        
    return status, density, roi  # ROI는 디버깅용 리턴

# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser, lidar
    
    # 카메라 초기화
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        # lidar = RPLidar(LIDAR_PORT) # 라이다 필요 시 주석 해제
        print("✅ 시스템 연결 성공")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_CAR1
    timer = 0
    
    print("🚀 [Vision Trigger] 주차 시스템 시작")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 영상 처리 수행
        status, density, roi_edges = detect_parking_slot(frame)
        
        cmd_speed = 0
        cmd_servo = SERVO_CENTER
        curr_time = time.time()
        msg = f"Dens: {density:.2f} [{status}]"

        # ---------------------------------------------------
        # [1] 시각적 탐색 (Visual Search)
        # ---------------------------------------------------
        if state == STATE_SEARCH:
            cmd_speed = SPEED_SEARCH
            cmd_servo = SERVO_CENTER
            
            # 1단계: 첫 번째 차 지나가는 중
            if search_step == STEP_CAR1:
                if status == 'EMPTY':
                    print("👀 빈 공간 진입 (Gap Start)")
                    search_step = STEP_GAP
            
            # 2단계: 빈 공간 지나가는 중
            elif search_step == STEP_GAP:
                if status == 'CAR':
                    # ★ 핵심 트리거: 빈 공간 지나다가 다시 '차(엣지)'가 보이면 즉시 정지!
                    print("🛑 두 번째 차 감지! (Visual Trigger)")
                    
                    # 급정거
                    ser.write(b"D,-100\n"); time.sleep(0.1)
                    ser.write(b"D,0\n"); time.sleep(1.0)
                    
                    state = STATE_SWING_OUT
                    timer = curr_time

        # ---------------------------------------------------
        # [2] Swing Out (기존 로직)
        # ---------------------------------------------------
        elif state == STATE_SWING_OUT:
            # 여기부터는 기존 Swing & Cut 로직과 동일
            # 카메라로 위치를 잡았으니, 정해진 패턴대로 움직이면 됨
            
            cmd_servo = SERVO_LEFT_MAX 
            if curr_time - timer < STEER_WAIT_TIME:
                cmd_speed = 0
                msg = "Wait (Steer Left)"
            else:
                cmd_speed = SPEED_SWING
                msg = "Swing Out"
                if curr_time - timer > (STEER_WAIT_TIME + 1.0): # 1초 스윙
                    state = STATE_REVERSE_ENTRY
                    timer = curr_time
                    ser.write(b"D,0\n"); time.sleep(0.5)

        # ---------------------------------------------------
        # [3] 후진 진입 (간소화)
        # ---------------------------------------------------
        elif state == STATE_REVERSE_ENTRY:
            cmd_servo = SERVO_RIGHT_MAX
            if curr_time - timer < STEER_WAIT_TIME:
                cmd_speed = 0
            else:
                cmd_speed = -SPEED_PARK
                
                # 2.5초 후진 후 종료 (또는 센서 체크)
                if curr_time - timer > (STEER_WAIT_TIME + 2.5):
                    state = STATE_DONE

        elif state == STATE_DONE:
            cmd_speed = 0
            ser.write(b"D,0\n")
            # break # 계속 화면 보려면 주석 처리

        # 명령 전송
        ser.write(f"S,{cmd_servo}\n".encode())
        ser.write(f"D,{cmd_speed}\n".encode())

        # 디버그 화면 (ROI 엣지 보여주기)
        # 원본에 ROI 사각형 표시
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (int(w*0.66), int(h*0.4)), (w, int(h*0.8)), (0, 255, 0), 2)
        cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow("Front Camera", frame)
        # ROI 엣지 화면도 작게 띄우기
        if roi_edges is not None:
             cv2.imshow("ROI Edges", roi_edges)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    if ser: ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()