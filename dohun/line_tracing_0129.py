import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 환경 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX = 0

MAX_SPEED = 210  # 속도 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# Fail-safe용 전역 변수
last_target_x = 320

# ==========================================
# [2] 영상 처리 보조 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [3] 메인 실행 루프
# ==========================================
def main():
    global last_target_x
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 노출을 -7 정도로 고정 (너무 밝으면 -9까지 조절)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)

    ser = None
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        print(f"✅ {PORT} 포트 연결 성공")
        time.sleep(2)
    except:
        print("⚠️ 아두이노 연결 실패 - 시뮬레이션 모드")

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- [STEP 1] 동적 임계값 계산 (핵심 로직) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 차선이 주로 위치하는 하단부(60% 아래)의 평균 밝기 계산
        roi_bottom = blur[int(height*0.6):, :]
        avg_brightness = np.mean(roi_bottom)
        
        # [동적 공식] 평균 밝기에 50(오프셋)을 더함. 
        # 밝은 곳에선 기준을 높이고, 어두운 곳에선 기준을 낮춤
        dynamic_thresh = int(avg_brightness + 55) 
        dynamic_thresh = max(110, min(dynamic_thresh, 220)) # 최소/최대값 제한

        # --- [STEP 2] 이진화 및 ROI ---
        _, thresh = cv2.threshold(blur, dynamic_thresh, 255, cv2.THRESH_BINARY)
        
        roi_v = np.array([[(0, height), (width*0.3, height*0.6), 
                           (width*0.7, height*0.6), (width, height)]], dtype=np.int32)
        masked_img = region_of_interest(thresh, roi_v)

        # --- [STEP 3] 차선 검출 및 연결 ---
        # maxLineGap=150으로 점들을 실선으로 이음
        lines = cv2.HoughLinesP(masked_img, 1, np.pi/180, 20, minLineLength=10, maxLineGap=160)

        left_pts, right_pts = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # 기울기로 좌우 분류
                if slope < -0.3:
                    left_pts.append(x2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif slope > 0.3:
                    right_pts.append(x2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # --- [STEP 4] 조향 계산 및 전송 ---
        if left_pts and right_pts:
            target_x = (np.mean(left_pts) + np.mean(right_pts)) / 2
        elif left_pts:
            target_x = np.mean(left_pts) + 160 # 왼쪽만 보이면 오른쪽으로 타겟 이동
        elif right_pts:
            target_x = np.mean(right_pts) - 160 # 오른쪽만 보이면 왼쪽으로 타겟 이동
        else:
            target_x = last_target_x # 차선 실종 시 유지

        last_target_x = target_x
        
        # 타겟 지점과의 각도 계산 (dx, dy 기반)
        dx = target_x - 320
        dy = -200 # 가상의 높이차
        angle = math.degrees(math.atan2(dx, abs(dy)))
        
        # 서보 제어값 맵핑
        servo_val = int(map_value(max(-40, min(40, angle)), -40, 40, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        if ser:
            ser.write(f"S,{servo_val}\n".encode())

        # --- [STEP 5] 상태 모니터링 ---
        debug_info = f"Bright:{int(avg_brightness)} Thresh:{dynamic_thresh}"
        cv2.putText(frame, debug_info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.circle(frame, (int(target_x), int(height*0.6)), 10, (255, 0, 255), -1)
        
        combined = np.hstack((frame, cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Dynamic Lane Tracking", combined)

        if cv2.waitKey(1) == ord('q'): break

    if ser:
        ser.write(b"D,0\n")
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()