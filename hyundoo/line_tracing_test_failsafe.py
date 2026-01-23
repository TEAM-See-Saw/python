import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 설정 (튜닝 포인트)
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

# 서보/속도 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 150 

# 노출값 (환경에 맞춰 조절: -4 ~ -10)
CAM_EXPOSURE = -7

# 해상도 (속도 최적화 640x480)
TARGET_W = 640
TARGET_H = 480

# ==========================================
# [2] 초기화
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
except:
    print("⚠️ 아두이노 연결 실패 (영상 테스트 모드)")
    ser = None

# MJPG + 자동노출 해제 설정
params = [
    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
    cv2.CAP_PROP_FRAME_WIDTH, 1920,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    cv2.CAP_PROP_FPS, 30,
    cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 
]
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW, params)
cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE)

# ==========================================
# [3] 로직 & 시각화 함수
# ==========================================

def make_coords(image, line_params):
    """ 기울기와 절편을 좌표로 변환 (그리기 용) """
    if line_params is None: return None
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0: slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def check_glare(img):
    """ 눈부심 감지 """
    h, w = img.shape[:2]
    # 상단 30% 영역의 평균 밝기 확인
    roi_glare = img[0:int(h*0.3), :]
    gray_glare = cv2.cvtColor(roi_glare, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_glare)
    return avg_brightness > 200

def process_image(img, is_glare):
    h, w = img.shape[:2]
    
    # [핵심] 눈부시면 시야를 낮춤 (ROI 조절)
    if is_glare:
        roi_top = int(h * 0.6) # 하단 40%만 봄
        roi_img = img[roi_top:h, :]
    else:
        roi_top = int(h * 0.4) # 하단 60% 봄
        roi_img = img[roi_top:h, :]

    # 전처리 (Top-Hat)
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 이진화 & 엣지
    _, binary = cv2.threshold(top_hat, 50, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150)
    
    return edges, roi_top

def calculate_data(edges, w, h, roi_top):
    """ 조향각 계산 및 시각화 데이터 리턴 """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=150)
    
    left_fit, right_fit = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            
            if abs(slope) < 0.4: continue # 수평선 제거
            if slope < -0.5: left_fit.append((slope, fit[1]))
            elif slope > 0.5: right_fit.append((slope, fit[1]))

    # 평균 라인 계산
    l_param = np.mean(left_fit, axis=0) if left_fit else None
    r_param = np.mean(right_fit, axis=0) if right_fit else None

    # 좌표 변환 (그리기 위해) -> ROI 기준 좌표임에 주의
    # make_coords 함수 내에서 shape[0]을 쓰는데, 이는 edges(ROI된 것)의 높이여야 함
    # 임시 빈 이미지 생성해서 shape 전달
    dummy_img = np.zeros((h - roi_top, w, 3), dtype=np.uint8)
    l_coords = make_coords(dummy_img, l_param)
    r_coords = make_coords(dummy_img, r_param)

    # 타겟 포인트 계산
    car_x = w // 2
    target_x = None

    if l_coords and r_coords:
        target_x = (l_coords[0][2] + r_coords[0][2]) / 2
    elif l_coords:
        target_x = l_coords[0][2] + (w * 0.22)
    elif r_coords:
        target_x = r_coords[0][2] - (w * 0.22)

    angle = None
    if target_x is not None:
        dx = target_x - car_x
        dy = (h - roi_top) * 0.7 
        angle = math.degrees(math.atan2(dx, dy))
        # ROI 상의 target_x를 전체 화면 기준으로 변환할 필요는 없음 (X축은 동일)
    
    return angle, l_coords, r_coords, target_x

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())
    
    last_servo = SERVO_CENTER
    no_line_count = 0 

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Resize
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        h, w = frame.shape[:2]

        # 2. 눈부심 체크 & 처리
        is_glare = check_glare(frame)
        edges, roi_top = process_image(frame, is_glare)

        # 3. 데이터 계산 (라인 좌표 포함)
        angle, l_coords, r_coords, target_x = calculate_data(edges, w, h, roi_top)

        # 4. 제어 로직 (Failsafe)
        status_text = "NORMAL"
        text_color = (0, 255, 0) # Green

        if angle is not None:
            clamped = max(-45, min(45, angle))
            servo = int(np.interp(clamped, [-45, 45], [SERVO_LEFT_MAX, SERVO_RIGHT_MAX]))
            last_servo = servo
            no_line_count = 0
        else:
            no_line_count += 1
            if no_line_count < 30:
                servo = last_servo
                status_text = "HOLDING (Line Lost)"
                text_color = (0, 255, 255) # Yellow
            else:
                servo = SERVO_CENTER
                status_text = "LOST LINE!"
                text_color = (0, 0, 255) # Red

        # 5. 명령 전송
        if ser: ser.write(f"S,{servo}\n".encode())

        # ==========================================
        # ★ [시각화] 화면 그리기 (여기가 추가됨)
        # ==========================================
        # (1) 검은 배경에 라인 그리기
        line_layer = np.zeros_like(frame)
        
        # ROI 오프셋(roi_top)을 더해서 전체 화면 위치에 맞게 그리기
        if l_coords:
            x1, y1, x2, y2 = l_coords[0]
            cv2.line(line_layer, (x1, y1 + roi_top), (x2, y2 + roi_top), (0, 255, 0), 5)
        if r_coords:
            x1, y1, x2, y2 = r_coords[0]
            cv2.line(line_layer, (x1, y1 + roi_top), (x2, y2 + roi_top), (0, 255, 0), 5)

        # (2) 원본 + 라인 합성
        display_frame = cv2.addWeighted(frame, 0.8, line_layer, 1, 1)

        # (3) 타겟 포인트(빨간 선) 그리기
        if target_x is not None:
            cv2.line(display_frame, (w//2, h), (int(target_x), int(h*0.6)), (0, 0, 255), 3)

        # (4) 정보 텍스트 표시
        cv2.putText(display_frame, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(display_frame, f"Servo: {servo}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if is_glare:
            cv2.putText(display_frame, "GLARE DETECTED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # ROI 경계선 표시 (노란선)
            cv2.line(display_frame, (0, roi_top), (w, roi_top), (0, 255, 255), 2)

        # 화면 출력
        cv2.imshow("Final Visual View", display_frame)
        # 엣지 화면도 같이 보면 좋음
        cv2.imshow("Edges", edges)

        if cv2.waitKey(1) == ord('q'): break

    if ser:
        ser.write(b"D,0\n")
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()