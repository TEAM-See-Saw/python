import cv2
import numpy as np
import math
import serial
import time
import datetime

# ==========================================
# [1] 설정 (Configuration)
# ==========================================
CAM_INDEX = 0           # 카메라 포트 (0 또는 1)
PORT = 'COM4'           # 아두이노 포트
BAUDRATE = 9600

# 서보 및 모터 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 80         # 최고 속도

# ★ Version 5 튜닝 값 (FHD 기준)
# 이 값들은 현장 상황(카메라 각도)에 따라 미세 조정 필요
ROI_TOP_WIDTH = 960
ROI_TOP_Y = 4013
ROI_BOTTOM_Y = 1057
L_MIN_WHITE = 169       # 흰색 인식 감도 (낮을수록 민감)

# 노출값 설정 (햇빛 대응용, 필요시 주석 해제 후 값 변경)
# CAM_EXPOSURE = -6 

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    # 테스트를 위해 exit() 보류 가능
    exit()

# ==========================================
# [3] 유틸리티 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
    
    # 이미지 높이 (Crop된 이미지 기준)
    y1 = image.shape[0]
    y2 = int(y1 * 0.6) # Look-ahead distance
    
    if slope == 0: slope = 0.001
    
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None: return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # 기울기 필터링 (수평선 제거)
            if abs(slope) < 0.4: continue 

            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def calculate_steering_angle(image, left_line, right_line):
    # 입력된 image는 Crop된 이미지임
    h, w = image.shape[:2]
    car_x = w / 2
    car_y = h

    if left_line is not None and right_line is not None:
        lx1, _, lx2, _ = left_line[0]
        rx1, _, rx2, _ = right_line[0]
        target_x = (lx2 + rx2) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (w * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (w * 0.25)
    else:
        target_x = car_x # 라인 없으면 직진

    dx = target_x - car_x
    dy = (h * 0.6) - car_y
    angle_radian = math.atan2(dx, abs(dy))
    angle_deg = math.degrees(angle_radian)
    return angle_deg, int(target_x)

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    # -------------------------------------------------------------
    # ★ [핵심] 선배님의 MJPG 강제 주입 (FPS 해결)
    # -------------------------------------------------------------
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.CAP_PROP_FRAME_WIDTH, 1920,
        cv2.CAP_PROP_FRAME_HEIGHT, 1080,
        cv2.CAP_PROP_FPS, 30
    ]
    
    print(f"📷 카메라 #{CAM_INDEX} (MJPG 모드) 연결 시도...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW, params)

    if not cap.isOpened():
        print("❌ 카메라 열기 실패!")
        return

    # 실제 적용된 코덱 확인
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"✅ 코덱: {codec} | 해상도: {cap.get(3)}x{cap.get(4)}")

    # 노출값 적용 (필요 시)
    # cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE) 

    print("🚀 실시간 고속 라인 트레이싱 시작")
    ser.write(f"D,{MAX_SPEED}\n".encode())

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 안전장치: 해상도가 안 맞으면 리사이즈 (MJPG 설정이 먹혔다면 불필요)
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))

        # ---------------------------------------------------------
        # ★ [핵심] Version 5의 ROI Crop 최적화
        # 전체 이미지를 처리하지 않고, 필요한 부분만 잘라내어 연산함
        # ---------------------------------------------------------
        crop_top = ROI_TOP_Y
        crop_bottom = ROI_BOTTOM_Y
        
        # proc: 실제 연산에 사용될 '잘린 이미지'
        proc = frame[crop_top:crop_bottom, :]
        proc_h, proc_w = proc.shape[:2]

        # 1. HLS & Masking
        hls = cv2.cvtColor(proc, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN_WHITE, 0], dtype=np.uint8)
        upper_white = np.array([179, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        # 2. Edge Detection
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask_white)

        # 3. ROI Masking (잘린 이미지 기준 좌표 사용)
        center_x = proc_w // 2
        roi_vertices = [
            (0, proc_h),
            (center_x - ROI_TOP_WIDTH, 0),
            (center_x + ROI_TOP_WIDTH, 0),
            (proc_w, proc_h)
        ]
        cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 4. Hough Transform
        lines = cv2.HoughLinesP(
            cropped_edges, 1, np.pi / 180, 50,
            minLineLength=80, # 노이즈 필터링 강화
            maxLineGap=100
        )
        
        # 5. Steering Calculation
        left_line, right_line = average_slope_intercept(proc, lines)
        steering_angle, target_x_proc = calculate_steering_angle(proc, left_line, right_line)

        # 6. Servo Control
        clamped_angle = max(-45, min(45, steering_angle))
        servo_value = map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX)
        servo_value = int(servo_value)

        ser.write(f"S,{servo_value}\n".encode())

        # ---------------------------------------------------------
        # 시각화 (Visualization) - 디버깅용 화면 만들기
        # ---------------------------------------------------------
        line_image_proc = np.zeros_like(proc)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line:
                cv2.line(line_image_proc, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line:
                cv2.line(line_image_proc, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # 잘린 이미지(proc) 위에 라인을 그림
        combo_proc = cv2.addWeighted(proc, 0.8, line_image_proc, 1, 1)
        
        # 타겟 포인트 표시
        cv2.line(combo_proc, (proc_w // 2, proc_h), (int(target_x_proc), int(proc_h * 0.6)), (0, 0, 255), 3)

        # 원본 프레임에 다시 붙여넣기 (사용자 확인용)
        display_frame = frame.copy()
        display_frame[crop_top:crop_bottom, :] = combo_proc

        # FPS 계산 및 정보 표시
        curr_time = time.time()
        dt = curr_time - prev_time
        fps = 1.0 / dt if dt > 0 else 0
        prev_time = curr_time

        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Angle: {steering_angle:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Servo: {servo_value}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Final Fast Lane Tracing', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 루틴
    print("🛑 프로그램 종료")
    ser.write(b"D,0\n")
    time.sleep(0.1)
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    
    ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()