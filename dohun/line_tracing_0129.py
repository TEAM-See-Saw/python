import cv2
import numpy as np
import math
import serial
import time
import datetime
import os

# ==========================================
# [1] 환경 설정
# ==========================================
PORT = 'COM4'  # 아두이노 포트
BAUDRATE = 9600
CAM_INDEX = 0

MAX_SPEED = 230  # 주행 속도 (안정성을 위해 약간 하향)
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 마지막 성공적인 조향값 기억 (차선 상실 대비)
last_servo_val = SERVO_CENTER

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 연결 성공")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")

# ==========================================
# [3] 영상 처리 핵심 함수들
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    # 분모가 0이 되는 것 방지
    curr_slope = slope if abs(slope) > 0.001 else 0.001
    x1 = int((y1 - intercept) / curr_slope)
    x2 = int((y2 - intercept) / curr_slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None: return None, None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # 기울기 기준 완화 (차선이 누워있을 때 대비)
            if slope < -0.3:
                left_fit.append((slope, intercept))
            elif slope > 0.3:
                right_fit.append((slope, intercept))
                
    left_line = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def calculate_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    car_position_x = width / 2

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        return None, int(car_position_x)

    dx = target_x - car_position_x
    dy = (height * 0.6) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [4] 메인 실행 루프
# ==========================================
def main():
    global last_servo_val
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # 노출을 -6 정도로 낮춰 빛 번짐 최소화
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("❌ 카메라 연결 확인 필요")
        return

    # 출발 신호
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())
        print("🚀 주행 시작!")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 동적 밝기 분석 (화면 하단 차선 영역 위주)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        avg_brightness = np.mean(l_channel[int(height*0.6):, :])
        
        # [핵심] 조명에 따른 동적 임계값 설정
        # 너무 밝으면 기준을 높이고, 어두우면 낮춤
        dynamic_l_min = int(avg_brightness + 30)
        dynamic_l_min = max(90, min(dynamic_l_min, 210)) # 범위 제한

        # 2. 흰색 필터링 (채도 S를 낮게 잡아 색상 노이즈 차단)
        lower_white = np.array([0, dynamic_l_min, 0])
        upper_white = np.array([180, 255, 60])
        mask = cv2.inRange(hls, lower_white, upper_white)

        # 3. 노이즈 제거 (점 형태로 나오는 현상 방지)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 끊긴 선 연결
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 자잘한 점 제거

        # 4. 엣지 및 ROI
        edges = cv2.Canny(mask, 50, 150)
        roi_vertices = [(0, height), (width//2-60, int(height*0.6)), 
                        (width//2+60, int(height*0.6)), (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 5. 차선 검출 (파라미터 완화: 선이 끊겨보여도 잘 잡도록)
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 25, minLineLength=20, maxLineGap=80)
        left, right = average_slope_intercept(frame, lines)

        # 6. 조향 계산 및 보정 (Fail-safe)
        angle, target = calculate_steering_angle(frame, left, right)
        
        if angle is not None:
            # 급격한 핸들 꺾임 방지 (최대 40도 제한)
            safe_angle = max(-40, min(40, angle))
            servo_val = int(map_value(safe_angle, -40, 40, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
            last_servo_val = servo_val
        else:
            # 차선을 놓치면 이전 조향 유지 (급정거 방지)
            servo_val = last_servo_val
            cv2.putText(frame, "LANE LOST - HOLDING", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if ser:
            ser.write(f"S,{servo_val}\n".encode())

        # 디스플레이 (상태 확인용)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))
        cv2.putText(combined, f"L_MIN: {dynamic_l_min} | Bright: {int(avg_brightness)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Lane Tracking", combined)

        if cv2.waitKey(1) == ord('q'):
            break

    # 종료
    if ser:
        ser.write(b"D,0\n")
        ser.write(f"S,{SERVO_CENTER}\n".encode())
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()