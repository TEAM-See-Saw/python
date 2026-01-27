import cv2
import numpy as np
import math
import serial
import time
import threading
import Function_Library as LiDAR_Lib

# ==========================================
# [1] 통합 환경 설정
# ==========================================
MOTOR_PORT = 'COM4'   # 아두이노 포트
LIDAR_PORT = 'COM3'   # 라이다 포트
BAUDRATE = 9600

MAX_SPEED = 255       # 평상시 속도
AVOID_SPEED = 160     # 회피 시 속도 (안전을 위해 감속)
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

AVOID_DISTANCE = 500  # 장애물 감지 거리 (mm)
DETECT_ANGLE = 30     # 전방 감시 각도

IS_SUNNY = True       # 조명 모드

# ==========================================
# [2] 전역 변수
# ==========================================
is_running = True
obstacle_detected = False

# ==========================================
# [3] LiDAR 스레드 (변경 없음)
# ==========================================
def lidar_thread_func():
    global obstacle_detected, is_running
    try:
        lidar = LiDAR_Lib.libLIDAR(LIDAR_PORT)
        lidar.init()
        print("✅ LiDAR 가동 시작")
    except:
        print("❌ LiDAR 연결 실패")
        return

    for scan in lidar.scanning():
        if not is_running: break
        valid_scan = scan[scan[:, 1] > 0]
        front_obstacles = valid_scan[
            ((valid_scan[:, 0] < DETECT_ANGLE) | (valid_scan[:, 0] > (360 - DETECT_ANGLE))) &
            (valid_scan[:, 1] < AVOID_DISTANCE)
        ]
        obstacle_detected = True if len(front_obstacles) > 0 else False
    lidar.stop()

# ==========================================
# [4] 영상 처리 함수 (수정됨: 좌표 반환 추가)
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
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
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
            if slope < -0.5: left_fit.append((slope, fit[1]))
            elif slope > 0.5: right_fit.append((slope, fit[1]))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

# [수정된 함수] 각도뿐만 아니라, 양쪽 차선의 X좌표 위치도 반환합니다.
def get_lane_info(image, left_line, right_line):
    height, width, _ = image.shape
    car_center = width / 2
    
    # 1. 차선 위치 파악 (없으면 가상의 끝점 부여)
    # 왼쪽 차선이 없으면 0(맨 왼쪽), 오른쪽이 없으면 width(맨 오른쪽)으로 가정
    left_x = left_line[0][2] if left_line is not None else 0
    right_x = right_line[0][2] if right_line is not None else width
    
    # 2. 목표 지점 계산 (기본 주행용)
    if left_line is not None and right_line is not None:
        target_x = (left_x + right_x) / 2
    elif left_line is not None:
        target_x = left_x + (width * 0.25)
    elif right_line is not None:
        target_x = right_x - (width * 0.25)
    else:
        target_x = car_center

    # 3. 조향각 계산
    dx = target_x - car_center
    dy = (height * 0.6) - height
    angle = math.degrees(math.atan2(dx, abs(dy)))
    
    return angle, left_x, right_x, car_center

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [5] 메인 실행 함수
# ==========================================
def main():
    global is_running, obstacle_detected
    
    # 카메라 설정
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 밝기 설정
    if IS_SUNNY:
        cap.set(cv2.CAP_PROP_EXPOSURE, -9)
        L_MIN, S_MAX, MORPH_SIZE = 160, 50, (5, 5)
    else:
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        L_MIN, S_MAX, MORPH_SIZE = 100, 60, (3, 3)

    # 아두이노 연결
    ser = None
    try:
        ser = serial.Serial(MOTOR_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
    except: pass

    # LiDAR 스레드 시작
    t_lidar = threading.Thread(target=lidar_thread_func)
    t_lidar.start()

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    print("🚀 주행 시작")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (width, height))
        
        # --- 영상 처리 ---
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, L_MIN, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        edges = cv2.Canny(mask, 50, 150)
        
        roi_vertices = [(0, height), (width//2 - 50, int(height*0.6)), (width//2 + 50, int(height*0.6)), (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))
        
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)
        
        # [중요] 차선 정보 받아오기 (각도, 왼쪽좌표, 오른쪽좌표, 차량중심)
        lane_angle, left_x, right_x, car_center = get_lane_info(frame, left, right)

        # --- 제어 로직 ---
        final_servo = SERVO_CENTER
        final_speed = MAX_SPEED
        status = "NORMAL"
        
        # 🚨 상황 1: 장애물 감지됨 -> "공간 판단" 회피
        if obstacle_detected:
            final_speed = AVOID_SPEED # 감속
            
            # 내 차 기준 여유 공간 계산
            space_left = car_center - left_x   # 왼쪽 공간 크기
            space_right = right_x - car_center # 오른쪽 공간 크기
            
            # 더 넓은 쪽으로 회피 (가중치 적용 가능)
            if space_left > space_right:
                status = "AVOID -> LEFT (Wide)"
                # 왼쪽으로 꺾되, 차선을 벗어나지 않게 너무 과하게 꺾진 않음 (45도 대신 30도 정도 권장)
                avoid_angle = -35 
            else:
                status = "AVOID -> RIGHT (Wide)"
                avoid_angle = 35

            # 각도를 서보 값으로 매핑
            final_servo = int(map_value(avoid_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 🚗 상황 2: 장애물 없음 -> "차선 유지" 주행
        else:
            status = "LANE KEEPING"
            final_servo = int(map_value(max(-45, min(45, lane_angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 아두이노 전송
        if ser:
            ser.write(f"S,{final_servo}\n".encode())
            ser.write(f"D,{final_speed}\n".encode())

        # --- 디버깅 화면 ---
        if left is not None: cv2.line(frame, (left[0][0], left[0][1]), (left[0][2], left[0][3]), (0, 0, 255), 5)
        if right is not None: cv2.line(frame, (right[0][0], right[0][1]), (right[0][2], right[0][3]), (0, 0, 255), 5)
        
        cv2.putText(frame, f"Mode: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"L_Space: {int(car_center - left_x)} | R_Space: {int(right_x - car_center)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Smart Drive", frame)
        if cv2.waitKey(1) == ord('q'):
            is_running = False
            break

    # 종료
    if ser:
        ser.write(b"D,0\n")
        ser.write(b"S,570\n")
        ser.close()
    t_lidar.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()