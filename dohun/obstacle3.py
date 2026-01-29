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
MOTOR_PORT = 'COM4'   
LIDAR_PORT = 'COM3'   
BAUDRATE = 9600

MAX_SPEED = 255       
AVOID_SPEED = 160     
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

AVOID_DISTANCE = 400  # mm 단위 (너무 가까우면 늦으니 40cm 정도로 여유 확보)
DETECT_ANGLE = 60     # 전방 감시 각도 (좌우 60도씩)

IS_SUNNY = False       

# ==========================================
# [2] 전역 변수 (수정됨)
# ==========================================
is_running = True
# 장애물 위치 상태: "NONE", "LEFT", "RIGHT", "CENTER"
obstacle_status = "NONE" 

# ==========================================
# [3] LiDAR 스레드 (핵심 수정 구역)
# ==========================================
def lidar_thread_func():
    global obstacle_status, is_running
    try:
        lidar = LiDAR_Lib.libLIDAR(LIDAR_PORT)
        lidar.init()
        print("✅ LiDAR 가동 시작")
    except:
        print("❌ LiDAR 연결 실패")
        return

    for scan in lidar.scanning():
        if not is_running: break
        
        # 유효 데이터 필터링 (거리 > 0)
        valid_scan = scan[scan[:, 1] > 0]
        
        # [수정] 좌측/우측 영역 분리 감지
        # 보통 LiDAR는 0도가 정면, 0~N도가 좌측(또는 우측), 360-N~360도가 반대측
        # 여기서는 0~DETECT_ANGLE을 'Left_Side', 360-DETECT_ANGLE~360을 'Right_Side'로 가정
        
        # 1. 거리 내에 있는 점들만 추출
        nearby_points = valid_scan[valid_scan[:, 1] < AVOID_DISTANCE]
        
        if len(nearby_points) == 0:
            obstacle_status = "NONE"
            continue

        # 2. 좌/우 포인트 개수 세기
        # 좌측 영역 (0 ~ 60도)
        left_points = nearby_points[
            (nearby_points[:, 0] >= 0) & (nearby_points[:, 0] < DETECT_ANGLE)
        ]
        # 우측 영역 (300 ~ 360도)
        right_points = nearby_points[
            (nearby_points[:, 0] > (360 - DETECT_ANGLE)) & (nearby_points[:, 0] <= 360)
        ]
        
        l_count = len(left_points)
        r_count = len(right_points)

        # 3. 위치 판단 로직
        if l_count == 0 and r_count == 0:
            obstacle_status = "NONE"
        elif l_count > r_count * 1.5:  # 왼쪽이 훨씬 많으면
            obstacle_status = "LEFT"
        elif r_count > l_count * 1.5:  # 오른쪽이 훨씬 많으면
            obstacle_status = "RIGHT"
        else:                          # 비슷하면 정면으로 간주
            obstacle_status = "CENTER"
            
    lidar.stop()

# ==========================================
# [4] 영상 처리 함수 (기존 유지)
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

def get_lane_info(image, left_line, right_line):
    height, width, _ = image.shape
    car_center = width / 2
    
    left_x = left_line[0][2] if left_line is not None else 0
    right_x = right_line[0][2] if right_line is not None else width
    
    if left_line is not None and right_line is not None:
        target_x = (left_x + right_x) / 2
    elif left_line is not None:
        target_x = left_x + (width * 0.25)
    elif right_line is not None:
        target_x = right_x - (width * 0.25)
    else:
        target_x = car_center

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
    global is_running, obstacle_status
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if IS_SUNNY:
        cap.set(cv2.CAP_PROP_EXPOSURE, -9)
        L_MIN, S_MAX, MORPH_SIZE = 160, 50, (5, 5)
    else:
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        L_MIN, S_MAX, MORPH_SIZE = 100, 60, (3, 3)

    ser = None
    try:
        ser = serial.Serial(MOTOR_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
    except: pass

    t_lidar = threading.Thread(target=lidar_thread_func)
    t_lidar.start()

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    print("🚀 주행 시작")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (width, height))
        
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, L_MIN, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        edges = cv2.Canny(mask, 50, 150)
        
        roi_vertices = [(0, height), (width//2 - 50, int(height*0.6)), (width//2 + 50, int(height*0.6)), (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))
        
        lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)
        lane_angle, left_x, right_x, car_center = get_lane_info(frame, left, right)

        # --- 제어 로직 (수정됨) ---
        final_servo = SERVO_CENTER
        final_speed = MAX_SPEED
        status_msg = f"Normal | Obs: {obstacle_status}"
        
        # 🚨 상황 1: 장애물 감지 -> 위치에 따른 회피
        if obstacle_status != "NONE":
            final_speed = AVOID_SPEED
            
            # [전략 1] 장애물이 왼쪽에 있으면 -> 무조건 오른쪽으로
            if obstacle_status == "LEFT":
                status_msg = "AVOID -> RIGHT (Obs Left)"
                avoid_angle = 35  # 우회전
                
            # [전략 2] 장애물이 오른쪽에 있으면 -> 무조건 왼쪽으로
            elif obstacle_status == "RIGHT":
                status_msg = "AVOID -> LEFT (Obs Right)"
                avoid_angle = -35 # 좌회전
                
            # [전략 3] 장애물이 정중앙에 있으면 -> 기존처럼 넓은 공간으로
            else: # CENTER
                space_left = car_center - left_x
                space_right = right_x - car_center
                if space_left > space_right:
                    status_msg = "AVOID -> LEFT (Wide Space)"
                    avoid_angle = -35
                else:
                    status_msg = "AVOID -> RIGHT (Wide Space)"
                    avoid_angle = 35
            
            # 계산된 회피 각도 적용
            final_servo = int(map_value(avoid_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 🚗 상황 2: 장애물 없음 -> 차선 유지
        else:
            status_msg = "LANE KEEPING"
            final_servo = int(map_value(max(-45, min(45, lane_angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        if ser:
            ser.write(f"S,{final_servo}\n".encode())
            ser.write(f"D,{final_speed}\n".encode())

        # --- 디버깅 ---
        if left is not None: cv2.line(frame, (left[0][0], left[0][1]), (left[0][2], left[0][3]), (0, 0, 255), 5)
        if right is not None: cv2.line(frame, (right[0][0], right[0][1]), (right[0][2], right[0][3]), (0, 0, 255), 5)
        
        # 장애물 상태 색상 표시 (빨강:감지 / 초록:안전)
        obs_color = (0, 0, 255) if obstacle_status != "NONE" else (0, 255, 0)
        cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, obs_color, 2)
        
        cv2.imshow("Smart Drive", frame)
        if cv2.waitKey(1) == ord('q'):
            is_running = False
            break

    if ser:
        ser.write(b"D,0\n")
        ser.write(b"S,570\n")
        ser.close()
    t_lidar.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()