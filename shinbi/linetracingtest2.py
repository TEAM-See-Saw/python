import cv2
import numpy as np
import math
import serial
import time
import datetime
import os
import threading

# ==========================================
# [1] 환경 설정 (사용자 환경에 맞게 수정)
# ==========================================
IS_SUNNY = True  # True: 햇빛 쨍쨍 / False: 실내, 흐림

# ⚙️ 포트 설정 (장치 관리자 확인 필수!)
PORT_ARDUINO = 'COM4'   # 아두이노
PORT_LIDAR   = 'COM5'   # 라이다 (없으면 None으로 두면 무시됨)
BAUDRATE     = 9600
CAM_INDEX    = 0

# ⚙️ 주행 값 설정
MAX_SPEED      = 120   # 평소 주행 속도 (안전을 위해 조금 줄임)
SERVO_CENTER   = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX= 480

# ⚙️ 미션 임계값
LIDAR_STOP_DIST = 600  # 600mm(60cm) 이내 장애물 감지 시 정지
TRAFFIC_ROI_H   = 0.4  # 화면 상단 40%만 신호등으로 인식

# ==========================================
# [2] 라이브러리 로드 (RPLidar)
# ==========================================
try:
    from rplidar import RPLidar
    HAS_LIDAR = True
except ImportError:
    print("⚠️ 'rplidar' 라이브러리가 없습니다. (pip install rplidar-types)")
    HAS_LIDAR = False

# ==========================================
# [3] 모드별 자동 튜닝값
# ==========================================
if IS_SUNNY:
    print(f"☀️ [모드: SUNNY] 강력한 햇빛 대응 설정")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print(f"🌙 [모드: NORMAL] 실내/저녁 설정")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)

# ==========================================
# [4] 공유 변수 및 스레드 (라이다, 신호등)
# ==========================================
shared_data = {
    "front_dist": 9999,  # 전방 장애물 거리 (mm)
    "traffic": "NONE",   # 신호등 상태: NONE, RED, GREEN
    "running": True      # 프로그램 종료 플래그
}

def thread_lidar():
    """ 라이다 센서 값을 백그라운드에서 계속 읽어옴 """
    if not HAS_LIDAR: return
    
    lidar = RPLidar(PORT_LIDAR)
    print(f"📡 라이다 연결 성공: {PORT_LIDAR}")
    
    try:
        for scan in lidar.iter_scans():
            if not shared_data["running"]: break
            
            min_dist = 9999
            for (_, angle, dist) in scan:
                # 전방 30도 부채꼴 (0~15도, 345~360도)
                if (angle < 15 or angle > 345) and dist > 0:
                    if dist < min_dist:
                        min_dist = dist
            
            shared_data["front_dist"] = min_dist
    except Exception as e:
        print(f"❌ 라이다 오류: {e}")
    finally:
        lidar.stop()
        lidar.disconnect()

def detect_traffic_light(frame):
    """ 화면 상단에서 신호등 색상 인식 """
    h, w = frame.shape[:2]
    # 화면 상단 40%만 잘라서 확인 (신호등은 위에 있으므로)
    roi = frame[0:int(h * TRAFFIC_ROI_H), :]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 빨간색 범위 (두 개로 나뉨)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 초록색 범위
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # 픽셀 수 카운트 (노이즈 제거를 위해 일정 개수 이상이어야 인식)
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    threshold = 200 # 인식 최소 픽셀 수
    
    status = "NONE"
    if red_pixels > threshold and red_pixels > green_pixels:
        status = "RED"
    elif green_pixels > threshold and green_pixels > red_pixels:
        status = "GREEN"
        
    return status, roi

# ==========================================
# [5] 기존 영상 처리 함수 (라인트레이싱)
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
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
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
        target_x = car_position_x

    dx = target_x - car_position_x
    dy = (height * 0.6) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [6] 메인 실행 함수
# ==========================================
def main():
    # 1. 아두이노 연결
    ser = None
    try:
        ser = serial.Serial(PORT_ARDUINO, BAUDRATE, timeout=1)
        print(f"✅ 아두이노 연결 성공 ({PORT_ARDUINO})")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 아두이노 연결 실패: {e}")

    # 2. 카메라 설정
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE) # 밝기 설정

    if not cap.isOpened():
        print("❌ 카메라 실패")
        return

    # 3. 녹화 설정
    if not os.path.exists('dataset'): os.makedirs('dataset')
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(f"dataset/mission_{now}.mp4", 
                          cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width*2, height))

    # 4. 라이다 스레드 시작
    if HAS_LIDAR:
        t_lidar = threading.Thread(target=thread_lidar)
        t_lidar.start()

    print(f"🚀 미션 주행 시작! 속도: {MAX_SPEED}")
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    # ==========================
    # 메인 루프
    # ==========================
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 리사이즈 안전장치
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))

            # ---------------------------
            # [A] 센서 판단 (신호등 & 장애물)
            # ---------------------------
            
            # 1. 신호등 인식
            traffic_status, roi_frame = detect_traffic_light(frame)
            shared_data["traffic"] = traffic_status
            
            # 2. 장애물 거리 읽기
            lidar_d = shared_data["front_dist"]
            
            # 3. 주행 상태 결정
            # 우선순위: 빨간불(정지) > 장애물(정지) > 녹색불/없음(주행)
            
            current_speed = MAX_SPEED
            status_msg = "DRIVE"
            
            if traffic_status == "RED":
                current_speed = 0
                status_msg = "🔴 RED LIGHT"
                
            elif lidar_d < LIDAR_STOP_DIST:
                current_speed = 0
                status_msg = f"⚠️ OBSTACLE ({int(lidar_d)}mm)"
                
            else:
                # 정상 주행
                status_msg = "🟢 GO"
            
            # ---------------------------
            # [B] 라인트레이싱 (영상처리)
            # ---------------------------
            # 기존 코드 로직 그대로 사용
            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            lower_white = np.array([0, L_MIN, 0])
            upper_white = np.array([179, 255, S_MAX])
            mask = cv2.inRange(hls, lower_white, upper_white)
            
            kernel = np.ones(MORPH_SIZE, np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            edges = cv2.Canny(mask, 50, 150)
            
            roi_v = [(0, height), (width//2 - 50, int(height*0.6)), 
                     (width//2 + 50, int(height*0.6)), (width, height)]
            cropped = region_of_interest(edges, np.array([roi_v], np.int32))
            
            lines = cv2.HoughLinesP(cropped, 1, np.pi/180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)
            
            # 조향 계산
            angle, target = calculate_steering_angle(frame, left, right)
            
            # 서보값 매핑
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            # ---------------------------
            # [C] 모터 제어 명령 전송
            # ---------------------------
            if ser:
                # 상태에 따라 속도 0 또는 설정 속도 전송
                ser.write(f"S,{servo_val}\n".encode())
                ser.write(f"D,{current_speed}\n".encode())

            # ---------------------------
            # [D] 디버깅 화면
            # ---------------------------
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # 화면에 정보 표시
            cv2.putText(frame, f"State: {status_msg}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, f"Lidar: {int(lidar_d)}mm", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Traffic: {traffic_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            
            # 신호등 ROI 영역 표시 (노란 박스)
            cv2.rectangle(frame, (0, 0), (width, int(height*TRAFFIC_ROI_H)), (0, 255, 255), 2)
            
            combined = np.hstack((frame, mask_bgr))
            out.write(combined)
            cv2.imshow("Mission Drive", combined)
            # 신호등 인식 확인용 (작은 창)
            # cv2.imshow("Traffic ROI", roi_frame) 

            if cv2.waitKey(1) == ord('q'):
                print("🛑 사용자 중지")
                break

    finally:
        shared_data["running"] = False
        if HAS_LIDAR: t_lidar.join()
        
        if ser:
            ser.write(b"D,0\n")
            ser.write(b"S,570\n")
            ser.close()
            
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("시스템 종료")

if __name__ == "__main__":
    main()