import serial
from rplidar import RPLidar
import time
import cv2
import numpy as np
import math
from threading import Thread

# ==========================================
# [1] 설정값
# ==========================================
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'

# ★ 카메라 인덱스 (현장에 맞게 수정)
CAM_IDX_TRAFFIC = 0  # 신호등 (저화질)
CAM_IDX_LANE = 1     # 차선 (고화질)

# --- ★ 핵심: 해상도 차별화 ---
# 신호등용: 작고 빠르게
RES_TRAFFIC_W, RES_TRAFFIC_H = 320, 240
# 차선용: 크고 정밀하게
RES_LANE_W, RES_LANE_H = 640, 480

# --- ROI 설정 (비율로 설정하므로 해상도 바뀌어도 OK) ---
ROI_LANE_H_RATIO = 0.6
ROI_TL_H_RATIO = 0.6 

# --- 하드웨어 설정 ---
SPEED_NORMAL = 200
SPEED_SLOW = 100
SPEED_STOP = 0
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# --- 장애물 회피 ---
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.8 
SIDE_CLEAR_THRES = 600

# ==========================================
# [2] 스레딩 카메라 클래스 (해상도 개별 설정 가능)
# ==========================================
class WebcamVideoStream:
    def __init__(self, src=0, width=640, height=480, exposure=-6):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # 중요: 차선 카메라는 어둡게, 신호등은 밝게
        self.stream.set(cv2.CAP_PROP_EXPOSURE, exposure)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped: return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# [3] 유틸리티 함수
# ==========================================

# --- 신호등 인식 (320x240 전용) ---
lower_red1 = np.array([0, 150, 150]); upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150]); upper_red2 = np.array([179, 255, 255])
lower_green = np.array([40, 150, 150]); upper_green = np.array([90, 255, 255])

def detect_traffic_light_hybrid(frame):
    if frame is None: return "UNKNOWN"
    # 320x240 이미지이므로 ROI도 작게 잡힘
    h, w = frame.shape[:2]
    roi = frame[0:int(h*ROI_TL_H_RATIO), :] 
    
    # 가우시안 블러 (작은 이미지라 3x3이면 충분)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (3, 3), 0)

    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_g = cv2.inRange(hsv, lower_green, upper_green)

    count_r = cv2.countNonZero(mask_r)
    count_g = cv2.countNonZero(mask_g)
    
    # ★ 중요: 해상도가 320x240이므로 감지 면적 기준을 낮춰야 함 (20픽셀)
    MIN_PIXELS = 20
    
    if count_r > MIN_PIXELS and count_r > count_g: return "RED"
    if count_g > MIN_PIXELS and count_g > count_r: return "GREEN"
    return "UNKNOWN"

# --- 차선 인식 (640x480 전용) ---
def process_lane_lines(frame):
    h, w = frame.shape[:2] # 480, 640
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # 해상도 높으니 블러 좀 더 줌
    edges = cv2.Canny(blur, 50, 150)
    
    # ROI (사다리꼴) - 640x480 기준 좌표
    roi_points = np.array([[
        (0, h), (w, h), 
        (int(w*0.75), int(h*ROI_LANE_H_RATIO)), 
        (int(w*0.25), int(h*ROI_LANE_H_RATIO))
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_points, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=100)
    
    left_lines, right_lines = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            slope = (y2-y1)/(x2-x1)
            if slope < -0.5: left_lines.append(line[0])
            elif slope > 0.5: right_lines.append(line[0])
            
    return left_lines, right_lines

def calculate_steering(frame, lefts, rights, obstacle_dist, last_angle):
    h, w = frame.shape[:2] # 480, 640
    car_x = w / 2          # 320
    
    l_x = np.mean([l[2] for l in lefts]) if lefts else None
    r_x = np.mean([r[2] for r in rights]) if rights else None
    
    if l_x and r_x: target = (l_x + r_x) / 2
    elif l_x: target = l_x + (w * 0.25) # 왼쪽만 보이면 폭의 1/4만큼 오른쪽으로
    elif r_x: target = r_x - (w * 0.25)
    else: return last_angle, int(car_x)
    
    # 장애물 회피 (해상도에 맞춰 Shift 양 조절)
    shift = 0
    if 0 < obstacle_dist < OBSTACLE_START_DIST:
        # 해상도가 커졌으니 Gain도 픽셀 수에 맞춰야 함
        # 예: 640폭에서는 100~200픽셀 이동이 적당
        shift = (OBSTACLE_START_DIST - obstacle_dist) * (SHIFT_GAIN * 1.5) 
        target -= shift
        
    # 조향각 계산 (h * 0.6 위치를 바라봄)
    dy = h * 0.4 
    angle = math.degrees(math.atan2(target - car_x, dy))
    return angle, int(target)

def map_servo(angle):
    angle = max(-45, min(45, angle))
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)

# 초음파
us_right = 9999
def read_sonar():
    global us_right
    if ser and ser.in_waiting:
        try:
            line = ser.readline().decode().strip()
            if line.startswith("US:"):
                parts = line.split(",")
                if len(parts) >= 6: us_right = int(parts[5].strip()) # RM
        except: pass

# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    global ser, us_right
    
    try:
        ser = serial.Serial(ARDUINO_PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        
        # ★★★ 핵심 변경 사항 ★★★
        print("📷 카메라 스레드 초기화...")
        # 1. 신호등용: 320x240, 밝게 (-5)
        cam_traffic = WebcamVideoStream(src=CAM_IDX_TRAFFIC, width=RES_TRAFFIC_W, height=RES_TRAFFIC_H, exposure=-5).start()
        # 2. 주행용: 640x480, 어둡게 (-6 or -7)
        cam_lane = WebcamVideoStream(src=CAM_IDX_LANE, width=RES_LANE_W, height=RES_LANE_H, exposure=-6).start()
        
        time.sleep(2)
        print("✅ 하이브리드 해상도 시스템 준비 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        return

    last_angle = 0
    frame_count = 0
    traffic_state = "UNKNOWN"
    prev_time = time.time()

    try:
        for scan in lidar.iter_scans():
            read_sonar()
            
            raw_dist = 2000
            for (_, angle, dist) in scan:
                if 200 < dist < 1200 and (angle > 330 or angle < 30):
                    if dist < raw_dist: raw_dist = dist
            
            # 각각 다른 해상도의 프레임 가져오기
            frame_t = cam_traffic.read() # 320x240
            frame_l = cam_lane.read()    # 640x480
            
            frame_count += 1
            
            # [최적화] 신호등은 5프레임마다 1번 처리 (작은 이미지라 더 빠름)
            if frame_count % 5 == 0:
                traffic_state = detect_traffic_light_hybrid(frame_t)
            
            # [주행] 차선은 매번 고해상도로 처리
            lefts, rights = process_lane_lines(frame_l)
            angle, target_x = calculate_steering(frame_l, lefts, rights, raw_dist, last_angle)
            last_angle = angle
            
            # 속도 결정
            speed = SPEED_NORMAL
            if traffic_state == "RED":
                speed = SPEED_STOP
                print("🛑 RED LIGHT")
            elif raw_dist < 600:
                speed = SPEED_SLOW
            
            # 측면 안전장치
            if raw_dist > 1000 and us_right < SIDE_CLEAR_THRES:
                 angle = -20 # 복귀 방지
            
            # 아두이노 전송
            pwm = map_servo(angle)
            if ser:
                ser.write(f"S,{pwm}\n".encode())
                if frame_count % 3 == 0: 
                    ser.write(f"D,{speed}\n".encode())

            # [디버깅]
            if frame_count % 2 == 0:
                # FPS 표시
                curr_fps = 1/(time.time()-prev_time)
                cv2.putText(frame_l, f"FPS: {curr_fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame_l, f"TL: {traffic_state}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                # 타겟 포인트 (중앙은 320)
                cv2.circle(frame_l, (target_x, int(RES_LANE_H * 0.6)), 8, (0,0,255), -1)
                cv2.line(frame_l, (int(RES_LANE_W/2), RES_LANE_H), (target_x, int(RES_LANE_H * 0.6)), (255,0,0), 2)
                
                # 신호등 이미지는 작으니까 주행화면 구석에 붙여서 보여주기 (PIP 기능)
                # 320x240 -> 160x120으로 줄여서 우측 상단에 붙임
                mini_traffic = cv2.resize(frame_t, (160, 120))
                frame_l[0:120, RES_LANE_W-160:RES_LANE_W] = mini_traffic
                cv2.rectangle(frame_l, (RES_LANE_W-160, 0), (RES_LANE_W, 120), (255,255,0), 2)

                cv2.imshow("Main Driving View (640p)", frame_l)
                if cv2.waitKey(1) == ord('q'): break
            
            prev_time = time.time()

    except KeyboardInterrupt: pass
    finally:
        cam_traffic.stop(); cam_lane.stop()
        if lidar: lidar.stop(); lidar.disconnect()
        if ser: ser.write(b"D,0\n"); ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()