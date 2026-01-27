import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar

# ==========================================
# [1] 설정 (팀장님 튜닝값 + 미션 설정)
# ==========================================
# 1-1. 통신 포트
PORT = 'COM4'           # 아두이노
LIDAR_PORT = 'COM3'     # 라이다
BAUDRATE = 9600

# 1-2. 팀장님 코드 튜닝값 (fine_tuned_line_tracing2.py 기반)
ROI_TOP_WIDTH = 960
ROI_TOP_Y = 401
ROI_BOTTOM_Y = 1057
L_MIN_WHITE = 169
CAM_EXPOSURE = -6       # (환경에 맞춰 조절)

# 1-3. 미션 파라미터
NORMAL_SPEED = 150      # 평상시 속도
AVOID_SPEED = 130       # 회피 시 감속
OBS_DIST_THRES = 600    # 전방 장애물 감지 (60cm)
SIDE_CLEAR_THRES = 400  # 측면 안전 거리 (40cm 이상이면 빈 공간)
OFFSET_VAL = 150        # 차선 변경 시 목표점 이동량
LANE_CHANGE_TIME = 2.0  # 차선 변경 시간 (보조용)

# 1-4. 서보/모터 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 1-5. 상태 정의
STATE_NORMAL = 0      # 2차선 주행
STATE_DETECTED = 1    # 감지됨 -> 1차선 변경 중
STATE_AVOIDING = 2    # 1차선 주행 (장애물 옆 통과 중)
STATE_RETURNING = 3   # 장애물 통과 -> 2차선 복귀 중
STATE_FINISHED = 4    # 미션 끝

# ==========================================
# [2] 하드웨어 연결
# ==========================================
print("🔄 하드웨어 연결 중...")
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    lidar = RPLidar(LIDAR_PORT)
    print("✅ 아두이노 & 라이다 연결 성공")
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    ser = None
    lidar = None

# 카메라 설정 (MJPG 모드)
params = [
    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
    cv2.CAP_PROP_FRAME_WIDTH, 1920,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    cv2.CAP_PROP_FPS, 30,
    cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 
]
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW, params)
cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE)

if not cap.isOpened():
    print("❌ 카메라 연결 실패")
    exit()

# ==========================================
# [3] 유틸리티 함수 (팀장님 로직)
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
            if slope < -0.5: left_fit.append((slope, intercept))
            elif slope > 0.5: right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def calculate_steering_angle_with_offset(image, left_line, right_line, offset=0):
    """ 기본 조향각 계산에 Offset(회피용 편향) 추가 """
    height, width, _ = image.shape
    car_x = width / 2
    
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = car_x 

    final_target_x = target_x + offset # ★ 핵심: 오프셋 적용

    dx = final_target_x - car_x
    dy = (height * 0.6) - height
    angle = math.degrees(math.atan2(dx, abs(dy)))
    return angle, int(final_target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [4] 센서 및 시각화 함수
# ==========================================
us_left = 9999
us_right = 9999

def read_sonar_sensors():
    global us_left, us_right
    if ser is None: return
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 2:
                    us_left = int(parts[0])
                    us_right = int(parts[1])
        except: pass

def draw_lidar_map(scan_data, max_dist=2000):
    img_size = 500
    lidar_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    center_x, center_y = img_size // 2, img_size // 2
    scale = (img_size / 2) / max_dist 

    # 내 차 (중앙)
    cv2.circle(lidar_img, (center_x, center_y), 5, (0, 0, 255), -1) 
    cv2.arrowedLine(lidar_img, (center_x, center_y), (center_x, center_y - 20), (0,0,255), 2)

    # 데이터 점 찍기
    for (_, angle, dist) in scan_data:
        if dist > 0 and dist < max_dist:
            # 180도가 정면이므로 -270도 회전하여 화면 위쪽이 정면이 되게 함
            theta = math.radians(angle - 270)
            x = int(center_x + dist * scale * math.cos(theta))
            y = int(center_y + dist * scale * math.sin(theta))
            if 0 <= x < img_size and 0 <= y < img_size:
                cv2.circle(lidar_img, (x, y), 2, (255, 255, 255), -1)

    # 전방 60cm 감지 구역 표시 (초록색 아크)
    cv2.ellipse(lidar_img, (center_x, center_y), 
                (int(OBS_DIST_THRES*scale), int(OBS_DIST_THRES*scale)), 
                -90, -10, 10, (0, 255, 0), 1) 
    
    cv2.putText(lidar_img, f"Danger Zone: {OBS_DIST_THRES}mm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return lidar_img

# ==========================================
# [5] 메인 루프
# ==========================================
def main():
    state = STATE_NORMAL
    state_timer = 0
    lidar_iter = lidar.iter_scans() if lidar else None
    
    print("🚀 통합 시스템 시작 (Line Tracing + Obstacle Avoidance + Visualization)")
    if ser: ser.write(f"D,{NORMAL_SPEED}\n".encode())

    while True:
        # [A] 데이터 수집
        ret, frame = cap.read()
        if not ret: break
        read_sonar_sensors()
        
        # 라이다 데이터 (시각화용 전체 + 전방 거리용)
        scan_data = []
        front_dist = 9999
        if lidar_iter:
            try:
                scan_data = next(lidar_iter)
                for (_, angle, dist) in scan_data:
                    # 전방 20도 (170~190) 거리 추출
                    if (angle >= 170 and angle <= 190) and dist > 0:
                        if dist < front_dist: front_dist = dist
            except: pass

        # [B] 상태 머신 (장애물 회피 로직)
        offset = 0
        speed = NORMAL_SPEED
        curr_time = time.time()
        
        if state == STATE_NORMAL:
            if front_dist < OBS_DIST_THRES:
                print(f"🚨 장애물 감지({int(front_dist)}mm) -> 회피")
                state = STATE_DETECTED
                state_timer = curr_time

        elif state == STATE_DETECTED:
            offset = -OFFSET_VAL # 좌측 이동
            speed = AVOID_SPEED
            if curr_time - state_timer > LANE_CHANGE_TIME:
                state = STATE_AVOIDING

        elif state == STATE_AVOIDING:
            offset = -OFFSET_VAL
            # 우측 초음파로 통과 확인
            if us_right > SIDE_CLEAR_THRES and front_dist > 800:
                print(f"✅ 통과 확인(Side Clear) -> 복귀")
                state = STATE_RETURNING
                state_timer = curr_time

        elif state == STATE_RETURNING:
            offset = +OFFSET_VAL # 우측 이동
            speed = AVOID_SPEED
            if curr_time - state_timer > LANE_CHANGE_TIME:
                state = STATE_FINISHED

        elif state == STATE_FINISHED:
            offset = 0
            speed = NORMAL_SPEED

        # [C] 영상 처리 및 제어 (팀장님 코드 기반)
        crop_top, crop_bottom = ROI_TOP_Y, ROI_BOTTOM_Y
        proc = frame[crop_top:crop_bottom, :]
        
        hls = cv2.cvtColor(proc, cv2.COLOR_BGR2HLS)
        mask_white = cv2.inRange(hls, np.array([0, L_MIN_WHITE, 0]), np.array([179, 255, 255]))
        
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask_white)
        
        h, w = proc.shape[:2]
        center_x = w // 2
        roi_verts = np.array([[(0, h), (center_x - 300, 0), (center_x + 300, 0), (w, h)]], dtype=np.int32)
        masked_edges = region_of_interest(edges, roi_verts)
        
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, minLineLength=60, maxLineGap=100)
        left_line, right_line = average_slope_intercept(proc, lines)
        
        # 오프셋 적용하여 최종 조향각 계산
        angle, target_x = calculate_steering_angle_with_offset(proc, left_line, right_line, offset)
        
        clamped_angle = max(-45, min(45, angle))
        servo_val = int(map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
        
        if ser:
            ser.write(f"S,{servo_val}\n".encode())
            ser.write(f"D,{speed}\n".encode())

        # [D] 시각화 (Dual Display)
        # 1. 카메라 화면
        line_img = np.zeros_like(proc)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line: cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line: cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        combo = cv2.addWeighted(proc, 0.8, line_img, 1, 1)
        cv2.line(combo, (w//2, h), (target_x, int(h*0.6)), (0, 0, 255), 3) # 타겟 라인
        
        display_frame = frame.copy()
        display_frame[crop_top:crop_bottom, :] = combo
        
        state_str = ["NORMAL", "DETECTED", "AVOIDING", "RETURNING", "FINISHED"]
        cv2.putText(display_frame, f"State: {state_str[state]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Lidar: {int(front_dist)}mm", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Sonar R: {us_right}mm", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        cv2.imshow("Mission Monitor", display_frame)

        # 2. 라이다 맵 화면
        if scan_data:
            lidar_visual = draw_lidar_map(scan_data, max_dist=1500)
            cv2.imshow("LiDAR Map", lidar_visual)

        if cv2.waitKey(1) == ord('q'):
            break

    # 종료 처리
    if ser:
        ser.write(b"D,0\n")
        ser.write(f"S,{SERVO_CENTER}\n".encode())
        ser.close()
    if lidar:
        lidar.stop()
        lidar.disconnect()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()