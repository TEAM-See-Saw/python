import cv2
import numpy as np
import math
import serial
import time
from rplidar import RPLidar

# ==========================================
# [1] 설정 (Configuration)
# ==========================================
# 1-1. 통신 포트
ARDUINO_PORT = 'COM4'
LIDAR_PORT = 'COM3'
BAUDRATE = 9600

# 1-2. 카메라 설정 (선배님 코드 적용)
CAM_INDEX = 0
CAM_EXPOSURE = -6  # ★ 튜닝했던 노출값 유지

# 1-3. ROI 및 주행 설정
ROI_TOP_WIDTH = 960
ROI_TOP_Y = 401
ROI_BOTTOM_Y = 1057
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 1-4. 속도 설정
NORMAL_SPEED = 150
AVOID_SPEED = 130

# 1-5. 장애물 감지 기준 (LiDAR)
OBS_ANGLE_MIN = 175
OBS_ANGLE_MAX = 185
OBS_DIST_THRES = 600 

# ==========================================
# [2] 초기화
# ==========================================
try:
    ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
    lidar = RPLidar(LIDAR_PORT)
    print(f"✅ 장치 연결 성공: Arduino, LiDAR")
    time.sleep(2)
except Exception as e:
    print(f"❌ 하드웨어 연결 실패: {e}")
    exit()

# ==========================================
# [3] 비전 처리 함수 (Shadow/Sunlight Robust)
# ==========================================
def preprocess_lane(img):
    """ Top-Hat 변환으로 그림자/햇빛 영향을 제거하고 차선만 추출 """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 커널 크기는 차선 두께보다 조금 커야 함
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    # 이진화
    _, binary = cv2.threshold(top_hat, 40, 255, cv2.THRESH_BINARY)
    
    # 노이즈 제거
    kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    return clean_binary

def calculate_base_angle(frame):
    """ 현재 차선 위치를 기반으로 기본 조향각 계산 """
    height, width = frame.shape[:2]
    
    # 1. 전처리 (Top-Hat)
    edges = cv2.Canny(preprocess_lane(frame), 50, 150)
    
    # 2. ROI Crop
    mask = np.zeros_like(edges)
    center_x = width // 2
    roi_verts = np.array([[(0, height), (center_x - 350, 0), (center_x + 350, 0), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_verts, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    # 3. HoughLinesP
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    # 4. 차선 구분 및 평균 계산
    left_fit, right_fit = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            
            if abs(slope) < 0.4: continue # 그림자 경계선 제거
            
            if slope < -0.5: left_fit.append(line[0])
            elif slope > 0.5: right_fit.append(line[0])
    
    # 5. 타겟 포인트 계산
    car_x = width / 2
    target_x = car_x
    
    if left_fit and right_fit:
        l_avg = np.mean(left_fit, axis=0)
        r_avg = np.mean(right_fit, axis=0)
        target_x = (l_avg[2] + r_avg[2]) / 2 
    elif left_fit:
        l_avg = np.mean(left_fit, axis=0)
        target_x = l_avg[2] + (width * 0.23)
    elif right_fit:
        r_avg = np.mean(right_fit, axis=0)
        target_x = r_avg[2] - (width * 0.23)
        
    dx = target_x - car_x
    dy = height * 0.6
    angle = math.degrees(math.atan2(dx, dy))
    return angle, int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

# ==========================================
# [4] 메인 로직 (State Machine)
# ==========================================
def main():
    # ---------------------------------------------------------
    # ★ [핵심 변경] 선배님의 MJPG 강제 주입 코드 적용
    # ---------------------------------------------------------
    params = [
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.CAP_PROP_FRAME_WIDTH, 1920,
        cv2.CAP_PROP_FRAME_HEIGHT, 1080,
        cv2.CAP_PROP_FPS, 30
    ]
    # DSHOW + Params 주입으로 고속 모드 진입
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW, params)

    # 연결 확인
    if not cap.isOpened():
        print("❌ 카메라 연결 실패 (포트 혹은 사용중 여부 확인)")
        return

    # ★ 노출값 적용 (MJPG 모드 진입 후 설정해야 안전함)
    cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE) 

    # 디버깅: 실제 적용된 코덱 확인
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"📷 Camera Init: Codec={codec}, Res={cap.get(3)}x{cap.get(4)}")

    if codec != "MJPG":
        print("⚠️ 경고: MJPG 모드 진입 실패. FPS가 낮을 수 있습니다.")
    # ---------------------------------------------------------

    stage = 0
    state_timer = 0
    
    print("🚀 장애물 회피 미션 시작 (High FPS Mode)")
    ser.write(f"D,{NORMAL_SPEED}\n".encode())
    
    iterator = lidar.iter_scans()
    
    prev_time = time.time()

    try:
        while True:
            # 1. 센서 데이터 수집
            ret, frame = cap.read()
            if not ret: break
            
            # FPS 계산 (확인용)
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = curr_time
            
            # 라이다 데이터
            try:
                scan = next(iterator)
            except StopIteration: continue
            
            front_dist = 9999
            for (_, angle, dist) in scan:
                if OBS_ANGLE_MIN <= angle <= OBS_ANGLE_MAX:
                    if 0 < dist < front_dist: front_dist = dist

            # 2. 비전 처리
            # 연산 부하를 줄이기 위해 ROI만 자름
            proc_frame = cv2.resize(frame, (1920, 1080))
            roi_frame = proc_frame[ROI_TOP_Y:ROI_BOTTOM_Y, :]
            base_angle, target_x = calculate_base_angle(roi_frame)
            
            # 3. 판단 및 제어
            final_angle = base_angle
            speed = NORMAL_SPEED
            
            # [Stage 0] 2차선 직진
            if stage == 0:
                if front_dist < OBS_DIST_THRES:
                    print(f"⚠️ 장애물 발견({int(front_dist)}mm) -> 좌측 회피 시작")
                    stage = 1
                    state_timer = curr_time
            
            # [Stage 1] 2 -> 1 차선 변경
            elif stage == 1:
                final_angle = base_angle - 22
                speed = AVOID_SPEED
                if curr_time - state_timer > 2.5:
                    print("✅ 1차선 진입 완료")
                    stage = 2
            
            # [Stage 2] 1차선 직진
            elif stage == 2:
                if front_dist < OBS_DIST_THRES:
                    print(f"⚠️ 장애물 발견({int(front_dist)}mm) -> 우측 복귀 시작")
                    stage = 3
                    state_timer = curr_time
                    
            # [Stage 3] 1 -> 2 차선 변경
            elif stage == 3:
                final_angle = base_angle + 22
                speed = AVOID_SPEED
                if curr_time - state_timer > 2.5:
                    print("✅ 2차선 복귀 완료")
                    stage = 4
            
            # [Stage 4] 완료
            elif stage == 4:
                final_angle = base_angle
                speed = NORMAL_SPEED

            # 4. 명령 전송
            final_angle = max(-45, min(45, final_angle))
            servo_val = map_value(final_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX)
            
            ser.write(f"S,{int(servo_val)}\n".encode())
            ser.write(f"D,{speed}\n".encode())
            
            # 5. 디버깅
            cv2.putText(roi_frame, f"FPS: {fps:.1f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(roi_frame, f"Stage: {stage}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Mission View", roi_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        ser.write(b"D,0\n")
        ser.close()
        lidar.stop()
        lidar.disconnect()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()