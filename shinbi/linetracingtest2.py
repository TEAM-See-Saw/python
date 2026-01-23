'''
1. BEV 변환: 도로를 위에서 본 평면도로 만듭니다(11자 차선이 됨)
2. 화이트 필터: 흰색 차선만 강력하게 뽑아냅니다
3. 히스토그램 분석: 차선이 어디에 뭉쳐있는지 찾아냅니다(끊긴 차선에 강함)
4. PID 제어: 사람이 운전하듯 부드럽게 곡선을 타도록 제어합니다

<튜닝>
1. BEV 영역 맞추기 (노란색 네모): 64~69줄 src_points 숫자
2. 흰색 잘 따지는지 확인
   오른쪽 화면(검은 배경)에 흰색 차선이 선명하게 나오는지 확인
   -만약 바닥 전체가 희게 나오면 코드 91줄 lower_white의 180->200
   -차선이 잘 안 보이면 180을 150으로 
3. 주행감 조절 (PID)
   -지그재그가 심하다? Kd = 0.15를 0.2나 0.3으로 올리기(브레이크를 더 강하게 검)
   -코너를 너무 작게 돈다(반응이 느리다)? Kp = 0.4를 0.5나 0.6으로 올리세요.
'''

import cv2
import numpy as np
import serial
import time

# ==========================================
# [1] 설정 (내 차에 맞게 튜닝 필수)
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

# 서보 & 속도
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680   # 좌회전 최대값
SERVO_RIGHT_MAX = 480  # 우회전 최대값
MAX_SPEED = 200        # 테스트 속도

# 해상도 (BEV 변환을 위해 640x480 권장)
WIDTH = 640
HEIGHT = 480

# ★ PID 제어 상수 (사람 같은 주행의 핵심) ★
# 차가 비틀거리면 Kp를 줄이고(0.3), 코너를 못 돌면 Kp를 늘리세요(0.5)
Kp = 0.4  # 비례항: 오차만큼 꺾기
Kd = 0.15 # 미분항: 급발진/진동 억제 (핸들 떨림 방지)
Ki = 0.0  # 적분항: 보통 0으로 둠

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print(f"✅ 아두이노 연결 성공: {PORT}")
except Exception as e:
    print(f"⚠️ 연결 실패 (시뮬레이션 모드): {e}")

# ==========================================
# [3] 카메라 설정 (MJPG 고속 모드)
# ==========================================
params = [
    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
    cv2.CAP_PROP_FRAME_WIDTH, 1920,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    cv2.CAP_PROP_FPS, 30
]
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW, params)
if not cap.isOpened():
    print("❌ 카메라 오류")
    exit()

# ==========================================
# [4] 핵심 알고리즘 함수들
# ==========================================

def warp_perspective(img):
    """
    [핵심 1] Bird's Eye View (탑뷰) 변환
    사다리꼴 이미지를 직사각형으로 펴서 차선을 평행하게 만듦
    """
    h, w = img.shape[:2]
    
    # [중요] 소스 좌표 (사다리꼴): 화면에서 '도로 바닥' 영역 지정
    # 이 좌표가 안 맞으면 탑뷰가 찌그러집니다. 화면 보면서 조절 필요.
    src_points = np.float32([
        [w * 0.15, h * 0.85],  # 좌하 (화면 아래쪽 넓게)
        [w * 0.85, h * 0.85],  # 우하
        [w * 0.35, h * 0.45],  # 좌상 (화면 위쪽 좁게 - 멀리 있는 도로)
        [w * 0.65, h * 0.45]   # 우상
    ])
    
    # 목적지 좌표 (직사각형): 이미지를 꽉 채우도록 폄
    dst_points = np.float32([
        [w * 0.2, h],       # 좌하
        [w * 0.8, h],       # 우하
        [w * 0.2, 0],       # 좌상
        [w * 0.8, 0]        # 우상
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points) # 복구용
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return warped, src_points

def color_filter_white(img):
    """
    [핵심 2] 흰색 차선 추출 (HSV 색상 공간)
    자연광 아래서는 RGB보다 HSV가 더 유리함
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 흰색 정의: 채도(S)는 낮고, 명도(V)는 높은 색
    # 햇빛이 강하면 180을 200으로 올리세요. 그늘지면 150으로 내리세요.
    lower_white = np.array([0, 0, 180])  
    upper_white = np.array([179, 40, 255]) 
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask

def get_lane_center_histogram(binary_warped):
    """
    [핵심 3] 히스토그램 분석
    화면 하단의 흰색 픽셀 분포를 분석해 차선 위치 결정
    """
    h, w = binary_warped.shape
    
    # 화면 하단 50%만 분석 (가장 가까운 도로)
    bottom_half = binary_warped[h//2:, :]
    
    # 세로로 픽셀을 다 더함 -> 막대그래프처럼 됨
    histogram = np.sum(bottom_half, axis=0)
    
    midpoint = int(histogram.shape[0] / 2)
    
    # 왼쪽 절반에서 가장 높은 봉우리 / 오른쪽 절반에서 가장 높은 봉우리 찾기
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 예외 처리: 차선이 거의 안 보이면(픽셀 합이 너무 작으면) 감지 실패로 간주
    if np.max(histogram[:midpoint]) < 100: leftx_base = None
    if np.max(histogram[midpoint:]) < 100: rightx_base = None

    image_center = w // 2
    lane_center = image_center # 기본값: 직진
    
    # 양쪽 차선 다 보임 -> 그 정중앙이 목표
    if leftx_base is not None and rightx_base is not None:
        lane_center = (leftx_base + rightx_base) // 2
        
    # 왼쪽만 보임 -> 왼쪽 차선에서 일정 거리 떨어진 곳이 목표
    elif leftx_base is not None:
        lane_center = leftx_base + 260 # 260은 차선 폭의 절반 정도 (튜닝 필요)
        
    # 오른쪽만 보임 -> 오른쪽 차선에서 일정 거리 떨어진 곳이 목표
    elif rightx_base is not None:
        lane_center = rightx_base - 260
        
    return lane_center, leftx_base, rightx_base

# PID 제어 변수 (전역)
prev_error = 0
integral = 0

def pid_control(error):
    """
    [핵심 4] PID 제어기
    단순 비례 제어가 아니라, 급격한 변화를 막아주는(D) 기능 포함
    """
    global prev_error, integral
    
    # P: 오차만큼 꺾어라
    p_term = Kp * error
    
    # D: 오차의 변화 속도를 줄여라 (진동 방지)
    d_term = Kd * (error - prev_error)
    
    # I: 누적 오차 (여기선 잘 안 씀)
    # integral += error
    # i_term = Ki * integral
    
    prev_error = error
    
    return p_term + d_term # + i_term

# ==========================================
# [5] 메인 루프
# ==========================================
def main():
    print("🚀 BEV + PID 주행 시작")
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())
    
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. 리사이즈 (속도 최적화)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        
        # 2. BEV 변환 (탑뷰 만들기)
        warped, src_points = warp_perspective(frame)
        
        # 3. 흰색 차선 추출
        binary = color_filter_white(warped)
        
        # 4. 차선 중심 찾기
        target_center, lx, rx = get_lane_center_histogram(binary)
        
        # 5. 오차 계산 (화면중앙 - 차선중앙)
        # 값이 양수면 차선이 오른쪽에 있음 -> 오른쪽으로 핸들 돌려야 함
        image_center = WIDTH // 2
        error = target_center - image_center
        
        # 6. PID 계산
        control_val = pid_control(error)
        
        # 7. 서보 값 매핑
        # control_val을 서보 각도로 변환
        steering = SERVO_CENTER + int(control_val)
        
        # 안전장치 (최대/최소값 제한)
        steering = max(SERVO_RIGHT_MAX, min(SERVO_LEFT_MAX, steering))
        
        if ser: ser.write(f"S,{steering}\n".encode())

        # --- 디버깅 화면 (튜닝용) ---
        cur_time = time.time()
        dt = cur_time - prev_time
        if dt == 0: dt = 0.001
        fps = 1.0 / dt
        prev_time = cur_time
        
        # [왼쪽 화면] 원본에 BEV 영역 표시
        debug_ori = frame.copy()
        cv2.polylines(debug_ori, [np.int32(src_points)], True, (0, 255, 255), 2)
        cv2.putText(debug_ori, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # [오른쪽 화면] BEV 결과 + 차선 인식 상태
        debug_bev = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if lx: cv2.line(debug_bev, (lx, 0), (lx, HEIGHT), (255, 0, 0), 3) # 왼쪽 파랑
        if rx: cv2.line(debug_bev, (rx, 0), (rx, HEIGHT), (0, 0, 255), 3) # 오른쪽 빨강
        cv2.line(debug_bev, (target_center, 0), (target_center, HEIGHT), (0, 255, 0), 2) # 목표 초록
        cv2.line(debug_bev, (image_center, 0), (image_center, HEIGHT), (255, 255, 255), 1) # 내 차 중심
        
        cv2.imshow('Left: Raw(Area) / Right: BEV', cv2.hconcat([debug_ori, debug_bev]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    if ser:
        ser.write(b"D,0\n")
        ser.write(f"S,{SERVO_CENTER}\n".encode())
        ser.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()