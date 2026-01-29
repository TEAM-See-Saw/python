import cv2
import numpy as np
import math
import serial
import time
import datetime
import os

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = True  # True: 햇빛 모드, False: 실내 모드

# ⚙️ 통신 설정
PORT = 'COM4'
BAUDRATE = 9600
# ★ [핵심 1] 통신 딜레이 (0.05초 = 50ms 마다 전송) -> 렉 방지 필수!
SERIAL_DELAY = 0.03

# 📷 카메라 및 주행 설정
CAM_INDEX = 0
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# ★ [핵심 2] ROI(관심 영역) 높이 비율 설정
# 카메라를 들어 올리면(신호등 보기 위해), 차선은 화면 아래쪽으로 내려갑니다.
# 0.6 (기존): 화면의 60% 지점부터 바닥으로 인식 (위 40%는 버림)
# 0.7 : 화면의 70% 지점부터 바닥으로 인식 (위 50% 버림 -> 카메라를 많이 들었을 때 추천)
ROI_HEIGHT_RATIO = 0.6

# 영상 처리 임계값 설정
if IS_SUNNY:
    print("☀️ 모드: SUNNY")
    EXPOSURE = -9;
    L_MIN = 160;
    S_MAX = 50;
    MORPH_SIZE = (5, 5)
else:
    print("🌙 모드: NORMAL")
    EXPOSURE = -4;
    L_MIN = 100;
    S_MAX = 60;
    MORPH_SIZE = (3, 3)

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    # timeout을 0.1로 짧게 설정하여 읽기 지연 방지
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}")


# ==========================================
# [3] 영상 처리 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters):
    if line_parameters is None: return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    # ★ ROI 비율 변수 적용
    y2 = int(y1 * ROI_HEIGHT_RATIO)
    if slope == 0: slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = [];
    right_fit = []
    if lines is None: return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0];
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
    car_x = width / 2

    # ★ 목표 지점 Y좌표도 ROI 비율에 맞춰 자동 조절
    target_y = int(height * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = (height - target_y)  # 차 앞부분부터 목표지점까지 거리

    # 아크탄젠트로 각도 계산
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width = 640;
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened(): print("❌ 카메라 오류"); return

    # 녹화 설정
    if not os.path.exists('dataset'): os.makedirs('dataset')
    filename = f"dataset/drive_{datetime.datetime.now().strftime('%H%M%S')}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 20.0, (width * 2, height))
    print(f"🎥 녹화 시작: {filename}")

    # 카운트다운
    print("\n🚀 3초 후 출발!");
    for i in range(3, 0, -1): print(f"{i}.."); time.sleep(1)

    # 출발!
    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    # ★ 타이머 초기화
    last_serial_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))

        # 1. 영상 전처리
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, L_MIN, 0]), np.array([179, 255, S_MAX]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
        edges = cv2.Canny(mask, 50, 150)

        # 2. ROI 설정 (비율 변수 적용)
        # 사다리꼴 모양: 위쪽은 좁게, 아래쪽은 넓게
        roi_points = np.array([[
            (0, height),
            (width, height),
            (width // 2 + 50, int(height * ROI_HEIGHT_RATIO)),
            (width // 2 - 50, int(height * ROI_HEIGHT_RATIO))
        ]], dtype=np.int32)
        cropped = region_of_interest(edges, roi_points)

        # 3. 라인 검출
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)

        # 4. 조향 계산
        angle, target = calculate_steering_angle(frame, left, right)
        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 5. ★ [핵심] 시리얼 통신 주기 제어 (멈춤 해결)
        current_time = time.time()
        if ser and (current_time - last_serial_time > SERIAL_DELAY):
            ser.write(f"S,{servo_val}\n".encode())
            # 필요 시 속도 명령 주석 해제 (지금은 시작 시 한 번만 전송 중)
            # ser.write(f"D,{MAX_SPEED}\n".encode())
            last_serial_time = current_time

        # 6. 디스플레이
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # ROI 박스 그리기 (파란색) - 디버깅용
        cv2.polylines(frame, roi_points, True, (255, 0, 0), 2)
        # 목표 지점 그리기 (빨간 점)
        cv2.circle(frame, (target, int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

        combined = np.hstack((frame, mask_bgr))
        cv2.putText(combined, f"Servo: {servo_val}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(combined)
        cv2.imshow("Auto Drive", combined)

        if cv2.waitKey(1) == ord('q'): break

    # 종료
    if ser:
        ser.write(b"D,0\n")
        time.sleep(0.1)
        ser.write(b"S,570\n")
        ser.close()
    out.release();
    cap.release();
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()