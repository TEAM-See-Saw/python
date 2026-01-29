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
IS_SUNNY = False  # True: 햇빛 강함, False: 실내/흐림

# ⚙️ 통신 설정
PORT = 'COM4'  # 아두이노 포트 확인
BAUDRATE = 9600
SERIAL_DELAY = 0.05  # ★ 멈춤 방지: 0.05초마다 명령 전송

# 📷 카메라 및 주행 설정
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_SPEED = 255  # 주행 속도 (0~255)

# 📐 서보 모터 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 🖼️ ROI (관심 영역) 설정 - 카메라 각도 대응 ★
# 0.0(맨위) ~ 1.0(맨아래)
# 카메라를 들었으므로, 위쪽 50%는 잘라내서 신호등/배경 간섭을 막음
ROI_Y_TOP_RATIO = 0.5
ROI_Y_BOTTOM_RATIO = 1.0
ROI_X_MARGIN = 50  # 사다리꼴 윗변 여백

# 영상 처리 임계값 (모드별 자동 설정)
if IS_SUNNY:
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)

# 동적 임계값 초기값
L_MIN_DYNAMIC = float(L_MIN)
S_MAX_DYNAMIC = float(S_MAX)
ALPHA = 0.9  # 0~1, 높을수록 천천히 변함

# ==========================================
# [2] 하드웨어 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"⚠️ 아두이노 연결 실패 ({e}) -> 영상 처리만 진행")


# ==========================================
# [3] 영상 처리 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * ROI_Y_TOP_RATIO)  # ROI 상단까지만 선 그리기
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
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
    car_x = width / 2

    target_y = int(height * ROI_Y_TOP_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.20)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.20)
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = (height - target_y)

    angle_deg = math.degrees(math.atan2(dx, dy))
    return angle_deg, int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def update_thresholds(hls, roi_vertices, prev_L_MIN, prev_S_MAX):
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    mask = np.zeros_like(L, dtype=np.uint8)
    cv2.fillPoly(mask, roi_vertices, 255)

    roi_L = L[mask > 0]
    roi_S = S[mask > 0]

    if roi_L.size == 0 or roi_S.size == 0:
        return prev_L_MIN, prev_S_MAX

    p75_L = np.percentile(roi_L, 75)
    p75_S = np.percentile(roi_S, 75)

    new_L_MIN = np.clip(p75_L * 0.7, 60, 200)
    new_S_MAX = np.clip(p75_S * 1.1, 40, 120)

    return new_L_MIN, new_S_MAX


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    global L_MIN_DYNAMIC, S_MAX_DYNAMIC

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened():
        print("❌ 카메라 오류")
        return

    # 녹화 파일 설정
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    filename = f"dataset/line_trace_{datetime.datetime.now().strftime('%H%M%S')}.mp4"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (FRAME_WIDTH * 2, FRAME_HEIGHT))
    print(f"🎥 녹화 시작: {filename}")

    last_serial_time = 0

    print("🚀 출발!")
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            roi_points = np.array([[
                (0, int(h * ROI_Y_BOTTOM_RATIO)),
                (w, int(h * ROI_Y_BOTTOM_RATIO)),
                (w // 2 + ROI_X_MARGIN, int(h * ROI_Y_TOP_RATIO)),
                (w // 2 - ROI_X_MARGIN, int(h * ROI_Y_TOP_RATIO))
            ]], dtype=np.int32)

            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

            new_L, new_S = update_thresholds(hls, roi_points, L_MIN_DYNAMIC, S_MAX_DYNAMIC)
            L_MIN_DYNAMIC = ALPHA * L_MIN_DYNAMIC + (1 - ALPHA) * new_L
            S_MAX_DYNAMIC = ALPHA * S_MAX_DYNAMIC + (1 - ALPHA) * new_S

            mask = cv2.inRange(
                hls,
                np.array([0, int(L_MIN_DYNAMIC), 0]),
                np.array([179, 255, int(S_MAX_DYNAMIC)])
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))
            edges = cv2.Canny(mask, 50, 150)

            cropped = region_of_interest(edges, roi_points)

            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            angle, target_x = calculate_steering_angle(frame, left, right)

            clamped_angle = max(-45, min(45, angle))
            servo_val = int(map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            current_time = time.time()
            if ser and (current_time - last_serial_time > SERIAL_DELAY):
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.polylines(frame, roi_points, True, (255, 0, 0), 2)
            cv2.circle(frame, (target_x, int(h * ROI_Y_TOP_RATIO)), 10, (0, 0, 255), -1)

            combined = np.hstack((frame, mask_bgr))
            cv2.putText(
                combined,
                f"Angle: {int(angle)} | Servo: {servo_val} | L_MIN: {int(L_MIN_DYNAMIC)} | S_MAX: {int(S_MAX_DYNAMIC)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            out.write(combined)
            cv2.imshow("Line Tracing", combined)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

    finally:
        if ser:
            ser.write(b"D,0\n")
            ser.write(f"S,{SERVO_CENTER}\n".encode())
            ser.close()
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 종료됨")


if __name__ == "__main__":
    main()
