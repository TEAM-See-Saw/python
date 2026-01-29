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
IS_SUNNY = True

# ⚙️ 모터/서보 설정
PORT = 'COM4'
BAUDRATE = 9600
CAM_INDEX = 0

MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# ★ [수정 1] 시리얼 통신 딜레이 설정 (0.05초 = 50ms)
# 이 시간 간격으로만 명령을 보냅니다. (멈춤 방지 핵심)
SERIAL_DELAY = 0.05

# ==========================================
# [2] 모드별 자동 튜닝값 적용
# ==========================================
if IS_SUNNY:
    print(f"☀️ [모드: SUNNY] 강력한 햇빛 대응 설정 적용")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print(f"🌙 [모드: NORMAL] 저녁/실내 설정 적용")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)

# ==========================================
# [3] 시리얼 연결
# ==========================================
ser = None
try:
    # ★ [수정 2] timeout을 0.1로 짧게 설정
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    print("⚠️ 주의: 영상 처리만 진행됩니다.")


# ==========================================
# [4] 영상 처리 함수들 (기존과 동일)
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
# [5] 메인 실행 함수
# ==========================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    # 녹화 설정
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset/autodrive_{now}.mp4"
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20.0
    out = cv2.VideoWriter(filename, fourcc, fps, (width * 2, height))
    print(f"🎥 녹화 준비 완료: {filename}")

    # 카운트다운
    print("\n" + "=" * 30)
    print(f"🚀 {MAX_SPEED} 속도로 출발합니다!")
    print("=" * 30)
    for i in range(3, 0, -1):
        print(f"Count: {i}")
        time.sleep(1)
    print("GO!!!")

    # 출발 명령
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())
        time.sleep(0.1)

    # ★ [수정 3] 통신 타이머 초기화
    last_serial_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # 영상 처리 파이프라인
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN, 0])
        upper_white = np.array([179, 255, S_MAX])
        mask = cv2.inRange(hls, lower_white, upper_white)

        kernel = np.ones(MORPH_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(mask, 50, 150)

        roi_vertices = [(0, height), (width // 2 - 50, int(height * 0.6)), (width // 2 + 50, int(height * 0.6)),
                        (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)

        angle, target = calculate_steering_angle(frame, left, right)
        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # ==========================================================
        # ★ [수정 4] 시리얼 통신 최적화 (가장 중요한 부분)
        # ==========================================================
        current_time = time.time()

        # 이전 전송으로부터 SERIAL_DELAY(0.05초) 이상 지났을 때만 전송
        if ser and (current_time - last_serial_time > SERIAL_DELAY):
            ser.write(f"S,{servo_val}\n".encode())
            last_serial_time = current_time  # 타이머 갱신

        # 화면 디버깅
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))
        info_text = f"Mode: {'SUNNY' if IS_SUNNY else 'NORMAL'} | Servo: {servo_val}"
        cv2.putText(combined, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(combined)
        cv2.imshow("Autonomous Driving", combined)

        if cv2.waitKey(1) == ord('q'):
            print("🛑 긴급 정지!")
            break

    if ser:
        ser.write(b"D,0\n")
        time.sleep(0.1)
        ser.write(b"S,570\n")
        ser.close()

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("💾 녹화 완료 및 프로그램 종료")


if __name__ == "__main__":
    main()