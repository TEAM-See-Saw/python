import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 설정값 (✅ 사용자 튜닝값 그대로)
# ==========================================
PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
MAX_SPEED = 255

# --- 서보 모터 설정 ---
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 440

# --- ROI 설정 ---
ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

# --- 오프셋 설정 ---
STEERING_OFFSET = -2

# --- 필터링 고정값 (✅ 유지) ---
L_THRESHOLD = 230

# ==========================================
# [2] 시리얼 연결
# ==========================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    ser = None


# ==========================================
# ✅ [추가] 카메라 AUTO 설정 (DirectShow 기준)
# - 여기서 '영상 밝기/노출'을 카메라 드라이버가 자동으로 맞추게 함
# - 장치/드라이버마다 지원 여부가 다를 수 있음 (안 먹으면 set이 무시됨)
# ==========================================
def configure_camera_auto(cap):
    # 해상도는 main에서 set하지만, 혹시 몰라 여기서도 안전하게 둠(중복 set 무해)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Auto Exposure:
    # OpenCV+DirectShow에서 CAP_PROP_AUTO_EXPOSURE 값 해석이 드라이버마다 다름.
    # 보통 0.75=Auto, 0.25=Manual 로 많이 동작함.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    # Auto White Balance (가능한 드라이버에서만 적용)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    # Auto Focus (차선 인식은 오히려 흔들릴 수 있어, 필요하면 0으로 꺼도 됨)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Brightness/Contrast/Gain 같은 건 "오토를 믿겠다"는 요청이므로 굳이 고정 안 함.
    # 다만 어떤 카메라는 오토 노출이 너무 튀면 아래처럼 범위 내로 제한값을 줄 수도 있음.
    # cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # <- 수동개입이므로 지금은 사용 안 함


# ==========================================
# [3] 영상 처리 함수들
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
    y2 = int(y1 * ROI_HEIGHT_RATIO)
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
            if x1 == x2:
                continue
            if math.hypot(x2 - x1, y2 - y1) < 80:
                continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if abs(slope) < 0.55:
                continue

            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


last_target_x = 320


def calculate_steering_angle(image, left_line, right_line):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        base_target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        base_target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        base_target_x = right_line[0][2] - (width * 0.25)
    else:
        base_target_x = width / 2  # 못 찾으면 중앙

    last_target_x = base_target_x

    final_target_x = base_target_x + STEERING_OFFSET
    final_target_x = max(0, min(width, final_target_x))

    dx = final_target_x - car_x
    dy = (height - target_y)

    return math.degrees(math.atan2(dx, abs(dy))), int(final_target_x)


def map_servo(angle):
    return int((angle - (-45)) * (SERVO_RIGHT_MAX - SERVO_LEFT_MAX) / (45 - (-45)) + SERVO_LEFT_MAX)


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)

    # ✅ 카메라 오토 설정 적용
    configure_camera_auto(cap)

    if not cap.isOpened():
        print("❌ 카메라 오류")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('lane_record.avi', fourcc, 20.0, (1280, 480))
    print("🎥 녹화 시작: lane_record.avi")

    print("\n🚀 3초 후 출발!")
    for i in range(3, 0, -1):
        print(f"{i}..")
        time.sleep(1)

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0.0
    last_speed_time = 0.0

    try:
        while True:
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            h, w = frame.shape[:2]

            # ==========================================
            # ✅ 필터링 (L_THRESHOLD=230 고정 유지)
            # ==========================================
            blurred = cv2.medianBlur(frame, 5)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            # L 채널 기준 230 이상을 흰색으로 판단
            mask = cv2.inRange(hls,
                               np.array([0, L_THRESHOLD, 0]),
                               np.array([179, 255, 255]))

            # 차선 인식
            edges = cv2.Canny(mask, 50, 150)

            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            cropped = region_of_interest(edges, roi_points)

            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50,
                                    minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            # 조향 계산 (놓치면 중앙)
            angle, target = calculate_steering_angle(frame, left, right)
            servo_val = map_servo(max(-45, min(45, angle)))

            # 통신
            if ser:
                curr_time = time.time()
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = curr_time
                if curr_time - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = curr_time

            # 디스플레이
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(frame, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            combined = np.hstack((frame, mask_bgr))
            cv2.putText(combined,
                        f"Angle: {angle:.1f} Offset: {STEERING_OFFSET} L_TH:{L_THRESHOLD}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            out.write(combined)

            cv2.imshow("Auto Exposure Lane Tracing (L_THRESHOLD fixed)", combined)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지 & 녹화 저장 완료")
        if ser:
            try:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(b"S,570\n")
                    time.sleep(0.05)
                ser.close()
            except:
                pass

        if 'out' in locals() and out.isOpened():
            out.release()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
