import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 설정값
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

# --- 필터링 고정값 ---
L_THRESHOLD = 170
# L_THRESHOLD = 230 # 융기원

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


# ==========================================
# [3] 영상 처리 함수들
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters):
    if line_parameters is None: return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
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
            if x1 == x2: continue
            if math.hypot(x2 - x1, y2 - y1) < 80: continue

            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0];
            intercept = fit[1]

            if abs(slope) < 0.55: continue

            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


# last_target_x 변수는 더 이상 필요 없지만 호환성을 위해 남겨둠 (사용 안 함)
last_target_x = 320


def calculate_steering_angle(image, left_line, right_line):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        # 양쪽 다 보임 -> 중간 지점
        base_target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        # 왼쪽만 보임 -> 왼쪽 + 차폭 절반
        base_target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        # 오른쪽만 보임 -> 오른쪽 - 차폭 절반
        base_target_x = right_line[0][2] - (width * 0.25)
    else:
        # ★ [수정] 차선 못 찾으면 무조건 중앙(직진)으로 복귀
        # 기존: base_target_x = last_target_x
        base_target_x = width / 2  # 중앙

    # 값 저장 (다음 프레임에 쓰진 않지만 변수 업데이트용)
    last_target_x = base_target_x

    # 오프셋 적용
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
    width = 640;
    height = 480
    cap.set(3, width);
    cap.set(4, height)
    # cap.set(15, -6)

    if not cap.isOpened(): print("❌ 카메라 오류"); return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('lane_record.avi', fourcc, 20.0, (1280, 480))
    print("🎥 녹화 시작: lane_record.avi")

    print("\n🚀 3초 후 출발!");
    for i in range(3, 0, -1): print(f"{i}.."); time.sleep(1)

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0
    last_speed_time = 0

    try:
        while True:
            if ser:
                try:
                    if ser.in_waiting > 0: ser.read(ser.in_waiting)
                except:
                    pass

            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            # 필터링
            blurred = cv2.medianBlur(frame, 5)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            mask = cv2.inRange(hls, np.array([0, L_THRESHOLD, 0]), np.array([179, 255, 255]))

            # 차선 인식
            edges = cv2.Canny(mask, 50, 150)
            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)
            cropped = region_of_interest(edges, roi_points)

            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)

            # ★ 조향 계산 (놓치면 중앙)
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
            cv2.putText(combined, f"Angle: {angle:.1f} Offset: {STEERING_OFFSET}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            out.write(combined)

            cv2.imshow("Auto Exposure Lane Tracing", combined)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지 & 녹화 저장 완료")
        if ser:
            for _ in range(3): ser.write(b"D,0\n"); ser.write(b"S,570\n"); time.sleep(0.05)
            ser.close()

        if 'out' in locals() and out.isOpened():
            out.release()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()