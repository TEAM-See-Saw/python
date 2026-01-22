import cv2
import numpy as np
import math
import serial
import time
import datetime

# ==========================================
# [1] 설정 (카메라 및 아두이노)
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 255

# FHD 기준 튜닝 값
ROI_TOP_WIDTH = 960
ROI_TOP_Y = 401
ROI_BOTTOM_Y = 1057
L_MIN_WHITE = 169

# 처리용 해상도 (다운스케일)
PROC_W = 960
PROC_H = 540

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()


# ==========================================
# [3] 함수 정의
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
            slope, intercept = fit[0], fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def calculate_steering_angle(image, left_line, right_line):
    h, w, _ = image.shape
    car_x = w / 2
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (w * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (w * 0.25)
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = (h * 0.6) - h
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)


# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    print(f"📷 카메라 #{CAM_INDEX} 연결 시도 중...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    target_width, target_height = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    # [녹화 기능 주석 처리]
    # now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"drive_log_{now}.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps_out = 30.0
    # out = cv2.VideoWriter(filename, fourcc, fps_out, (target_width, target_height))

    print("🚀 실시간 라인 트레이싱 시작!")
    ser.write(f"D,{MAX_SPEED}\n".encode())

    # 스케일 비율 계산
    sx, sy = PROC_W / float(target_width), PROC_H / float(target_height)

    # FPS 계산용 변수
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- FPS 계산 ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 1. 전처리 (다운스케일링으로 속도 확보)
        proc = cv2.resize(frame, (PROC_W, PROC_H))
        hls = cv2.cvtColor(proc, cv2.COLOR_BGR2HLS)
        mask_white = cv2.inRange(hls, np.array([0, L_MIN_WHITE, 0]), np.array([179, 255, 255]))

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask_white)

        # 2. ROI 설정
        cx = PROC_W // 2
        roi_vertices = [
            (0, int(ROI_BOTTOM_Y * sy)),
            (cx - int(ROI_TOP_WIDTH * sx), int(ROI_TOP_Y * sy)),
            (cx + int(ROI_TOP_WIDTH * sx), int(ROI_TOP_Y * sy)),
            (PROC_W, int(ROI_BOTTOM_Y * sy))
        ]
        cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 3. 차선 검출 및 조향 계산
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left_line, right_line = average_slope_intercept(proc, lines)
        steering_angle, target_x_small = calculate_steering_angle(proc, left_line, right_line)

        # 4. 아두이노 명령 전송
        clamped_angle = max(-45, min(45, steering_angle))
        servo_value = int(map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
        ser.write(f"S,{servo_value}\n".encode())

        # 5. 시각화 (결과 합성)
        line_small = np.zeros_like(proc)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line: cv2.line(line_small, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line: cv2.line(line_small, (x1, y1), (x2, y2), (0, 255, 0), 5)

        line_fhd = cv2.resize(line_small, (target_width, target_height))
        combo_image = cv2.addWeighted(frame, 0.8, line_fhd, 1, 1)

        # 조향 가이드 라인 및 정보 출력
        target_x_fhd = int(target_x_small / sx)
        cv2.line(combo_image, (target_width // 2, target_height), (target_x_fhd, int(target_height * 0.6)), (0, 0, 255),
                 3)

        # 정보 표시 (FPS 추가)
        cv2.putText(combo_image, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combo_image, f"Angle: {steering_angle:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        cv2.putText(combo_image, f"Servo: {servo_value}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # [녹화 저장 주석 처리]
        # out.write(combo_image)

        # 화면 표시 (FHD가 너무 크면 resize해서 보세요)
        cv2.imshow('Live Lane Tracing', cv2.resize(combo_image, (960, 540)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 처리
    print("🛑 프로그램 종료")
    ser.write(b"D,0\n")
    time.sleep(0.1)
    ser.write(f"S,{SERVO_CENTER}\n".encode())

    # [녹화 종료 주석 처리]
    # out.release()

    ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()