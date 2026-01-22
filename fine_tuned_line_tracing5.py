import cv2
import numpy as np
import math
import serial
import time
import datetime

# ==========================================
# [1] 설정
# ==========================================
CAM_INDEX = 0

PORT = 'COM4'
BAUDRATE = 9600

SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 255

# FHD 튜닝 값
ROI_TOP_WIDTH = 960
ROI_TOP_Y = 401
ROI_BOTTOM_Y = 1057
L_MIN_WHITE = 169

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"{PORT} 포트에 연결되었습니다.")
    time.sleep(2)
except Exception as e:
    print(f"아두이노 연결 실패: {e}")
    exit()


# ==========================================
# [3] 함수 정의
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
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


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def calculate_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    car_position_x = width / 2
    car_position_y = height

    if left_line is not None and right_line is not None:
        left_x1, _, left_x2, _ = left_line[0]
        right_x1, _, right_x2, _ = right_line[0]
        target_x = (left_x2 + right_x2) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = car_position_x

    dx = target_x - car_position_x
    dy = (height * 0.6) - car_position_y
    angle_radian = math.atan2(dx, abs(dy))
    angle_deg = math.degrees(angle_radian)

    return angle_deg, int(target_x)


# ==========================================
# [4] 메인 루프
# ==========================================
def main():
    print(f"카메라 #{CAM_INDEX} 연결 시도 중...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    target_width = 1920
    target_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if not cap.isOpened():
        print(f"카메라 #{CAM_INDEX}를 열 수 없습니다.")
        return

    print("실시간 라인 트레이싱 시작")

    ser.write(f"D,{MAX_SPEED}\n".encode())

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 신호 없음. 종료.")
            break

        frame = cv2.resize(frame, (target_width, target_height))

        crop_top = ROI_TOP_Y
        crop_bottom = ROI_BOTTOM_Y
        proc = frame[crop_top:crop_bottom, :]
        proc_h, proc_w, _ = proc.shape

        hls = cv2.cvtColor(proc, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN_WHITE, 0], dtype=np.uint8)
        upper_white = np.array([179, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask_white)

        center_x = proc_w // 2
        roi_vertices = [
            (0, proc_h),
            (center_x - ROI_TOP_WIDTH, 0),
            (center_x + ROI_TOP_WIDTH, 0),
            (proc_w, proc_h)
        ]
        cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

        lines = cv2.HoughLinesP(
            cropped_edges,
            1,
            np.pi / 180,
            50,
            minLineLength=80,
            maxLineGap=100
        )
        left_line, right_line = average_slope_intercept(proc, lines)

        steering_angle, target_x_proc = calculate_steering_angle(proc, left_line, right_line)

        clamped_angle = max(-45, min(45, steering_angle))
        servo_value = map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX)
        servo_value = int(servo_value)

        cmd = f"S,{servo_value}\n"
        ser.write(cmd.encode())

        line_image_proc = np.zeros_like(proc)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line:
                cv2.line(line_image_proc, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line:
                cv2.line(line_image_proc, (x1, y1), (x2, y2), (0, 255, 0), 5)

        combo_proc = cv2.addWeighted(proc, 0.8, line_image_proc, 1, 1)
        cv2.line(
            combo_proc,
            (proc_w // 2, proc_h),
            (int(target_x_proc), int(proc_h * 0.6)),
            (0, 0, 255),
            3
        )

        display_frame = frame.copy()
        display_frame[crop_top:crop_bottom, :] = combo_proc

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps_now = 1.0 / dt
        else:
            fps_now = 0.0
        prev_time = now

        cv2.putText(display_frame, f"Angle: {steering_angle:.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Servo: {servo_value}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps_now:.1f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Lane Tracing', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("프로그램 종료: 정지 및 초기화")
    ser.write(b"D,0\n")
    time.sleep(0.1)
    ser.write(f"S,{SERVO_CENTER}\n".encode())

    ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
