import cv2
import numpy as np
import math
import serial
import time

CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480
MAX_SPEED = 255

# ===============================
# FHD 튜닝값 → 640x480 변환 결과
# ===============================
ROI_TOP_WIDTH = 320     # 960 * (640 / 1920)
ROI_TOP_Y = 178         # 401 * (480 / 1080)
ROI_BOTTOM_Y = 469      # 1057 * (480 / 1080)
L_MIN = 169

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
except:
    print("아두이노 연결 실패")
    ser = None


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
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line


def calculate_steering_angle(image, left_line, right_line):
    height, width, _ = image.shape
    car_x = width / 2
    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + width * 0.25
    elif right_line is not None:
        target_x = right_line[0][2] - width * 0.25
    else:
        target_x = car_x

    dx = target_x - car_x
    dy = height * 0.6 - height
    angle = math.degrees(math.atan2(dx, abs(dy)))
    return angle, int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN, 0], dtype=np.uint8)
        upper_white = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hls, lower_white, upper_white)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        edges = cv2.Canny(mask, 50, 150)

        h, w = edges.shape
        cx = w // 2
        roi_vertices = [
            (0, ROI_BOTTOM_Y),
            (cx - ROI_TOP_WIDTH, ROI_TOP_Y),
            (cx + ROI_TOP_WIDTH, ROI_TOP_Y),
            (w, ROI_BOTTOM_Y)
        ]

        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))
        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, 40, 100)

        left, right = average_slope_intercept(frame, lines)
        angle, _ = calculate_steering_angle(frame, left, right)

        servo = int(map_value(max(-45, min(45, angle)),
                              -45, 45,
                              SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
        if ser:
            ser.write(f"S,{servo}\n".encode())

        # ROI 시각화
        cv2.polylines(frame, [np.array(roi_vertices, np.int32)], True, (0, 255, 0), 2)

        cv2.imshow("ROI Visualization (640x480)", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    if ser:
        ser.write(b"D,0\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
