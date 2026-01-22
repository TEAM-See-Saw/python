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

# =========================================================
# 🎯 [핵심] 대회장 환경 맞춤 임의 조정값 (시나리오 기반)
# =========================================================
# 전략: 그림자 속 차선도 봐야 하므로 L_MIN을 너무 높이면 안 됨.
# 대신, S(채도)를 엄격하게 잡아서 햇빛 받은 바닥을 무시해야 함.

# 1. 밝기 최소값 (L): 그림자 속 차선이 안 보일 수 있으므로 170 -> 150으로 낮춤
L_MIN = 150

# 2. 채도 최대값 (S): 아스팔트나 초록 배경이 햇빛 받으면 채도가 올라감.
# 흰색은 채도가 거의 0에 가까우므로, 이 값을 낮춰서 색깔 있는 모든 걸 날림.
S_MAX = 30
# =========================================================

try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
except:
    print("아두이노 연결 실패")


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


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    # [중요] 카메라 노출을 아주 낮게 설정해서 전체적으로 어둡게 만드세요.
    # 그래야 햇빛 받은 바닥이 '흰색'으로 인식되는 걸 막습니다.
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. HLS 변환
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # 2. 필터링 (임의 조정된 값 적용)
        # L: 150 이상 (그림자 속 차선도 잡기 위해 조금 낮춤)
        # S: 30 이하 (색감이 조금이라도 있으면 가차 없이 제거)
        lower_white = np.array([0, L_MIN, 0])
        upper_white = np.array([179, 255, S_MAX])

        mask = cv2.inRange(hls, lower_white, upper_white)

        # 3. 모폴로지 연산 (노이즈 제거 강화)
        # 자잘한 빛 반사 점들을 지워버림
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 4. Canny & Hough
        edges = cv2.Canny(mask, 50, 150)

        height, width = edges.shape
        roi_vertices = [(0, height), (width // 2 - 50, int(height * 0.6)), (width // 2 + 50, int(height * 0.6)),
                        (width, height)]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)
        angle, target = calculate_steering_angle(frame, left, right)

        # 아두이노 전송 및 시각화 (기존 동일)
        servo = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))
        if ser: ser.write(f"S,{servo}\n".encode())

        # 디버깅용 화면 분할 (왼쪽: 원본, 오른쪽: 마스크)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))
        cv2.imshow("Original vs Mask", combined)

        if cv2.waitKey(1) == ord('q'): break

    if ser: ser.write(b"D,0\n")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()