import cv2
import numpy as np
import math
import serial
import time

PORT = 'COM4'
BAUDRATE = 9600
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 0
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.6

L_MIN = 170
S_MAX_VAL = 80
MORPH_SIZE = (3, 3)
BLUR_K = 5

ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
    print(f"[SERIAL] {PORT} 연결 성공")
    time.sleep(2)
except Exception as e:
    print(f"[SERIAL] 연결 실패: {e}")


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
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
    height, width = image.shape[:2]
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
    dy = int(height * ROI_HEIGHT_RATIO) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)
    cap.set(15, -6)

    if not cap.isOpened():
        print("[CAM] 카메라 오픈 실패")
        return

    print("[MAIN] 3초 후 출발")
    for i in range(3, 0, -1):
        print(f"[MAIN] {i}..")
        time.sleep(1)

    if ser:
        for _ in range(3):
            try:
                ser.write(f"D,{MAX_SPEED}\n".encode())
                time.sleep(0.05)
            except Exception as e:
                print(f"[SERIAL] 초기 D 전송 실패: {e}")
        print(f"[SERIAL] 주행 명령 전송 D,{MAX_SPEED}")

    last_serial_time = 0
    last_speed_time = 0
    frame_idx = 0

    try:
        while True:
            frame_idx += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[CAM] 프레임 읽기 실패")
                time.sleep(0.01)
                continue

            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            lower_white = np.array([0, L_MIN, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            roi_vertices = np.array([
                (0, h),
                (w // 2 - 50, int(h * ROI_HEIGHT_RATIO)),
                (w // 2 + 50, int(h * ROI_HEIGHT_RATIO)),
                (w, h)
            ], dtype=np.int32)

            edges = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges, roi_vertices)
            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50,
                                    minLineLength=40, maxLineGap=100)
            left, right = average_slope_intercept(frame, lines)
            angle, target = calculate_steering_angle(frame, left, right)

            servo_val = int(map_value(max(-45, min(45, angle)),
                                      -45, 45,
                                      SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            if ser:
                curr_time = time.time()
                if curr_time - last_serial_time > SERIAL_DELAY:
                    try:
                        ser.write(f"S,{servo_val}\n".encode())
                    except Exception as e:
                        print(f"[SERIAL] S 전송 실패: {e}")
                    last_serial_time = curr_time

                if curr_time - last_speed_time > SPEED_REFRESH_DELAY:
                    try:
                        ser.write(f"D,{MAX_SPEED}\n".encode())
                    except Exception as e:
                        print(f"[SERIAL] D 전송 실패: {e}")
                    last_speed_time = curr_time

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.polylines(frame, [roi_vertices], True, (255, 0, 0), 2)
            cv2.circle(frame, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            if left is not None:
                x1, y1, x2, y2 = left[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if right is not None:
                x1, y1, x2, y2 = right[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            combined = np.hstack((frame, mask_bgr))
            cv2.imshow("HLS + ROI Visualized", combined)

            key = cv2.waitKey(1)
            if key == ord('q'):
                print("[MAIN] 종료")
                break

            if frame_idx % 30 == 0:
                print(f"[DEBUG] frame={frame_idx}, angle={angle:.2f}, servo={servo_val}, "
                      f"left={left is not None}, right={right is not None}")

    except Exception as e:
        print(f"[MAIN] 예외 발생: {e}")

    finally:
        print("[MAIN] 정리 중")
        if ser:
            try:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(f"S,{SERVO_CENTER}\n".encode())
                    time.sleep(0.05)
                ser.close()
                print("[SERIAL] 정지 완료")
            except Exception as e:
                print(f"[SERIAL] 정지 중 오류: {e}")
        cap.release()
        cv2.destroyAllWindows()
        print("[MAIN] 자원 해제 완료")


if __name__ == "__main__":
    main()
