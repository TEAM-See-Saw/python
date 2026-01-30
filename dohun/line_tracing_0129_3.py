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
IS_SUNNY = False

# 통신 설정
PORT = 'COM4'
BAUDRATE = 9600
SERIAL_DELAY = 0.05

# 카메라 설정
CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_SPEED = 255

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# 영상 처리 임계값 (날씨 모드 없이 고정값 사용)
L_MIN = 100
S_MAX = 60
MORPH_SIZE = (3, 3)

# 전역 변수 (ROI 좌표 저장용)
pts = []

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
# [3] 유틸리티 함수
# ==========================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def make_points(image, line_parameters, roi_y_top):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(roi_y_top)
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines, roi_y_top):
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

    left_line = make_points(image, np.mean(left_fit, axis=0), roi_y_top) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0), roi_y_top) if len(right_fit) > 0 else None
    return left_line, right_line


def calculate_steering_angle(image, left_line, right_line, roi_y_top):
    height, width, _ = image.shape
    car_x = width / 2
    target_y = int(roi_y_top)

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


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상
    rect[2] = pts[np.argmax(s)]  # 우하
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상
    rect[3] = pts[np.argmax(diff)]  # 좌하
    return rect.astype("int32")


# ==========================================
# [★] 사다리꼴 설정 인터페이스
# ==========================================
def click_event(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append([x, y])


def setup_trapezoid_roi(cap):
    global pts
    pts = []

    print("\n-------------------------------------------------")
    print("🖱️ [사다리꼴 ROI 설정] 화면에 점 4개를 찍으세요!")
    print("   - 마우스 클릭: 점 찍기 (좌상->우상->우하->좌하 순 추천)")
    print("   - 'r' 키: 초기화 (다시 찍기)")
    print("   - 'c' 키: 설정 완료 및 출발")
    print("-------------------------------------------------\n")

    cv2.namedWindow("Setup ROI")
    cv2.setMouseCallback("Setup ROI", click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            return None

        display = frame.copy()

        for pt in pts:
            cv2.circle(display, tuple(pt), 5, (0, 0, 255), -1)

        if len(pts) == 4:
            ordered_pts = order_points(np.array(pts))
            cv2.polylines(display, [ordered_pts], True, (0, 255, 0), 2)
            cv2.putText(display, "Press 'c' to Start", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display, f"Points: {len(pts)}/4", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Setup ROI", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            pts = []
            print("🔄 초기화되었습니다. 다시 찍으세요.")
        elif key == ord('c'):
            if len(pts) == 4:
                cv2.destroyWindow("Setup ROI")
                return order_points(np.array(pts))
            else:
                print("⚠️ 점 4개를 모두 찍어야 합니다!")


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # 윈도우 기본 자동 노출/자동 설정 사용
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    except:
        pass

    if not cap.isOpened():
        print("❌ 카메라 오류")
        return

    roi_points = setup_trapezoid_roi(cap)
    if roi_points is None:
        print("❌ 설정 취소됨")
        return

    # ROI 윗변을 프레임 폭 전체로 확장
    ret, sample = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        return
    h, w, _ = sample.shape
    roi_points[0][0] = 0
    roi_points[1][0] = w - 1

    # 필요하면 아랫변도 끝까지 늘리고 싶으면 아래 두 줄도 켜면 됨
    roi_points[3][0] = 0
    roi_points[2][0] = w - 1

    roi_y_top = (roi_points[0][1] + roi_points[1][1]) // 2

    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    filename = f"dataset/trapezoid_{datetime.datetime.now().strftime('%H%M%S')}.mp4"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                          20.0, (FRAME_WIDTH * 2, FRAME_HEIGHT))

    last_serial_time = 0

    print("🚀 3초 후 출발합니다!")
    time.sleep(3)
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
            mask = cv2.inRange(
                hls,
                np.array([0, L_MIN, 0]),
                np.array([179, 255, S_MAX])
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones(MORPH_SIZE, np.uint8))
            edges = cv2.Canny(mask, 50, 150)

            cropped = region_of_interest(edges,
                                         np.array([roi_points], np.int32))

            lines = cv2.HoughLinesP(
                cropped, 1, np.pi / 180, 50,
                minLineLength=40, maxLineGap=100
            )
            left, right = average_slope_intercept(frame, lines, roi_y_top)

            angle, target_x = calculate_steering_angle(
                frame, left, right, roi_y_top
            )

            clamped_angle = max(-45, min(45, angle))
            servo_val = int(map_value(
                clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX
            ))

            current_time = time.time()
            if ser and (current_time - last_serial_time > SERIAL_DELAY):
                ser.write(f"S,{servo_val}\n".encode())
                last_serial_time = current_time

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

            combined = np.hstack((frame, mask_bgr))
            cv2.putText(combined, f"Angle: {int(angle)} | Servo: {servo_val}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

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
