import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 설정 (본인 환경에 맞게 수정)
# ==========================================
PORT = 'COM4'  # 아두이노 연결 포트
BAUDRATE = 9600  # 통신 속도

# 영상 경로 (사용자 kimmi 환경에 맞춤)
VIDEO_PATH = r'C:\Users\kimmi\Downloads\curv.mp4'

# ⚙️ 아두이노 제어 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680  # 좌회전 한계값
SERVO_RIGHT_MAX = 480  # 우회전 한계값
MAX_SPEED = 255  # 주행 최고 속도 (0~255)

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)  # 아두이노 리셋 대기
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    # 아두이노 없이 영상만 테스트하려면 아래 exit()를 주석 처리하세요.
    exit()


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


def map_value(x, in_min, in_max, out_min, out_max):
    # 범위 변환 함수 (Arduino map과 동일)
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


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 영상 파일 열기 확인
    if not cap.isOpened():
        print(f"❌ 오류: 영상 파일을 열 수 없습니다.\n👉 경로 확인: {VIDEO_PATH}")
        return

    print(f"🚀 시뮬레이션 시작! 최고 속도(PWM {MAX_SPEED})로 뒷바퀴가 회전합니다.")
    print("⚠️ 주의: 차체가 공중에 떠 있는지 확인하세요!")

    # [핵심] 시작하자마자 최고 속도 명령 전송
    ser.write(f"D,{MAX_SPEED}\n".encode())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # 영상이 끝나면 처음으로 되감기 (무한 루프)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. 전처리 (Resize & Edge Detection)
        frame = cv2.resize(frame, (640, 480))
        height, width, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. ROI 설정 (관심 영역)
        roi_vertices = [
            (0, height),
            (width // 2 - 50, int(height * 0.6)),
            (width // 2 + 50, int(height * 0.6)),
            (width, height)
        ]
        cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 3. 차선 검출 (Hough Transform)
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left_line, right_line = average_slope_intercept(frame, lines)

        # 4. 조향각 계산
        steering_angle, target_x = calculate_steering_angle(frame, left_line, right_line)

        # 5. [Mapping] 각도 -> 아두이노 서보 값 변환
        # 영상 각도: -45(Left) ~ 45(Right) 가정
        # 서보 값: 680(Left) ~ 480(Right)

        clamped_angle = max(-45, min(45, steering_angle))  # 각도 제한
        servo_value = map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX)
        servo_value = int(servo_value)

        # 6. 아두이노로 조향 명령 전송 (S,값\n)
        cmd = f"S,{servo_value}\n"
        ser.write(cmd.encode())

        # --- 시각화 (화면에 그리기) ---
        line_image = np.zeros_like(frame)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        # 주행 방향 가이드라인 (Red)
        cv2.line(combo_image, (int(width / 2), height), (int(target_x), int(height * 0.6)), (0, 0, 255), 3)

        # 정보 텍스트 표시
        cv2.putText(combo_image, f"Angle: {steering_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        cv2.putText(combo_image, f"Servo: {servo_value}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combo_image, f"Motor: {MAX_SPEED} (MAX)", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Lane Assist Simulation (Running)', combo_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ------------------------------------------
    # [종료 절차] 안전하게 정지
    # ------------------------------------------
    print("🛑 시뮬레이션 종료: 모터 및 조향 초기화")
    ser.write(b"D,0\n")  # 모터 정지
    time.sleep(0.1)
    ser.write(b"S,570\n")  # 조향 중앙 정렬

    ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()