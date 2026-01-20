import cv2
import numpy as np
import math
import serial
import time
import datetime  # [추가] 파일명에 시간을 넣기 위해 필요

# ==========================================
# [1] 설정 (카메라 및 아두이노)
# ==========================================
# ★ 카메라 번호 선택
CAM_INDEX = 0
# CAM_INDEX = 1

PORT = 'COM4'  # 아두이노 연결 포트
BAUDRATE = 9600  # 통신 속도

# ⚙️ 아두이노 제어 설정
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680  # 좌회전 한계값
SERVO_RIGHT_MAX = 480  # 우회전 한계값
MAX_SPEED = 255  # 주행 최고 속도

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)  # 아두이노 리셋 대기
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()


# ==========================================
# [3] 함수 정의 (기존 로직 유지)
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
# [4] 메인 루프 (카메라 처리)
# ==========================================
def main():
    print(f"📷 카메라 #{CAM_INDEX} 연결 시도 중...")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    # 해상도 강제 설정 (640x480)
    target_width = 640
    target_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    if not cap.isOpened():
        print(f"❌ 오류: 카메라 #{CAM_INDEX}를 열 수 없습니다.")
        return

    # ========================================================
    # [녹화 설정] VideoWriter 초기화
    # ========================================================
    # 현재 시간을 파일명에 포함 (예: recording_20260120_143000.avi)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drive_log_{now}.avi"

    # 코덱 설정 (Windows는 'XVID' 또는 'MJPG' 권장)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20.0  # 녹화 프레임 속도

    # VideoWriter 객체 생성 (파일명, 코덱, FPS, 해상도)
    out = cv2.VideoWriter(filename, fourcc, fps, (target_width, target_height))

    print(f"🎥 녹화 시작: {filename} 파일로 저장됩니다.")
    # ========================================================

    print("🚀 실시간 라인 트레이싱 시작!")
    print("⚠️ 주의: 차체가 공중에 떠 있는지 확인하세요!")

    # 모터 구동 명령 (현재 주석 해제됨)
    ser.write(f"D,{MAX_SPEED}\n".encode())

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠️ 카메라 신호 없음. 종료합니다.")
            break

        # 1. 전처리
        frame = cv2.resize(frame, (target_width, target_height))
        height, width, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. ROI 설정
        roi_vertices = [
            (0, height),
            (width // 2 - 50, int(height * 0.6)),
            (width // 2 + 50, int(height * 0.6)),
            (width, height)
        ]
        cropped_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

        # 3. 차선 검출
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left_line, right_line = average_slope_intercept(frame, lines)

        # 4. 조향각 계산
        steering_angle, target_x = calculate_steering_angle(frame, left_line, right_line)

        # 5. 아두이노 매핑
        clamped_angle = max(-45, min(45, steering_angle))
        servo_value = map_value(clamped_angle, -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX)
        servo_value = int(servo_value)

        # 6. 명령 전송
        cmd = f"S,{servo_value}\n"
        ser.write(cmd.encode())

        # --- 시각화 ---
        line_image = np.zeros_like(frame)
        if left_line is not None:
            for x1, y1, x2, y2 in left_line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        if right_line is not None:
            for x1, y1, x2, y2 in right_line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.line(combo_image, (int(width / 2), height), (int(target_x), int(height * 0.6)), (0, 0, 255), 3)

        cv2.putText(combo_image, f"Angle: {steering_angle:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        cv2.putText(combo_image, f"Servo: {servo_value}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ========================================================
        # [녹화 저장] 현재 처리된 프레임(combo_image)을 파일에 쓰기
        # ========================================================
        out.write(combo_image)
        # ========================================================

        cv2.imshow('Live Lane Tracing', combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 처리
    print("🛑 프로그램 종료: 정지 및 초기화")
    ser.write(b"D,0\n")  # 모터 정지
    time.sleep(0.1)
    ser.write(b"S,570\n")  # 조향 중앙

    # [녹화 종료] 파일 닫기
    out.release()
    print(f"💾 녹화 저장 완료: {filename}")

    ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()