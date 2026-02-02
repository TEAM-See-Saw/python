import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

ROI_HEIGHT_RATIO = 0.5

# 직사각형 ROI (와이드 모드)
ROI_X_LEFT_RATIO = 0.0
ROI_X_RIGHT_RATIO = 1.0

last_target_x = 320

# 자동 튜닝 목표 비율 (3% ~ 10%)
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

# HLS 필터 기준값 (초기값은 함수로 계산됨)
# current_l_min = 160  <-- 이 값은 이제 무시되고 자동 설정됨
MIN_L_VAL = 80       
MAX_L_VAL = 220      
S_MAX_VAL = 80       

# 노이즈 제거 필터값
MORPH_SIZE = (3, 3)
BLUR_K = 3

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
# [3] 영상 처리 함수
# ==========================================

# ★ [핵심] 절대 죽지 않는 '초기 밝기 분석' 함수
def find_optimal_l_min(cap):
    print("🔎 출발 전 밝기 분석 중...")
    
    try:
        # 1. 카메라 워밍업 (20프레임 동안 빛 적응 대기)
        for i in range(20):
            ret, _ = cap.read()
            if not ret:
                time.sleep(0.05) # 읽기 실패 시 잠깐 대기

        # 2. 분석용 프레임 진짜로 읽기
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ 화면 읽기 실패 -> 기본값(140) 사용")
            return 140

        # 3. 전처리 (메인 루프와 동일한 방식)
        frame = cv2.resize(frame, (640, 480))
        blurred = cv2.medianBlur(frame, BLUR_K)
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1] # 밝기(L) 채널만 뽑기

        # 4. ROI 영역(바닥)만 잘라내기
        h, w = frame.shape[:2]
        roi_l = l_channel[int(h * ROI_HEIGHT_RATIO):h, :]

        # 5. 밝기 통계 분석
        pixels = roi_l.flatten()      # 1줄로 쫙 펴기
        pixels = np.sort(pixels)      # 밝은 순서대로 정렬
        
        # 바닥에 흰색 차선이 약 5~10% 있다고 가정하고, 상위 10% 지점의 밝기를 찾음
        target_idx = int(len(pixels) * 0.90)
        detected_l = pixels[target_idx]

        # 6. 값 설정 (감지된 흰색 밝기보다 30 정도 낮게 잡아서 여유를 둠)
        optimal_l = detected_l - 30
        
        # 7. 안전장치 (너무 어둡거나 밝으면 강제 고정)
        final_l = max(MIN_L_VAL, min(optimal_l, MAX_L_VAL))
        
        print(f"✅ 분석 완료! (감지:{detected_l} -> 설정:{final_l})")
        return int(final_l)

    except Exception as e:
        print(f"⚠️ 분석 중 에러 발생({e}) -> 기본값(140) 사용")
        return 140


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
    left_fit = []
    right_fit = []
    if lines is None: return None, None
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            # 횡단보도(가로선) 무시
            if abs(slope) < 0.7:
                continue

            if slope < -0.7:
                left_fit.append((slope, intercept))
            elif slope > 0.7:
                right_fit.append((slope, intercept))

    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line


def calculate_steering_angle(image, left_line, right_line):
    global last_target_x
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = last_target_x

    last_target_x = target_x
    dx = target_x - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy))), int(target_x)


def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    global current_l_min

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width = 640
    height = 480
    cap.set(3, width)
    cap.set(4, height)

    if not cap.isOpened(): print("❌ 카메라 오류"); return

    # ★★★★ 여기서 밝기 분석 함수 호출 ★★★★
    current_l_min = find_optimal_l_min(cap)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★

    print("\n🚀 3초 후 출발!");
    for i in range(3, 0, -1): print(f"{i}.."); time.sleep(1)

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0
    last_speed_time = 0

    try:
        while True:
            # 시리얼 버퍼 비우기
            if ser:
                ser.reset_input_buffer()

            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != width: frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            # ------------------------------------------------
            # 1. 전처리
            # ------------------------------------------------
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

            lower_white = np.array([0, current_l_min, 0])
            upper_white = np.array([179, 255, S_MAX_VAL])
            mask = cv2.inRange(hls, lower_white, upper_white)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_SIZE, np.uint8))

            # ------------------------------------------------
            # 2. ROI 및 자동 튜닝
            # ------------------------------------------------
            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            roi_mask_poly = np.zeros_like(mask)
            cv2.fillPoly(roi_mask_poly, [roi_points], 255)
            roi_pixels = cv2.bitwise_and(mask, roi_mask_poly)

            white_count = cv2.countNonZero(roi_pixels)
            total_area = cv2.contourArea(roi_points)
            if total_area == 0: total_area = 1
            ratio = white_count / total_area

            # 자동 튜닝 (주행 중 미세조정)
            if ratio > TARGET_RATIO_MAX:
                current_l_min = min(current_l_min + 2, MAX_L_VAL)
            elif ratio < TARGET_RATIO_MIN:
                current_l_min = max(current_l_min - 2, MIN_L_VAL)

            # ------------------------------------------------
            # 3. 주행 계산
            # ------------------------------------------------
            edges = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges, roi_points)
            lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
            
            left, right = average_slope_intercept(frame, lines)
            
            angle, target = calculate_steering_angle(frame, left, right)
            servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

            # ------------------------------------------------
            # 4. 통신
            # ------------------------------------------------
            if ser:
                curr_time = time.time()
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = curr_time
                if curr_time - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = curr_time

            # ------------------------------------------------
            # 5. 디스플레이
            # ------------------------------------------------
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            cv2.polylines(frame, [roi_points], True, (255, 0, 0), 2)
            cv2.circle(frame, (target, int(h * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            if left is not None:
                x1, y1, x2, y2 = left[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            if right is not None:
                x1, y1, x2, y2 = right[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            combined = np.hstack((frame, mask_bgr))

            info_text = f"L-Min: {current_l_min} | Ratio: {ratio * 100:.1f}%"
            color = (0, 255, 0)
            if ratio > TARGET_RATIO_MAX:
                color = (0, 0, 255)
            elif ratio < TARGET_RATIO_MIN:
                color = (0, 255, 255)

            cv2.putText(combined, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Full Feature Mode", combined)

            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지")
        if ser:
            for _ in range(3):
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")
                time.sleep(0.05)
            ser.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()