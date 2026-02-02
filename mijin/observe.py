import cv2
import numpy as np
import math
import serial
import time

# ==========================================
# [1] 환경 및 튜닝 설정
# ==========================================
IS_SUNNY = True

PORT = 'COM4'
BAUDRATE = 115200
SERIAL_DELAY = 0.05
SPEED_REFRESH_DELAY = 1.0

CAM_INDEX = 1
MAX_SPEED = 255

# ✅ (이전 대화 기준) 왼쪽 480 / 오른쪽 680 / 센터 570
SERVO_CENTER = 570
SERVO_LEFT_MAX = 480
SERVO_RIGHT_MAX = 680

ROI_HEIGHT_RATIO = 0.6
ROI_X_LEFT_RATIO = 0.3125
ROI_X_RIGHT_RATIO = 0.6875

# 타겟 필터(시간 저역통과)
target_x_f = 320  # filtered target
TARGET_LPF_ALPHA = 0.2  # 0.1~0.3 추천 (작을수록 더 부드러움)

# Auto tuning 목표(“퍼센타일 기반”이지만, 하한값을 천천히 조절하는 용도)
current_l_min = 200
TARGET_RATIO_MIN = 0.03
TARGET_RATIO_MAX = 0.10

# ratio EMA + 튜닝 주기 제한
ratio_ema = None
RATIO_EMA_ALPHA = 0.15         # 0.1~0.2 추천
TUNE_PERIOD_SEC = 0.2          # ✅ 프레임마다 튜닝 금지
last_tune_time = 0.0

# 퍼센타일 기반 임계값 클램프
L_THR_MIN = 140
L_THR_MAX = 235

# 채도(S) 하한 클램프(글레어 억제)
S_THR_MIN = 20
S_THR_MAX = 90

# 전처리 파라미터
if IS_SUNNY:
    print("☀️ 모드: SUNNY")
    MIN_L_VAL = 150
    MAX_L_VAL = 240
    S_MAX_VAL = 50
    MORPH_OPEN_SIZE = (5, 5)
    MORPH_CLOSE_SIZE = (7, 7)
    BLUR_K = 7
else:
    print("🌙 모드: NORMAL")
    MIN_L_VAL = 80
    MAX_L_VAL = 220
    S_MAX_VAL = 80
    MORPH_OPEN_SIZE = (3, 3)
    MORPH_CLOSE_SIZE = (5, 5)
    BLUR_K = 5


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
    ser = None


# ==========================================
# [3] 영상 처리 함수들
# ==========================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
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
            if x1 == x2:
                continue
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


def estimate_target_x(image, left_line, right_line, last_x):
    """기존 로직대로 target_x만 산출"""
    height, width = image.shape[:2]

    if left_line is not None and right_line is not None:
        target_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        target_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        target_x = right_line[0][2] - (width * 0.25)
    else:
        target_x = last_x

    return int(target_x)


def angle_from_target(image, target_x):
    height, width = image.shape[:2]
    car_x = width / 2
    target_y = int(height * ROI_HEIGHT_RATIO)
    dx = target_x - car_x
    dy = (height - target_y)
    return math.degrees(math.atan2(dx, abs(dy)))


def map_servo(angle_deg):
    """
    angle_deg: 음수=왼쪽, 양수=오른쪽
    PWM: left=SERVO_LEFT_MAX, right=SERVO_RIGHT_MAX
    """
    angle_deg = max(-45.0, min(45.0, angle_deg))
    if angle_deg < 0:
        return int(SERVO_CENTER + (SERVO_LEFT_MAX - SERVO_CENTER) * (abs(angle_deg) / 45.0))
    else:
        return int(SERVO_CENTER + (SERVO_RIGHT_MAX - SERVO_CENTER) * (angle_deg / 45.0))


# ==========================================
# [4] 메인 실행
# ==========================================
def main():
    global current_l_min, ratio_ema, last_tune_time, target_x_f

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480

    cap.set(3, width)
    cap.set(4, height)
    cap.set(15, -6)  # 기존 유지

    # ✅ 가능할 때만: 캡쳐 버퍼를 줄여 “과거 프레임” 처리 방지(장치에 따라 무시될 수 있음)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    if not cap.isOpened():
        print("❌ 카메라 오류")
        return

    print("\n🚀 3초 후 출발!")
    for i in range(3, 0, -1):
        print(f"{i}..")
        time.sleep(1)

    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    last_serial_time = 0.0
    last_speed_time = 0.0

    # 타겟 유지용(라인 못 잡을 때)
    last_target_raw = target_x_f

    try:
        while True:
            # Arduino RX 비우기
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            h, w = frame.shape[:2]

            # -------------------------
            # (1) ROI 폴리곤
            # -------------------------
            roi_points = np.array([[
                (0, h), (w, h),
                (int(w * ROI_X_RIGHT_RATIO), int(h * ROI_HEIGHT_RATIO)),
                (int(w * ROI_X_LEFT_RATIO), int(h * ROI_HEIGHT_RATIO))
            ]], dtype=np.int32)

            # ROI 마스크(폴리곤)
            roi_mask_poly = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask_poly, [roi_points], 255)

            # -------------------------
            # (2) 조명 변화 강한 환경용 마스크 생성(퍼센타일 + S + Edge 결합)
            # -------------------------
            blurred = cv2.medianBlur(frame, BLUR_K)
            hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
            H, L, S = cv2.split(hls)

            # ROI 내부 분포로 임계값 결정(퍼센타일)
            L_roi = L[roi_mask_poly == 255]
            S_roi = S[roi_mask_poly == 255]

            # 예외 방지
            if L_roi.size < 50:
                L_thr = current_l_min
                S_thr = 30
            else:
                L_thr = int(np.percentile(L_roi, 90))  # 상위 10% 밝기 기준
                S_thr = int(np.percentile(S_roi, 30))  # 하위 30% 기준(너무 낮은 채도 배제)

            # 클램프 + 하한(current_l_min) 반영
            L_thr = clamp(L_thr, L_THR_MIN, L_THR_MAX)
            L_thr = max(L_thr, current_l_min)
            S_thr = clamp(S_thr, S_THR_MIN, S_THR_MAX)

            # Color mask (HLS 기반)
            color_mask = ((L >= L_thr) & (S >= S_thr)).astype(np.uint8) * 255

            # Edge mask (조명 변화에 비교적 강함)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(gray, 50, 150)

            # 결합 (둘 중 하나라도 강하면 후보)
            mask = cv2.bitwise_or(color_mask, edges)

            # ROI 적용
            mask = cv2.bitwise_and(mask, roi_mask_poly)

            # 노이즈 정리(OPEN + CLOSE)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(MORPH_OPEN_SIZE, np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(MORPH_CLOSE_SIZE, np.uint8))

            # -------------------------
            # (3) ratio 계산(색 마스크 기반으로만 계산하는 게 안정적)
            # -------------------------
            roi_pixels_for_ratio = cv2.bitwise_and(color_mask, roi_mask_poly)
            white_count = cv2.countNonZero(roi_pixels_for_ratio)
            total_area = cv2.contourArea(roi_points) or 1
            ratio = white_count / total_area

            # ratio EMA
            if ratio_ema is None:
                ratio_ema = ratio
            else:
                ratio_ema = (1.0 - RATIO_EMA_ALPHA) * ratio_ema + RATIO_EMA_ALPHA * ratio

            # ✅ Auto tuning은 0.2초마다, 변화폭도 1씩만
            now = time.time()
            if now - last_tune_time >= TUNE_PERIOD_SEC:
                if ratio_ema > TARGET_RATIO_MAX:
                    current_l_min = min(current_l_min + 1, MAX_L_VAL)
                elif ratio_ema < TARGET_RATIO_MIN:
                    current_l_min = max(current_l_min - 1, MIN_L_VAL)
                last_tune_time = now

            # -------------------------
            # (4) 라인 검출 + 타겟 추정
            # -------------------------
            # 허프 입력은 "mask에서 Canny"로 (결합 마스크 기반)
            edges2 = cv2.Canny(mask, 50, 150)
            cropped = region_of_interest(edges2, roi_points)

            lines = cv2.HoughLinesP(
                cropped, 1, np.pi / 180, 50,
                minLineLength=40, maxLineGap=100
            )

            left, right = average_slope_intercept(frame, lines)
            target_raw = estimate_target_x(frame, left, right, last_target_raw)
            last_target_raw = target_raw

            # ✅ 타겟 저역통과 필터(튐 방지)
            target_x_f = int((1.0 - TARGET_LPF_ALPHA) * target_x_f + TARGET_LPF_ALPHA * target_raw)
            target_x_f = clamp(target_x_f, 0, w - 1)

            # 필터 타겟 기준으로 조향각 계산
            angle = angle_from_target(frame, target_x_f)

            # servo 값 계산
            servo_val = map_servo(angle)

            # -------------------------
            # (5) 통신(Heartbeat)
            # -------------------------
            if ser:
                curr_time = time.time()
                if curr_time - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = curr_time
                if curr_time - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{MAX_SPEED}\n".encode())
                    last_speed_time = curr_time

            # -------------------------
            # (6) 디스플레이
            # -------------------------
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)

            # raw 타겟(초록), filtered 타겟(빨강)
            y_ref = int(h * ROI_HEIGHT_RATIO)
            cv2.circle(frame, (int(target_raw), y_ref), 6, (0, 255, 0), -1)
            cv2.circle(frame, (int(target_x_f), y_ref), 10, (0, 0, 255), -1)

            combined = np.hstack((frame, mask_bgr))

            cv2.putText(
                combined,
                f"L_thr(p90)~{L_thr} S_thr(p30)~{S_thr} | L-min:{current_l_min}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            cv2.putText(
                combined,
                f"ratio:{ratio*100:.1f}% ema:{ratio_ema*100:.1f}% | angle:{angle:.1f} servo:{servo_val}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )

            cv2.imshow("Robust LineTracing (Percentile+S+Edge+LPF)", combined)
            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        print("\n🛑 안전 정지")
        if ser:
            try:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(f"S,{SERVO_CENTER}\n".encode())
                    time.sleep(0.05)
                ser.close()
            except:
                pass

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()