import cv2
import numpy as np
import time

# ==========================================
# [1] 설정값 (현장 튜닝 필수!)
# ==========================================
# 신호등 카메라 번호 (보통 주행이 0이면, 이건 1)
TL_CAM_INDEX = 0

# 해상도
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ★ 튜닝 포인트 1: 관심 영역 (ROI)
# 신호등은 보통 하늘(화면 상단)에 있습니다. 불필요한 바닥을 잘라냅니다.
ROI_Y_TOP = 0      # 화면 맨 위부터
ROI_Y_BOTTOM = int(FRAME_HEIGHT * 0.6) # 상단 60% 영역만 감지

# ★ 튜닝 포인트 2: 색상 임계값 (HSV 범위)
# 환경(조명, 날씨)에 따라 이 값을 조절해야 합니다.
# H:색상(0~179), S:채도(0~255), V:명도(0~255)

# 빨강 (Red는 H값이 0 근처와 179 근처 두 군데에 걸쳐 있음)
lower_red1 = np.array([0, 150, 150])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 150, 150])
upper_red2 = np.array([179, 255, 255])

# 노랑 (Yellow)
lower_yellow = np.array([15, 150, 150])
upper_yellow = np.array([35, 255, 255])

# 초록 (Green) - 신호등 초록은 약간 청록색(Cyan)에 가까울 수 있음
lower_green = np.array([40, 150, 150])
upper_green = np.array([90, 255, 255])

# 최소 감지 면적 (노이즈 필터링용)
MIN_AREA = 100  # 이 픽셀 수보다 작은 불빛은 무시

# 상태 정의
TL_STATE_RED = "RED"
TL_STATE_YELLOW = "YELLOW"
TL_STATE_GREEN = "GREEN"
TL_STATE_UNKNOWN = "UNKNOWN"


# ==========================================
# [2] 신호등 감지 함수 (핵심)
# ==========================================
def detect_traffic_light(frame):
    """
    프레임을 받아 현재 가장 유력한 신호등 상태를 반환
    Return: state(문자열), debug_image(이미지)
    """
    if frame is None: return TL_STATE_UNKNOWN, None

    # 1. ROI 설정 (상단부만 잘라냄)
    roi = frame[ROI_Y_TOP:ROI_Y_BOTTOM, :]
    debug_img = roi.copy()
    
    # 2. 전처리 (HSV 변환 + 블러링)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # 3. 색상별 마스크 생성
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2) # 두 범위 합치기
    
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 4. 노이즈 제거 (모폴로지 연산)
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.erode(mask_red, kernel, iterations=1)
    mask_red = cv2.dilate(mask_red, kernel, iterations=2)
    
    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=2)
    
    mask_green = cv2.erode(mask_green, kernel, iterations=1)
    mask_green = cv2.dilate(mask_green, kernel, iterations=2)

    # 5. 각 색상의 덩어리(Contour) 면적 계산 함수
    def get_max_area(mask, color_bgr, debug_frame):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                # 시각화: 감지된 영역에 박스 그리기
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color_bgr, 2)
                if area > max_area:
                    max_area = area
        return max_area

    # 색상별 최대 면적 계산
    area_r = get_max_area(mask_red, (0, 0, 255), debug_img)
    area_y = get_max_area(mask_yellow, (0, 255, 255), debug_img)
    area_g = get_max_area(mask_green, (0, 255, 0), debug_img)

    # 6. 최종 판단 (가장 큰 면적을 가진 색상 선택)
    detected_state = TL_STATE_UNKNOWN
    max_detected_area = max(area_r, area_y, area_g)

    if max_detected_area > MIN_AREA:
        if max_detected_area == area_r:
            detected_state = TL_STATE_RED
        elif max_detected_area == area_y:
            detected_state = TL_STATE_YELLOW
        elif max_detected_area == area_g:
            detected_state = TL_STATE_GREEN

    return detected_state, debug_img


# ==========================================
# [3] 메인 테스트 루프
# ==========================================
def main():
    print("🚦 신호등 감지 카메라 시작 (CAM Index: {})".format(TL_CAM_INDEX))
    
    cap = cv2.VideoCapture(TL_CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # ★ 중요: 신호등 카메라는 노출을 적절히 조절해야 불빛이 하얗게 번지지 않습니다.
    # 현장에서 -3 ~ -7 사이로 조절해 보세요.
    cap.set(cv2.CAP_PROP_EXPOSURE, -5) 

    if not cap.isOpened():
        print("❌ 신호등 카메라 연결 실패")
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 감지 수행
        state, debug_frame = detect_traffic_light(frame)
        
        # FPS 계산
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # 결과 출력
        # 상태에 따른 텍스트 색상
        text_color = (255, 255, 255)
        if state == TL_STATE_RED: text_color = (0, 0, 255)
        elif state == TL_STATE_YELLOW: text_color = (0, 255, 255)
        elif state == TL_STATE_GREEN: text_color = (0, 255, 0)

        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(debug_frame, f"STATE: {state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

        # 전체 화면에 ROI 영역 표시 (참고용)
        cv2.rectangle(frame, (0, ROI_Y_TOP), (FRAME_WIDTH, ROI_Y_BOTTOM), (255, 0, 255), 2)
        
        cv2.imshow("Traffic Light Camera (Raw + ROI)", frame)
        cv2.imshow("Detection Result (ROI View)", debug_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()