import serial
import time
import numpy as np
import cv2

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
CAM_INDEX = 0

# 속도
SPEED_SEARCH = 80
SPEED_SWING = 80
SPEED_PARK = 75
SPEED_STOP = 0

# 서보
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
STEER_WAIT_TIME = 0.8

# ★ 튜닝 포인트: 감지 민감도
# 종이 박스 인식이 잘 안되면 THRES_HIGH를 0.05 정도로 낮춰보세요.
EDGE_THRES_HIGH = 0.05  # 5% 이상 엣지면 '차'
EDGE_THRES_LOW = 0.02  # 2% 이하면 '빈 공간'

# ROI 설정 (화면 오른쪽 영역)
ROI_X_RATIO = 0.60  # 화면 가로 60% 지점부터 오른쪽을 봄
ROI_Y_MIN = 0.30  # 화면 세로 상단 30% 지점부터 (바닥/하늘 제외)
ROI_Y_MAX = 0.80  # 화면 세로 하단 80% 지점까지

# 상태
STATE_SEARCH = 0
STATE_SWING_OUT = 1
STATE_REVERSE_ENTRY = 2
STATE_DONE = 3

# 탐색 단계
STEP_CAR1 = 0
STEP_GAP = 1
STEP_CAR2 = 2


# ==========================================
# [2] 영상 처리 및 시각화 함수
# ==========================================
def process_vision(frame):
    """
    원본 프레임을 받아 엣지 분석 후,
    1. 감지 상태 (CAR/EMPTY)
    2. 엣지 밀도 (density)
    3. 시각화용 엣지 이미지 (edge_visual)
    를 반환합니다.
    """
    if frame is None: return 'UNKNOWN', 0, None

    h, w = frame.shape[:2]

    # 1. ROI 좌표 계산
    x_start = int(w * ROI_X_RATIO)
    y_start = int(h * ROI_Y_MIN)
    y_end = int(h * ROI_Y_MAX)

    # ROI 잘라내기
    roi = frame[y_start:y_end, x_start:w]

    # 2. 전처리 (Gray -> Blur -> Canny)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ★ Canny 임계값 조절 (박스 인식률 높이려면 30, 100 정도로 낮춰보세요)
    edges = cv2.Canny(blur, 30, 80)

    # 3. 밀도 계산
    total_pixels = edges.size
    white_pixels = cv2.countNonZero(edges)
    density = white_pixels / total_pixels if total_pixels > 0 else 0

    # 4. 판단
    status = 'UNKNOWN'
    if density > EDGE_THRES_HIGH:
        status = 'CAR'
    elif density < EDGE_THRES_LOW:
        status = 'EMPTY'
    else:
        status = 'UNCERTAIN'  # 애매한 구간

    # ==========================================
    # ★ [시각화 생성] 검은 배경에 엣지만 그리기
    # ==========================================
    # 1) 전체 화면 크기의 검은 도화지 생성
    edge_visual = np.zeros((h, w, 3), dtype=np.uint8)

    # 2) 엣지 이미지는 흑백(1채널)이므로 컬러(3채널)로 변환
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3) 상태에 따라 엣지 색상 변경 (빨강=차, 초록=빈공간)
    if status == 'CAR':
        edges_bgr[edges > 0] = (0, 0, 255)  # Red Edge
    elif status == 'EMPTY':
        edges_bgr[edges > 0] = (0, 255, 0)  # Green Edge
    else:
        edges_bgr[edges > 0] = (0, 255, 255)  # Yellow Edge

    # 4) 검은 도화지의 ROI 위치에 채색된 엣지 붙여넣기
    edge_visual[y_start:y_end, x_start:w] = edges_bgr

    # 5) ROI 박스 그리기 (시각화 화면에도)
    cv2.rectangle(edge_visual, (x_start, y_start), (w, y_end), (255, 255, 255), 1)

    return status, density, edge_visual


# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    global ser

    # 카메라 설정
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, 640);
    cap.set(4, 480)

    # 아두이노 연결
    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        print("✅ 아두이노 연결됨")
        time.sleep(2)
    except:
        print("⚠️ 아두이노 없음 (시뮬레이션 모드)")
        ser = None

    state = STATE_SEARCH
    search_step = STEP_CAR1
    timer = 0

    print("🚀 시각화 모드 시작")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ★ 비전 처리 및 시각화 이미지 받기
        status, density, edge_view = process_vision(frame)

        cmd_speed = 0
        cmd_servo = SERVO_CENTER
        curr_time = time.time()
        msg = f"Dens: {density:.2f} [{status}]"

        # ---------------------------------------------------
        # [로직] (기존과 동일)
        # ---------------------------------------------------
        if state == STATE_SEARCH:
            cmd_speed = SPEED_SEARCH

            if search_step == STEP_CAR1:
                if status == 'EMPTY':
                    print("👀 빈 공간 시작 (Gap Start)")
                    search_step = STEP_GAP

            elif search_step == STEP_GAP:
                if status == 'CAR':
                    print("🛑 2번 차 감지! (Trigger)")
                    if ser:
                        ser.write(b"D,-150\n");
                        time.sleep(0.1)
                        ser.write(b"D,0\n");
                        time.sleep(1.0)
                    state = STATE_SWING_OUT
                    timer = curr_time

        elif state == STATE_SWING_OUT:
            cmd_servo = SERVO_LEFT_MAX
            if curr_time - timer < STEER_WAIT_TIME:
                cmd_speed = 0
            else:
                cmd_speed = SPEED_SWING

            if curr_time - timer > (STEER_WAIT_TIME + 1.2):
                state = STATE_REVERSE_ENTRY
                timer = curr_time
                if ser: ser.write(b"D,0\n"); time.sleep(0.5)

        elif state == STATE_REVERSE_ENTRY:
            cmd_servo = SERVO_RIGHT_MAX
            if curr_time - timer < STEER_WAIT_TIME:
                cmd_speed = 0
            else:
                cmd_speed = -SPEED_PARK

            if curr_time - timer > (STEER_WAIT_TIME + 3.0):
                state = STATE_DONE

        elif state == STATE_DONE:
            cmd_speed = 0
            if ser: ser.write(b"D,0\n")

        # 명령 전송
        if ser:
            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

        # ---------------------------------------------------
        # [화면 출력] 2개의 창 띄우기
        # ---------------------------------------------------

        # 1. 원본 화면 (Main Camera)
        h, w = frame.shape[:2]
        # ROI 박스 표시
        roi_color = (0, 255, 0) if status == 'EMPTY' else (0, 0, 255)
        x_s = int(w * ROI_X_RATIO);
        y_s = int(h * ROI_Y_MIN);
        y_e = int(h * ROI_Y_MAX)
        cv2.rectangle(frame, (x_s, y_s), (w, y_e), roi_color, 2)
        cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)
        cv2.imshow("1. Main Camera", frame)

        # 2. 엣지 마스크 화면 (Obstacle Mask)
        # 여기가 '종이 박스'가 보이는지 확인하는 곳입니다.
        if edge_view is not None:
            cv2.putText(edge_view, "Obstacle Edge View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("2. Obstacle Mask (Edge)", edge_view)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()