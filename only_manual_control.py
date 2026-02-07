import serial
import time
import keyboard
import cv2
import numpy as np  # ★ 화면 합치기용 추가
# ==========================================
# [1] 설정
# ==========================================
CAM_INDEX = 1
PORT = 'COM4'
BAUDRATE = 115200

# 조향값
VAL_LEFT = 680
VAL_RIGHT = 480
VAL_CENTER = 570

# 속도값
SPEED_FWD = 255
SPEED_STOP = 0
SPEED_BWD = -255

SERIAL_INTERVAL = 0.1

# ==========================================
# [2] 초기화
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print("✅ 아두이노 연결 성공")
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

print("🚗 [조종 모드] 시작 (Binary View 포함)")
print("🎮 조작: ↑(전진), ↓(후진), ← →(조향), ESC(종료)")

# ==========================================
# [3] 메인 루프
# ==========================================
last_steer_val = VAL_CENTER
last_motor_speed = SPEED_STOP
last_serial_send_time = 0

try:
    while True:
        # ★ 아두이노 버퍼 비우기 (필수)
        if ser.in_waiting > 0:
            ser.reset_input_buffer()

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()

        # -------------------------------------------------
        # ★ [추가] 바이너리 이미지 생성 (차선 인식 시뮬레이션)
        # -------------------------------------------------
        # 1. 블러 처리 (노이즈 제거)
        blurred = cv2.medianBlur(frame, 5)

        # 2. HLS 변환
        hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

        # 3. 흰색 필터링 (L값 150 이상인 것만 추출)
        # 환경에 따라 (0, 150, 0)의 150을 조절하세요 (어두우면 낮추고, 밝으면 높임)
        mask = cv2.inRange(hls, np.array([0, 150, 0]), np.array([179, 255, 255]))

        # 4. 화면 합치기를 위해 바이너리(1채널)를 컬러(3채널) 형식으로 변환
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # -------------------------------------------------

        # --- 1. 키보드 입력 ---
        if keyboard.is_pressed('left'):
            curr_steer = VAL_LEFT;
            steer_text = "LEFT"
        elif keyboard.is_pressed('right'):
            curr_steer = VAL_RIGHT;
            steer_text = "RIGHT"
        else:
            curr_steer = VAL_CENTER;
            steer_text = "CENTER"

        if keyboard.is_pressed('up'):
            curr_speed = SPEED_FWD;
            motor_text = "FWD"
        elif keyboard.is_pressed('down'):
            curr_speed = SPEED_BWD;
            motor_text = "BWD"
        else:
            curr_speed = SPEED_STOP;
            motor_text = "STOP"

        # --- 2. 전송 ---
        should_send = False
        if (curr_steer != last_steer_val) or (curr_speed != last_motor_speed):
            should_send = True
        elif (current_time - last_serial_send_time > SERIAL_INTERVAL):
            should_send = True

        if should_send:
            try:
                ser.write(f"S,{curr_steer}\n".encode())
                ser.write(f"D,{curr_speed}\n".encode())
                last_steer_val = curr_steer
                last_motor_speed = curr_speed
                last_serial_send_time = current_time
            except Exception as e:
                print(f"전송 오류: {e}")

        # --- 3. 화면 표시 (좌: 원본, 우: 바이너리) ---
        color = (0, 0, 255) if curr_speed < 0 else (0, 255, 0)

        # 텍스트는 원본 화면에만 표시
        cv2.putText(frame, f"Steer: {steer_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Motor: {motor_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 두 화면을 가로로 붙이기 (Horizontal Stack)
        combined_view = np.hstack((frame, mask_bgr))

        # 화면이 너무 크면 0.8배로 축소 (선택 사항)
        # combined_view = cv2.resize(combined_view, (int(1280*0.8), int(480*0.8)))

        cv2.imshow("Dual View: [Camera] + [Binary]", combined_view)

        if keyboard.is_pressed('esc') or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

except KeyboardInterrupt:
    print("강제 종료")

finally:
    if ser.is_open:
        ser.write(b'D,0\n');
        ser.write(b'S,570\n');
        ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 종료.")