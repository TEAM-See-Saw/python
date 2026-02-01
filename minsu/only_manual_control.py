import serial
import time
import keyboard
import cv2

# ==========================================
# [1] 설정
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 115200 # ★ 아두이노와 속도 일치

# 조향값
VAL_LEFT = 680
VAL_RIGHT = 480
VAL_CENTER = 570

# 속도값
SPEED_FWD = 255
SPEED_STOP = 0
SPEED_BWD = -255

# 통신 주기 (0.1초마다 재전송)
SERIAL_INTERVAL = 0.1

# ==========================================
# [2] 초기화
# ==========================================
# 1. 아두이노 연결
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer() # 시작 전 청소
    print("✅ 아두이노 연결 성공")
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()

# 2. 카메라 설정
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

print("🚗 [조종 모드] 시작 (녹화 X)")
print("🎮 조작: ↑(전진), ↓(후진), ← →(조향), ESC(종료)")

# ==========================================
# [3] 메인 루프
# ==========================================
last_steer_val = VAL_CENTER
last_motor_speed = SPEED_STOP
last_serial_send_time = 0

try:
    while True:
        # ==========================================================
        # ★ [핵심 추가] 아두이노 데이터 쓰레기통 비우기
        # ==========================================================
        # 녹화를 안 해도 아두이노는 데이터를 계속 보내므로,
        # 안 비워주면 19초 뒤에 버퍼가 꽉 차서 멈춥니다.
        if ser.in_waiting > 0:
            ser.reset_input_buffer()
        # ==========================================================

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (640, 480))
        current_time = time.time()

        # --- 1. 키보드 입력 처리 ---
        if keyboard.is_pressed('left'):
            curr_steer = VAL_LEFT
            steer_text = "LEFT"
        elif keyboard.is_pressed('right'):
            curr_steer = VAL_RIGHT
            steer_text = "RIGHT"
        else:
            curr_steer = VAL_CENTER
            steer_text = "CENTER"

        if keyboard.is_pressed('up'):
            curr_speed = SPEED_FWD
            motor_text = "FWD"
        elif keyboard.is_pressed('down'):
            curr_speed = SPEED_BWD
            motor_text = "BWD"
        else:
            curr_speed = SPEED_STOP
            motor_text = "STOP"

        # --- 2. 아두이노 전송 (하트비트 적용) ---
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

        # --- 3. 화면 표시 ---
        color = (0, 0, 255) if curr_speed < 0 else (0, 255, 0)
        cv2.putText(frame, f"Steer: {steer_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Motor: {motor_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("RC Car Control (No Rec)", frame)

        if keyboard.is_pressed('esc') or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

except KeyboardInterrupt:
    print("강제 종료")

finally:
    if ser.is_open:
        ser.write(b'D,0\n')
        ser.write(b'S,570\n')
        ser.close()

    cap.release()
    cv2.destroyAllWindows()
    print("👋 조종 종료.")