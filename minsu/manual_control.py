import serial
import time
import keyboard
import cv2
import datetime
import csv
import os

# ==========================================
# [1] 설정
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

# 조향값
VAL_LEFT = 680
VAL_RIGHT = 480
VAL_CENTER = 570

# 속도값 (음수값을 넣으면 후진합니다)
SPEED_FWD = 255  # 전진 속도
SPEED_STOP = 0  # 정지
SPEED_BWD = -255  # ★ [추가] 후진 속도 (너무 빠르면 -150 정도로 줄이세요)

# ==========================================
# [2] 초기화
# ==========================================
# 1. 아두이노
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()

# 2. 카메라
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# 3. 파일 저장 설정
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"drive_data_{now}.mp4"
csv_filename = f"drive_data_{now}.csv"

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['elapsed_time', 'servo_val', 'motor_speed'])

print(f"🎥 녹화 및 로깅 시작: {video_filename}, {csv_filename}")
print("🚗 조작: ↑(전진), ↓(후진), ← →(조향), ESC(종료)")

# ==========================================
# [3] 메인 루프
# ==========================================
last_steer_val = VAL_CENTER
last_motor_speed = SPEED_STOP
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))

        # --- 1. 키보드 입력 처리 ---
        # (1) 조향
        if keyboard.is_pressed('left'):
            curr_steer = VAL_LEFT
            steer_text = "LEFT"
        elif keyboard.is_pressed('right'):
            curr_steer = VAL_RIGHT
            steer_text = "RIGHT"
        else:
            curr_steer = VAL_CENTER
            steer_text = "CENTER"

        # (2) 구동 (후진 로직 추가됨)
        if keyboard.is_pressed('up'):
            curr_speed = SPEED_FWD
            motor_text = "FWD"
        elif keyboard.is_pressed('down'):
            curr_speed = SPEED_BWD  # ★ [수정] 후진 속도 적용
            motor_text = "BWD"  # ★ [수정] 텍스트 변경
        else:
            curr_speed = SPEED_STOP
            motor_text = "STOP"

        # --- 2. 아두이노 전송 ---
        if curr_steer != last_steer_val:
            ser.write(f"S,{curr_steer}\n".encode())
            last_steer_val = curr_steer

        if curr_speed != last_motor_speed:
            ser.write(f"D,{curr_speed}\n".encode())
            last_motor_speed = curr_speed

        # --- 3. 데이터 로깅 ---
        elapsed_time = round(time.time() - start_time, 3)
        # CSV에는 음수 속도(-200)도 그대로 기록되어 나중에 학습할 때 유용합니다.
        csv_writer.writerow([elapsed_time, last_steer_val, last_motor_speed])

        # --- 4. 화면 표시 ---
        cv2.putText(frame, f"Time: {elapsed_time}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 후진일 때 빨간색 글씨로 표시
        color = (0, 0, 255) if curr_speed < 0 else (0, 255, 0)
        cv2.putText(frame, f"Cmd: {steer_text} | {motor_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        out.write(frame)
        cv2.imshow("Recording...", frame)

        if keyboard.is_pressed('esc') or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

except KeyboardInterrupt:
    print("종료 중...")

finally:
    ser.write(b'D,0\n')
    ser.write(b'S,570\n')
    ser.close()
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("💾 데이터 저장 완료.")