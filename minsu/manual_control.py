import serial
import time
import keyboard
import cv2
import datetime
import csv  # ★ 엑셀 저장을 위해 추가
import os

# ==========================================
# [1] 설정
# ==========================================
CAM_INDEX = 0
PORT = 'COM4'
BAUDRATE = 9600

VAL_LEFT = 680
VAL_RIGHT = 480
VAL_CENTER = 570
SPEED_FWD = 255
SPEED_STOP = 0

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

# 3. 파일 저장 설정 (영상 + 로그)
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"drive_data_{now}.mp4"
csv_filename = f"drive_data_{now}.csv"  # ★ 로그 파일 이름

# 영상 라이터
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

# ★ 로그 파일 생성 및 헤더 작성
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
# 헤더: 경과시간, 조향각, 모터속도
csv_writer.writerow(['elapsed_time', 'servo_val', 'motor_speed'])

print(f"🎥 녹화 및 로깅 시작: {video_filename}, {csv_filename}")
print("🚗 주행을 시작하세요! (ESC: 종료)")

# ==========================================
# [3] 메인 루프
# ==========================================
last_steer_val = VAL_CENTER
last_motor_speed = SPEED_STOP

# ★ 시작 시간 기준점
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 480))

        # --- 1. 키보드 입력 처리 ---
        # 조향
        if keyboard.is_pressed('left'):
            curr_steer = VAL_LEFT
            steer_text = "LEFT"
        elif keyboard.is_pressed('right'):
            curr_steer = VAL_RIGHT
            steer_text = "RIGHT"
        else:
            curr_steer = VAL_CENTER
            steer_text = "CENTER"

        # 구동
        if keyboard.is_pressed('up'):
            curr_speed = SPEED_FWD
            motor_text = "FWD"
        elif keyboard.is_pressed('down'):
            curr_speed = SPEED_STOP  # 후진 구현 시 변경 가능
            motor_text = "STOP"
        else:
            curr_speed = SPEED_STOP
            motor_text = "STOP"

        # --- 2. 아두이노 전송 (값이 바뀔 때만 보내면 통신 부하 줄임) ---
        # 하지만 정밀한 리플레이를 위해 매 프레임 기록하거나,
        # 여기서는 "명령을 보낸 시점"을 기록합니다.

        # 조향 명령
        if curr_steer != last_steer_val:
            ser.write(f"S,{curr_steer}\n".encode())
            last_steer_val = curr_steer

        # 속도 명령
        if curr_speed != last_motor_speed:
            ser.write(f"D,{curr_speed}\n".encode())
            last_motor_speed = curr_speed

        # --- 3. ★ 데이터 로깅 (핵심) ---
        # 현재 시간 - 시작 시간 = 경과 시간
        elapsed_time = round(time.time() - start_time, 3)

        # CSV에 한 줄 저장: [0.123초, 570, 255]
        csv_writer.writerow([elapsed_time, last_steer_val, last_motor_speed])

        # --- 4. 화면 표시 및 영상 저장 ---
        cv2.putText(frame, f"Time: {elapsed_time}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Cmd: {steer_text} | {motor_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
    csv_file.close()  # ★ 파일 닫기 중요
    cv2.destroyAllWindows()
    print("💾 데이터 저장 완료.")