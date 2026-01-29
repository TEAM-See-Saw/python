import serial
import time
import keyboard  # pip install keyboard
import cv2  # pip install opencv-python
import datetime  # 파일명 날짜 생성용
import numpy as np

# ==========================================
# [1] 설정
# ==========================================
# ★ 카메라 번호 확인 (0 또는 1)
CAM_INDEX = 0

PORT = 'COM4'  # 본인 포트 번호
BAUDRATE = 9600

# ⚙️ 제어 값 설정
VAL_LEFT = 680
VAL_RIGHT = 480
VAL_CENTER = 570
SPEED_FWD = 255
SPEED_STOP = 0

# ==========================================
# [2] 장치 연결 (아두이노 + 카메라)
# ==========================================
# 1. 아두이노 연결
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    # 카메라만 테스트하려면 아래 exit() 주석 처리
    exit()

# 2. 카메라 연결
print(f"📷 카메라 #{CAM_INDEX} 연결 시도 중...")
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
target_w, target_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

# 3. 녹화 설정 (MP4, H.264)
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"manual_drive_{now}.mp4"
try:
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 맥/윈도우 호환
except:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비상용

fps = 20.0
out = cv2.VideoWriter(filename, fourcc, fps, (target_w, target_h))
print(f"🎥 녹화 시작: {filename}")

# ==========================================
# [3] 메인 루프
# ==========================================
last_steer_val = -1
last_motor_speed = -1
current_action = "STOP"  # 화면 표시용 텍스트

print("\n🚗 [주행 및 녹화 시작]")
print("  키보드 방향키로 조종하세요. (ESC: 종료)")

try:
    while True:
        # ----------------------------------
        # 0. 카메라 프레임 읽기 & 녹화
        # ----------------------------------
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 카메라 신호 끊김")
            break

        frame = cv2.resize(frame, (target_w, target_h))

        # ----------------------------------
        # 1. 조향 (Steering)
        # ----------------------------------
        if keyboard.is_pressed('left'):
            target_val = VAL_LEFT
            steer_text = "LEFT"
        elif keyboard.is_pressed('right'):
            target_val = VAL_RIGHT
            steer_text = "RIGHT"
        else:
            target_val = VAL_CENTER
            steer_text = "CENTER"

        if target_val != last_steer_val:
            cmd = f"S,{target_val}\n"
            ser.write(cmd.encode())
            last_steer_val = target_val
            # time.sleep은 영상 끊김 방지를 위해 최소화하거나 제거

        # ----------------------------------
        # 2. 구동 (Drive)
        # ----------------------------------
        if keyboard.is_pressed('up'):
            target_speed = SPEED_FWD
            motor_text = "FWD"
        elif keyboard.is_pressed('down'):
            target_speed = SPEED_STOP
            motor_text = "STOP"
        else:
            target_speed = SPEED_STOP  # 키 떼면 정지
            motor_text = "STOP"

        if target_speed != last_motor_speed:
            cmd = f"D,{target_speed}\n"
            ser.write(cmd.encode())
            last_motor_speed = target_speed

        # ----------------------------------
        # 3. 화면 오버레이 및 저장
        # ----------------------------------
        # 현재 상태를 영상에 글씨로 씀 (나중에 로그 분석할 때 편함)
        info_text = f"Steer: {steer_text} | Motor: {motor_text}"
        cv2.putText(frame, info_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        # 파일 저장
        out.write(frame)

        # 화면 출력
        cv2.imshow("Manual Driving & Recording", frame)

        # ----------------------------------
        # 4. 종료 조건
        # ----------------------------------
        # ESC 키를 누르거나, OpenCV 창에서 q를 누르면 종료
        if keyboard.is_pressed('esc') or (cv2.waitKey(1) & 0xFF == ord('q')):
            print("🛑 프로그램 종료")
            ser.write(b'D,0\n')  # 모터 정지
            time.sleep(0.1)
            ser.write(b'S,570\n')  # 핸들 중앙
            break

        # 루프 속도 조절 (time.sleep 대신 waitKey로 딜레이 대체하여 영상 부드럽게)
        # cv2.waitKey(1)이 이미 1ms 딜레이 역할을 함

except KeyboardInterrupt:
    print("\n강제 종료됨")

finally:
    # 자원 해제 (중요)
    if ser: ser.close()
    if cap: cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    print("💾 저장 완료 및 종료")