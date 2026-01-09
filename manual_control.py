import serial
import time
import keyboard  # pip install keyboard

# ==========================================
# [1] 설정 (본인 환경에 맞게 수정 필수)
# ==========================================
PORT = 'COM3'  # 아두이노가 연결된 포트 번호 (장치관리자 확인)
BAUDRATE = 9600  # 아두이노 코드의 Serial.begin 값과 일치해야 함

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)  # 아두이노 리셋 대기
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    exit()

# ==========================================
# [3] 상태 관리 변수
# ==========================================
# 중복 명령 전송을 막기 위해 현재 상태를 저장합니다.
last_steer_cmd = None
last_motor_cmd = None

print("\n🚗 [조작 설명서]")
print("  ⬆️  (위)   : 전진 (전속력)")
print("  ⬇️  (아래) : 정지")
print("  ⬅️  (좌)   : 좌회전")
print("  ➡️  (우)   : 우회전")
print("  (좌우 뗌)  : 중앙 정렬")
print("  ESC       : 종료")
print("===========================\n")

try:
    while True:
        # 1. 조향 (Steering) 제어
        # 키보드를 누르고 있는 동안 해당 방향으로 꺾고, 떼면 중앙으로 복귀
        if keyboard.is_pressed('left'):
            current_steer = 'L'
        elif keyboard.is_pressed('right'):
            current_steer = 'R'
        else:
            current_steer = 'C'  # 아무것도 안 누르면 중앙

        # 2. 모터 (Motor) 제어
        # 위쪽 화살표를 누르면 전진, 아래쪽은 정지
        if keyboard.is_pressed('up'):
            current_motor = '2'  # 아두이노 코드의 '2' (moveRearMotors 200)
        elif keyboard.is_pressed('down'):
            current_motor = '0'  # 아두이노 코드의 '0' (stopRearMotors)
        else:
            # 키를 뗐을 때 멈추게 하려면 아래 주석 해제 (지금은 누를 때만 동작)
            # current_motor = '0'
            pass
            # (현재 로직: 위를 누르면 전진, 아래를 눌러야 정지.
            #  전진 키 떼도 계속 가는 게 편하면 pass 유지, 떼면 멈추게 하려면 위 주석 해제)

        # 3. 명령 전송 (상태가 바뀌었을 때만 전송)

        # (1) 조향 명령 전송
        if current_steer != last_steer_cmd:
            ser.write(current_steer.encode())
            print(f"Steering: {current_steer}")
            last_steer_cmd = current_steer

        # (2) 모터 명령 전송 (값이 할당되었을 때만)
        if 'current_motor' in locals():
            if current_motor != last_motor_cmd:
                ser.write(current_motor.encode())
                print(f"Motor: {current_motor}")
                last_motor_cmd = current_motor

        # 종료 조건
        if keyboard.is_pressed('esc'):
            print("🛑 프로그램 종료")
            ser.write(b'0')  # 모터 정지 명령 보내고 종료
            ser.write(b'C')  # 조향 중앙 복귀
            break

        time.sleep(0.05)  # CPU 점유율 방지 (50ms 대기)

except KeyboardInterrupt:
    print("\n강제 종료됨")
finally:
    ser.close()