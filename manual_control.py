import serial
import time
import keyboard  # pip install keyboard

# ==========================================
# [1] 설정 (아두이노 코드의 const 값과 일치시킴)
# ==========================================
PORT = 'COM4'       # 본인 포트 번호로 수정
BAUDRATE = 9600

# ⚙️ 포텐쇼미터 목표값 설정 (아두이노 코드 참고)
# LIMIT_MAX가 680(좌), LIMIT_MIN이 480(우), CENTER가 570
VAL_LEFT = 680      # 좌회전 목표값
VAL_RIGHT = 480     # 우회전 목표값
VAL_CENTER = 570    # 중앙 목표값

# ⚙️ 모터 속도 설정 (0 ~ 200, 아두이노 MAX가 200임)
SPEED_FWD = 150     # 전진 속도
SPEED_STOP = 0      # 정지

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    print(f"✅ {PORT} 포트에 연결되었습니다.")
    time.sleep(2)  # 아두이노 리셋 대기
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    exit()

# ==========================================
# [3] 상태 관리 변수
# ==========================================
last_steer_val = -1
last_motor_speed = -1

print("\n🚗 [포텐쇼미터 피드백 제어 모드]")
print(f"  ⬆️  (위)   : 전진 (속도 {SPEED_FWD})")
print(f"  ⬇️  (아래) : 정지 (속도 0)")
print(f"  ⬅️  (좌)   : 좌회전 (목표값 {VAL_LEFT})")
print(f"  ➡️  (우)   : 우회전 (목표값 {VAL_RIGHT})")
print("  ESC        : 종료")
print("===========================\n")

try:
    while True:
        # ----------------------------------
        # 1. 조향 (Steering) - 'S' 명령
        # ----------------------------------
        if keyboard.is_pressed('left'):
            target_val = VAL_LEFT
        elif keyboard.is_pressed('right'):
            target_val = VAL_RIGHT
        else:
            target_val = VAL_CENTER  # 키 떼면 중앙 복귀

        # 값이 바뀌었을 때만 아두이노로 전송 (통신 부하 감소)
        if target_val != last_steer_val:
            # 프로토콜: "S,숫자" (예: S,680)
            cmd = f"S,{target_val}\n"
            ser.write(cmd.encode())
            print(f"Steering -> {cmd}")
            last_steer_val = target_val
            time.sleep(0.02) # 전송 안정화 대기

        # ----------------------------------
        # 2. 구동 (Drive) - 'D' 명령
        # ----------------------------------
        if keyboard.is_pressed('up'):
            target_speed = SPEED_FWD
        elif keyboard.is_pressed('down'):
            target_speed = SPEED_STOP
        else:
            # 키를 떼면 멈추게 하려면 아래 주석 해제 (지금은 누르고 있어야 전진)
            target_speed = SPEED_STOP
            pass

        # 값이 바뀌었을 때만 전송
        if target_speed != last_motor_speed:
            # 프로토콜: "D,숫자" (예: D,150)
            cmd = f"D,{target_speed}\n"
            ser.write(cmd.encode())
            print(f"Motor    -> {cmd}")
            last_motor_speed = target_speed
            time.sleep(0.02) # 전송 안정화 대기

        # ----------------------------------
        # 3. 종료
        # ----------------------------------
        if keyboard.is_pressed('esc'):
            print("🛑 프로그램 종료")
            ser.write(b'D,0\n')     # 모터 정지
            time.sleep(0.1)
            ser.write(b'S,570\n')   # 핸들 중앙
            break

        time.sleep(0.05)  # 루프 주기

except KeyboardInterrupt:
    print("\n강제 종료됨")
finally:
    ser.close()