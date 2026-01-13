import serial
import time
import keyboard
import json
import os

# 아두이노 코드를 steering.ino로 변경 후 실행할 것!

# ==========================================
# [1] 설정
# ==========================================
PORT = 'COM4'  # 포트 번호 확인 필수
BAUDRATE = 9600
LOG_FILE = 'driving_log.json'  # 기록 저장 파일명

# ==========================================
# [2] 시리얼 연결
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE)
    print(f"✅ {PORT} 포트 연결 성공!")
    time.sleep(2)
except Exception as e:
    print(f"❌ 연결 실패: {e}")
    exit()


# ==========================================
# [3] 기능 정의
# ==========================================

def record_mode():
    """ 사람이 운전하는 내용을 시간과 함께 기록 """
    print("\n🔴 [녹화 모드] 시작! (운전을 시작하세요)")
    print("ESC를 누르면 녹화가 종료되고 파일로 저장됩니다.")

    log_data = []  # [(경과시간, 조향명령, 모터명령), ...]
    start_time = time.time()

    last_steer = 'C'
    last_motor = '0'

    # 초기 상태 기록 (0초, 정지)
    log_data.append({'time': 0.0, 'steer': 'C', 'motor': '0'})

    try:
        while True:
            # --- 키보드 입력 감지 ---
            current_steer = 'C'
            if keyboard.is_pressed('left'):
                current_steer = 'L'
            elif keyboard.is_pressed('right'):
                current_steer = 'R'

            current_motor = '0'  # 기본 정지 (키 떼면 멈춤 방식)
            if keyboard.is_pressed('up'):
                current_motor = '2'
            elif keyboard.is_pressed('down'):
                current_motor = '0'

            # --- 상태 변화 감지 및 전송 ---
            # 명령이 바뀔 때만 기록합니다 (데이터 최적화)
            if current_steer != last_steer or current_motor != last_motor:
                elapsed = time.time() - start_time

                # 1. 아두이노 전송
                if current_steer != last_steer: ser.write(current_steer.encode())
                if current_motor != last_motor: ser.write(current_motor.encode())

                # 2. 로그 리스트에 저장
                log_data.append({
                    'time': elapsed,
                    'steer': current_steer,
                    'motor': current_motor
                })

                print(f"[{elapsed:.2f}s] Steer:{current_steer}, Motor:{current_motor}")

                last_steer = current_steer
                last_motor = current_motor

            # 종료
            if keyboard.is_pressed('esc'):
                # 마지막에 정지 명령 추가
                elapsed = time.time() - start_time
                log_data.append({'time': elapsed, 'steer': 'C', 'motor': '0'})
                break

            time.sleep(0.01)  # 빠른 반응속도

    finally:
        # 파일 저장
        with open(LOG_FILE, 'w') as f:
            json.dump(log_data, f)
        ser.write(b'0');
        ser.write(b'C')  # 안전 정지
        print(f"\n💾 녹화 완료! '{LOG_FILE}'에 저장되었습니다.")


def replay_mode():
    """ 저장된 로그를 읽어서 그대로 실행 """
    if not os.path.exists(LOG_FILE):
        print(f"❌ 기록 파일({LOG_FILE})이 없습니다. 녹화부터 하세요!")
        return

    print(f"\n🟢 [재생 모드] '{LOG_FILE}' 불러오는 중...")
    with open(LOG_FILE, 'r') as f:
        log_data = json.load(f)

    print("ready...", end='', flush=True)
    time.sleep(1)
    print("GO! 🚀")

    start_time = time.time()

    # 로그 데이터 순차 실행
    for entry in log_data:
        target_time = entry['time']
        cmd_steer = entry['steer']
        cmd_motor = entry['motor']

        # 해당 시간이 될 때까지 대기 (Sync 맞추기)
        while True:
            current_elapsed = time.time() - start_time
            if current_elapsed >= target_time:
                break
            # 너무 빡빡하게 돌면 CPU 점유율 높으니 살짝 쉼, 하지만 정밀도를 위해 아주 짧게
            time.sleep(0.001)

            # 명령 전송
        ser.write(cmd_steer.encode())
        ser.write(cmd_motor.encode())
        print(f"[{target_time:.2f}s] 재생 -> Steer:{cmd_steer}, Motor:{cmd_motor}")

    # 끝난 후 정지
    ser.write(b'0');
    ser.write(b'C')
    print("\n🏁 재생 완료 (Ghost Drive Finished)")


# ==========================================
# [4] 메인 메뉴
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n--- 🏎️ Ghost Driver ---")
        print("1. 🔴 주행 녹화 (Record)")
        print("2. 🟢 주행 재생 (Replay)")
        print("3. 종료")
        choice = input("선택 >> ")

        if choice == '1':
            record_mode()
        elif choice == '2':
            replay_mode()
        elif choice == '3':
            ser.close()
            break
        else:
            print("다시 선택하세요.")