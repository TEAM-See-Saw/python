import serial
import time
import csv
import keyboard  # 비상 정지용

# ==========================================
# [설정] 재생할 로그 파일 이름 입력
# ==========================================
LOG_FILE = "drive_data_20260129_193005.csv"  # <-- 아까 생성된 파일명으로 변경하세요! (날짜 부분만 바꾸기)
PORT = 'COM4'
BAUDRATE = 9600

# 아두이노 연결
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    print("✅ 아두이노 연결 성공. 리플레이 준비...")
except:
    print("❌ 아두이노 연결 실패")
    exit()

print(f"📂 {LOG_FILE} 파일을 읽어옵니다...")

# 데이터 읽기
commands = []
try:
    with open(LOG_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # 첫 줄(헤더) 건너뛰기
        for row in reader:
            # CSV 내용은 문자열이므로 숫자로 변환해서 저장
            # [경과시간, 서보값, 속도값]
            commands.append([float(row[0]), int(row[1]), int(row[2])])
except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다. 파일명을 확인하세요.")
    exit()

print(f"🚀 리플레이 시작! (총 {len(commands)}개의 명령)")
print("⚠️ 비상 정지: q 키를 누르세요.")

# 시작 기준 시간
start_reference = time.time()

try:
    for cmd in commands:
        # 1. 비상 정지 체크
        if keyboard.is_pressed('q'):
            print("🛑 비상 정지!")
            break

        # 2. 저장된 데이터 가져오기
        target_time = cmd[0]  # 기록된 시간 (예: 1.5초)
        servo_val = cmd[1]
        motor_spd = cmd[2]

        # 3. 시간 동기화 (Sync)
        # 현재 경과 시간이 기록된 시간보다 작으면 기다림
        while (time.time() - start_reference) < target_time:
            time.sleep(0.001)  # 1ms 대기 (CPU 부하 방지)

        # 4. 명령 전송
        # (아두이노가 과부하 걸리지 않게, 값이 변할 때만 보내도 되지만 여기선 그냥 보냄)
        ser.write(f"S,{servo_val}\n".encode())
        ser.write(f"D,{motor_spd}\n".encode())

        print(f"[{target_time:.2f}s] Servo:{servo_val}, Speed:{motor_spd}")

except KeyboardInterrupt:
    print("강제 종료")

finally:
    ser.write(b'D,0\n')
    ser.write(b'S,570\n')
    ser.close()
    print("🏁 리플레이 종료")