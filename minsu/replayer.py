import serial
import time
import csv
import keyboard  # pip install keyboard

# ==========================================
# [설정] 재생할 로그 파일 이름 입력
# ==========================================
LOG_FILE = "far_1_3.csv"  # <-- 실제 파일명으로 수정하세요
PORT = 'COM4'
BAUDRATE = 115200  # ★ 아두이노와 속도 일치

# 아두이노 연결
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)
    # 시작 전 버퍼 한번 비우기
    ser.reset_input_buffer()
    print("✅ 아두이노 연결 성공. 리플레이 준비...")
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    exit()

print(f"📂 {LOG_FILE} 파일을 읽어옵니다...")

# 데이터 읽기
commands = []
try:
    with open(LOG_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 건너뛰기
        for row in reader:
            if not row: continue
            # [경과시간, 서보값, 속도값]
            commands.append([float(row[0]), int(row[1]), int(row[2])])
except FileNotFoundError:
    print("❌ 파일을 찾을 수 없습니다.")
    exit()
except Exception as e:
    print(f"❌ 파일 읽기 오류: {e}")
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

        target_time = cmd[0]
        servo_val = cmd[1]
        motor_spd = cmd[2]

        # 2. 시간 동기화 (Sync) & ★ [핵심] 버퍼 비우기
        # 다음 명령 시간까지 기다리는 동안, 아두이노가 보내는 데이터를 계속 버림
        while (time.time() - start_reference) < target_time:
            # ★ 아두이노가 멈추지 않도록 데이터가 오면 즉시 비움
            if ser.in_waiting > 0:
                try:
                    ser.read(ser.in_waiting)
                except:
                    pass

            # CPU 점유율 낮추기 위해 짧게 대기
            time.sleep(0.001)

        # 3. 명령 전송
        try:
            ser.write(f"S,{servo_val}\n".encode())
            ser.write(f"D,{motor_spd}\n".encode())
        except Exception as e:
            print(f"전송 오류: {e}")
            break

        print(f"[{target_time:.2f}s] Servo:{servo_val}, Speed:{motor_spd}")

except KeyboardInterrupt:
    print("강제 종료")

finally:
    if ser.is_open:
        ser.write(b'D,0\n')
        ser.write(b'S,570\n')
        ser.close()
    print("🏁 리플레이 종료")