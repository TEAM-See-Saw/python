import cv2
import numpy as np
import serial
import time

# ==========================================
# [1] 설정
# ==========================================
PORT = 'COM4'
BAUDRATE = 9600

# ---------------------------------------------------------
# ★ 현장 튜닝 포인트
# ---------------------------------------------------------
# 1. 정렬 판단 기준 (mm)
# RM과 RT 차이가 이 값보다 작으면 "평행하다"고 판단
ALIGN_THRES = 50   # 5cm 이내 차이면 평행으로 간주

# 2. 직진 후진 시간 (주차 깊이 결정)
# 속도(-120)로 후진할 때 150cm 공간에 딱 맞게 들어가는 시간
# 너무 깊으면 줄이고, 덜 들어가면 늘리세요.
PARKING_DEPTH_TIME = 1.8 

# 속도
SEARCH_SPEED = 100
REVERSE_SPEED = -120  # 후진 속도 (일정해야 함)
EXIT_SPEED = 130

# 거리 기준 (mm)
EMPTY_THRES = 600     # 빈 공간 인식
SIDE_CHECK_MAX = 800  # 옆 차와의 거리가 너무 멀면(80cm) 센서 데이터 무시

# 서보 값
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 480 # 진입용
SERVO_LEFT_MAX = 680  # 출차용
# ---------------------------------------------------------

# 상태 정의
STATE_SEARCH = 0    # 빈자리 찾기
STATE_READY = 1     # 위치 잡기
STATE_ENTRY = 2     # 진입 (회전 후진 + 평행 감지)
STATE_PARKING = 3   # 주차 (직진 후진 + 시간 체크)
STATE_PARKED = 4    # 주차 완료
STATE_EXIT_TURN = 5 # 출차 1
STATE_EXIT_GO = 6   # 출차 2
STATE_DONE = 7      # 종료

# ==========================================
# [2] 초기화
# ==========================================
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.05)
    print("✅ 아두이노 연결 성공")
except:
    ser = None
    print("❌ 아두이노 연결 실패")

# 센서 데이터 (LF, LM, LT, RF, RM, RT)
sensors = [0]*6

def read_sensors():
    global sensors
    if ser and ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                for i in range(min(6, len(parts))): 
                    sensors[i] = int(parts[i])
        except: pass

def send_cmd(servo, speed):
    if ser:
        ser.write(f"S,{servo}\n".encode())
        ser.write(f"D,{speed}\n".encode())

# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    state = STATE_SEARCH
    timer = 0
    gap_found = False
    
    print("🅿️ 주차 미션 시작 (Sensor Align + Timer Stop)")

    while True:
        read_sensors()
        
        # 우측 센서 (옆 차와의 거리)
        rm_dist = sensors[4] # 중간
        rt_dist = sensors[5] # 꼬리

        servo_cmd = SERVO_CENTER
        motor_cmd = 0

        # [1] 빈 공간 탐색
        if state == STATE_SEARCH:
            servo_cmd = SERVO_CENTER
            motor_cmd = SEARCH_SPEED
            if rm_dist > EMPTY_THRES:
                gap_found = True
                print("🔍 빈 공간 발견")
                state = STATE_READY

        # [2] 위치 잡기 (RT 센서 기준)
        elif state == STATE_READY:
            servo_cmd = SERVO_CENTER
            motor_cmd = SEARCH_SPEED
            
            # 빈 공간 지나서 다음 장애물 감지 시 정지
            # (차 엉덩이가 빈 공간을 지났다는 뜻)
            if rt_dist < 500 and gap_found:
                print("🛑 위치 확보. 정지!")
                send_cmd(SERVO_CENTER, 0)
                time.sleep(1.0)
                state = STATE_ENTRY
                timer = time.time() # 진입 시작 시간 (혹시 모를 타임아웃용)

        # [3] 진입 & 평행 감지 (핵심 로직)
        elif state == STATE_ENTRY:
            servo_cmd = SERVO_RIGHT_MAX
            motor_cmd = REVERSE_SPEED
            
            # 우측 센서값 차이 계산
            diff = abs(rm_dist - rt_dist)
            
            # 조건 1: 두 센서 모두 유효한 거리(옆 차)를 보고 있어야 함
            valid_sensing = (rm_dist < SIDE_CHECK_MAX and rt_dist < SIDE_CHECK_MAX)
            
            # 조건 2: 두 센서 값의 차이가 작음 (평행)
            is_parallel = (diff < ALIGN_THRES)
            
            # ★ 평행하거나, 진입한 지 3초가 넘었으면(안전장치) 다음 단계로
            if (valid_sensing and is_parallel) or (time.time() - timer > 3.0):
                print(f"🔄 정렬 완료! (Diff: {diff}mm) -> 직진 후진")
                state = STATE_PARKING
                timer = time.time() # 직진 후진 타이머 시작

        # [4] 주차 확정 (시간제 후진)
        elif state == STATE_PARKING:
            servo_cmd = SERVO_CENTER # 핸들 11자
            motor_cmd = REVERSE_SPEED
            
            # 설정한 시간만큼만 후진
            if time.time() - timer > PARKING_DEPTH_TIME:
                state = STATE_PARKED

        # [5] 주차 완료
        elif state == STATE_PARKED:
            send_cmd(SERVO_CENTER, 0)
            print("🅿️ 주차 완료 (3초 대기)")
            time.sleep(3.0)
            print("🚗 출차 시작")
            state = STATE_EXIT_TURN
            timer = time.time()
            continue

        # [6] 출차 1 (좌회전)
        elif state == STATE_EXIT_TURN:
            servo_cmd = SERVO_LEFT_MAX
            motor_cmd = EXIT_SPEED
            if time.time() - timer > 2.0:
                state = STATE_EXIT_GO
                timer = time.time()

        # [7] 출차 2 (직진)
        elif state == STATE_EXIT_GO:
            servo_cmd = SERVO_CENTER
            motor_cmd = EXIT_SPEED
            if time.time() - timer > 1.5:
                print("🎉 미션 성공")
                state = STATE_DONE

        # [8] 종료
        elif state == STATE_DONE:
            send_cmd(SERVO_CENTER, 0)
            break
        
        send_cmd(servo_cmd, motor_cmd)
        
        # 디버깅
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        cv2.putText(img, f"RM:{rm_dist} RT:{rt_dist}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(img, f"Diff: {abs(rm_dist-rt_dist)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(img, f"State: {state}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Debug", img)
        if cv2.waitKey(1) == ord('q'): break

    if ser: ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()