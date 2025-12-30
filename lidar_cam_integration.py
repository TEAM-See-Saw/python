import serial
from rplidar import RPLidar
import time
import cv2
from Function_Library import libCAMERA

# ==========================================
# 1. 포트 설정 (본인 환경에 맞게 수정!)
arduino_port = 'COM4'
lidar_port = 'COM3'
# ==========================================

# 2. 장치 연결
try:
    ser = serial.Serial(arduino_port, 9600)
    lidar = RPLidar(lidar_port)
    camera = libCAMERA()
    cam0, _ = camera.initial_setting(capnum=1)  # 안되면 0으로 변경

    if not cam0.isOpened():
        print("❌ 카메라 연결 실패")
        exit()

    print("✅ 시스템 시작 (로직: RED=정지 / 그 외=라이다 주행)")
    time.sleep(1)

except Exception as e:
    print(f"연결 오류: {e}")
    exit()

try:
    # 3. 메인 루프
    for scan in lidar.iter_scans():

        # --- [A] 카메라 신호등 인식 ---
        ret, frame = cam0.read()
        if not ret: break

        # 신호등 인식 (결과: "RED", "GREEN", "YELLOW" 또는 None)
        traffic_light = camera.object_detection(frame, sample=5, print_enable=True)
        if camera.loop_break(): break

        # --- [B] 라이다 거리 측정 ---
        min_distance = 999999
        for (_, angle, distance) in scan:
            if 170 <= angle < 190:  # 전방
                if distance > 0 and distance < min_distance:
                    min_distance = distance

        if min_distance == 999999: continue

        # --- [C] 수정된 제어 로직 ---
        command = b'0'
        msg = ""

        # 상황 1: 빨간불 발견 (최우선 순위 -> 정지)
        if traffic_light == "RED":
            command = b'0'
            msg = "🔴 RED Light -> STOP (Traffic Signal)"

            # 상황 2: 그 외 모든 경우 (초록불 OR 신호등 안 보임)
        else:
            status = "Green" if traffic_light == "GREEN" else "No Light"
            # [수정됨] 위험한 순서대로 판단해야 합니다!

            # 1. 30cm 이내: 완전 정지 (가장 위험하므로 1순위 체크)
            if min_distance < 300:
                command = b'0'
                msg = f"🛑 {status} & Danger({int(min_distance)}mm) -> STOP"

            # 2. 30cm ~ 70cm: 감속 (위의 300 조건은 통과했으므로 300 이상임)
            elif min_distance < 700:
                command = b'1'
                msg = f"⚠️ {status} & Warning({int(min_distance)}mm) -> SLOW"

            # 3. 70cm 이상: 고속 주행 (안전)
            else:
                command = b'2'
                msg = f"🚀 {status} & Clear({int(min_distance)}mm) -> GO!"

        # 아두이노 전송
        ser.write(command)
        print(msg)

except KeyboardInterrupt:
    print("\n종료 중...")
    lidar.stop()
    lidar.disconnect()
    ser.close()
    cam0.release()
    cv2.destroyAllWindows()