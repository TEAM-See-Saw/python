import serial
from rplidar import RPLidar
import time

# 오프라인 실습2: 라이다를 활용한 모터 제어 실습
# 아두이노, 라이다 pc에 연결 후 실습 진행

# 1. 포트 설정 (본인 환경에 맞게 수정 필요!)
arduino_port = 'COM4'
lidar_port = 'COM3'

# 2. 연결 시작
ser = serial.Serial(arduino_port, 9600)
lidar = RPLidar(lidar_port)

print("시스템 시작...")

try:
    for scan in lidar.iter_scans():
        min_distance = 999999  # 초기화 (아주 큰 값)

        # 3. 스캔 데이터 분석
        for (_, angle, distance) in scan:
            # 실습 조건 1: 각도 170도 이상 ~ 190도 미만만 사용
            if 170 <= angle < 190:
                if distance > 0:  # 0은 측정 에러일 수 있으므로 제외
                    if distance < min_distance:
                        min_distance = distance

        # 데이터가 없으면 계속 진행
        if min_distance == 999999:
            continue

        # 4. 실습 조건 로직 적용 (단위: 라이다는 보통 mm 사용)
        # 30cm = 300mm, 70cm = 700mm

        if min_distance < 300:  # 30cm 이내
            print(f"장애물 감지! 거리: {min_distance}mm -> 정지")
            ser.write(b'0')  # 아두이노에 '0' 전송

        elif 300 <= min_distance < 700:  # 30~70cm
            print(f"주의 구간. 거리: {min_distance}mm -> 1/2 속도")
            ser.write(b'1')  # 아두이노에 '1' 전송

        else:  # 70cm 이상 (장애물 없음)
            print(f"안전. 거리: {min_distance}mm -> 최대 속도")
            ser.write(b'2')  # 아두이노에 '2' 전송

except KeyboardInterrupt:
    print("중지 중...")
    lidar.stop()
    lidar.disconnect()
    ser.close()