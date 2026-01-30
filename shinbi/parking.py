import time
import math
import numpy as np
import cv2

# ==========================================
# 1. 설정 및 튜닝 파라미터
# ==========================================
IS_SIMULATION = False  # False: 실제 차 (라이다 연결 필수)

# 주차 로직 설정
PARKING_DEPTH_THRESHOLD = 1.2  # (m) 이보다 깊으면 주차공간
SIDE_WALL_DIST = 0.5
READY_TIME = 1.5  # (초) 공간 발견 후 더 나가는 시간
STOP_DIST_CM = 20  # (cm) 후방 정지 거리
LIDAR_PORT = 'COM3'

# 시각화 설정
WINDOW_SIZE = 600
SCALE = 100


# ==========================================
# 2. 하드웨어 인터페이스 (Non-blocking)
# ==========================================
class RobotHardware:
    def __init__(self, simulation=False):
        self.sim = simulation
        self.lidar = None
        self.start_time = time.time()

        if not self.sim:
            from rplidar import RPLidar
            try:
                self.lidar = RPLidar(LIDAR_PORT)
                # ★ [핵심 1] 버퍼를 비우고 시작
                self.lidar.clean_input()
                print(f"Lidar Connected: {LIDAR_PORT}")
            except Exception as e:
                print(f"Lidar 연결 실패: {e}")

    def get_lidar_scan(self):
        """
        라이다 데이터를 Non-blocking으로 가져옵니다.
        데이터가 없으면 None을 반환하여 루프가 멈추지 않게 합니다.
        """
        scan_data = [0.0] * 360

        if self.sim:
            # [시뮬레이션 데이터 생성]
            elapsed = time.time() - self.start_time
            for i in range(360): scan_data[i] = 0.0

            # 3~6초 사이에 공간 등장 (가상 시나리오)
            is_parking_space = (3.0 < elapsed < 6.0)
            dist = 3.0 if is_parking_space else 0.5

            for i in range(80, 100):
                scan_data[i] = dist + np.random.uniform(-0.02, 0.02)
            return scan_data

        else:
            # [실제 라이다]
            try:
                # ★ [핵심 2] max_buf_meas를 5000으로 늘려서 버퍼 오버플로우 방지
                # iterator를 한 번만 호출하고 바로 리턴 (루프가 빠를수록 좋음)
                for scan in self.lidar.iter_scans(max_buf_meas=5000):
                    for (_, angle, distance) in scan:
                        angle_int = int(angle) % 360
                        scan_data[angle_int] = distance / 1000.0
                    return scan_data  # 최신 스캔 데이터 반환
            except Exception as e:
                print(f"Lidar Error: {e}")
                # 에러 발생 시 재연결 시도 로직 등을 넣을 수 있음
                return None
        return scan_data

    def get_ultrasonic_rear(self):
        if self.sim:
            elapsed = time.time() - self.start_time
            if elapsed > 8.0: return max(10, 100 - (elapsed - 8.0) * 30)
            return 200
        else:
            # 실제 초음파 센서 코드 (GPIO 등)
            return 999

    def set_motor(self, speed, angle):
        # 실제 모터 드라이버 연결 시 여기에 작성
        # 너무 잦은 출력 방지를 위해 상태 변경 시에만 출력하거나 디버그용으로 둠
        # print(f"🎮 Motor: {speed}, {angle}")
        pass

    def stop(self):
        self.set_motor(0, 0)
        if self.lidar:
            self.lidar.stop()
            self.lidar.disconnect()


# ==========================================
# 3. 메인 로직 (상태 머신)
# ==========================================
def draw_visualization(scan_data, state, rear_dist):
    if scan_data is None: return 0  # 데이터 없으면 패스

    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    center = WINDOW_SIZE // 2

    # 차량 (빨간색)
    cv2.rectangle(img, (center - 15, center - 25), (center + 15, center + 25), (0, 0, 255), -1)
    cv2.line(img, (center, center), (center, center - 30), (0, 255, 255), 2)  # 헤딩

    # 라이다 점 (녹색)
    for angle in range(360):
        dist = scan_data[angle]
        if 0.1 < dist < 5.0:
            theta = math.radians(angle - 90)
            x = int(center + dist * SCALE * math.cos(theta))
            y = int(center + dist * SCALE * math.sin(theta))
            if 0 <= x < WINDOW_SIZE and 0 <= y < WINDOW_SIZE:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 정보 텍스트
    cv2.putText(img, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"REAR: {rear_dist:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)

    cv2.imshow("Parking System", img)
    return cv2.waitKey(1) & 0xFF


def main():
    robot = RobotHardware(simulation=IS_SIMULATION)

    # 초기 상태
    state = "SEARCH"
    state_start_time = time.time()  # 상태 진입 시간 기록용

    # 루프 속도 제어용
    prev_time = 0

    print("🚀 주차 미션 시작 (Non-blocking Mode)")

    try:
        while True:
            current_time = time.time()

            # 1. 센서 데이터 읽기 (이 부분이 가장 중요 - 절대 멈추면 안됨)
            lidar_scan = robot.get_lidar_scan()
            if lidar_scan is None:
                continue  # 데이터 없으면 건너뛰고 계속 루프 돔

            rear_dist_cm = robot.get_ultrasonic_rear()

            # 2. 데이터 처리 (우측 거리 평균)
            right_dists = [lidar_scan[i] for i in range(85, 95) if lidar_scan[i] > 0.1]
            avg_right_dist = np.mean(right_dists) if right_dists else 0.0

            # 3. 상태 머신 (State Machine) - time.sleep 사용 금지!

            if state == "SEARCH":
                robot.set_motor(30, 0)  # 계속 직진 명령

                # 공간 발견 조건
                if avg_right_dist > PARKING_DEPTH_THRESHOLD:
                    print(f"✨ 공간 발견! ({avg_right_dist:.2f}m)")
                    state = "READY"
                    state_start_time = current_time  # 타이머 시작

            elif state == "READY":
                robot.set_motor(30, 0)  # 조금 더 직진

                # ★ [핵심] sleep 대신 타이머 체크
                # READY_TIME(1.5초) 동안만 이 상태 유지
                if current_time - state_start_time > READY_TIME:
                    print("🛑 위치 확보. 정지 후 후진 준비")
                    state = "PAUSE_BEFORE_REVERSE"  # 잠깐 멈춤 상태로 이동
                    state_start_time = current_time

            elif state == "PAUSE_BEFORE_REVERSE":
                robot.set_motor(0, 0)  # 정지

                # 1초간 정지 (기어 변속 시뮬레이션)
                if current_time - state_start_time > 1.0:
                    state = "REVERSE"
                    state_start_time = current_time

            elif state == "REVERSE":
                robot.set_motor(-20, 50)  # 우측 핸들 후진

                # 후방 센서 감지 시 정지
                if 0 < rear_dist_cm <= STOP_DIST_CM:
                    print(f"🛑 주차 완료 (후방 {rear_dist_cm}cm)")
                    state = "STOP"
                    state_start_time = current_time

            elif state == "STOP":
                robot.set_motor(0, 0)
                # 3초 대기 후 종료
                if current_time - state_start_time > 3.0:
                    print("👋 미션 종료")
                    break

            # 4. 시각화 (너무 자주 그리면 느려지므로 30ms마다 갱신)
            if current_time - prev_time > 0.03:
                key = draw_visualization(lidar_scan, state, rear_dist_cm)
                if key == ord('q'): break
                prev_time = current_time

    except KeyboardInterrupt:
        print("\n강제 종료 요청됨")
    finally:
        robot.stop()
        cv2.destroyAllWindows()
        print("시스템 안전 종료")


if __name__ == "__main__":
    main()