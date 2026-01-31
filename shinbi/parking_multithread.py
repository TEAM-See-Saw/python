import time
import math
import numpy as np
import cv2
import threading

# ==========================================
# 1. 설정 및 튜닝 파라미터
# ==========================================
IS_SIMULATION = False  # False: 실제 라이다 사용

# 주차 로직 설정
PARKING_DEPTH_THRESHOLD = 1.2
READY_TIME = 1.5
STOP_DIST_CM = 20
LIDAR_PORT = 'COM3'
LIDAR_BAUDRATE = 115200  # ★ 라이다 통신 속도 (A1: 115200, A2/A3: 256000 확인 필요)

# 시각화 설정
WINDOW_SIZE = 600
SCALE = 100


# ==========================================
# 2. 하드웨어 인터페이스 (멀티스레딩 적용)
# ==========================================
class RobotHardware:
    def __init__(self, simulation=False):
        self.sim = simulation
        self.lidar = None
        self.start_time = time.time()

        # 최신 스캔 데이터를 저장할 공유 메모리 (0~359도)
        self.scan_data = [0.0] * 360
        self.running = True  # 스레드 제어용 플래그

        if not self.sim:
            from rplidar import RPLidar
            try:
                # baudrate 명시적 설정 (중요)
                self.lidar = RPLidar(LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
                self.lidar.clean_input()
                print(f"✅ Lidar Connected: {LIDAR_PORT} ({LIDAR_BAUDRATE})")

                # ★ [핵심] 라이다 데이터 수신을 별도 스레드로 분리
                self.thread = threading.Thread(target=self._lidar_thread_func)
                self.thread.daemon = True  # 메인 프로그램 종료 시 같이 종료
                self.thread.start()

            except Exception as e:
                print(f"❌ Lidar 연결 실패: {e}")
                self.lidar = None

    def _lidar_thread_func(self):
        """백그라운드에서 쉬지 않고 라이다 데이터를 받아오는 함수"""
        print("🔄 라이다 데이터 수신 스레드 시작")
        while self.running:
            try:
                # iter_scans는 내부적으로 버퍼를 관리함
                for scan in self.lidar.iter_scans(max_buf_meas=5000):
                    if not self.running: break

                    # 데이터 갱신
                    for (_, angle, distance) in scan:
                        angle_int = int(angle) % 360
                        # mm -> m 변환하여 저장
                        self.scan_data[angle_int] = distance / 1000.0
            except Exception as e:
                print(f"⚠️ 라이다 읽기 오류 (재시도 중): {e}")
                if self.lidar:
                    try:
                        self.lidar.clean_input()
                    except:
                        pass
                time.sleep(0.1)

    def get_latest_scan(self):
        """메인 루프에서는 이미 저장된 최신 데이터만 쏙 가져감 (즉시 리턴)"""
        if self.sim:
            # 시뮬레이션 데이터 생성
            elapsed = time.time() - self.start_time
            sim_data = [0.0] * 360
            is_parking_space = (3.0 < elapsed < 6.0)
            dist = 3.0 if is_parking_space else 0.5
            for i in range(80, 100):
                sim_data[i] = dist + np.random.uniform(-0.02, 0.02)
            return sim_data
        else:
            # 실제 데이터 반환 (스레드가 채워둔 것)
            return list(self.scan_data)  # 리스트 복사해서 반환

    def get_ultrasonic_rear(self):
        # 실제 초음파 센서 코드 필요
        return 999

    def set_motor(self, speed, angle):
        # 모터 제어 코드
        pass

    def stop(self):
        self.set_motor(0, 0)
        self.running = False  # 스레드 종료 신호
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
            except:
                pass
        print("장치 연결 해제 완료")


# ==========================================
# 3. 메인 로직
# ==========================================
def draw_visualization(scan_data, state, rear_dist):
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    center = WINDOW_SIZE // 2

    cv2.rectangle(img, (center - 15, center - 25), (center + 15, center + 25), (0, 0, 255), -1)
    cv2.line(img, (center, center), (center, center - 30), (0, 255, 255), 2)

    # 라이다 점 찍기
    for angle in range(360):
        dist = scan_data[angle]
        if 0.1 < dist < 5.0:
            theta = math.radians(angle - 90)
            x = int(center + dist * SCALE * math.cos(theta))
            y = int(center + dist * SCALE * math.sin(theta))
            if 0 <= x < WINDOW_SIZE and 0 <= y < WINDOW_SIZE:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 우측 감지 영역 표시 (85~95도)
    theta_start = math.radians(85 - 90)
    theta_end = math.radians(95 - 90)
    x1 = int(center + 1.5 * SCALE * math.cos(theta_start))
    y1 = int(center + 1.5 * SCALE * math.sin(theta_start))
    cv2.line(img, (center, center), (x1, y1), (50, 50, 50), 1)

    cv2.putText(img, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"REAR: {rear_dist:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)

    cv2.imshow("Parking System", img)
    return cv2.waitKey(1) & 0xFF


def main():
    robot = RobotHardware(simulation=IS_SIMULATION)

    state = "SEARCH"
    state_start_time = time.time()
    prev_time = 0

    print("🚀 멀티스레드 주차 시스템 시작")
    print("초기화 대기 중 (2초)...")
    time.sleep(2)  # 스레드가 데이터 모을 시간 주기

    try:
        while True:
            current_time = time.time()

            # 1. 센서 데이터 가져오기 (즉시 반환됨)
            lidar_scan = robot.get_latest_scan()
            rear_dist_cm = robot.get_ultrasonic_rear()

            # 2. 데이터 처리 (우측 85~95도 평균)
            # 0.1m 이상인 유효값만 필터링
            right_dists = [lidar_scan[i] for i in range(85, 95) if lidar_scan[i] > 0.1]
            avg_right_dist = np.mean(right_dists) if right_dists else 0.0

            # 3. 상태 머신
            if state == "SEARCH":
                robot.set_motor(30, 0)
                # 디버깅 출력 (너무 자주 찍히면 주석 처리)
                # print(f"우측 거리: {avg_right_dist:.2f}m")

                if avg_right_dist > PARKING_DEPTH_THRESHOLD:
                    print(f"✨ 공간 발견! ({avg_right_dist:.2f}m)")
                    state = "READY"
                    state_start_time = current_time

            elif state == "READY":
                robot.set_motor(30, 0)
                if current_time - state_start_time > READY_TIME:
                    print("🛑 위치 확보. 정지.")
                    state = "PAUSE_BEFORE_REVERSE"
                    state_start_time = current_time

            elif state == "PAUSE_BEFORE_REVERSE":
                robot.set_motor(0, 0)
                if current_time - state_start_time > 1.0:
                    state = "REVERSE"
                    state_start_time = current_time

            elif state == "REVERSE":
                robot.set_motor(-20, 50)
                if 0 < rear_dist_cm <= STOP_DIST_CM:
                    print(f"🛑 주차 완료 (후방 {rear_dist_cm}cm)")
                    state = "STOP"
                    state_start_time = current_time

            elif state == "STOP":
                robot.set_motor(0, 0)
                if current_time - state_start_time > 3.0:
                    print("👋 미션 종료")
                    break

            # 4. 시각화 (30ms 주기)
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