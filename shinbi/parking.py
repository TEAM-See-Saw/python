import time
import math
import numpy as np
import cv2
import threading
import serial

# ==========================================
# 1. 설정 파라미터
# ==========================================
IS_SIMULATION = False  # 실제 주행 시 False

# 포트 설정 (장치관리자 확인 필수)
LIDAR_PORT = 'COM3'       # 라이다 포트
ARDUINO_PORT = 'COM4'     # 아두이노 포트
ARDUINO_BAUDRATE = 9600   # ★ 아두이노 코드와 일치 (9600)

# 주차 로직 튜닝
PARKING_DEPTH_THRESHOLD = 1.2  # (m) 주차 공간 깊이
READY_TIME = 1.5               # (초) 공간 발견 후 더 전진하는 시간
STOP_DIST_MM = 200             # (mm) 후방 정지 거리 (아두이노가 mm 단위로 줌)

# 조향(Steering) 매핑 설정 (아두이노 코드 기준)
STEER_CENTER = 570
STEER_LIMIT_MIN = 480  # 좌측 최대
STEER_LIMIT_MAX = 680  # 우측 최대
# 논리적 각도(-50~50)를 실제 값으로 변환하기 위한 비율
# 대략 50도 꺾을 때 값이 110 변하므로 비율은 약 2.2
STEER_RATIO = 2.2 

# 시각화 설정
WINDOW_SIZE = 600
SCALE = 100 # 1m = 100px

# ==========================================
# 2. 하드웨어 인터페이스 클래스
# ==========================================
class RobotHardware:
    def __init__(self):
        self.lidar = None
        self.arduino = None
        self.running = True
        
        # 데이터 저장소
        self.scan_data = [0.0] * 360
        # 초음파 6개 [LF, LM, LT, RF, RM, RT] 순서 (mm 단위)
        self.sonar_data = [9999] * 6 

        # 1. 아두이노 연결
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=0.1)
            time.sleep(2) # 아두이노 리셋 대기
            print(f"✅ Arduino Connected: {ARDUINO_PORT}")
            
            # 수신 스레드 시작
            self.serial_thread = threading.Thread(target=self._arduino_rx_thread)
            self.serial_thread.daemon = True
            self.serial_thread.start()
        except Exception as e:
            print(f"❌ Arduino 연결 실패: {e}")

        # 2. 라이다 연결
        if not IS_SIMULATION:
            from rplidar import RPLidar
            try:
                self.lidar = RPLidar(LIDAR_PORT, baudrate=115200)
                self.lidar.clean_input()
                print(f"✅ Lidar Connected: {LIDAR_PORT}")
                
                self.lidar_thread = threading.Thread(target=self._lidar_thread)
                self.lidar_thread.daemon = True
                self.lidar_thread.start()
            except Exception as e:
                print(f"❌ Lidar 연결 실패: {e}")

    def _arduino_rx_thread(self):
        """아두이노에서 오는 초음파 데이터 수신 (US:d1,d2,...)"""
        while self.running and self.arduino:
            try:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("US:"):
                        # "US:100,200,300..." -> 파싱
                        parts = line[3:].split(',')
                        if len(parts) == 6:
                            self.sonar_data = [int(p) for p in parts]
            except Exception:
                pass
            time.sleep(0.01)

    def _lidar_thread(self):
        """라이다 데이터 수신"""
        while self.running and self.lidar:
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=5000):
                    if not self.running: break
                    for (_, angle, distance) in scan:
                        angle_int = int(angle) % 360
                        self.scan_data[angle_int] = distance / 1000.0 # m 단위로 변환
            except:
                # 에러 발생 시 재접속 시도
                try:
                    self.lidar.clean_input()
                except: pass
                time.sleep(0.1)

    def get_lidar(self):
        return list(self.scan_data)

    def get_rear_distance(self):
        """
        후방 거리 반환 (mm)
        아두이노 코드 순서: LF, LM, LT, RF, RM, RT
        보통 마지막 2개(5, 6번)가 후방일 가능성이 높음.
        여기서는 가장 마지막 센서(RT, index 5)를 후방이라고 가정.
        필요하면 평균값 사용: (self.sonar_data[4] + self.sonar_data[5]) / 2
        """
        return self.sonar_data[5] 

    def send_command(self, speed, angle_deg):
        """
        speed: -255 ~ 255 (음수: 후진)
        angle_deg: -50(좌) ~ 50(우) -> 아두이노 POT 값으로 변환하여 전송
        """
        if not self.arduino: return

        # 1. 조향각 변환 (Angle -> Potentiometer Value)
        # angle_deg가 양수(우회전)면 POT값 증가, 음수면 감소
        target_pot = STEER_CENTER + (angle_deg * STEER_RATIO)
        
        # 안전 범위 제한 (Clamp)
        target_pot = max(STEER_LIMIT_MIN, min(STEER_LIMIT_MAX, target_pot))
        target_pot = int(target_pot)

        # 2. 속도 제한
        target_speed = int(speed)

        # 3. 명령 전송 (S:xxx\n 그리고 D:xxx\n)
        try:
            cmd_steer = f"S:{target_pot}\n"
            cmd_drive = f"D:{target_speed}\n"
            
            self.arduino.write(cmd_steer.encode())
            self.arduino.write(cmd_drive.encode())
        except Exception as e:
            print(f"Tx Error: {e}")

    def stop(self):
        self.send_command(0, 0) # 정지 명령
        self.running = False
        time.sleep(0.5)
        if self.lidar:
            try: self.lidar.stop(); self.lidar.disconnect()
            except: pass
        if self.arduino:
            self.arduino.close()
        print("Hardware Stopped.")

# ==========================================
# 3. 시각화 및 메인 로직
# ==========================================
def draw_screen(scan, state, rear_mm, steer_deg):
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    cx, cy = WINDOW_SIZE // 2, WINDOW_SIZE // 2

    # 1. 차량 그리기
    cv2.rectangle(img, (cx-20, cy-30), (cx+20, cy+30), (0, 0, 255), -1) # 차체
    
    # 조향 표시 (노란 선)
    rad = math.radians(steer_deg - 90) # -90은 화면 좌표계 보정
    ex = int(cx + 40 * math.cos(rad))
    ey = int(cy + 40 * math.sin(rad))
    cv2.line(img, (cx, cy-30), (ex, ey), (0, 255, 255), 3)

    # 2. 라이다 점 찍기
    for i in range(360):
        d = scan[i]
        if 0.1 < d < 5.0:
            th = math.radians(i - 90)
            x = int(cx + d * SCALE * math.cos(th))
            y = int(cy + d * SCALE * math.sin(th))
            if 0 <= x < WINDOW_SIZE and 0 <= y < WINDOW_SIZE:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 3. 텍스트 정보
    cv2.putText(img, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"REAR: {rear_mm} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    cv2.putText(img, f"STEER: {steer_deg} deg", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    cv2.imshow("Car View", img)
    return cv2.waitKey(1) & 0xFF

def main():
    robot = RobotHardware()
    state = "SEARCH"
    state_start_time = time.time()
    
    # 현재 명령 상태 변수
    current_speed = 0
    current_steer = 0

    print("🚀 자율 주차 시스템 시작")
    print("안전을 위해 잠시 대기 (2초)...")
    time.sleep(2)

    try:
        while True:
            # 시간 측정
            loop_start = time.time()

            # 1. 센서 데이터 획득
            scan = robot.get_lidar()
            rear_mm = robot.get_rear_distance()

            # 2. 데이터 가공 (우측 거리 측정: 85~95도 평균)
            right_points = [scan[i] for i in range(85, 95) if scan[i] > 0.1]
            avg_right_m = np.mean(right_points) if right_points else 0.0

            # 3. 상태 머신 (로직)
            if state == "SEARCH":
                current_speed = 60    # 천천히 전진 (PWM)
                current_steer = 0     # 직진
                
                # 우측에 공간이 생겼다면 (깊이가 깊어짐)
                if avg_right_m > PARKING_DEPTH_THRESHOLD:
                    print(f"✨ 공간 발견! (깊이: {avg_right_m:.2f}m)")
                    state = "READY"
                    state_start_time = time.time()

            elif state == "READY":
                current_speed = 60
                current_steer = 0
                # 차체 길이만큼 더 전진해서 평행주차 준비 위치 잡기
                if time.time() - state_start_time > READY_TIME:
                    print("🛑 위치 확보 완료. 정지 후 후진 준비.")
                    state = "PAUSE"
                    state_start_time = time.time()

            elif state == "PAUSE":
                current_speed = 0
                current_steer = 0
                if time.time() - state_start_time > 1.0: # 1초 정지
                    state = "REVERSE"

            elif state == "REVERSE":
                current_speed = -70   # 후진 (PWM)
                current_steer = 50    # 핸들 우측 최대 (주차 공간으로 진입)

                # 후방 센서 감지 시 정지
                if 0 < rear_mm <= STOP_DIST_MM:
                    print(f"🛑 후방 장애물 감지 ({rear_mm}mm). 주차 완료.")
                    state = "STOP"
                    state_start_time = time.time()

            elif state == "STOP":
                current_speed = 0
                current_steer = 0
                if time.time() - state_start_time > 5.0:
                    print("시스템 종료")
                    break

            # 4. 명령 전송 (★ 중요: 루프마다 계속 보내야 Failsafe 안 걸림)
            robot.send_command(current_speed, current_steer)

            # 5. 시각화
            key = draw_screen(scan, state, rear_mm, current_steer)
            if key == ord('q'):
                break
            
            # 루프 주기 조절 (약 20Hz)
            dt = time.time() - loop_start
            if dt < 0.05:
                time.sleep(0.05 - dt)

    except KeyboardInterrupt:
        print("\n강제 종료")
    finally:
        robot.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()