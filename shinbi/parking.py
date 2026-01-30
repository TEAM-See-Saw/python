import time
import math
import numpy as np
import cv2  # 시각화를 위한 OpenCV

# ==========================================
# 1. 설정 및 튜닝 파라미터 (여기를 수정하세요)
# ==========================================
IS_SIMULATION = True  # True: 내 컴퓨터에서 테스트, False: 실제 차에서 실행

# 차량 및 주차 설정
PARKING_DEPTH_THRESHOLD = 1.2  # (m) 이 깊이 이상 뚫려있으면 주차공간으로 인식
SIDE_WALL_DIST = 0.5           # (m) 주행 중 우측 벽과의 거리
READY_TIME = 1.5               # (초) 공간 발견 후 앞으로 더 나가는 시간 (오버런)
STOP_DIST_CM = 20              # (cm) 후방 벽과 이 거리 남으면 정지
LIDAR_PORT = 'COM3'            # 라이다 포트 (윈도우: COMx, 리눅스: /dev/ttyUSB0)

# 시각화 설정 (창 크기)
WINDOW_SIZE = 600
SCALE = 100  # 화면 배율 (1m = 100픽셀)

# ==========================================
# 2. 하드웨어 인터페이스 (라이다/모터/초음파)
# ==========================================
class RobotHardware:
    def __init__(self, simulation=False):
        self.sim = simulation
        self.lidar = None
        self.start_time = time.time()
        
        if not self.sim:
            # [실제 장비 연결]
            from rplidar import RPLidar
            try:
                self.lidar = RPLidar(LIDAR_PORT)
                print(f"Lidar Connected: {LIDAR_PORT}")
            except:
                print("Lidar 연결 실패! 설정 포트 확인")
        
    def get_lidar_scan(self):
        """360도 거리 정보를 리스트로 반환 [0도~359도] (단위: m)"""
        scan_data = [0.0] * 360
        
        if self.sim:
            # [시뮬레이션] 가짜 벽 데이터 생성
            # 평소엔 우측(90도)에 0.5m 벽이 있다가, 특정 시간(3~6초)에 구멍이 뚫림
            elapsed = time.time() - self.start_time
            for i in range(360):
                scan_data[i] = 0.0 # 기본 허공
                
            # 우측(80~100도)에 벽 생성
            is_parking_space = (3.0 < elapsed < 6.0) # 3초~6초 사이에 공간 등장
            dist = 3.0 if is_parking_space else 0.5 # 공간이면 3m, 아니면 0.5m
            
            # 노이즈 추가해서 리얼하게
            for i in range(80, 100):
                scan_data[i] = dist + np.random.uniform(-0.02, 0.02)
                
        else:
            # [실제 라이다] 데이터 읽기 (RPLidar 라이브러리 사용)
            # (Note: 실제 루프에서는 iterator 방식이 더 효율적이나 편의상 구조 단순화)
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=500):
                    for (_, angle, distance) in scan:
                        angle_int = int(angle) % 360
                        scan_data[angle_int] = distance / 1000.0 # mm -> m 변환
                    break # 한 바퀴 스캔 후 반환
            except Exception as e:
                print(f"Lidar Error: {e}")
                
        return scan_data

    def get_ultrasonic_rear(self):
        """후방 거리(cm) 반환"""
        if self.sim:
            # 시뮬레이션: 7초 뒤부터 후방 벽이 가까워짐 (후진 상황 가정)
            elapsed = time.time() - self.start_time
            if elapsed > 8.0: 
                return max(10, 100 - (elapsed - 8.0) * 30) # 점점 줄어듦
            return 200 # 평소엔 2m
        else:
            # [실제 센서] GPIO 읽어서 거리 계산 코드 여기에 작성
            # distance_L = ...
            # distance_R = ...
            # return min(distance_L, distance_R)
            return 999 # 임시

    def set_motor(self, speed, angle):
        """모터 제어"""
        print(f"🎮 Motor: Speed={speed}, Angle={angle}")
        if not self.sim:
            # 실제 모터 드라이버 코드 작성
            pass

    def stop(self):
        self.set_motor(0, 0)
        if self.lidar:
            self.lidar.stop()
            self.lidar.disconnect()

# ==========================================
# 3. 주차 로직 및 시각화 엔진
# ==========================================
def draw_visualization(scan_data, state, rear_dist):
    # 검은색 배경 생성
    img = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8)
    center = WINDOW_SIZE // 2
    
    # 1. 차 그리기 (중앙에 빨간 박스)
    cv2.rectangle(img, (center-15, center-25), (center+15, center+25), (0, 0, 255), -1)
    # 차 앞쪽 표시 (노란 선)
    cv2.line(img, (center-15, center-25), (center+15, center-25), (0, 255, 255), 2)

    # 2. LiDAR 데이터 점 찍기
    for angle in range(360):
        dist = scan_data[angle]
        if 0.1 < dist < 5.0: # 유효 거리만
            # 극좌표 -> 직교좌표 변환 (화면 좌표계: Y가 아래로 증가하므로 주의)
            # 라이다 0도가 위쪽(전방)이라고 가정
            theta = math.radians(angle - 90) 
            x = int(center + dist * SCALE * math.cos(theta))
            y = int(center + dist * SCALE * math.sin(theta))
            
            if 0 <= x < WINDOW_SIZE and 0 <= y < WINDOW_SIZE:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    # 3. 감지 영역(ROI) 표시 (우측 90도 방향 박스)
    # 우측 1m 지점에 박스를 그려서 거기가 비었는지 시각적으로 확인
    roi_x = int(center + 1.0 * SCALE) # 우측 1m
    roi_y = center
    color = (0, 255, 0) if state == "SEARCH" else (100, 100, 100)
    cv2.rectangle(img, (roi_x-20, roi_y-20), (roi_x+20, roi_y+20), color, 2)

    # 4. 상태 텍스트 출력
    cv2.putText(img, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"REAR: {rear_dist:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 1)

    cv2.imshow("Lidar Parking View", img)
    return cv2.waitKey(1) & 0xFF

def main():
    robot = RobotHardware(simulation=IS_SIMULATION)
    state = "SEARCH"
    state_start_time = time.time()
    
    print("🚀 주차 미션 시작!")

    try:
        while True:
            # 1. 센서 값 읽기
            lidar_scan = robot.get_lidar_scan()
            rear_dist_cm = robot.get_ultrasonic_rear()

            # 2. 데이터 처리 (우측 90도 거리 평균)
            # 실제로는 85~95도 사이값 평균 내는게 좋음
            right_dists = []
            for i in range(85, 95):
                d = lidar_scan[i]
                if d > 0.1: right_dists.append(d)
            
            avg_right_dist = np.mean(right_dists) if right_dists else 0.0

            # 3. 상태 머신 로직
            if state == "SEARCH":
                robot.set_motor(30, 0) # 직진 탐색
                if avg_right_dist > PARKING_DEPTH_THRESHOLD:
                    print(f"✨ 공간 발견! ({avg_right_dist:.2f}m)")
                    state = "READY"
                    state_start_time = time.time()

            elif state == "READY":
                robot.set_motor(30, 0) # 조금 더 가서 정차 위치 잡기
                if time.time() - state_start_time > READY_TIME:
                    print("🛑 위치 확보. 정지 후 후진 준비")
                    state = "REVERSE"
                    robot.set_motor(0, 0)
                    time.sleep(1) # 기어 변속 시간 벌기

            elif state == "REVERSE":
                # 핸들 우측 최대(+50), 후진(-20)
                robot.set_motor(-20, 50) 
                
                # 시뮬레이션이 아닌 실제 상황에서는 후방 센서가 튈 수 있으므로 필터링 필요
                if 0 < rear_dist_cm <= STOP_DIST_CM:
                    print("🛑 주차 완료 (후방 감지)")
                    state = "STOP"
                    state_start_time = time.time()

            elif state == "STOP":
                robot.set_motor(0, 0)
                if time.time() - state_start_time > 3.0: # 3초 대기
                    state = "EXIT"

            elif state == "EXIT":
                print("👋 출차 중...")
                robot.set_motor(30, -30) # 좌회전하며 나감 (진입 반대)
                # 탈출 조건 (예: 일정 시간 후 종료)
                if time.time() - state_start_time > 5.0: # 대충 3초뒤에 종료
                    break

            # 4. 시각화 업데이트
            key = draw_visualization(lidar_scan, state, rear_dist_cm)
            if key == ord('q'): # 'q' 누르면 종료
                break
                
            time.sleep(0.05) # 루프 속도 조절

    except KeyboardInterrupt:
        print("강제 종료")
    finally:
        robot.stop()
        cv2.destroyAllWindows()
        print("시스템 종료")

if __name__ == "__main__":
    main()