import time
import math
import numpy as np
import cv2

# ==========================================
# 1. 시뮬레이션 설정
# ==========================================
IS_SIMULATION = True
WINDOW_W, WINDOW_H = 1000, 600  # 화면 크기 (가로, 세로)
SCALE = 50                      # 1미터 = 50픽셀

# 주차장 맵 설정 (가상의 벽 좌표)
# (시작점 x, y) -> (끝점 x, y)
WALLS = [
    [(0, 150), (400, 150)],    # 첫 번째 벽
    [(600, 150), (1000, 150)], # 두 번째 벽 (중간에 400~600이 빈 공간)
]

# 차량 설정
CAR_LENGTH = 0.4  # 차 길이 (m)
CAR_WIDTH = 0.2   # 차 폭 (m)

# ==========================================
# 2. 로봇 하드웨어 (시뮬레이터 포함)
# ==========================================
class RobotHardware:
    def __init__(self, simulation=False):
        self.sim = simulation
        self.lidar = None
        
        # [시뮬레이션용 로봇 상태]
        self.x = 50.0   # 시작 위치 X (픽셀)
        self.y = 300.0  # 시작 위치 Y (픽셀)
        self.angle = 0.0 # 바라보는 각도 (라디안, 0=오른쪽)
        self.speed = 0.0 # 현재 속도
        self.steer = 0.0 # 조향 각도
        self.last_time = time.time()

        if not self.sim:
            # 실제 장비 연결 코드 (생략)
            pass

    def update_physics(self):
        """시뮬레이션: 모터 값에 따라 차 위치 이동"""
        if not self.sim: return

        dt = time.time() - self.last_time
        self.last_time = time.time()

        # 간단한 자전거 모델 (Bicycle Model)
        self.angle += (self.speed * math.tan(math.radians(self.steer)) / (CAR_LENGTH * SCALE)) * dt
        self.x += self.speed * math.cos(self.angle) * dt
        self.y += self.speed * math.sin(self.angle) * dt

    def get_lidar_scan(self):
        """현재 위치에서 벽까지의 거리 계산 (Ray Casting 흉내)"""
        scan = [0.0] * 360
        if self.sim:
            # 시뮬레이션: 우측(90도)에 벽이 있는지 확인
            # 내 위치(x)가 400~600 사이면 빈 공간(주차장 입구)
            # Y좌표가 150(벽 위치)이므로 거리 = 내 Y - 150
            
            # 간단하게 우측 90도 방향만 계산
            dist_to_wall = (self.y - 150) / SCALE # 미터 단위 변환
            
            # 주차 공간(X좌표 400~600)에 있으면 벽이 멀리 있음
            if 400 < self.x < 600:
                dist_right = 3.0 # 빈 공간 (3m)
            else:
                dist_right = dist_to_wall # 벽 있음
            
            # 노이즈 추가
            for i in range(85, 95):
                scan[i] = dist_right + np.random.uniform(-0.02, 0.02)
        else:
            # 실제 라이다 코드
            pass
            
        return scan

    def get_ultrasonic_rear(self):
        """후방 센서 (Y좌표가 0에 가까워지면 값 작아짐)"""
        if self.sim:
            # 화면 위쪽(Y=0)을 벽이라고 가정
            dist_cm = (self.y - 100) / SCALE * 100 # 임의의 후방 벽
            return max(0, dist_cm)
        return 999

    def set_motor(self, speed, angle):
        """속도 및 조향 설정 (시뮬레이션 물리엔진에 반영)"""
        # 시뮬레이션상 속도 배율 보정
        self.speed = speed * 3.0 
        self.steer = angle

    def stop(self):
        self.set_motor(0, 0)

# ==========================================
# 3. 시각화 (Top-Down Map)
# ==========================================
def draw_global_map(robot, state):
    # 회색 배경
    img = np.full((WINDOW_H, WINDOW_W, 3), 50, dtype=np.uint8)

    # 1. 벽 그리기 (주차 라인)
    for wall in WALLS:
        cv2.line(img, wall[0], wall[1], (0, 255, 255), 5) # 노란색 벽

    # 2. 로봇 그리기 (회전된 사각형)
    # 로봇 중심 좌표 및 회전 행렬 계산
    rect = ((robot.x, robot.y), (CAR_LENGTH*SCALE*2, CAR_WIDTH*SCALE*2), math.degrees(robot.angle))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # 색상: 탐색중(초록), 주차중(파랑), 완료(빨강)
    color = (0, 255, 0)
    if state == "REVERSE": color = (255, 100, 0)
    elif state == "STOP": color = (0, 0, 255)
    
    cv2.drawContours(img, [box], 0, color, -1)
    
    # 헤딩 방향 표시 (차 앞머리)
    front_x = robot.x + 30 * math.cos(robot.angle)
    front_y = robot.y + 30 * math.sin(robot.angle)
    cv2.line(img, (int(robot.x), int(robot.y)), (int(front_x), int(front_y)), (0, 0, 0), 2)

    # 3. 정보 텍스트
    cv2.putText(img, f"STATE: {state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"POS: ({int(robot.x)}, {int(robot.y)})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Parking Simulation (Top-Down)", img)
    return cv2.waitKey(1) & 0xFF

# ==========================================
# 4. 메인 로직
# ==========================================
def main():
    robot = RobotHardware(simulation=IS_SIMULATION)
    state = "SEARCH"
    state_start_time = time.time()
    
    # 파라미터 설정
    PARKING_DEPTH_THRESHOLD = 1.5 # 1.5m 이상 뚫리면 주차공간
    READY_TIME = 2.0  # 공간 발견 후 더 가는 시간

    print("🚀 시뮬레이션 시작! (화면을 클릭하고 'q'를 누르면 종료)")

    try:
        while True:
            # [중요] 시뮬레이션 물리 업데이트
            robot.update_physics()

            # 1. 센서 값 읽기
            lidar_scan = robot.get_lidar_scan()
            rear_dist_cm = robot.get_ultrasonic_rear()

            # 우측 거리 평균 계산
            right_dists = [d for d in lidar_scan[85:95] if d > 0.1]
            avg_right_dist = np.mean(right_dists) if right_dists else 0.0

            # 2. 상태 머신 (로직)
            if state == "SEARCH":
                # 직진하며 우측 탐색
                robot.set_motor(20, 0) 
                
                # 우측 벽이 갑자기 멀어지면 (공간 발견)
                if avg_right_dist > PARKING_DEPTH_THRESHOLD:
                    print(f"✨ 주차 공간 발견! (거리: {avg_right_dist:.2f}m)")
                    state = "READY"
                    state_start_time = time.time()

            elif state == "READY":
                # 차를 주차 공간보다 조금 더 앞으로 보냄 (오버런)
                robot.set_motor(20, 0)
                if time.time() - state_start_time > READY_TIME:
                    print("🛑 정지! 후진 준비")
                    state = "REVERSE"
                    robot.set_motor(0, 0)
                    time.sleep(0.5)

            elif state == "REVERSE":
                # 핸들을 오른쪽으로 꺾고 후진
                # 시뮬레이션 좌표계상: 핸들(+), 속도(-) -> 우측 후방으로 휨
                robot.set_motor(-15, 40) 

                # 차가 충분히 안쪽(Y좌표 기준)으로 들어오면 정지
                if robot.y < 200: 
                    print("🛑 주차 완료!")
                    state = "STOP"
                    state_start_time = time.time()

            elif state == "STOP":
                robot.set_motor(0, 0)
                if time.time() - state_start_time > 3.0:
                    state = "EXIT"
            
            elif state == "EXIT":
                 # 왼쪽으로 꺾어서 나감
                robot.set_motor(20, -40)
                if robot.y > 350: # 도로로 복귀하면 끝
                    break

            # 3. 화면 그리기
            key = draw_global_map(robot, state)
            if key == ord('q'):
                break
            
            time.sleep(0.03) # 30ms 딜레이

    except KeyboardInterrupt:
        print("종료")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()