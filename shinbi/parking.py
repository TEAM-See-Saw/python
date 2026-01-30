import time

class ParkingSystem:
    def __init__(self):
        # 상태 정의: SEARCH(탐색) -> READY(준비) -> PARK(진입) -> STOP(정지) -> EXIT(탈출)
        self.state = "SEARCH"
        self.start_time = 0
        
        # 튜닝 파라미터 (실제 차에 맞춰서 수정 필요!)
        self.PARKING_DEPTH_THRESHOLD = 1.0  # (m) 이 이상 깊으면 주차 공간으로 인식
        self.STOP_DISTANCE_CM = 15          # (cm) 후방 벽과 이 거리 남으면 정지
        self.WAIT_TIME = 3.0                # (초) 주차 후 대기 시간

    def process(self, lidar_data, us_rear_left, us_rear_right):
        """
        메인 루프에서 실행될 함수
        :param lidar_data: 라이다 거리 배열 (예: lidar[90]은 우측 90도 거리)
        :param us_rear_left: 후방 좌측 초음파 거리 (cm)
        :param us_rear_right: 후방 우측 초음파 거리 (cm)
        """
        
        # 후방 거리 중 더 위험한(가까운) 값을 기준으로 잡음
        rear_distance = min(us_rear_left, us_rear_right)
        
        # ---------------------------------------------------------
        # 1. 탐색 단계 (SEARCH): 우측을 보며 주차 공간 찾기
        # ---------------------------------------------------------
        if self.state == "SEARCH":
            # 라이다 우측(80~100도) 평균 거리를 구함
            # (인덱스는 사용하는 라이다 장비에 따라 다를 수 있음, 여기선 우측이 90도라 가정)
            right_side_dist = lidar_data[90] 

            print(f"탐색 중... 우측 거리: {right_side_dist}m")

            # 평소엔 벽(0.3m)이 있다가, 갑자기 1m 이상 뚫리면 주차 공간 발견!
            if right_side_dist > self.PARKING_DEPTH_THRESHOLD:
                print("!! 주차 공간 발견 !! -> 위치 잡기")
                self.state = "READY"
                self.start_time = time.time()
                return {"speed": 30, "angle": 0} # 발견 즉시 멈추지 말고 조금 더 가서 멈춰야 함

            return {"speed": 50, "angle": 0} # 직진하며 탐색

        # ---------------------------------------------------------
        # 2. 진입 준비 (READY): 차를 꺾기 좋은 위치로 이동
        # ---------------------------------------------------------
        elif self.state == "READY":
            # 공간을 발견하고 나서 차체 길이만큼 조금 더 앞으로 가야 후진 각이 나옴
            # 여기서는 간단하게 시간(0.5초)으로 처리 (엔코더가 있다면 거리로 처리 추천)
            if time.time() - self.start_time > 1.5: 
                print("위치 확보 완료. 정지 후 후진 준비")
                self.state = "PARK"
                return {"speed": 0, "angle": 0} # 잠시 정지
            
            return {"speed": 30, "angle": 0} # 조금 더 직진

        # ---------------------------------------------------------
        # 3. 주차 진입 (PARK): 핸들 꺾고 후진
        # ---------------------------------------------------------
        elif self.state == "PARK":
            print(f"후진 중... 뒤 벽까지: {rear_distance}cm")

            # 초음파 센서값이 위험 거리보다 가까워지면 정지!
            if rear_distance <= self.STOP_DISTANCE_CM:
                print("!! 후방 장애물 감지 -> 정지 !!")
                self.state = "STOP"
                self.start_time = time.time() # 정지 시간 기록
                return {"speed": 0, "angle": 0}

            # 우측으로 핸들을 꺾고(-50) 후진(-30)
            # (모터 방향 설정에 따라 부호는 반대일 수 있음)
            return {"speed": -30, "angle": 50} 

        # ---------------------------------------------------------
        # 4. 정지 및 미션 수행 (STOP)
        # ---------------------------------------------------------
        elif self.state == "STOP":
            elapsed = time.time() - self.start_time
            print(f"주차 완료. 대기 중... {elapsed:.1f}초")

            if elapsed >= self.WAIT_TIME:
                self.state = "EXIT"
            
            return {"speed": 0, "angle": 0}

        # ---------------------------------------------------------
        # 5. 탈출 (EXIT): 다시 도로로 복귀
        # ---------------------------------------------------------
        elif self.state == "EXIT":
            print("주차장 탈출 중")
            
            # 주차장을 완전히 빠져나왔는지 판단 (여기선 간단히 시간으로, 실제론 라인 인식 활용)
            # 핸들을 좌측으로 꺾고 전진
            return {"speed": 40, "angle": -30} 

        return {"speed": 0, "angle": 0}
