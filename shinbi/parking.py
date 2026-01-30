import time
import numpy as np

class VerticalParking:
    def __init__(self):
        # 상태 머신: SEARCH -> MOVE_FORWARD -> REVERSE_ENTRY -> STOP -> EXIT
        self.state = "SEARCH"
        self.timer_start = 0
        
        # --- [튜닝 파라미터: 실제 차에 맞춰 값 수정 필수] ---
        self.DETECT_DIST = 1.5       # (m) 이 거리보다 멀면 '빈 공간'으로 인식
        self.WALL_DIST = 0.5         # (m) 평소 벽과의 거리 (참고용)
        self.READY_TIME = 2.0        # (초) 빈 공간 발견 후 앞으로 더 나가는 시간 (매우 중요!)
        self.STOP_DIST_CM = 20       # (cm) 후방 벽과 이 거리 남으면 정지
        self.PARKING_WAIT = 3.0      # (초) 주차 완료 후 대기 시간
        self.EXIT_TIME = 2.5         # (초) 탈출 시 전진하는 시간
        
        # RPLidar A1 각도 설정 (우측 90도를 보기 위해 85~95도 범위 평균 사용)
        self.LIDAR_ANGLE_MIN = 85
        self.LIDAR_ANGLE_MAX = 95

    def process(self, lidar_scan, us_rear_L, us_rear_R):
        """
        :param lidar_scan: 라이다 거리 배열 (360개 혹은 그 이상)
        :param us_rear_L: 후방 좌측 초음파 (cm)
        :param us_rear_R: 후방 우측 초음파 (cm)
        """
        
        # 1. 센서 데이터 전처리
        # 라이다 우측(90도 부근) 데이터 추출 및 노이즈 제거
        right_side_ranges = []
        # 인덱스 에러 방지 (lidar 개수가 360개가 아닐 수 있으므로)
        scan_len = len(lidar_scan)
        
        for deg in range(self.LIDAR_ANGLE_MIN, self.LIDAR_ANGLE_MAX):
            # 배열 인덱스 매핑 (혹시 모를 인덱스 초과 방지)
            idx = deg % scan_len 
            dist = lidar_scan[idx]
            # 0.0이나 inf(무한대)는 노이즈거나 측정 불가이므로 제외
            if 0.1 < dist < 10.0: 
                right_side_ranges.append(dist)

        # 유효한 값이 없으면 벽이 아주 가깝거나 먼 것으로 간주 (기본값 설정)
        current_right_dist = np.mean(right_side_ranges) if right_side_ranges else 0.0
        
        # 후방 거리: 둘 중 더 가까운 장애물 기준 (안전 제일)
        rear_dist = min(us_rear_L, us_rear_R)
        # 초음파 노이즈(0) 처리
        if rear_dist == 0: rear_dist = 999 

        # -------------------------------------------------------
        # 로직 시작
        # -------------------------------------------------------
        
        # [1단계] 탐색: 우측 벽을 보며 가다가 공간 발견
        if self.state == "SEARCH":
            print(f"[SEARCH] 우측 거리: {current_right_dist:.2f}m")
            
            # 우측 거리가 갑자기 임계값(1.5m) 이상으로 뚫리면 주차공간 입구!
            if current_right_dist > self.DETECT_DIST:
                print(">>> 빈 공간 발견! 위치 잡으러 이동 (MOVE_FORWARD)")
                self.state = "MOVE_FORWARD"
                self.timer_start = time.time()
                # 멈추지 않고 그대로 진행
                return {"speed": 40, "angle": 0} 
            
            # 평소 주행 (벽 타고 가기)
            return {"speed": 40, "angle": 0}

        # [2단계] 위치 잡기 (Over-run): 공간을 지나쳐서 정차 위치로 이동
        elif self.state == "MOVE_FORWARD":
            elapsed = time.time() - self.timer_start
            print(f"[READY] 진입 각도 만드는 중... {elapsed:.1f}초")
            
            if elapsed > self.READY_TIME:
                print(">>> 위치 확보 완료. 정지 후 후진 준비 (REVERSE_ENTRY)")
                self.state = "REVERSE_ENTRY"
                return {"speed": 0, "angle": 0} # 잠시 완전 정지
            
            # 시간을 튜닝해서 차 엉덩이가 주차 공간 끝라인을 살짝 지나게 해야 함
            return {"speed": 40, "angle": 0} 

        # [3단계] 진입: 핸들 다 꺾고 후진
        elif self.state == "REVERSE_ENTRY":
            print(f"[PARK] 후진 중... 후방 거리: {rear_dist}cm")
            
            # 후방 벽 감지 시 정지
            if rear_dist <= self.STOP_DIST_CM:
                print(">>> !! 쿵! 하기 전에 정지 (STOP) !!")
                self.state = "STOP"
                self.timer_start = time.time()
                return {"speed": 0, "angle": 0}
            
            # 후륜구동은 조향각을 최대로 하고 후진하면 됨
            # angle: +50 (우측 최대), speed: -30 (후진)
            return {"speed": -30, "angle": 50} 

        # [4단계] 주차 완료 및 대기
        elif self.state == "STOP":
            elapsed = time.time() - self.timer_start
            print(f"[WAIT] 주차 완료 대기... {elapsed:.1f}초")
            
            if elapsed > self.PARKING_WAIT:
                self.state = "EXIT"
                self.timer_start = time.time()
            
            return {"speed": 0, "angle": 0}

        # [5단계] 탈출: 반대 방향으로 핸들 꺾고 나감
        elif self.state == "EXIT":
            elapsed = time.time() - self.timer_start
            print("[EXIT] 빠져나가는 중...")
            
            # 탈출 완료 조건 (시간 or 라인 인식)
            if elapsed > self.EXIT_TIME:
                return "MISSION_COMPLETE" # 메인 로직에 끝났음을 알림
            
            # 들어올 때 우회전했으니, 나갈 땐 우회전하며 전진 (T자니까)
            # 만약 차가 일자로 이쁘게 서있다면 우회전으로 나가야 도로 방향임
            return {"speed": 40, "angle": 30} 

        return {"speed": 0, "angle": 0}