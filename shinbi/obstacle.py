import serial
import time
import math
import cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

# common_lane_base에서 필요한 함수와 변수들을 가져옵니다.
from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, MAX_SPEED, 
    configure_camera_auto, compute_lane, base_target_from_lines,
    angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO
)

# ==========================================
# [1] 튜닝 파라미터 (현장에서 조절 필요)
# ==========================================

# 하드웨어 설정
LIDAR_PORT = 'COM3'
WIDTH, HEIGHT = 640, 480

# 장애물 감지 거리 (mm) 
DIST_TRIGGER_1 = 1200  # 첫 번째 장애물 감지 거리
DIST_TRIGGER_2 = 1000  # 두 번째 장애물 감지 거리
# 차가 너무 가까이 붙어서 피하기 시작하면 값을 키우기

# 측면 통과 확인 거리 (mm)
# 이 거리보다 멀어지면 "장애물을 완전히 지났다"고 판단
SIDE_CLEAR_THRESHOLD = 600 
# 차가 장애물을 다 지나지도 않았는데 다시 차선으로 들어오면 값 줄이기
# 장애물을 다 지났는데도 복귀 안 하면 값을 키우기

# 라이다 각도 설정
ANGLE_FRONT_RANGE = 30     # 전방 ±30도
ANGLE_RIGHT_SIDE_MIN = 40  # 우측 측면 감지 시작 각도
ANGLE_RIGHT_SIDE_MAX = 110 # 우측 측면 감지 끝 각도
ANGLE_LEFT_SIDE_MIN = 250  # 좌측 측면 감지 시작 각도 (360-110)
ANGLE_LEFT_SIDE_MAX = 320  # 좌측 측면 감지 끝 각도 (360-40)

# 회피량 설정 (Pixel 단위) - 고정값 사용이 훨씬 안정적임
SHIFT_AMOUNT_LEFT = -140  # 1번 장애물 피할 때 (왼쪽으로 이동)
SHIFT_AMOUNT_RIGHT = 140  # 2번 장애물 피할 때 (오른쪽으로 이동)
# 차가 장애물을 충분히 피하지 못하고 모서리에 걸리면 절대값을 키우기

# 속도 설정
SPEED_NORMAL = MAX_SPEED
SPEED_AVOID = int(MAX_SPEED * 0.8) # 회피 중엔 약간 감속

# ==========================================
# [2] 라이다 데이터 처리 함수
# ==========================================
def get_lidar_view(scan):
    """
    스캔 데이터에서 전방/우측/좌측의 최소 거리를 각각 추출
    return: front_min, right_min, left_min
    """
    f_min = 2000
    r_min = 2000 # 우측 (장애물이 차의 오른쪽에 있을 때 확인용)
    l_min = 2000 # 좌측 (장애물이 차의 왼쪽에 있을 때 확인용)

    if scan:
        for (_, ang, dist) in scan:
            if dist < 10 or dist > 2500: continue # 노이즈 및 먼 거리 무시
            
            # 1. 전방 감지 (330~360 or 0~30)
            if ang >= (360 - ANGLE_FRONT_RANGE) or ang <= ANGLE_FRONT_RANGE:
                if dist < f_min: f_min = dist
            
            # 2. 우측 측면 (40 ~ 110도) -> 1번 장애물 확인용
            if ANGLE_RIGHT_SIDE_MIN <= ang <= ANGLE_RIGHT_SIDE_MAX:
                if dist < r_min: r_min = dist
                
            # 3. 좌측 측면 (250 ~ 320도) -> 2번 장애물 확인용
            if ANGLE_LEFT_SIDE_MIN <= ang <= ANGLE_LEFT_SIDE_MAX:
                if dist < l_min: l_min = dist
                
    return f_min, r_min, l_min

# ==========================================
# [3] 메인 루프
# ==========================================
def main():
    # --- 연결 초기화 ---
    ser = None
    lidar = None
    
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ 모터 연결 성공: {PORT}")
    except Exception as e:
        print(f"❌ 모터 연결 실패: {e}")

    try:
        lidar = RPLidar(LIDAR_PORT)
        print(f"✅ 라이다 연결 성공: {LIDAR_PORT}")
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 라이다 연결 실패: {e}")
        return

    # 카메라 초기화
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    configure_camera_auto(cap)

    # --- 상태 변수 ---
    # obstacle_step:
    # 0: 탐색 중 (첫번째 장애물 기다림)
    # 1: 1번 장애물 회피 중 (우측에 장애물, 왼쪽으로 피하는 중)
    # 2: 1번 통과 완료 -> 2번 장애물 탐색 중
    # 3: 2번 장애물 회피 중 (좌측에 장애물, 오른쪽으로 피하는 중)
    # 4: 모든 장애물 통과 완료
    obstacle_step = 0
    
    last_serial_time = 0
    last_speed_time = 0
    
    # 라이다 스캔 이터레이터
    scan_iter = lidar.iter_scans()

    print("🚀 장애물 회피 주행 시작!")

    try:
        # 모터 시동
        if ser: ser.write(f"D,{SPEED_NORMAL}\n".encode())

        while True:
            now = time.time()

            # 1. 센서 데이터 수집 (Lidar)
            # ----------------------------------------
            scan = None
            try:
                scan = next(scan_iter)
            except StopIteration:
                scan_iter = lidar.iter_scans()
            except RPLidarException:
                lidar.clean_input()
            
            # 전방, 우측, 좌측 거리 추출
            front_d, right_d, left_d = get_lidar_view(scan)

            # 2. 영상 처리 (Camera)
            # ----------------------------------------
            ret, frame = cap.read()
            if not ret: break
            if frame.shape[1] != WIDTH:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
            
            lane = compute_lane(frame)
            left_line, right_line = lane["left"], lane["right"]
            mask_disp = lane["mask_bgr"]
            
            # 기본 타겟(차선 중앙) 계산
            base_target = base_target_from_lines(WIDTH, left_line, right_line)
            
            # 3. 장애물 회피 로직 (State Machine)
            # ----------------------------------------
            shift_val = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            target_speed = SPEED_NORMAL

            # [STEP 0] 1번 장애물 탐색 (우측에 위치한다고 가정 -> 왼쪽 회피)
            if obstacle_step == 0:
                status_msg = "SEARCH OBS #1"
                if front_d < DIST_TRIGGER_1:
                    obstacle_step = 1
                    print(f"⚠️ 1번 장애물 발견! (전방 {int(front_d)}mm) -> 좌측 회피 시작")

            # [STEP 1] 1번 장애물 회피 (왼쪽으로 Shift)
            elif obstacle_step == 1:
                status_msg = "AVOIDING #1 (LEFT)"
                status_color = (0, 0, 255) # Red
                shift_val = SHIFT_AMOUNT_LEFT
                target_speed = SPEED_AVOID
                
                # 복귀 조건: 전방이 뚫려있고(안전) AND 우측 측면도 비었을 때
                # 우측 센서(right_d)가 SIDE_CLEAR_THRESHOLD보다 커지면 장애물이 없는 것임
                if front_d > (DIST_TRIGGER_1 + 200) and right_d > SIDE_CLEAR_THRESHOLD:
                    obstacle_step = 2
                    print("✅ 1번 장애물 통과 완료 -> 중앙 복귀")

            # [STEP 2] 2번 장애물 탐색 (좌측에 위치한다고 가정 -> 우측 회피)
            elif obstacle_step == 2:
                status_msg = "SEARCH OBS #2"
                status_color = (255, 255, 0) # Cyan
                if front_d < DIST_TRIGGER_2:
                    obstacle_step = 3
                    print(f"⚠️ 2번 장애물 발견! (전방 {int(front_d)}mm) -> 우측 회피 시작")

            # [STEP 3] 2번 장애물 회피 (오른쪽으로 Shift)
            elif obstacle_step == 3:
                status_msg = "AVOIDING #2 (RIGHT)"
                status_color = (255, 0, 255) # Magenta
                shift_val = SHIFT_AMOUNT_RIGHT
                target_speed = SPEED_AVOID

                # 복귀 조건: 전방 뚫림 AND 좌측 측면 비었음
                if front_d > (DIST_TRIGGER_2 + 200) and left_d > SIDE_CLEAR_THRESHOLD:
                    obstacle_step = 4
                    print("✅ 2번 장애물 통과 완료 -> 미션 클리어")

            # [STEP 4] 미션 완료 (일반 주행)
            elif obstacle_step == 4:
                status_msg = "ALL CLEAR"
                status_color = (0, 255, 0)
            
            # 최종 타겟 결정
            final_target = base_target + shift_val
            
            # 조향각 계산
            angle = angle_from_target(WIDTH, HEIGHT, final_target)
            servo_cmd = servo_from_angle(angle)

            # 4. 모터 제어 명령
            # ----------------------------------------
            if ser:
                # 조향 (자주 보냄)
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_cmd}\n".encode())
                    last_serial_time = now
                
                # 속도 (가끔 보냄)
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{target_speed}\n".encode())
                    last_speed_time = now

            # 5. 화면 디스플레이
            # ----------------------------------------
            # ROI 박스
            roi_y = int(HEIGHT * ROI_HEIGHT_RATIO)
            cv2.line(mask_disp, (0, roi_y), (WIDTH, roi_y), (100, 100, 100), 1)
            
            # 타겟 포인트 (빨간점: 최종 타겟, 파란점: 원래 타겟)
            cv2.circle(mask_disp, (int(base_target), roi_y), 5, (255, 0, 0), -1) 
            cv2.circle(mask_disp, (int(final_target), roi_y), 8, (0, 0, 255), -1)

            # 정보 텍스트
            info_text = f"Step:{obstacle_step} | F:{int(front_d)} R:{int(right_d)} L:{int(left_d)}"
            cv2.putText(mask_disp, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(mask_disp, f"{status_msg} (Shift:{shift_val})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow("Obstacle Mission Final", mask_disp)
            
            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("사용자 종료")
    
    finally:
        print("🛑 시스템 정지")
        if ser:
            ser.write(b"D,0\n")
            ser.write(b"S,570\n") # 센터값(환경따라 변경)
            ser.close()
        if lidar:
            lidar.stop()
            lidar.disconnect()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()