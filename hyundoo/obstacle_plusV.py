import serial
import time
from rplidar import RPLidar
import cv2

from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, MAX_SPEED
)
from common_lane_base import (
    configure_camera_auto, compute_lane, base_target_from_lines,
    angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO
)

# ==== obstacle 전용 설정 ====
LIDAR_PORT = 'COM3'

# ★ [수정 1] 2단계 감지 거리를 대폭 줄임 (1200 -> 900)
# 차가 2번 장애물에 충분히 가까워질 때까지(1번을 완전히 통과할 때까지) 반응하지 않게 함
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 900  

SHIFT_GAIN = 1.5

# 회피 후 유지 시간
OBSTACLE_CLEAR_TIME_STAGE1 = 2.5
OBSTACLE_CLEAR_TIME_STAGE2 = 2.0 # 2번 장애물은 좀 더 짧게 (복귀용)

# ★ [수정 2] 장애물 사이 쿨타임 변수화 및 증가 (1.5 -> 2.5)
# 1번 피하고 나서 2.5초 동안은 라이다를 무시하고 차선만 보고 달려서 자세를 잡음
INTER_OBSTACLE_COOLDOWN = 2.5 

LANE2_BIAS_RATIO = 0.12
LANE_BIAS_ONLY_WHEN_BOTH = True
BOTH_SEEN_HOLD_SEC = 0.6


def get_stage_trigger_and_clear(obstacle_count_now):
    if obstacle_count_now == 0:
        return OBSTACLE_DIST_STAGE1, OBSTACLE_CLEAR_TIME_STAGE1
    if obstacle_count_now == 1:
        return OBSTACLE_DIST_STAGE2, OBSTACLE_CLEAR_TIME_STAGE2
    return OBSTACLE_DIST_STAGE2, OBSTACLE_CLEAR_TIME_STAGE2


def calc_shift_px(raw_dist, trigger_dist):
    # 거리가 가까울수록 shift 값이 커짐
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (SHIFT_GAIN / (trigger_dist ** 0.25))


def main():
    ser = None
    lidar = None

    # --- 연결 설정 ---
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        time.sleep(1)
    except: pass

    try:
        lidar = RPLidar(LIDAR_PORT)
    except: pass

    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap_lane.set(3, width)
    cap_lane.set(4, height)
    configure_camera_auto(cap_lane)

    # --- 변수 초기화 ---
    obstacle_count = 0
    is_obstacle_detected = False
    obs_clear_finished_time = 0.0
    obstacle_last_seen_time = 0.0

    last_serial_time = 0.0
    last_speed_time = 0.0

    both_seen_streak = 0
    last_both_seen_time = 0.0
    desired_lane = 2

    print(f"🚀 주행 시작 (ZigZag Fix Mode)")
    print(f"Dist1: {OBSTACLE_DIST_STAGE1}, Dist2: {OBSTACLE_DIST_STAGE2}, Cooldown: {INTER_OBSTACLE_COOLDOWN}")

    try:
        if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

        scan_iter = lidar.iter_scans() if lidar else [None] * 1000

        for scan in scan_iter:
            if ser:
                try:
                    if ser.in_waiting > 0: ser.read(ser.in_waiting)
                except: pass

            now = time.time()

            # 1. 라이다 거리 측정
            raw_dist = 2000
            if scan:
                for (_, ang, dist) in scan:
                    # ★ [수정 3] 시야각을 좁힘 (330~30 -> 345~15)
                    # 차가 비스듬히 있을 때 옆에 있는 장애물을 정면으로 착각하지 않게 함
                    if 200 < dist < 2500:
                        if ang >= 345 or ang <= 15: 
                            if dist < raw_dist: raw_dist = dist

            # 2. 카메라 처리
            ret, frame_lane = cap_lane.read()
            if not ret: break
            if frame_lane.shape[1] != width: frame_lane = cv2.resize(frame_lane, (width, height))

            lane = compute_lane(frame_lane)
            left, right = lane["left"], lane["right"]
            mask_bgr = lane["mask_bgr"]
            roi_points = lane["roi_points"]

            if left is not None and right is not None:
                both_seen_streak = min(10, both_seen_streak + 1)
                last_both_seen_time = now
            else:
                both_seen_streak = max(0, both_seen_streak - 1)

            # ===== 장애물 회피 로직 =====
            stage_trigger, stage_clear_time = get_stage_trigger_and_clear(obstacle_count)

            avoid_direction = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED
            calc_dist_input = raw_dist

            # 거리 조건 만족 시
            if raw_dist < stage_trigger:
                obstacle_last_seen_time = now
                calc_dist_input = raw_dist

                # ★ 쿨타임 체크 (기존 1.5 -> INTER_OBSTACLE_COOLDOWN 적용)
                if not is_obstacle_detected and (now - obs_clear_finished_time > INTER_OBSTACLE_COOLDOWN):
                    obstacle_count += 1
                    is_obstacle_detected = True
                    
                    if obstacle_count == 1:
                        desired_lane = 1
                    elif obstacle_count == 2:
                        desired_lane = 2
                    
                    print(f"🚨 장애물 #{obstacle_count} 발견! (Dist: {int(raw_dist)})")

                # 방향 설정 (감지 중일 때)
                if is_obstacle_detected:
                    if obstacle_count == 1:
                        avoid_direction = -1  # 왼쪽
                        status_msg = f"AVOID LEFT #{obstacle_count}"
                        status_color = (0, 255, 255)
                    elif obstacle_count == 2:
                        avoid_direction = 1   # 오른쪽
                        status_msg = f"AVOID RIGHT #{obstacle_count}"
                        status_color = (255, 0, 255)

            else:
                # 장애물이 라이다에서 사라짐 (혹은 쿨타임 중)
                
                # 쿨타임 중이라도 , 전방 60cm 이내에 뭔가 있으면 즉시 반응해야함.
                if (not is_obstacle_detected) and (raw_dist < 700):
                    print("🚨 긴급! 쿨타임 강제 종료 (너무 가까움)")
                    obs_clear_finished_time = now - 100 # 쿨타임 즉시 만료시킴

                if is_obstacle_detected:
                    # 유지 시간(Holding) 체크
                    if (now - obstacle_last_seen_time) < stage_clear_time:
                        calc_dist_input = 0 # 최대 회피각 유지

                        if obstacle_count == 1:
                            avoid_direction = -1
                        elif obstacle_count == 2:
                            avoid_direction = 1

                        status_msg = f"HOLDING... {stage_clear_time - (now - obstacle_last_seen_time):.1f}s"
                        status_color = (0, 100, 255)
                    else:
                        # 유지 시간 종료 -> 복귀 시작
                        is_obstacle_detected = False
                        obs_clear_finished_time = now
                        avoid_direction = 0
                        print(f"✅ 회피 #{obstacle_count} 종료 -> 쿨타임 시작")

            # ===== 타겟 및 Shift 계산 =====
            base_target = base_target_from_lines(width, left, right)
            shift_px = 0

            if avoid_direction != 0:
                shift_px = calc_shift_px(calc_dist_input, stage_trigger)
                if avoid_direction == -1:
                    base_target -= shift_px
                else:
                    base_target += shift_px

            # 2차선 복귀 Bias (2번 장애물 회피 후 복귀 도움)
            bias_allowed = (both_seen_streak >= 2) and ((now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC)
            if (desired_lane == 2) and (avoid_direction == 0) and bias_allowed:
                base_target += int(width * LANE2_BIAS_RATIO)

            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            # ===== 통신 & 디스플레이 =====
            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # 시각화
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(mask_bgr, (int(base_target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            
            # 정보 표시
            cv2.putText(mask_bgr, f"Obs#{obstacle_count} Dist:{int(raw_dist)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_bgr, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # 쿨타임 표시 (디버깅용)
            if not is_obstacle_detected and (now - obs_clear_finished_time < INTER_OBSTACLE_COOLDOWN):
                cool_remain = INTER_OBSTACLE_COOLDOWN - (now - obs_clear_finished_time)
                cv2.putText(mask_bgr, f"Cooldown: {cool_remain:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

            cv2.imshow("ZigZag Master", mask_bgr)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cap_lane.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()