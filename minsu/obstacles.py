import serial
import time
from rplidar import RPLidar
import cv2
import numpy as np

# common_lane_base에서 필요한 변수/함수들 import
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

# 감지 거리 설정 (1차, 2차)
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 1200

# 감지 각도 설정 (40도 미만, 320도 초과)
LIDAR_ANGLE_MAX = 40
LIDAR_ANGLE_MIN = 320

# 회피 강도
SHIFT_GAIN = 1.5

# 다음 장애물 카운트까지 대기 시간 (같은 장애물 중복 카운트 방지)
OBSTACLE_COUNT_COOLDOWN = 2.0

# 센서 노이즈 방지용 디바운싱 (장애물이 사라져도 이 시간만큼은 기다림)
SENSOR_DEBOUNCE_TIME = 0.2

LANE2_BIAS_RATIO = 0.12
LANE_BIAS_ONLY_WHEN_BOTH = True
BOTH_SEEN_HOLD_SEC = 0.6


def calc_shift_px(raw_dist, trigger_dist):
    # 거리가 가까울수록 shift 값이 커짐 (최대치 제한 없음, 거리에 반비례)
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
    except:
        pass

    try:
        lidar = RPLidar(LIDAR_PORT)
    except:
        pass

    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap_lane.set(3, width);
    cap_lane.set(4, height)
    configure_camera_auto(cap_lane)

    # --- 변수 초기화 ---
    obstacle_count = 0  # 장애물 발견 횟수 (0 -> 1 -> 2)
    is_avoiding = False  # 현재 회피 동작 중인가?

    last_obstacle_seen_time = 0.0  # 장애물이 마지막으로 관측된 시각
    avoid_end_time = 0.0  # 회피가 완전히 끝난 시각 (쿨다운용)

    last_serial_time = 0.0
    last_speed_time = 0.0

    both_seen_streak = 0
    last_both_seen_time = 0.0
    desired_lane = 2

    print("🚀 주행 시작 (Real-time Avoidance Mode)")
    print(f"Angle Range: < {LIDAR_ANGLE_MAX}° or > {LIDAR_ANGLE_MIN}°")

    try:
        if ser: ser.write(f"D,{MAX_SPEED}\n".encode())

        scan_iter = lidar.iter_scans() if lidar else [None] * 1000

        for scan in scan_iter:
            if ser:
                try:
                    if ser.in_waiting > 0: ser.read(ser.in_waiting)
                except:
                    pass

            now = time.time()

            # ------------------------------------------------
            # 1. 라이다 거리 측정 (각도 필터링 적용)
            # ------------------------------------------------
            raw_dist = 9999
            detected = False

            if scan:
                for (_, ang, dist) in scan:
                    # 거리 필터 (20cm ~ 2.5m) 및 각도 필터 (40도 미만 OR 320도 초과)
                    if 200 < dist < 2500:
                        if ang < LIDAR_ANGLE_MAX or ang > LIDAR_ANGLE_MIN:
                            if dist < raw_dist:
                                raw_dist = dist
                                detected = True

            # ------------------------------------------------
            # 2. 장애물 상태 판단 로직 (리팩토링됨)
            # ------------------------------------------------
            trigger_dist = OBSTACLE_DIST_STAGE1 if obstacle_count == 0 else OBSTACLE_DIST_STAGE2

            # [A] 장애물 감지됨 (설정 거리 이내)
            if detected and raw_dist < trigger_dist:
                last_obstacle_seen_time = now

                # 새로운 장애물인지 판단 (회피 중이 아니고, 쿨다운 지났을 때)
                if not is_avoiding:
                    if (now - avoid_end_time) > OBSTACLE_COUNT_COOLDOWN:
                        obstacle_count += 1
                        is_avoiding = True
                        print(f"⚠️ 장애물 #{obstacle_count} 발견! 회피 시작")
                        if obstacle_count == 1:
                            desired_lane = 1
                        elif obstacle_count == 2:
                            desired_lane = 2
                    else:
                        # 쿨다운 중이면 카운트 안 올리고 그냥 감속/무시
                        pass

            # [B] 장애물 없음 (또는 멀어짐)
            else:
                # 회피 중이었다면 종료 조건 체크
                if is_avoiding:
                    # 센서가 잠깐 튀는걸 방지하기 위해 아주 짧게(0.2초) 기다림
                    if (now - last_obstacle_seen_time) > SENSOR_DEBOUNCE_TIME:
                        is_avoiding = False
                        avoid_end_time = now
                        print(f"✅ 장애물 #{obstacle_count} 통과 -> 차선 복귀")

            # ------------------------------------------------
            # 3. 회피 방향 및 Shift 값 설정
            # ------------------------------------------------
            avoid_direction = 0
            shift_px = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            if is_avoiding:
                # 실제 거리에 기반하여 Shift 계산 (더 이상 0으로 강제하지 않음)
                shift_px = calc_shift_px(raw_dist, trigger_dist)

                if obstacle_count == 1:
                    avoid_direction = -1  # 왼쪽 회피
                    status_msg = f"AVOID LEFT #{obstacle_count}"
                    status_color = (0, 255, 255)
                elif obstacle_count == 2:
                    avoid_direction = 1  # 오른쪽 회피
                    status_msg = f"AVOID RIGHT #{obstacle_count}"
                    status_color = (255, 0, 255)
                else:
                    final_speed = 100  # 3번째 이상은 감속

            # ------------------------------------------------
            # 4. 카메라 및 조향 계산
            # ------------------------------------------------
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

            # 기본 타겟
            base_target = base_target_from_lines(width, left, right)

            # 회피 Shift 적용
            if avoid_direction == -1:
                base_target -= shift_px
            elif avoid_direction == 1:
                base_target += shift_px

            # 2차선 복귀 Bias (Lane 2 복귀 시에만, 장애물 없을 때)
            bias_allowed = (both_seen_streak >= 2) and ((now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC)
            if (desired_lane == 2) and (not is_avoiding) and bias_allowed:
                base_target += int(width * LANE2_BIAS_RATIO)

            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            # ------------------------------------------------
            # 5. 통신 & 디스플레이
            # ------------------------------------------------
            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(mask_bgr, (int(base_target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            cv2.putText(mask_bgr, f"Obs#{obstacle_count} Dist:{int(raw_dist)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_bgr, f"{status_msg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(mask_bgr, f"Shift:{int(shift_px)} Angle:{int(angle)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Lane + Realtime Avoid", mask_bgr)
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