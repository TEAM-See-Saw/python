import serial
import time
from rplidar import RPLidar
import cv2

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

# 감지 거리 설정
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 1400

# [가변 시야각 Adaptive FOV]
# 평상시: 좁게 봐서 옆 차선 장애물 무시
ANGLE_DETECT_MAX = 10
ANGLE_DETECT_MIN = 350
# 회피중: 넓게 봐서 놓침 방지
ANGLE_KEEP_MAX = 40
ANGLE_KEEP_MIN = 320

# 회피 강도 (Gain)
SHIFT_GAIN_BASE = 1.5
SHIFT_GAIN_STRONG = 1.8  # 2번 장애물 복귀 시 강하게

# 다음 장애물 카운트 쿨다운
OBSTACLE_COUNT_COOLDOWN = 2.0

# 센서 디바운싱
SENSOR_DEBOUNCE_TIME = 0.3

# 차선 Bias (2차선 유지용)
LANE2_BIAS_RATIO = 0.12
LANE_BIAS_ONLY_WHEN_BOTH = True
BOTH_SEEN_HOLD_SEC = 0.6


def calc_shift_px(raw_dist, trigger_dist, gain):
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (gain / (trigger_dist ** 0.25))


def main():
    ser = None
    lidar = None

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
    cap_lane.set(3, width)
    cap_lane.set(4, height)
    configure_camera_auto(cap_lane)

    # 변수 초기화
    obstacle_count = 0
    is_avoiding = False

    last_obstacle_seen_time = 0.0
    avoid_end_time = 0.0

    last_serial_time = 0.0
    last_speed_time = 0.0

    both_seen_streak = 0
    last_both_seen_time = 0.0

    # ★ 핵심 상태 변수: 시작은 2차선
    desired_lane = 2

    print("🚀 시나리오 3 주행 시작")
    print("Logic: Start Lane 2 -> Obs1(Left) -> Stay Lane 1 -> Obs2(Right) -> Return Lane 2")

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
            # 1. FOV (시야각) 모드 설정
            # ------------------------------------------------
            if is_avoiding:
                cur_angle_max = ANGLE_KEEP_MAX
                cur_angle_min = ANGLE_KEEP_MIN
            else:
                cur_angle_max = ANGLE_DETECT_MAX
                cur_angle_min = ANGLE_DETECT_MIN

            # ------------------------------------------------
            # 2. 라이다 감지
            # ------------------------------------------------
            raw_dist = 9999
            detected = False

            if scan:
                for (_, ang, dist) in scan:
                    if 200 < dist < 2500:
                        if ang < cur_angle_max or ang > cur_angle_min:
                            if dist < raw_dist:
                                raw_dist = dist
                                detected = True

            # ------------------------------------------------
            # 3. 장애물 판단 및 차선 상태 변경
            # ------------------------------------------------
            trigger_dist = OBSTACLE_DIST_STAGE1 if obstacle_count == 0 else OBSTACLE_DIST_STAGE2

            # [A] 감지됨
            if detected and raw_dist < trigger_dist:
                last_obstacle_seen_time = now

                if not is_avoiding:
                    if (now - avoid_end_time) > OBSTACLE_COUNT_COOLDOWN:
                        obstacle_count += 1
                        is_avoiding = True
                        print(f"⚠️ 장애물 #{obstacle_count} 감지! -> 회피 시작")

                        # ★ 시나리오 로직 적용
                        if obstacle_count == 1:
                            desired_lane = 1  # 1번 발견하면 1차선으로 변경 (Bias 해제)
                        elif obstacle_count == 2:
                            desired_lane = 2  # 2번 발견하면 2차선으로 복귀 (Bias 적용)

            # [B] 사라짐
            else:
                if is_avoiding:
                    if (now - last_obstacle_seen_time) > SENSOR_DEBOUNCE_TIME:
                        is_avoiding = False
                        avoid_end_time = now
                        print(f"✅ 장애물 #{obstacle_count} 통과 -> 현재 목표 차선: {desired_lane}차선")

            # ------------------------------------------------
            # 4. 회피 방향 및 Shift 계산
            # ------------------------------------------------
            avoid_direction = 0
            shift_px = 0
            status_msg = f"LANE {desired_lane}"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            # 장애물 번호별 회피 강도 설정
            current_gain = SHIFT_GAIN_BASE
            if obstacle_count == 2:
                current_gain = SHIFT_GAIN_STRONG

            if is_avoiding:
                shift_px = calc_shift_px(raw_dist, trigger_dist, current_gain)

                if obstacle_count == 1:
                    avoid_direction = -1  # 왼쪽 회피
                    status_msg = f"LEFT #{obstacle_count} -> LANE 1"
                    status_color = (0, 255, 255)
                elif obstacle_count == 2:
                    avoid_direction = 1  # 오른쪽 회피
                    status_msg = f"RIGHT #{obstacle_count} -> LANE 2"
                    status_color = (255, 0, 255)
                else:
                    final_speed = 100

            # ------------------------------------------------
            # 5. 카메라 처리 및 조향 (Bias 로직 포함)
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

            base_target = base_target_from_lines(width, left, right)

            # (1) Shift 적용 (회피)
            if avoid_direction == -1:
                base_target -= shift_px
            elif avoid_direction == 1:
                base_target += shift_px

            # (2) Bias 적용 (차선 유지)
            # desired_lane이 2일 때만 우측 Bias를 줌.
            # desired_lane이 1이면 Bias 없이 중앙 주행 (1차선 유지)
            bias_allowed = (both_seen_streak >= 2) and ((now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC)

            if (desired_lane == 2) and (not is_avoiding) and bias_allowed:
                base_target += int(width * LANE2_BIAS_RATIO)

            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            # ------------------------------------------------
            # 6. 통신 & 디스플레이
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

            # 상태 표시
            cv2.putText(mask_bgr, f"Target Lane: {desired_lane}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(mask_bgr, f"{status_msg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if is_avoiding:
                cv2.putText(mask_bgr, f"Dist:{int(raw_dist)} Gain:{current_gain}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(mask_bgr, "Cruising...", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Scenario 3 Control", mask_bgr)
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