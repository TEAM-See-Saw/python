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

# ★ [수정 1] 감지 거리 늘림 (1.2m -> 1.4m)
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 1400

# ★ [수정 2] 감지 각도 대폭 확대 (40도 -> 60도)
# 회피하려고 꺾었을 때 시야에서 사라지는 것 방지
LIDAR_ANGLE_MAX = 60
LIDAR_ANGLE_MIN = 300

# ★ [수정 3] 회피 강도 (기본값)
SHIFT_GAIN_BASE = 1.5
SHIFT_GAIN_STRONG = 2.2  # 2차 회피용 강력한 값

# 다음 장애물 카운트 쿨다운
OBSTACLE_COUNT_COOLDOWN = 2.0

# 센서 디바운싱 (잠깐 사라져도 0.3초는 유지)
SENSOR_DEBOUNCE_TIME = 0.3

LANE2_BIAS_RATIO = 0.12
LANE_BIAS_ONLY_WHEN_BOTH = True
BOTH_SEEN_HOLD_SEC = 0.6


def calc_shift_px(raw_dist, trigger_dist, gain):
    # gain 파라미터 추가됨
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
    cap_lane.set(3, width);
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
    desired_lane = 2

    print("🚀 주행 시작 (Improved Real-time Avoidance)")
    print(f"Dist Settings: #1={OBSTACLE_DIST_STAGE1}, #2={OBSTACLE_DIST_STAGE2}")

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
            # 1. 라이다 거리 측정 (각도 필터링 확대됨)
            # ------------------------------------------------
            raw_dist = 9999
            detected = False

            if scan:
                for (_, ang, dist) in scan:
                    if 200 < dist < 2500:
                        # 60도 범위로 넓게 봄
                        if ang < LIDAR_ANGLE_MAX or ang > LIDAR_ANGLE_MIN:
                            if dist < raw_dist:
                                raw_dist = dist
                                detected = True

            # ------------------------------------------------
            # 2. 장애물 판단
            # ------------------------------------------------
            trigger_dist = OBSTACLE_DIST_STAGE1 if obstacle_count == 0 else OBSTACLE_DIST_STAGE2

            # [A] 감지됨
            if detected and raw_dist < trigger_dist:
                last_obstacle_seen_time = now

                if not is_avoiding:
                    if (now - avoid_end_time) > OBSTACLE_COUNT_COOLDOWN:
                        obstacle_count += 1
                        is_avoiding = True
                        print(f"⚠️ 장애물 #{obstacle_count} 발견! (거리: {int(raw_dist)})")
                        if obstacle_count == 1:
                            desired_lane = 1
                        elif obstacle_count == 2:
                            desired_lane = 2

            # [B] 사라짐 (디바운싱 적용)
            else:
                if is_avoiding:
                    # 0.3초 정도는 안 보여도 유지 (회피 중 사각지대 보완)
                    if (now - last_obstacle_seen_time) > SENSOR_DEBOUNCE_TIME:
                        is_avoiding = False
                        avoid_end_time = now
                        print(f"✅ 장애물 #{obstacle_count} 통과 완료")

            # ------------------------------------------------
            # 3. 회피 방향 및 Shift 계산
            # ------------------------------------------------
            avoid_direction = 0
            shift_px = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            # ★ [수정] 장애물 번호에 따라 회피 강도(Gain) 다르게 적용
            current_gain = SHIFT_GAIN_BASE
            if obstacle_count == 2:
                current_gain = SHIFT_GAIN_STRONG  # 2번 장애물은 더 세게!

            if is_avoiding:
                shift_px = calc_shift_px(raw_dist, trigger_dist, current_gain)

                if obstacle_count == 1:
                    avoid_direction = -1
                    status_msg = f"LEFT #{obstacle_count} (G:{current_gain})"
                    status_color = (0, 255, 255)
                elif obstacle_count == 2:
                    avoid_direction = 1
                    status_msg = f"RIGHT #{obstacle_count} (G:{current_gain})"
                    status_color = (255, 0, 255)
                else:
                    final_speed = 100

            # ------------------------------------------------
            # 4. 카메라 및 조향
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

            # Shift 적용
            if avoid_direction == -1:
                base_target -= shift_px
            elif avoid_direction == 1:
                base_target += shift_px

            # Bias 적용
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

            cv2.imshow("Realtime Avoid (Wide Angle)", mask_bgr)
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