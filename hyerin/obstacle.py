import serial
import time
from rplidar import RPLidar
import cv2

# common_lane_base에서 필요한 변수/함수들 import (기존과 동일)
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
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 1200
SHIFT_GAIN = 1.5
# 회피 후 차선에 완전히 진입할 때까지 충분히 길게 유지해야 합니다
OBSTACLE_CLEAR_TIME_STAGE1 = 3.5
OBSTACLE_CLEAR_TIME_STAGE2 = 2.2

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
    # 거리가 가까울수록(0에 수렴할수록) shift 값이 커짐
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (SHIFT_GAIN / (trigger_dist ** 0.25))


def main():
    ser = None
    lidar = None

    # --- 연결 설정 (기존과 동일) ---
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
    obstacle_count = 0
    is_obstacle_detected = False
    obs_clear_finished_time = 0.0
    obstacle_last_seen_time = 0.0

    last_serial_time = 0.0
    last_speed_time = 0.0

    both_seen_streak = 0
    last_both_seen_time = 0.0
    desired_lane = 2

    print("🚀 주행 시작 (Obstacle Avoid Mode - Fixed)")

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

            # 1. 라이다 거리 측정
            raw_dist = 2000
            if scan:
                for (_, ang, dist) in scan:
                    if 200 < dist < 2500 and (ang >= 330 or ang <= 30):
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

            # ===== [핵심 수정] 장애물 회피 로직 =====
            stage_trigger, stage_clear_time = get_stage_trigger_and_clear(obstacle_count)

            avoid_direction = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            # ★ 조향 계산용 거리 변수 (실제 거리와 분리)
            # 평소에는 실제 거리를 쓰지만, 회피 유지(Clearing) 중에는 0으로 강제합니다.
            calc_dist_input = raw_dist

            if raw_dist < stage_trigger:
                # [장애물 감지 중]
                obstacle_last_seen_time = now
                calc_dist_input = raw_dist  # 실제 거리 사용

                # 장애물 카운팅 (쿨다운 1.5초)
                if not is_obstacle_detected and (now - obs_clear_finished_time > 1.5):
                    obstacle_count += 1
                    is_obstacle_detected = True
                    if obstacle_count == 1:
                        desired_lane = 1
                    elif obstacle_count == 2:
                        desired_lane = 2

                # 방향 설정
                if obstacle_count == 1:
                    avoid_direction = -1  # 왼쪽 회피
                    status_msg = f"AVOID LEFT #{obstacle_count}"
                    status_color = (0, 255, 255)
                elif obstacle_count == 2:
                    avoid_direction = 1  # 오른쪽 회피
                    status_msg = f"AVOID RIGHT #{obstacle_count}"
                    status_color = (255, 0, 255)
                else:
                    final_speed = 100  # 감속

            else:
                # [장애물 라이다에서 사라짐 -> 회피 유지 체크]
                if is_obstacle_detected:
                    if (now - obstacle_last_seen_time) < stage_clear_time:
                        # ★ 여기가 핵심입니다 ★
                        # 회피 시간을 채우기 전까지는 '거리가 0인 것처럼' 속여서 최대 조향을 유지합니다.
                        calc_dist_input = 0

                        if obstacle_count == 1:
                            avoid_direction = -1
                        elif obstacle_count == 2:
                            avoid_direction = 1

                        status_msg = f"HOLDING... {stage_clear_time - (now - obstacle_last_seen_time):.1f}s"
                        status_color = (0, 100, 255)
                    else:
                        # 유지 시간 종료 -> 복귀
                        is_obstacle_detected = False
                        obs_clear_finished_time = now
                        avoid_direction = 0
                        print("✅ 회피 종료 -> 차선 복귀")

            # ===== 타겟 및 Shift 계산 =====
            base_target = base_target_from_lines(width, left, right)
            shift_px = 0

            if avoid_direction != 0:
                # ★ 수정된 calc_dist_input 사용 (유지 구간에서는 0이 들어가서 Max Shift 발생)
                shift_px = calc_shift_px(calc_dist_input, stage_trigger)

                if avoid_direction == -1:
                    base_target -= shift_px
                else:
                    base_target += shift_px

            # 2차선 복귀 Bias (Lane 2 복귀 시에만)
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
            cv2.putText(mask_bgr, f"Obs#{obstacle_count} RawDist:{int(raw_dist)} CalcDist:{int(calc_dist_input)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(mask_bgr, f"{status_msg}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(mask_bgr, f"Shift:{int(shift_px)} Angle:{int(angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            cv2.imshow("Lane + Obstacle Avoid", mask_bgr)
            if cv2.waitKey(1) == ord('q'): break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 종료 처리 (기존과 동일)
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cap_lane.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()