# run_obstacle_avoid.py
import serial
import time
from rplidar import RPLidar
import cv2

from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, MAX_SPEED, width, height,  # ❌ width/height는 common에 없으므로 아래에서 정의
)

from common_lane_base import (
    configure_camera_auto, compute_lane, base_target_from_lines,
    angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO, CROSSWALK_RATIO_MIN, CROSSWALK_COOLDOWN,
    detect_stop_line
)

# ==== obstacle 전용 설정 ====
LIDAR_PORT = 'COM3'
OBSTACLE_DIST_STAGE1 = 1200
OBSTACLE_DIST_STAGE2 = 1000
SHIFT_GAIN = 2.0
OBSTACLE_CLEAR_TIME_STAGE1 = 1.5
OBSTACLE_CLEAR_TIME_STAGE2 = 2.2

# 2차선 유지 bias (필요시 그대로 유지)
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
    # 원래 코드의 (d ** 1.25) * (SHIFT_GAIN / (trigger ** 0.25))
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (SHIFT_GAIN / (trigger_dist ** 0.25))


def main():
    # ====== 연결 ======
    ser = None
    lidar = None

    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ {PORT} 포트 연결 성공! (1초 대기)")
        time.sleep(1)
    except Exception as e:
        print(f"❌ 시리얼 연결 실패: {e}")
        ser = None

    try:
        lidar = RPLidar(LIDAR_PORT)
    except Exception as e:
        print(f"❌ 라이다 초기화 실패: {e}")
        lidar = None

    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap_lane.set(3, width); cap_lane.set(4, height)
    if not cap_lane.isOpened():
        print("❌ 차선 카메라 오류")
        return
    configure_camera_auto(cap_lane)

    # ====== 상태 변수 ======
    obstacle_count = 0
    is_obstacle_detected = False
    obs_clear_finished_time = 0.0
    obstacle_last_seen_time = 0.0

    last_serial_time = 0.0
    last_speed_time = 0.0

    # both-line 기반 bias
    both_seen_streak = 0
    last_both_seen_time = 0.0
    desired_lane = 2

    try:
        if ser:
            ser.write(f"D,{MAX_SPEED}\n".encode())

        print("🚀 주행 시작 (Obstacle Avoid Mode)")
        scan_iter = lidar.iter_scans() if lidar is not None else [None] * (10**9)

        for scan in scan_iter:
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            now = time.time()

            # ===== 라이다 전방거리 =====
            raw_dist = 2000
            if scan is not None:
                for (_, ang, dist) in scan:
                    if 200 < dist < 2500 and (ang >= 330 or ang <= 30):
                        if dist < raw_dist:
                            raw_dist = dist

            # ===== 카메라 =====
            ret, frame_lane = cap_lane.read()
            if not ret:
                print("❌ 카메라 신호 끊김")
                break
            if frame_lane.shape[1] != width:
                frame_lane = cv2.resize(frame_lane, (width, height))

            lane = compute_lane(frame_lane)
            left, right = lane["left"], lane["right"]
            mask_bgr = lane["mask_bgr"]
            roi_points = lane["roi_points"]

            # both-line 업데이트
            if left is not None and right is not None:
                both_seen_streak = min(10, both_seen_streak + 1)
                last_both_seen_time = now
            else:
                both_seen_streak = max(0, both_seen_streak - 1)

            # ===== 장애물 회피 상태결정 =====
            stage_trigger, stage_clear_time = get_stage_trigger_and_clear(obstacle_count)

            avoid_direction = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            if raw_dist < stage_trigger:
                obstacle_last_seen_time = now

                if not is_obstacle_detected and (now - obs_clear_finished_time > 1.5):
                    obstacle_count += 1
                    is_obstacle_detected = True
                    print(f"⚠️ 장애물 #{obstacle_count} 감지!")
                    if obstacle_count == 1:
                        desired_lane = 1
                    elif obstacle_count == 2:
                        desired_lane = 2

                if obstacle_count == 1:
                    avoid_direction = -1
                    status_msg = "AVOID LEFT (#1)"
                    status_color = (0, 255, 255)
                elif obstacle_count == 2:
                    avoid_direction = 1
                    status_msg = "AVOID RIGHT (#2)"
                    status_color = (255, 0, 255)
                else:
                    status_msg = f"OBSTACLE #{obstacle_count}"
                    final_speed = min(final_speed, 120)

            else:
                if is_obstacle_detected:
                    if (now - obstacle_last_seen_time) < stage_clear_time:
                        if obstacle_count == 1:
                            avoid_direction = -1
                        elif obstacle_count == 2:
                            avoid_direction = 1
                        status_msg = f"CLEARING... ({now - obstacle_last_seen_time:.1f}s)"
                        status_color = (0, 100, 255)
                    else:
                        is_obstacle_detected = False
                        obs_clear_finished_time = now
                        avoid_direction = 0
                        print("✅ 복귀")

            is_avoiding_now = (avoid_direction != 0) or is_obstacle_detected

            # ===== 타겟 계산(기본 + shift + (원하면 bias)) =====
            base_target = base_target_from_lines(width, left, right)
            shift_px = 0

            if avoid_direction != 0:
                shift_px = calc_shift_px(raw_dist, stage_trigger)
                if avoid_direction == -1:
                    base_target -= shift_px
                else:
                    base_target += shift_px

            # 2차선 유지 bias (회피가 끝나고 desired_lane=2일 때만)
            bias_allowed = True
            if LANE_BIAS_ONLY_WHEN_BOTH:
                bias_allowed = (both_seen_streak >= 2) and ((now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC)

            if (desired_lane == 2) and (avoid_direction == 0) and bias_allowed:
                base_target += int(width * LANE2_BIAS_RATIO)

            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            # ===== 통신 =====
            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # ===== 디스플레이 =====
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(mask_bgr, (int(base_target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            cv2.putText(mask_bgr, f"obs#{obstacle_count} dist:{raw_dist} trig:{stage_trigger}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mask_bgr, f"{status_msg} | angle:{angle:.1f} shift:{shift_px:.0f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow("Lane + Obstacle Avoid", mask_bgr)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        print("\n🛑 안전 정지")
        if ser:
            try:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(b"S,570\n")
                    time.sleep(0.05)
                ser.close()
            except:
                pass
        if lidar is not None:
            try:
                lidar.stop()
                lidar.disconnect()
            except:
                pass
        cap_lane.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
