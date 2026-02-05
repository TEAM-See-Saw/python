# run_obstacle_only_merged.py
# - Camera/ROI: 1번 코드(common_lane_base) 방식 그대로
# - Obstacle state/trigger/clear: 2번 코드 방식 그대로(단일 임계값 + clear time)
# - Traffic/crosswalk: 전부 제거
# - Lane tracing: 1번 코드 방식(base_target_from_lines + (desired_lane==2) bias + both-line gate)
# - Shift(회피 이동량): 2번 코드 방식(선형) 적용

import time
import serial
import cv2
from rplidar import RPLidar

from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, MAX_SPEED,
    configure_camera_auto, compute_lane, base_target_from_lines,
    angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO,
)

# ===== 장애물 회피 설정 (2번 코드와 동일) =====
LIDAR_PORT = "COM3"
OBSTACLE_START_DIST = 1000
SHIFT_GAIN = 1.2
OBSTACLE_CLEAR_TIME = 1.5

# ===== 2차선 복귀/유지 보완 (1번 코드 방식) =====
LANE2_BIAS_RATIO = 0.12            # 화면 폭 대비 오른쪽 bias 비율
LANE_BIAS_ONLY_WHEN_BOTH = True    # 양쪽 라인 보일 때만 bias 허용
BOTH_SEEN_HOLD_SEC = 0.6           # both-line 유지 시간
BOTH_SEEN_STREAK_MIN = 2           # both-line 최소 연속 프레임(가볍게)

# ===== 안전/기본 =====
WIDTH, HEIGHT = 640, 480


def calc_shift_px_linear(raw_dist: float) -> int:
    """
    2번 코드 shift 모델(선형):
      shift_amount = (OBSTACLE_START_DIST - min(dist, OBSTACLE_START_DIST)) * SHIFT_GAIN
    """
    calc_dist = min(float(raw_dist), float(OBSTACLE_START_DIST))
    shift = (float(OBSTACLE_START_DIST) - calc_dist) * float(SHIFT_GAIN)
    if shift < 0:
        shift = 0.0
    return int(shift)


def main():
    ser = None
    lidar = None
    cap_lane = None

    try:
        # ===== 시리얼 연결 =====
        try:
            ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
            print(f"✅ Serial connected: {PORT} (wait 1s)")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Serial connect failed: {e}")
            ser = None

        # ===== 라이다 연결 =====
        try:
            lidar = RPLidar(LIDAR_PORT)
            print(f"✅ LiDAR connected: {LIDAR_PORT}")
        except Exception as e:
            print(f"❌ LiDAR init failed: {e}")
            lidar = None

        # ===== 카메라(1번 코드 방식) =====
        cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        cap_lane.set(3, WIDTH)
        cap_lane.set(4, HEIGHT)
        if not cap_lane.isOpened():
            print("❌ Lane camera open failed")
            return
        configure_camera_auto(cap_lane)

        # ===== 상태 변수 =====
        last_serial_time = 0.0
        last_speed_time = 0.0

        # both-line gate
        both_seen_streak = 0
        last_both_seen_time = 0.0

        # obstacle states (2번 코드 방식)
        obstacle_count = 0
        is_obstacle_detected = False
        obs_clear_finished_time = 0.0
        obstacle_last_seen_time = 0.0

        # lane preference (1번 코드 보완)
        desired_lane = 2  # 시작은 2차선 기준(필요하면 1로 바꿔도 됨)

        # 초기 속도
        if ser:
            ser.write(f"D,{MAX_SPEED}\n".encode())

        print("🚀 Start: Obstacle Avoid ONLY (Merged)")

        scan_iter = lidar.iter_scans() if lidar is not None else [None] * (10**9)

        for scan in scan_iter:
            now = time.time()

            # 수신 버퍼 정리(필수는 아니지만 지연 줄이기)
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            # ===== 라이다 전방 최소거리 =====
            raw_dist = 2000
            if scan is not None:
                for (_, ang, dist) in scan:
                    # 2번 코드 전방 조건(대략 동일): 200<dist<1500, ang 전방(330~30)
                    if 200 < dist < 1500 and (ang >= 330 or ang <= 30):
                        if dist < raw_dist:
                            raw_dist = dist

            # ===== 카메라 =====
            ret, frame_lane = cap_lane.read()
            if not ret:
                print("❌ Camera frame failed")
                break
            if frame_lane.shape[1] != WIDTH:
                frame_lane = cv2.resize(frame_lane, (WIDTH, HEIGHT))

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

            # ===== 장애물 회피 상태결정 (2번 코드와 동일) =====
            avoid_direction = 0
            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            if raw_dist < OBSTACLE_START_DIST:
                obstacle_last_seen_time = now

                if not is_obstacle_detected:
                    if now - obs_clear_finished_time > 1.5:
                        obstacle_count += 1
                        is_obstacle_detected = True
                        print(f"⚠️ Obstacle #{obstacle_count} detected")
                        # 1번 코드 보완: 장애물 1개째면 1차선, 2개째면 2차선 복귀
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
                    if now - obstacle_last_seen_time < OBSTACLE_CLEAR_TIME:
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
                        print("✅ Back to lane (obstacle cleared)")

            # ===== 타겟 계산 (1번 코드 방식 + 2번 코드 shift) =====
            base_target = base_target_from_lines(WIDTH, left, right)
            shift_px = 0

            # 회피 중 shift 적용(2번 코드 방식: 선형)
            if avoid_direction != 0:
                shift_px = calc_shift_px_linear(raw_dist)
                if avoid_direction == -1:
                    base_target -= shift_px
                else:
                    base_target += shift_px

            # 2차선 유지 bias (회피 종료 & desired_lane=2일 때)
            bias_allowed = True
            if LANE_BIAS_ONLY_WHEN_BOTH:
                bias_allowed = (
                    both_seen_streak >= BOTH_SEEN_STREAK_MIN
                    and (now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC
                )

            if (desired_lane == 2) and (avoid_direction == 0) and bias_allowed:
                base_target += int(WIDTH * LANE2_BIAS_RATIO)

            # 조향 변환
            angle = angle_from_target(WIDTH, HEIGHT, base_target)
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
            cv2.circle(mask_bgr, (int(base_target), int(HEIGHT * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            cv2.putText(
                mask_bgr,
                f"obs#{obstacle_count} dist:{raw_dist} trig:{OBSTACLE_START_DIST} desired_lane:{desired_lane}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                mask_bgr,
                f"{status_msg} | angle:{angle:.1f} shift:{shift_px} both:{both_seen_streak}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

            cv2.imshow("Lane + Obstacle (Merged)", mask_bgr)
            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        print("\n🛑 SAFE STOP")
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

        if cap_lane is not None:
            try:
                cap_lane.release()
            except:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()