import serial
import time
import cv2
from rplidar import RPLidar

# 공통 모듈 임포트 (기존과 동일)
from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, CAM_INDEX_TRAFFIC, MAX_SPEED,
    configure_camera_auto, compute_lane,
    base_target_from_lines, angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO,
    CROSSWALK_RATIO_MIN, CROSSWALK_MAX_WAIT, CROSSWALK_COOLDOWN, CROSSWALK_CONFIRM_TIME,
    detect_stop_line,
)

# [설정] 스케줄링 간격
TRAFFIC_CHECK_INTERVAL = 0.3  # 0.3초마다 신호등 검사 (약 3~4 FPS)

# [설정] 장애물 관련
LIDAR_PORT = 'COM3'
OBSTACLE_DIST_STAGE1 = 1300
OBSTACLE_DIST_STAGE2 = 900
SHIFT_GAIN = 1.5
INTER_OBSTACLE_COOLDOWN = 2.5
LANE2_BIAS_RATIO = 0.12
BOTH_SEEN_HOLD_SEC = 0.6


# --- 보조 함수들 (기존과 동일, 생략 가능하지만 실행 위해 포함) ---
def get_stage_trigger_and_clear(obstacle_count_now):
    if obstacle_count_now == 0: return OBSTACLE_DIST_STAGE1, 2.5
    return OBSTACLE_DIST_STAGE2, 2.0


def calc_shift_px(raw_dist, trigger_dist):
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (SHIFT_GAIN / (trigger_dist ** 0.25))


def detect_traffic_lr_robust(frame_bgr):
    # (traffic.py의 신호등 인식 함수 내용 그대로 사용한다고 가정)
    # ... 코드 생략 (위의 코드와 동일) ...
    # 실제 실행 시엔 traffic.py의 내용을 여기에 복사하거나 import 해야 합니다.
    return "NONE", {}
    # ※ 주의: 여기서는 공간상 생략했습니다. 위쪽 답변의 detect_traffic_lr_robust를 꼭 넣으세요!


def main():
    ser = None
    lidar = None

    # 1. 연결
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1); time.sleep(1)
    except:
        print("⚠️ 시리얼 없음")
    try:
        lidar = RPLidar(LIDAR_PORT); print("✅ LiDAR 연결")
    except:
        print("⚠️ LiDAR 없음")

    # 2. 카메라 설정
    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)

    width, height = 640, 480
    cap_lane.set(3, width);
    cap_lane.set(4, height)
    cap_traffic.set(3, width);
    cap_traffic.set(4, height)
    configure_camera_auto(cap_lane)
    configure_camera_auto(cap_traffic)

    # 3. 변수 초기화
    # [스케줄링]
    last_traffic_check_time = 0.0
    traffic_state_cached = "NONE"  # 가장 최근에 본 신호등 상태 기억

    # [장애물]
    obstacle_count = 0
    is_obstacle_detected = False
    obs_clear_finished_time = 0.0
    obstacle_last_seen_time = 0.0
    both_seen_streak = 0
    last_both_seen_time = 0.0
    desired_lane = 2

    # [신호등/정지]
    is_crosswalk_stop = False
    crosswalk_start_time = 0.0
    crosswalk_cooldown_timer = 0.0
    crosswalk_detect_timer = 0.0

    # [통신]
    last_serial_time = 0.0
    last_speed_time = 0.0

    print(f"🚀 통합 주행 시작 (스케줄링: {TRAFFIC_CHECK_INTERVAL}s)")

    if ser: ser.write(f"D,{MAX_SPEED}\n".encode())
    scan_iter = lidar.iter_scans() if lidar else [None] * 1000

    try:
        for scan in scan_iter:
            now = time.time()
            if ser and ser.in_waiting: ser.read(ser.in_waiting)

            # ==========================================
            # 1. 필수 센서 처리 (매 루프)
            # ==========================================
            ret_l, frame_lane = cap_lane.read()
            if not ret_l: break
            if frame_lane.shape[1] != width: frame_lane = cv2.resize(frame_lane, (width, height))

            lane = compute_lane(frame_lane)  # 차선 인식은 항상!
            mask_bgr = lane["mask_bgr"]

            # 카메라 버퍼 비우기 (중요: 처리는 안 해도 읽어는 둬야 렉 안 걸림)
            ret_t, frame_traffic = cap_traffic.read()

            # ==========================================
            # 2. 신호등 처리 (스케줄링: 0.3초마다)
            # ==========================================
            if ret_t and (now - last_traffic_check_time > TRAFFIC_CHECK_INTERVAL):
                if frame_traffic.shape[1] != width: frame_traffic = cv2.resize(frame_traffic, (width, height))

                # 무거운 인식 함수 실행
                traffic_state_cached, _ = detect_traffic_lr_robust(frame_traffic)
                last_traffic_check_time = now

                # 디버깅용 표시 (갱신될 때만)
                cv2.putText(frame_traffic, f"Traffic: {traffic_state_cached}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
                cv2.imshow("Traffic Cam (Low FPS)", frame_traffic)

            # ==========================================
            # 3. 로직 통합 (장애물 + 신호등)
            # ==========================================

            # A. 장애물 회피 계산 (LiDAR)
            # ----------------------------------
            raw_dist = 2000
            if scan:
                for (_, ang, dist) in scan:
                    if 200 < dist < 2500 and (ang >= 345 or ang <= 15):
                        raw_dist = min(raw_dist, dist)

            stage_trigger, stage_clear_time = get_stage_trigger_and_clear(obstacle_count)
            shift_px = 0
            obs_msg = "No Obs"

            # ... (장애물 감지/카운트 로직은 이전과 동일) ...
            # 요약: 거리가 가까우면 is_obstacle_detected = True, shift_px 계산
            if raw_dist < stage_trigger:
                obstacle_last_seen_time = now
                if not is_obstacle_detected and (now - obs_clear_finished_time > INTER_OBSTACLE_COOLDOWN):
                    obstacle_count += 1
                    is_obstacle_detected = True
                    desired_lane = 1 if obstacle_count == 1 else 2

                if is_obstacle_detected:
                    direction = -1 if obstacle_count == 1 else 1
                    shift_px = calc_shift_px(raw_dist, stage_trigger)
                    if direction == -1: shift_px = -shift_px  # 왼쪽 (-)
                    obs_msg = f"AVOID #{obstacle_count}"
            else:
                if is_obstacle_detected:
                    if (now - obstacle_last_seen_time) < stage_clear_time:
                        # 홀딩 중
                        direction = -1 if obstacle_count == 1 else 1
                        shift_px = calc_shift_px(0, stage_trigger)  # Max shift
                        if direction == -1: shift_px = -shift_px
                        obs_msg = "HOLDING"
                    else:
                        is_obstacle_detected = False
                        obs_clear_finished_time = now

            # B. 주행 속도 및 정지 결정 (신호등 우선!)
            # ----------------------------------
            final_speed = MAX_SPEED
            status_msg = f"{obs_msg} | Traffic:{traffic_state_cached}"
            status_color = (0, 255, 0)

            # 정지 상태 처리
            if is_crosswalk_stop:
                final_speed = 0
                elapsed = now - crosswalk_start_time
                status_color = (0, 0, 255)
                status_msg = f"STOPPED ({elapsed:.1f}s)"

                # 출발 조건: 초록불(RIGHT) or 타임아웃
                if traffic_state_cached == "RIGHT" or elapsed > CROSSWALK_MAX_WAIT:
                    is_crosswalk_stop = False
                    crosswalk_cooldown_timer = now

            # 주행 중 정지 조건 체크 (정지선 + 빨간불)
            else:
                lane_ratio = lane["ratio"]
                # 쿨타임 지남 & 정지선 비율 충족
                if (lane_ratio > CROSSWALK_RATIO_MIN) and (now - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                    # detect_stop_line은 가벼우니 매번 해도 됨
                    if detect_stop_line(lane["mask"], mask_bgr, ROI_HEIGHT_RATIO):
                        if traffic_state_cached == "LEFT":  # 빨간불 (Cached된 최신 상태)
                            if crosswalk_detect_timer == 0:
                                crosswalk_detect_timer = now
                            elif now - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                                is_crosswalk_stop = True
                                crosswalk_start_time = now
                                final_speed = 0
                                crosswalk_detect_timer = 0
                        else:
                            crosswalk_detect_timer = 0
                    else:
                        crosswalk_detect_timer = 0
                else:
                    crosswalk_detect_timer = 0

            # C. 최종 조향 및 모터 제어
            # ----------------------------------
            base_target = base_target_from_lines(width, lane["left"], lane["right"])

            # 장애물 Shift 적용
            base_target += shift_px

            # 2차선 복귀 Bias (장애물 없을 때만)
            if not is_obstacle_detected and (desired_lane == 2):
                # ... Bias 로직 (생략 가능하거나 간단히 적용) ...
                base_target += int(width * LANE2_BIAS_RATIO * 0.5)  # 약하게 적용

            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # 화면 출력
            cv2.circle(mask_bgr, (int(base_target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            cv2.putText(mask_bgr, status_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.imshow("Loose Schedule Drive", mask_bgr)

            if cv2.waitKey(1) == ord('q'): break

    finally:
        if ser: ser.write(b"D,0\n"); ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cap_lane.release();
        cap_traffic.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()