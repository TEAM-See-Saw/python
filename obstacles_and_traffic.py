import serial
import time
import cv2
import numpy as np
from rplidar import RPLidar

# ==============================================================================
# [Import] 공통 모듈
# traffic.py와 obstacle_plusV.py가 사용하는 common_lane_base.py가 같은 폴더에 있어야 합니다.
# ==============================================================================
from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, CAM_INDEX_TRAFFIC, MAX_SPEED,
    configure_camera_auto, compute_lane,
    base_target_from_lines, angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO,
    # Traffic 관련 상수
    CROSSWALK_RATIO_MIN, CROSSWALK_MAX_WAIT, CROSSWALK_COOLDOWN, CROSSWALK_CONFIRM_TIME,
    detect_stop_line,
    TRAFFIC_BOX_X1, TRAFFIC_BOX_X2, TRAFFIC_BOX_Y1, TRAFFIC_BOX_Y2,
    TRAFFIC_HIGHLIGHT_PCTL, TRAFFIC_TH_MIN
)

# ==============================================================================
# [설정] 통합 파라미터
# ==============================================================================

# 1. 스케줄링 설정 (CPU 부하 감소용)
TRAFFIC_CHECK_INTERVAL = 0.3  # 0.3초마다 신호등 검사 (약 3fps)

# 2. 장애물(Obstacle) 설정
LIDAR_PORT = 'COM3'  # ★ 실행 전 포트 확인 필수
OBSTACLE_DIST_STAGE1 = 1300  # 1번 장애물 감지 거리 (mm)
OBSTACLE_DIST_STAGE2 = 900  # 2번 장애물 감지 거리 (mm)
SHIFT_GAIN = 1.5  # 회피 조향 민감도
INTER_OBSTACLE_COOLDOWN = 2.5  # 장애물 사이 쿨타임 (초)
LANE2_BIAS_RATIO = 0.12  # 2차선 복귀용 편향 비율
BOTH_SEEN_HOLD_SEC = 0.6  # 양쪽 차선이 보일 때 편향 유지 시간

# 3. 정지선 오인 방지
START_GRACE_PERIOD = 3.0  # 출발 후 3초간은 정지선 무시 (출발선 오인 방지)


# ==============================================================================
# [함수 1] 신호등 인식 (Traffic Logic)
# traffic.py의 detect_traffic_lr_robust 함수 원본
# ==============================================================================
def detect_traffic_lr_robust(frame_bgr):
    H, W = frame_bgr.shape[:2]

    # ROI 설정
    x1 = int(W * TRAFFIC_BOX_X1)
    x2 = int(W * TRAFFIC_BOX_X2)
    y1 = int(H * TRAFFIC_BOX_Y1)
    y2 = int(H * TRAFFIC_BOX_Y2)

    # 예외 처리
    x1 = max(0, min(W - 2, x1))
    x2 = max(x1 + 1, min(W - 1, x2))
    y1 = max(0, min(H - 2, y1))
    y2 = max(y1 + 1, min(H - 1, y2))

    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 5 or rw < 5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "ROI_TOO_SMALL"}

    third = max(1, rw // 3)

    # 1. HSV 색상 기반 검출
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S_GATE = 60
    V_GATE = 80
    COLOR_TH = 0.003

    sat_mask = cv2.inRange(hsv, (0, S_GATE, 0), (180, 255, 255))
    red1 = cv2.inRange(hsv, (0, S_GATE, V_GATE), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, S_GATE, V_GATE), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, (35, S_GATE, V_GATE), (85, 255, 255))

    red_mask = cv2.bitwise_and(red_mask, sat_mask)
    green_mask = cv2.bitwise_and(green_mask, sat_mask)

    k = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)

    red_left = red_mask[:, 0:third]
    green_right = green_mask[:, 2 * third:rw] if (2 * third) < rw else green_mask[:, third:rw]

    red_left_ratio = cv2.countNonZero(red_left) / float(red_left.size)
    green_right_ratio = cv2.countNonZero(green_right) / float(green_right.size)

    if red_left_ratio > COLOR_TH and green_right_ratio < COLOR_TH:
        return "LEFT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR", "val": red_left_ratio}

    if green_right_ratio > COLOR_TH and red_left_ratio < COLOR_TH:
        return "RIGHT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR", "val": green_right_ratio}

    # 2. Fallback: 밝기(Blob) 기반 검출
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    p = np.percentile(gray, TRAFFIC_HIGHLIGHT_PCTL)
    thr = int(max(TRAFFIC_TH_MIN, p))
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_NONE"}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 30:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_SMALL"}

    roi_area = float(rh * rw)
    if area > roi_area * 0.25:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_TOO_BIG"}

    x, y, ww, hh = cv2.boundingRect(c)
    if ww > hh * 2.5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_WIDE"}

    M = cv2.moments(c)
    if M["m00"] == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_BADMOM"}

    cx = int(M["m10"] / M["m00"])
    if cx < third:
        state = "LEFT"
    elif cx > 2 * third:
        state = "RIGHT"
    else:
        state = "NONE"

    return state, {"box": (x1, y1, x2, y2), "mode": "BRIGHT_BLOB", "cx": cx}


# ==============================================================================
# [함수 2] 장애물 계산 (Obstacle Logic)
# obstacle_plusV.py의 함수들
# ==============================================================================
def get_stage_trigger_and_clear(obstacle_count_now):
    # 0개 처리 전(첫번째 장애물) -> 거리 1300, 클리어타임 2.5초
    if obstacle_count_now == 0:
        return OBSTACLE_DIST_STAGE1, 2.5
    # 1개 처리 후(두번째 장애물) -> 거리 900, 클리어타임 2.0초
    return OBSTACLE_DIST_STAGE2, 2.0


def calc_shift_px(raw_dist, trigger_dist):
    # 거리가 가까울수록 shift 값이 기하급수적으로 커짐
    calc_dist = min(raw_dist, trigger_dist)
    d = max(0.0, float(trigger_dist - calc_dist))
    return (d ** 1.25) * (SHIFT_GAIN / (trigger_dist ** 0.25))

# [추가 함수] 라이더 스캔을 안전하게 가져오는 제너레이터
def robust_scan_iter(lidar):
    while True:
        try:
            # 기존 iter_scans 실행
            for scan in lidar.iter_scans():
                yield scan
        except Exception as e:
            print(f"⚠️ LiDAR 버퍼 오류 발생 ({e}). 재접속 중...")
            try:
                lidar.stop()
                lidar.clean_input() # 버퍼 비우기 (지원하는 버전의 경우)
                time.sleep(0.1)
                lidar.start_motor()
            except:
                pass
            time.sleep(0.5) # 잠시 대기 후 재시도


# ==============================================================================
# [메인] 실행 루프
# ==============================================================================
def main():
    ser = None
    lidar = None

    # ---------------------------------------------------------
    # 1. 하드웨어 연결
    # ---------------------------------------------------------
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ 시리얼 연결 성공: {PORT}")
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ 시리얼 연결 실패: {e}")

    try:
        lidar = RPLidar(LIDAR_PORT)
        print(f"✅ LiDAR 연결 성공: {LIDAR_PORT}")
    except Exception as e:
        print(f"⚠️ LiDAR 연결 실패: {e}")

    # ---------------------------------------------------------
    # 2. 카메라 초기화 (2대)
    # ---------------------------------------------------------
    # (1) 차선 인식용
    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap_lane.set(3, width)
    cap_lane.set(4, height)
    configure_camera_auto(cap_lane)

    # (2) 신호등 인식용
    cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)
    cap_traffic.set(3, width)
    cap_traffic.set(4, height)
    configure_camera_auto(cap_traffic)

    if not cap_lane.isOpened():
        print("❌ 차선 카메라 열기 실패")
        return

    # ---------------------------------------------------------
    # 3. 변수 초기화
    # ---------------------------------------------------------
    start_time_program = time.time()

    # [스케줄링]
    last_traffic_check_time = 0.0
    traffic_state_cached = "NONE"  # 0.3초간 유지될 신호등 상태

    # [장애물]
    obstacle_count = 0
    is_obstacle_detected = False
    obs_clear_finished_time = 0.0
    obstacle_last_seen_time = 0.0
    desired_lane = 2

    # [차선 복귀 Bias]
    both_seen_streak = 0
    last_both_seen_time = 0.0

    # [신호등 및 정지]
    is_crosswalk_stop = False
    crosswalk_start_time = 0.0
    crosswalk_cooldown_timer = 0.0
    crosswalk_detect_timer = 0.0

    # [통신 제어]
    last_serial_time = 0.0
    last_speed_time = 0.0

    print("🚀 통합 주행 시작 (Obstacle + Traffic + Loose Schedule)")
    print(f"ℹ️ 신호등 검사 주기: {TRAFFIC_CHECK_INTERVAL}초")

    # 초기 속도 전송
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())

    if lidar:
        scan_iter = robust_scan_iter(lidar)
    else:
        scan_iter = [None] * 10000  # 라이더 없으면 더미 데이터

    try:
        for scan in scan_iter:
            now = time.time()

            # 시리얼 버퍼 비우기
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            # -----------------------------------------------------
            # [Step 1] 센서 데이터 읽기 (Camera & LiDAR)
            # -----------------------------------------------------

            # (1) 차선 카메라 (매 루프 필수)
            ret_l, frame_lane = cap_lane.read()
            if not ret_l:
                print("❌ 차선 카메라 신호 없음")
                break
            if frame_lane.shape[1] != width:
                frame_lane = cv2.resize(frame_lane, (width, height))

            # (2) 신호등 카메라 (버퍼 비우기용 매 루프 read)
            ret_t, frame_traffic = cap_traffic.read()

            # -----------------------------------------------------
            # [Step 2] 기본 차선 인식 (Common Lane Base)
            # -----------------------------------------------------
            lane = compute_lane(frame_lane)
            left, right = lane["left"], lane["right"]
            mask_bgr = lane["mask_bgr"]
            roi_points = lane["roi_points"]
            lane_ratio = lane["ratio"]

            # 차선 인식 상태 업데이트 (복귀 Bias용)
            if left is not None and right is not None:
                both_seen_streak = min(10, both_seen_streak + 1)
                last_both_seen_time = now
            else:
                both_seen_streak = max(0, both_seen_streak - 1)

            # -----------------------------------------------------
            # [Step 3] 신호등 인식 (느슨한 스케줄링)
            # -----------------------------------------------------
            # 일정 시간(0.3초)이 지났고, 이미지가 유효할 때만 무거운 연산 수행
            if ret_t and (now - last_traffic_check_time > TRAFFIC_CHECK_INTERVAL):
                if frame_traffic.shape[1] != width:
                    frame_traffic = cv2.resize(frame_traffic, (width, height))

                # 신호등 인식 수행
                traffic_state_cached, tdbg = detect_traffic_lr_robust(frame_traffic)
                last_traffic_check_time = now

                # [Debug] 인식된 신호등 화면에 표시
                box = tdbg.get("box", None)
                if box:
                    bx1, by1, bx2, by2 = box
                    cv2.rectangle(frame_traffic, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
                cv2.putText(frame_traffic, f"T: {traffic_state_cached}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Traffic Cam (Low FPS)", frame_traffic)

            # -----------------------------------------------------
            # [Step 4] 통합 로직: 장애물 회피 + 속도 제어
            # -----------------------------------------------------

            # --- A. 장애물 회피 (Steering Control) ---
            raw_dist = 2000
            if scan:
                for (_, ang, dist) in scan:
                    # 유효 거리(20cm~2.5m) & 시야각(정면 30도 이내: 345~15도)
                    if 200 < dist < 2500:
                        if ang >= 345 or ang <= 15:
                            if dist < raw_dist:
                                raw_dist = dist

            stage_trigger, stage_clear_time = get_stage_trigger_and_clear(obstacle_count)
            shift_px = 0
            obs_msg = "OBS: None"
            avoid_direction = 0  # 0:직진, -1:왼쪽, 1:오른쪽

            # 1) 장애물 감지됨
            if raw_dist < stage_trigger:
                obstacle_last_seen_time = now

                # 쿨타임 체크 (이전 장애물 피하고 일정 시간 지났는지)
                if not is_obstacle_detected and (now - obs_clear_finished_time > INTER_OBSTACLE_COOLDOWN):
                    obstacle_count += 1
                    is_obstacle_detected = True
                    print(f"🚨 장애물 #{obstacle_count} 발견 (거리: {int(raw_dist)})")

                    if obstacle_count == 1:
                        desired_lane = 1
                    elif obstacle_count == 2:
                        desired_lane = 2

                # 회피 방향 설정
                if is_obstacle_detected:
                    avoid_direction = -1 if obstacle_count == 1 else 1
                    obs_msg = f"AVOID #{obstacle_count} ({int(raw_dist)})"

            # 2) 장애물 감지 안됨 (사라짐 or 쿨타임)
            else:
                if is_obstacle_detected:
                    # 유지 시간(Holding) 체크
                    if (now - obstacle_last_seen_time) < stage_clear_time:
                        avoid_direction = -1 if obstacle_count == 1 else 1
                        raw_dist = 0  # 강제로 최대 Shift 적용
                        obs_msg = "HOLDING..."
                    else:
                        # 회피 완료
                        is_obstacle_detected = False
                        obs_clear_finished_time = now
                        avoid_direction = 0
                        print(f"✅ 장애물 #{obstacle_count} 회피 완료")

            # Shift 계산
            if avoid_direction != 0:
                shift_val = calc_shift_px(raw_dist, stage_trigger)
                if avoid_direction == -1:  # 왼쪽
                    shift_px = -shift_val
                else:  # 오른쪽
                    shift_px = shift_val

            # --- B. 속도 및 정지 제어 (Speed Control) ---
            # 신호등/정지선이 장애물보다 우선순위가 높음 (멈춰야 하면 멈춤)

            final_speed = MAX_SPEED
            status_msg = f"{obs_msg} | T: {traffic_state_cached}"
            status_color = (0, 255, 0)  # Green

            # (1) 이미 정지 중인 경우
            if is_crosswalk_stop:
                final_speed = 0
                elapsed = now - crosswalk_start_time
                status_color = (0, 0, 255)  # Red
                status_msg = f"STOPPED ({elapsed:.1f}s)"

                # 출발 조건: 신호가 초록색(RIGHT)이거나, 타임아웃
                if traffic_state_cached == "RIGHT":
                    is_crosswalk_stop = False
                    crosswalk_cooldown_timer = now
                    print("🟢 신호 변경(RIGHT) -> 출발")
                elif elapsed > CROSSWALK_MAX_WAIT:
                    is_crosswalk_stop = False
                    crosswalk_cooldown_timer = now
                    print("⏰ 대기 시간 초과 -> 출발")

            # (2) 주행 중 정지 조건 체크
            else:
                # 시작 후 일정 시간(Grace Period)이 지나야 정지선 검사
                if (now - start_time_program > START_GRACE_PERIOD):
                    # 정지선이 보이고 & 쿨타임이 지났을 때
                    if (lane_ratio > CROSSWALK_RATIO_MIN) and (now - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                        if detect_stop_line(lane["mask"], mask_bgr, ROI_HEIGHT_RATIO):
                            # 저장된 신호등 상태 확인
                            if traffic_state_cached == "LEFT":  # 빨간불
                                if crosswalk_detect_timer == 0:
                                    crosswalk_detect_timer = now
                                elif now - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                                    is_crosswalk_stop = True
                                    crosswalk_start_time = now
                                    final_speed = 0
                                    crosswalk_detect_timer = 0
                                    print("🛑 정지선 + 적색신호 발견 -> 정지")
                                else:
                                    status_msg = "Checking Stop..."
                            else:
                                crosswalk_detect_timer = 0
                        else:
                            crosswalk_detect_timer = 0
                    else:
                        crosswalk_detect_timer = 0

            # -----------------------------------------------------
            # [Step 5] 최종 모터 제어
            # -----------------------------------------------------

            # 기본 타겟
            base_target = base_target_from_lines(width, left, right)

            # 장애물 Shift 적용
            base_target += shift_px

            # 2차선 복귀 Bias (장애물 없고, 원하는 차선이 2차선일 때)
            bias_allowed = (not is_obstacle_detected) and \
                           (desired_lane == 2) and \
                           (both_seen_streak >= 2) and \
                           ((now - last_both_seen_time) <= BOTH_SEEN_HOLD_SEC)

            if bias_allowed:
                base_target += int(width * LANE2_BIAS_RATIO)

            # 조향각 계산
            angle = angle_from_target(width, height, base_target)
            servo_val = servo_from_angle(angle)

            # 시리얼 전송
            if ser:
                # 조향 (Servo)
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now

                # 속도 (DC Motor)
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # -----------------------------------------------------
            # [Step 6] 디스플레이
            # -----------------------------------------------------
            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            # 타겟 포인트 (빨간 점)
            cv2.circle(mask_bgr, (int(base_target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)
            # 상태 메시지
            cv2.putText(mask_bgr, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            cv2.imshow("Integrated Drive", mask_bgr)

            if cv2.waitKey(1) == ord('q'):
                break

    except KeyboardInterrupt:
        print("사용자 종료")
    except Exception as e:
        print(f"Error Loop: {e}")

    finally:
        print("\n🛑 시스템 종료 및 안전 정지")
        if ser:
            try:
                ser.write(b"D,0\n")
                ser.write(b"S,570\n")  # 중앙 정렬 가정
                time.sleep(0.1)
                ser.close()
            except:
                pass

        if lidar:
            try:
                lidar.stop()
                lidar.disconnect()
            except:
                pass

        cap_lane.release()
        cap_traffic.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()