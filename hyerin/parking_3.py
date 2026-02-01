import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque

# ==========================================
# [1] 포트/속도/서보 설정
# ==========================================
PORT = "COM4"
LIDAR_PORT = "COM3"

SPEED_CRUISE = 70     # 빨간 경로 따라 좌로 직진(탐색)
SPEED_TURNIN = 60     # 우회전 진입
SPEED_PARKIN = 55     # 주차 직진
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680

STEER_WAIT_TIME = 0.6  # 조향 안정 대기

# ==========================================
# [2] 치수 반영 (mm)
# ==========================================
CAR_W = 750      # 차량 폭(짧은쪽) 75cm
CAR_L = 1000     # 차량 길이(긴쪽) 100cm
SLOT_W = 950     # 주차면 폭(짧은쪽) 95cm
SLOT_L = 1500    # 주차면 길이(긴쪽) 150cm

SIDE_CLEAR_TOTAL = SLOT_W - CAR_W        # 200mm
SIDE_CLEAR_EACH = SIDE_CLEAR_TOTAL / 2   # 100mm

FR_REAR_CLEAR_TOTAL = SLOT_L - CAR_L     # 500mm
TARGET_FRONT_CLEAR = 250                 # 앞쪽 250mm 남기고 정지 권장

# 전방(주차칸 안쪽 끝) 감지 기준: 25cm 남기고 정지
PARK_FRONT_STOP_DIST = 250  # (기존 500보다 타이트)

# ==========================================
# [3] 라이다 튐 완화/섹터 정의
# ==========================================
LIDAR_TEMPORAL_N = 5     # 최근 N개 median
LIDAR_PCTL = 10          # 섹터 거리 중 하위 10% 퍼센타일 (min보다 안정적)

# 오른쪽(주차칸 방향) 섹터: 270 ± 20
RIGHT_SECTOR = (250, 290)
# 왼쪽(반대쪽) 섹터: 90 ± 20
LEFT_SECTOR = (70, 110)
# 전방 섹터: 350~360 + 0~20
FRONT_SECTOR_WRAP = (350, 20)

# ==========================================
# [4] 주차칸(갭) 인식 임계값
# ==========================================
# "오른쪽에 차 있음" / "오른쪽이 확 넓음(갭)"
RIGHT_CAR_DIST = 1600     # 이하면 오른쪽에 차(장애물) 존재
RIGHT_GAP_DIST = 2600     # 이상이면 오른쪽이 비어있다(갭)
CAR_STREAK = 3            # 연속 N회면 확정
GAP_STREAK = 3

# 갭 중앙 보정/진입/주차 타임 파라미터
TIME_TO_CENTER_IN_GAP = 0.9  # 갭 감지 후 중앙까지 더 직진(좌로 이동)
TIME_TURN_IN = 2.0           # 우회전으로 주차칸 진입
TIME_STRAIGHT_IN_MAX = 4.0   # 직진 주차 최대 시간(안전 타임아웃)

# ==========================================
# [5] 폭이 타이트(여유 10cm/측면) → 센터링 제어
# ==========================================
CENTER_KP = 0.04          # mm 오차 -> 서보 보정량 스케일 (0.02~0.08 튜닝)
CENTER_SERVO_LIMIT = 28   # 중앙 기준 최대 ±28만 미세 조향(과조향 방지)
CENTER_DEADBAND = 30      # 오차가 ±30mm 이내면 보정하지 않음(진동 방지)

# ==========================================
# [6] 초음파(있으면 안전용)
# ==========================================
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
REAR_URGENT = 120  # 12cm 이내면 즉시 정지(안전)

# ==========================================
# [7] 상태 정의 (경로 기반)
# ==========================================
STATE_CRUISE = 0          # 빨간 경로대로 좌로 직진 이동(탐색 준비)
STATE_FIND_FIRST_CAR = 1  # 오른쪽 첫 번째 파란차 찾기
STATE_FIND_GAP = 2        # 첫 차 이후 갭(빈칸) 찾기
STATE_CENTER_GAP = 3      # 갭 중앙까지 이동
STATE_TURN_IN = 4         # 우회전 진입
STATE_STRAIGHT_IN = 5     # 직진 주차 마무리(센터링 포함)
STATE_PARKED = 6
STATE_PAUSE = 99

# ==========================================
# [8] 센서/라이다 유틸
# ==========================================
def read_sensors():
    """아두이노에서 'US:x,x,x,x,x,x' 형태로 초음파 6채널 수신."""
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data[:] = [int(p) for p in parts]
        except:
            pass


def percentile_in_sector(scan, start_angle, end_angle, percentile=10):
    """start_angle~end_angle 섹터의 dist들을 모아 하위 퍼센타일을 반환."""
    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if start_angle <= angle <= end_angle:
            dists.append(dist)
    if not dists:
        return 9999.0
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))


def percentile_wrap_sector(scan, start_angle, end_angle, percentile=10):
    """wrap 섹터 지원(예: 350~20)."""
    if start_angle <= end_angle:
        return percentile_in_sector(scan, start_angle, end_angle, percentile)

    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if angle >= start_angle or angle <= end_angle:
            dists.append(dist)

    if not dists:
        return 9999.0
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))


def temporal_median(buf: deque, v: float):
    buf.append(v)
    if not buf:
        return 9999.0
    return float(np.median(np.array(buf, dtype=np.float32)))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ==========================================
# [9] 메인
# ==========================================
def main():
    global ser, lidar
    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결")
        time.sleep(1.0)
        lidar.clean_input()
    except Exception as e:
        print(f"❌ 오류: {e}")
        return

    # 초기 정렬
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    ser.write(b"D,0\n")
    time.sleep(1.0)

    # 라이다 smoothing 버퍼
    buf_right = deque(maxlen=LIDAR_TEMPORAL_N)
    buf_left = deque(maxlen=LIDAR_TEMPORAL_N)
    buf_front = deque(maxlen=LIDAR_TEMPORAL_N)

    # 상태/카운터
    state = STATE_CRUISE
    state_timer = time.time()

    car_cnt = 0
    gap_cnt = 0

    # PAUSE 관리
    pause_start_time = 0.0
    pause_duration = 0.0
    next_state = STATE_CRUISE
    pause_msg = ""

    running = True

    while running:
        try:
            for scan in lidar.iter_scans():
                if not running:
                    break

                read_sensors()
                curr_time = time.time()

                # ---- 라이다: 오른쪽/왼쪽/전방 거리 ----
                right_raw = percentile_in_sector(scan, RIGHT_SECTOR[0], RIGHT_SECTOR[1], LIDAR_PCTL)
                left_raw = percentile_in_sector(scan, LEFT_SECTOR[0], LEFT_SECTOR[1], LIDAR_PCTL)
                front_raw = percentile_wrap_sector(scan, FRONT_SECTOR_WRAP[0], FRONT_SECTOR_WRAP[1], LIDAR_PCTL)

                right_dist = temporal_median(buf_right, right_raw)
                left_dist = temporal_median(buf_left, left_raw)
                front_dist = temporal_median(buf_front, front_raw)

                # 초음파(안전용)
                dist_LT = sonar_data[IDX_LT]
                dist_RT = sonar_data[IDX_RT]

                cmd_speed = 0
                cmd_servo = SERVO_CENTER
                msg = ""

                # =========================
                # PAUSE
                # =========================
                if state == STATE_PAUSE:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = f"WAIT... ({pause_msg})"
                    if curr_time - pause_start_time > pause_duration:
                        state = next_state
                        state_timer = curr_time
                        car_cnt = 0
                        gap_cnt = 0
                        lidar.clean_input()

                # =========================
                # [0] CRUISE: 빨간 경로대로 “좌로 직진”
                # =========================
                elif state == STATE_CRUISE:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_CRUISE
                    msg = "CRUISE -> SEARCH RIGHT CARS"

                    # 시작 잡음 제거용 1초
                    if curr_time - state_timer > 1.0:
                        state = STATE_FIND_FIRST_CAR
                        state_timer = curr_time
                        car_cnt = 0
                        gap_cnt = 0

                # =========================
                # [1] 오른쪽 첫 번째 파란차 찾기
                # =========================
                elif state == STATE_FIND_FIRST_CAR:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_CRUISE
                    msg = f"FIND FIRST CAR | R={int(right_dist)}"

                    if right_dist < RIGHT_CAR_DIST:
                        car_cnt += 1
                        if car_cnt >= CAR_STREAK:
                            print(f"🚗 오른쪽 1번차 감지 확정! R={int(right_dist)}")
                            state = STATE_FIND_GAP
                            state_timer = curr_time
                            gap_cnt = 0
                    else:
                        car_cnt = 0

                # =========================
                # [2] 갭(빈칸) 찾기
                # =========================
                elif state == STATE_FIND_GAP:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_CRUISE
                    msg = f"FIND GAP | R={int(right_dist)}"

                    if right_dist > RIGHT_GAP_DIST:
                        gap_cnt += 1
                        if gap_cnt >= GAP_STREAK:
                            print(f"🅿️ 갭 진입 감지! R={int(right_dist)} -> 중앙으로 이동")
                            state = STATE_CENTER_GAP
                            state_timer = curr_time
                    else:
                        gap_cnt = 0

                # =========================
                # [3] 갭 중앙까지 더 직진(좌로 이동)
                # =========================
                elif state == STATE_CENTER_GAP:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_CRUISE
                    dt = curr_time - state_timer
                    msg = f"CENTER GAP... {dt:.1f}/{TIME_TO_CENTER_IN_GAP}s"

                    if dt >= TIME_TO_CENTER_IN_GAP:
                        ser.write(b"D,0\n")
                        state = STATE_PAUSE
                        next_state = STATE_TURN_IN
                        pause_duration = 0.5
                        pause_start_time = curr_time
                        pause_msg = "Ready Turn-In (Right)"
                        print("➡️ 갭 중앙 도달. 우회전 진입 준비")

                # =========================
                # [4] 우회전으로 주차칸 진입
                # =========================
                elif state == STATE_TURN_IN:
                    dt = curr_time - state_timer
                    cmd_servo = SERVO_RIGHT_MAX

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align RIGHT..."
                    else:
                        cmd_speed = SPEED_TURNIN
                        msg = f"TURN IN RIGHT... {dt-STEER_WAIT_TIME:.1f}/{TIME_TURN_IN}s"

                        if (dt - STEER_WAIT_TIME) >= TIME_TURN_IN:
                            state = STATE_STRAIGHT_IN
                            state_timer = curr_time
                            print("⬆️ 진입 완료 -> 직진 주차 시작(센터링 활성)")

                # =========================
                # [5] 직진 주차 마무리 (센터링 포함)
                # =========================
                elif state == STATE_STRAIGHT_IN:
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = SERVO_CENTER
                        msg = "Align CENTER..."
                    else:
                        # 기본 전진
                        cmd_speed = SPEED_PARKIN

                        # (핵심) 좌/우 거리 균형 맞추기
                        error = left_dist - right_dist  # (+)이면 왼쪽이 멀다 = 오른쪽이 가깝다(상대적으로)
                        if abs(error) < CENTER_DEADBAND:
                            delta = 0
                        else:
                            delta = int(clamp(CENTER_KP * error, -CENTER_SERVO_LIMIT, CENTER_SERVO_LIMIT))

                        cmd_servo = SERVO_CENTER + delta
                        msg = f"PARK IN | L={int(left_dist)} R={int(right_dist)} err={int(error)} dS={delta}"

                    # 전방 끝/벽 감지로 정지 (앞 25cm 남기기 목표)
                    if front_dist < PARK_FRONT_STOP_DIST and dt > STEER_WAIT_TIME:
                        print(f"✅ 전방 감지로 주차 완료! F={int(front_dist)}mm (target~{PARK_FRONT_STOP_DIST})")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                    # 안전 타임아웃 (전방이 열려있어도 멈춤 보장)
                    if dt > TIME_STRAIGHT_IN_MAX:
                        print("✅ 주차 완료(타임아웃)")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                    # 초음파 긴급 정지(센서 배치가 다를 수 있어 “안전”으로만)
                    if dist_LT < REAR_URGENT or dist_RT < REAR_URGENT:
                        print("🧯 초음파 URGENT STOP!")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                elif state == STATE_PARKED:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "PARKED ✅ (press q to quit)"
                    ser.write(b"D,0\n")

                # ---- 명령 송신 ----
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                # ---- 디버그 UI ----
                debug_img = np.zeros((360, 920, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | {msg}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                cv2.putText(debug_img,
                            f"R(250~290): {int(right_dist)}  L(70~110): {int(left_dist)}  F(350~20): {int(front_dist)}",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(debug_img, f"US(LT/RT): {dist_LT}/{dist_RT}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(debug_img,
                            f"Car(WxL): {CAR_W}x{CAR_L}  Slot(WxL): {SLOT_W}x{SLOT_L}  SideClearEach: {int(SIDE_CLEAR_EACH)}mm",
                            (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

                cv2.putText(debug_img, "Press 'q' to quit", (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", debug_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                    break

        except RPLidarException as e:
            print(f"⚠️ 라이다 오류: {e}")
            try:
                lidar.clean_input()
            except:
                pass
            continue
        except KeyboardInterrupt:
            print("종료(KeyboardInterrupt)")
            break
        except Exception as e:
            print(f"시스템 오류: {e}")
            break

    # 종료 처리
    try:
        if ser:
            ser.write(b"D,0\n")
            ser.close()
    except:
        pass

    try:
        if lidar:
            lidar.stop()
            lidar.disconnect()
    except:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()