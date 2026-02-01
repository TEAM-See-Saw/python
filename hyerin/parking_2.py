import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque

# ==========================================
# [1] 설정값
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

SPEED_CRUISE = 70
SPEED_TURNIN = 60
SPEED_PARKIN = 55
SPEED_STOP = 0

# 서보 모터 설정
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555

STEER_WAIT_TIME = 0.6

# ---- 라이다 튐 완화 ----
LIDAR_TEMPORAL_N = 5
LIDAR_PCTL = 10  # 하위 10% 지점

# ---- “오른쪽에 주차된 차” 감지 임계값(튜닝 포인트) ----
RIGHT_CAR_DIST = 1600   # 오른쪽(주차칸 방향)에 물체가 이 거리 이하면 "차 있음"
RIGHT_GAP_DIST = 2600   # 오른쪽이 이 거리 이상이면 "갭(빈칸)로 간주"
GAP_STREAK = 3          # 몇 번 연속으로 갭이면 진짜 갭으로 인정
CAR_STREAK = 3          # 몇 번 연속으로 차면 진짜 차로 인정

# ---- 갭 중앙으로 들어가기 위한 시간 기반 보정(튜닝 포인트) ----
TIME_TO_CENTER_IN_GAP = 0.9   # 갭 진입 후 중앙까지 더 직진(좌로 이동) 시간
TIME_TURN_IN = 2.0            # 우회전으로 상단 주차칸 진입 시간
TIME_STRAIGHT_IN_MAX = 4.0    # 직진 주차 최대 시간(안전 타임아웃)

# ---- 주차 종료(전방 벽/끝) 감지 ----
PARK_FRONT_STOP_DIST = 500    # 전방(0도 섹터) 이내면 정지 (mm)

# 초음파(있으면 안전용으로만 사용) - 네 코드 그대로 유지
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
REAR_URGENT = 120

# ==========================================
# [2] 상태 정의
# ==========================================
STATE_CRUISE = 0          # 빨간선대로 좌로 직진 이동(탐색)
STATE_FIND_FIRST_CAR = 1  # 오른쪽 첫 번째 파란차 찾기
STATE_FIND_GAP = 2        # 첫 차 지나고 갭 찾기
STATE_CENTER_GAP = 3      # 갭 중앙까지 이동
STATE_TURN_IN = 4         # 우회전 진입
STATE_STRAIGHT_IN = 5     # 직진으로 주차 마무리
STATE_PARKED = 6
STATE_PAUSE = 99

# ==========================================
# [3] 센서/라이다 유틸
# ==========================================
def read_sensors():
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data[:] = [int(p) for p in parts]
        except:
            pass


def percentile_in_sector(scan, start_angle, end_angle, percentile=10):
    """start_angle~end_angle 섹터의 dist 모아서 하위 퍼센타일."""
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
    """
    wrap 지원 섹터 (예: 350~360 + 0~20 같은 전방)
    """
    if start_angle <= end_angle:
        return percentile_in_sector(scan, start_angle, end_angle, percentile)

    # wrap case
    d1 = []
    d2 = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if angle >= start_angle:
            d1.append(dist)
        elif angle <= end_angle:
            d2.append(dist)

    dists = d1 + d2
    if not dists:
        return 9999.0
    return float(np.percentile(np.array(dists, dtype=np.float32), percentile))


def temporal_median(buf: deque, v: float):
    buf.append(v)
    return float(np.median(np.array(buf, dtype=np.float32))) if buf else 9999.0


# ==========================================
# [4] 메인
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
    buf_right = deque(maxlen=LIDAR_TEMPORAL_N)  # 오른쪽(270 근처)
    buf_front = deque(maxlen=LIDAR_TEMPORAL_N)  # 전방(0 근처)

    # 상태/카운터
    state = STATE_CRUISE
    state_timer = time.time()

    car_cnt = 0
    gap_cnt = 0

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

                # 오른쪽(주차칸 방향) = 270 ± 20 정도를 섹터로 잡음
                right_raw = percentile_in_sector(scan, 250, 290, LIDAR_PCTL)
                right_dist = temporal_median(buf_right, right_raw)

                # 전방(0도 주변) = 350~360 + 0~20
                front_raw = percentile_wrap_sector(scan, 350, 20, LIDAR_PCTL)
                front_dist = temporal_median(buf_front, front_raw)

                # 초음파(긴급용)
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
                    msg = "CRUISE LEFT -> SEARCH RIGHT CARS"

                    # 바로 검색하면 시작구간 잡음이 있을 수 있어 1초는 그냥 감
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
                # [2] 갭(빈칸) 찾기: 차 -> 빈칸 패턴
                # =========================
                elif state == STATE_FIND_GAP:
                    cmd_servo = SERVO_CENTER
                    cmd_speed = SPEED_CRUISE
                    msg = f"FIND GAP | R={int(right_dist)}"

                    # 갭은 오른쪽이 확 넓어지는 구간
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
                        # 잠깐 멈추고 조향 안정
                        ser.write(b"D,0\n")
                        state = STATE_PAUSE
                        next_state = STATE_TURN_IN
                        pause_duration = 0.5
                        pause_start_time = curr_time
                        pause_msg = "Ready Turn-In (Right)"
                        print("➡️ 갭 중앙 도달. 우회전 진입 준비")

                # =========================
                # [4] 우회전으로 주차칸 진입 (상단 방향)
                # =========================
                elif state == STATE_TURN_IN:
                    cmd_servo = SERVO_RIGHT_MAX
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align RIGHT..."
                    else:
                        cmd_speed = SPEED_TURNIN
                        msg = f"TURN IN RIGHT... {dt-STEER_WAIT_TIME:.1f}/{TIME_TURN_IN}s"

                        if (dt - STEER_WAIT_TIME) >= TIME_TURN_IN:
                            # 바로 직진 주차로
                            state = STATE_STRAIGHT_IN
                            state_timer = curr_time
                            print("⬆️ 진입 완료 -> 직진 주차 시작")

                # =========================
                # [5] 직진으로 주차 마무리
                # =========================
                elif state == STATE_STRAIGHT_IN:
                    cmd_servo = SERVO_CENTER
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align CENTER..."
                    else:
                        cmd_speed = SPEED_PARKIN
                        msg = f"PARK IN STRAIGHT | F={int(front_dist)}"

                    # 전방 끝/벽 감지로 정지
                    if front_dist < PARK_FRONT_STOP_DIST and dt > STEER_WAIT_TIME:
                        print(f"✅ 전방 감지로 주차 완료! F={int(front_dist)}")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                    # 안전 타임아웃
                    if dt > TIME_STRAIGHT_IN_MAX:
                        print("✅ 주차 완료(타임아웃)")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                    # 초음파 긴급(혹시 후방 센서가 앞을 보게 달린 경우 대비)
                    if dist_LT < REAR_URGENT or dist_RT < REAR_URGENT:
                        print("🧯 초음파 URGENT STOP!")
                        ser.write(b"D,0\n")
                        state = STATE_PARKED
                        state_timer = curr_time

                elif state == STATE_PARKED:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "PARKED ✅ (q to quit)"
                    ser.write(b"D,0\n")

                # ---- 명령 송신 ----
                ser.write(f"S,{cmd_servo}\n".encode())
                ser.write(f"D,{cmd_speed}\n".encode())

                # ---- 디버그 UI ----
                debug_img = np.zeros((320, 820, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | {msg}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(debug_img, f"Right(250~290): {int(right_dist)}mm  | Front(350~20): {int(front_dist)}mm",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, f"US(LT/RT): {dist_LT}/{dist_RT}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, "Press 'q' to quit", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", debug_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
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

    # 종료
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
