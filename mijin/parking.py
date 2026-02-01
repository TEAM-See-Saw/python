import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================================
# [0] PORT / BAUD
# ==========================================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'
BAUD = 115200

# ==========================================================
# [1] SPEED / SERVO
# ==========================================================
SPEED_SEARCH = 80
SPEED_SETUP  = 80
SPEED_PARK   = 75
SPEED_EXIT   = 80

SERVO_CENTER    = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX  = 680

# ==========================================================
# [2] CAR DIM / STEERING MODEL (각도 계산용)
# ==========================================================
# 사용자가 준 차 길이/폭(단위: mm)
CAR_LEN_MM = 1100
CAR_W_MM   = 600

# 사용자가 측정한 L(휠베이스 또는 축간거리로 가정): 55~56cm -> 550~560mm
WHEELBASE_MM = 555

# ★ 서보 끝값이 만들어내는 "최대 조향각(바퀴 각도)"(deg) : 반드시 현장 캘리브레이션 필요
#   보수적으로 25~35deg 구간에서 먼저 테스트 추천
STEER_MAX_DEG = 30.0

def servo_from_delta_deg(delta_deg: float) -> int:
    """
    조향각 delta(deg) -> 서보값 변환.
    delta>0: 좌회전, delta<0: 우회전
    """
    delta_deg = max(-STEER_MAX_DEG, min(STEER_MAX_DEG, delta_deg))
    # 선형 매핑(캘리브레이션 전제)
    if delta_deg >= 0:
        # center -> left_max
        ratio = delta_deg / STEER_MAX_DEG
        s = SERVO_CENTER + ratio * (SERVO_LEFT_MAX - SERVO_CENTER)
    else:
        ratio = (-delta_deg) / STEER_MAX_DEG
        s = SERVO_CENTER - ratio * (SERVO_CENTER - SERVO_RIGHT_MAX)
    return int(max(SERVO_RIGHT_MAX, min(SERVO_LEFT_MAX, s)))

def delta_deg_from_radius_mm(R_mm: float) -> float:
    """
    자전거 모델: R = L / tan(delta)
    목표 회전 반경 R(mm) -> delta(deg)
    """
    if R_mm <= 1:
        return STEER_MAX_DEG
    delta_rad = np.arctan(WHEELBASE_MM / R_mm)
    return float(np.degrees(delta_rad))

# ==========================================================
# [3] SEARCH(차1-빈공간-차2) 임계값
# ==========================================================
CAR_EXIST_DIST   = 800
EMPTY_SPACE_DIST = 1200
GAP_STABLE_TH    = 3
CAR2_CONFIRM_TH  = 3
PASS_GAP_CONFIRM = 5

# ==========================================================
# [4] SETUP(좌대각 전진) 종료 조건
# ==========================================================
STEER_WAIT_TIME  = 0.8
RS_SETUP_TARGET  = 1200
RS_SETUP_CONFIRM = 3

# ==========================================================
# [5] LiDAR sector helper
# ==========================================================
def get_lidar_dist_in_sector(scan, ang_min, ang_max, invalid=9999, percentile=20):
    dists = []
    for m in scan:
        if len(m) != 3:
            continue
        _, angle, dist = m
        if dist <= 0:
            continue

        if ang_min <= ang_max:
            in_range = (ang_min <= angle <= ang_max)
        else:
            in_range = (angle >= ang_min or angle <= ang_max)

        if in_range:
            dists.append(dist)

    if not dists:
        return invalid
    return float(np.percentile(dists, percentile))

# ==========================================================
# [6] PARKING CONTROL PARAM (경량)
# ==========================================================
RS_TARGET   = 550        # 우측 목표 간격
KP_RS       = 0.25       # rs P-제어 민감도

RR_MIN_SAFE = 350        # 우후방 너무 가까우면 조향 완화
RR_RELEASE  = 25         # servo tick 완화(좌로 조금 풀기)

# ARC 구간에서 “최소 각도 성향”을 강제할 때 쓸 목표 회전반경(mm)
# - arc1: 후진 우회전(꼬리 넣기) -> 작은 반경
# - arc2: 후진 좌로 펴기(평행화) -> 큰 반경(거의 직선에 가깝게)
ARC1_RADIUS_MM = 900     # 보수적으로 시작(너무 작으면 확 꺾임)
ARC2_RADIUS_MM = 2000

# 평행 판정 임계(라이다 패턴)
PARALLEL_TH = 180
PARALLEL_CONFIRM = 3

# ==========================================================
# [7] STOP CONDITION (후방 라이다 90~270 기반)
# ==========================================================
STOP_DIST = 320
STOP_CONFIRM = 3

# ==========================================================
# [8] EXIT (시간 하드코딩 제거: 센서 조건 기반)
# ==========================================================
EXIT_CLEAR_RS = 1400
EXIT_CLEAR_RF = 1400
EXIT_CLEAR_CONFIRM = 3

ALIGN_PARALLEL_TH = 220
ALIGN_CONFIRM = 3

FRONT_SAFE_STOP = 350    # 전방 중앙이 너무 가까우면 즉시 정지(안전)

# ==========================================================
# [9] SONAR(안전용)
# ==========================================================
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
US_VALID_MIN = 50
US_VALID_MAX = 5000

def valid_us(d):
    return (d is not None) and (US_VALID_MIN <= d <= US_VALID_MAX)

# ==========================================================
# [10] SERIAL / LIDAR
# ==========================================================
ser = None
lidar = None

def read_sensors():
    global sonar_data, ser
    if ser is None:
        return
    for _ in range(5):
        if ser.in_waiting <= 0:
            break
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) == 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass

def send_cmd(servo, speed):
    global ser
    try:
        ser.write(f"S,{int(servo)}\n".encode())
        ser.write(f"D,{int(speed)}\n".encode())
    except:
        pass

def smooth_brake(from_speed, steps=6, dt=0.06):
    global ser
    if ser is None:
        return
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)

# ==========================================================
# [11] STATE
# ==========================================================
STATE_SEARCH       = 0
STATE_SETUP_LEFT   = 1
STATE_REV_ARC1     = 2
STATE_REV_ARC2     = 3
STATE_REV_ALIGN    = 4
STATE_PARK_WAIT    = 5
STATE_EXIT_ARC     = 6
STATE_EXIT_ALIGN   = 7
STATE_EXIT_STRAIGHT= 8
STATE_DONE         = 9

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP  = 2

def main():
    global ser, lidar, sonar_data

    cv2.namedWindow("Parking Monitor")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 성공")
        time.sleep(2)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1

    valid_gap_count = 0
    gap_stable_count = 0
    gap_ready = False
    car2_count = 0

    rs_setup_cnt = 0
    setup_t0 = time.time()

    arc_switch_cnt = 0
    parallel_cnt = 0
    stop_cnt = 0

    parked_t0 = None

    exit_clear_cnt = 0
    exit_align_cnt = 0

    done_sent = False

    print("🚀 주차 시스템 시작")

    try:
        scan_iter = lidar.iter_scans()

        while True:
            try:
                scan = next(scan_iter)
            except Exception as e:
                print(f"[WARN] LiDAR scan error: {e}")
                send_cmd(SERVO_CENTER, 0)
                break

            read_sensors()

            # --- LiDAR 우측 섹터 ---
            rf = get_lidar_dist_in_sector(scan, 30, 70, percentile=20)    # 우전방
            rs = get_lidar_dist_in_sector(scan, 70, 110, percentile=20)   # 우측
            rr = get_lidar_dist_in_sector(scan, 110, 160, percentile=20)  # 우후방
            lidar_side = min(rf, rs, rr)

            # --- 후방/전방 섹터 ---
            rear_center = get_lidar_dist_in_sector(scan, 150, 210, percentile=10)  # 후방 중앙
            rear_all    = get_lidar_dist_in_sector(scan, 90, 270, percentile=10)   # 후방 전체

            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=15)  # 전방 중앙(랩어라운드)

            # --- sonar 안전(옵션) ---
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]
            if not valid_us(dist_LT): dist_LT = 9999
            if not valid_us(dist_RT): dist_RT = 9999

            # 기본 명령
            cmd_speed = 0
            cmd_servo = SERVO_CENTER

            msg = ""
            sub_msg = f"rf:{int(rf)} rs:{int(rs)} rr:{int(rr)} | rearC:{int(rear_center)} front:{int(front_center)}"

            now = time.time()

            # ==========================================================
            # [S0] SEARCH
            # ==========================================================
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                if search_step == STEP_FIND_CAR1:
                    msg = "SEARCH CAR1"
                    if lidar_side < CAR_EXIST_DIST:
                        print("🚗 CAR1 감지 -> PASS")
                        search_step = STEP_PASS_CAR1
                        valid_gap_count = 0
                        gap_stable_count = 0
                        gap_ready = False
                        car2_count = 0

                elif search_step == STEP_PASS_CAR1:
                    msg = "PASS CAR1"
                    if lidar_side > EMPTY_SPACE_DIST:
                        valid_gap_count += 1
                        if valid_gap_count >= PASS_GAP_CONFIRM:
                            print("👀 GAP 진입")
                            search_step = STEP_FIND_GAP
                            gap_stable_count = 0
                            gap_ready = False
                            car2_count = 0
                            valid_gap_count = 0
                    else:
                        valid_gap_count = 0

                elif search_step == STEP_FIND_GAP:
                    msg = "GAP stable -> FIND CAR2"
                    if not gap_ready:
                        if lidar_side > EMPTY_SPACE_DIST:
                            gap_stable_count += 1
                        else:
                            gap_stable_count = 0
                        sub_msg = f"gap:{gap_stable_count}/{GAP_STABLE_TH} | " + sub_msg
                        if gap_stable_count >= GAP_STABLE_TH:
                            gap_ready = True
                            car2_count = 0
                            print("✅ GAP 안정화 -> CAR2 탐색")
                    else:
                        if lidar_side < CAR_EXIST_DIST:
                            car2_count += 1
                        else:
                            car2_count = 0
                        sub_msg = f"car2:{car2_count}/{CAR2_CONFIRM_TH} | " + sub_msg

                        if car2_count >= CAR2_CONFIRM_TH:
                            print("🛑 CAR2 감지 -> 정지, SETUP 시작")
                            smooth_brake(SPEED_SEARCH)
                            send_cmd(SERVO_CENTER, 0)
                            time.sleep(0.25)
                            state = STATE_SETUP_LEFT
                            setup_t0 = time.time()
                            rs_setup_cnt = 0
                            arc_switch_cnt = 0
                            parallel_cnt = 0
                            stop_cnt = 0
                            continue

            # ==========================================================
            # [S1] SETUP_LEFT (좌대각 전진)
            # ==========================================================
            elif state == STATE_SETUP_LEFT:
                msg = "SETUP: FORWARD-LEFT"
                cmd_servo = SERVO_LEFT_MAX

                if now - setup_t0 < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "SETUP: STEER LEFT(WAIT)"
                else:
                    cmd_speed = SPEED_SETUP

                if rs > RS_SETUP_TARGET:
                    rs_setup_cnt += 1
                else:
                    rs_setup_cnt = 0

                sub_msg = f"rs_setup:{rs_setup_cnt}/{RS_SETUP_CONFIRM} | " + sub_msg

                if rs_setup_cnt >= RS_SETUP_CONFIRM:
                    print("✅ SETUP 완료 -> 후진 ARC1")
                    send_cmd(SERVO_CENTER, 0)
                    time.sleep(0.2)
                    state = STATE_REV_ARC1
                    arc_switch_cnt = 0
                    continue

            # ==========================================================
            # [S2] REV_ARC1 (각도 계산 기반: 목표 반경으로 우회전 성향)
            # ==========================================================
            elif state == STATE_REV_ARC1:
                msg = "REV ARC1 (RIGHT)"
                # 목표 반경 -> delta 계산 (우회전이므로 음수)
                delta = -delta_deg_from_radius_mm(ARC1_RADIUS_MM)
                base_servo = servo_from_delta_deg(delta)

                # rs 목표 간격 보정(P): rs가 멀면(큰 값) 더 우회전(servo↓), 가까우면 덜 우회전(servo↑)
                err = RS_TARGET - rs
                servo = base_servo - KP_RS * err

                # rr 위험이면 조향 완화(좌로 조금)
                if rr < RR_MIN_SAFE:
                    servo += RR_RELEASE

                cmd_servo = int(max(SERVO_RIGHT_MAX, min(SERVO_LEFT_MAX, servo)))
                cmd_speed = -SPEED_PARK

                # arc1 -> arc2 전환(패턴 기반, 경량)
                # rf가 가까워지거나 rr이 많이 줄면(말려들어감) 전환
                cond = (rf < 650) or (rr < 650)
                arc_switch_cnt = arc_switch_cnt + 1 if cond else 0
                sub_msg = f"arc1_sw:{arc_switch_cnt}/3 servo:{cmd_servo} | " + sub_msg

                if arc_switch_cnt >= 3:
                    print("➡️ ARC1 -> ARC2(펴기)")
                    state = STATE_REV_ARC2
                    parallel_cnt = 0
                    continue

            # ==========================================================
            # [S3] REV_ARC2 (각도 계산 기반: 큰 반경으로 좌로 펴기)
            # ==========================================================
            elif state == STATE_REV_ARC2:
                msg = "REV ARC2 (LEFT/UNWIND)"
                # 큰 반경 -> 작은 delta(거의 직선). 펴는 성향(좌로)로 약간 양수 부여
                delta = +delta_deg_from_radius_mm(ARC2_RADIUS_MM)
                base_servo = servo_from_delta_deg(delta)

                # rs 목표 간격 보정(P)
                err = RS_TARGET - rs
                servo = base_servo - KP_RS * err

                if rr < RR_MIN_SAFE:
                    servo += RR_RELEASE

                cmd_servo = int(max(SERVO_RIGHT_MAX, min(SERVO_LEFT_MAX, servo)))
                cmd_speed = -SPEED_PARK

                # 평행 판정: rf≈rr
                if abs(rf - rr) < PARALLEL_TH:
                    parallel_cnt += 1
                else:
                    parallel_cnt = 0

                sub_msg = f"parallel:{parallel_cnt}/{PARALLEL_CONFIRM} servo:{cmd_servo} | " + sub_msg

                if parallel_cnt >= PARALLEL_CONFIRM:
                    print("✅ 평행화 -> ALIGN(센터 후진)")
                    state = STATE_REV_ALIGN
                    stop_cnt = 0
                    continue

            # ==========================================================
            # [S4] REV_ALIGN (센터 후진 + 후방 라이다로 STOP)
            # ==========================================================
            elif state == STATE_REV_ALIGN:
                msg = "REV ALIGN (CENTER)"
                cmd_servo = SERVO_CENTER
                cmd_speed = -SPEED_PARK

                if rear_center < STOP_DIST:
                    stop_cnt += 1
                else:
                    stop_cnt = 0

                sub_msg = f"stop:{stop_cnt}/{STOP_CONFIRM} rearC:{int(rear_center)} | " + sub_msg

                if stop_cnt >= STOP_CONFIRM:
                    print("🛑 슬롯 내 STOP -> PARK WAIT")
                    send_cmd(SERVO_CENTER, 0)
                    state = STATE_PARK_WAIT
                    parked_t0 = time.time()
                    continue

            # ==========================================================
            # [S5] PARK_WAIT (4초 정차)
            # ==========================================================
            elif state == STATE_PARK_WAIT:
                msg = "PARKED WAIT 4s"
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                if parked_t0 is not None and (time.time() - parked_t0) >= 4.0:
                    print("➡️ 출차 시작: 전진 우회전(반대방향)")
                    state = STATE_EXIT_ARC
                    exit_clear_cnt = 0
                    exit_align_cnt = 0
                    continue

            # ==========================================================
            # [S6] EXIT_ARC (시간X) - 센서로 슬롯 탈출 판단
            # ==========================================================
            elif state == STATE_EXIT_ARC:
                msg = "EXIT ARC (FORWARD RIGHT)"
                # 우회전 각도(목표 반경 기반)
                delta = -delta_deg_from_radius_mm(ARC1_RADIUS_MM)  # 출차도 비슷한 반경 사용
                cmd_servo = servo_from_delta_deg(delta)
                cmd_speed = SPEED_EXIT

                # 안전: 전방 너무 가까우면 정지
                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP (FRONT)"

                # 슬롯 탈출 조건: rs와 rf가 모두 충분히 커짐(연속)
                if (rs > EXIT_CLEAR_RS) and (rf > EXIT_CLEAR_RF):
                    exit_clear_cnt += 1
                else:
                    exit_clear_cnt = 0

                sub_msg = f"exit_clear:{exit_clear_cnt}/{EXIT_CLEAR_CONFIRM} | " + sub_msg

                if exit_clear_cnt >= EXIT_CLEAR_CONFIRM:
                    print("✅ 슬롯 탈출 -> EXIT ALIGN")
                    state = STATE_EXIT_ALIGN
                    exit_align_cnt = 0
                    continue

            # ==========================================================
            # [S7] EXIT_ALIGN (시간X) - 라이다 패턴으로 평행 맞추고 직진
            # ==========================================================
            elif state == STATE_EXIT_ALIGN:
                msg = "EXIT ALIGN (CENTER)"
                cmd_servo = SERVO_CENTER
                cmd_speed = SPEED_EXIT

                # 안전: 전방 너무 가까우면 정지
                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP (FRONT)"

                # 평행 조건: abs(rf-rr) 작아짐 연속
                if abs(rf - rr) < ALIGN_PARALLEL_TH:
                    exit_align_cnt += 1
                else:
                    exit_align_cnt = 0

                sub_msg = f"exit_align:{exit_align_cnt}/{ALIGN_CONFIRM} | " + sub_msg

                if exit_align_cnt >= ALIGN_CONFIRM:
                    print("✅ 출차 평행 정렬 -> STRAIGHT")
                    state = STATE_EXIT_STRAIGHT
                    continue

            # ==========================================================
            # [S8] EXIT_STRAIGHT (센서 기반 유지)
            # ==========================================================
            elif state == STATE_EXIT_STRAIGHT:
                msg = "EXIT STRAIGHT"
                cmd_servo = SERVO_CENTER
                cmd_speed = SPEED_EXIT

                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP (FRONT)"

                # 여기서 "완전 탈출" 조건을 더 붙일 수 있음:
                # 예) front_center가 매우 크고(rs도 큼) 일정 시간/프레임 유지하면 DONE
                # 지금은 간단히 충분히 열린 구간이면 종료하도록 설정(연속 10프레임)
                # (원하면 OUT 라인 구간에 맞춰 조건을 더 정교하게 넣어줄게)
                # ---
                # 열린 구간 판단
                if (front_center > 2000) and (rs > 1500):
                    done_cnt = getattr(main, "_done_cnt", 0) + 1
                    setattr(main, "_done_cnt", done_cnt)
                else:
                    setattr(main, "_done_cnt", 0)

                if getattr(main, "_done_cnt", 0) >= 10:
                    state = STATE_DONE
                    continue

            # ==========================================================
            # [S9] DONE
            # ==========================================================
            elif state == STATE_DONE:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                msg = "DONE"
                if not done_sent:
                    send_cmd(SERVO_CENTER, 0)
                    done_sent = True
                break

            # (옵션) 초음파 안전: 너무 가까우면 정지
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                msg = "SAFETY STOP (US)"

            send_cmd(cmd_servo, cmd_speed)

            # 디버그 화면
            img = np.zeros((340, 980, 3), dtype=np.uint8)
            cv2.putText(img, f"STATE:{state} STEP:{search_step}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(img, msg, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(img, sub_msg, (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            cv2.putText(img, f"servo:{cmd_servo} speed:{cmd_speed}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)

            cv2.imshow("Parking Monitor", img)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        try:
            if ser:
                send_cmd(SERVO_CENTER, 0)
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