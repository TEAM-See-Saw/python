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
# [1] SPEED / SERVO (사용자 확정값)
# ==========================================================
SPEED_SEARCH = 80
SPEED_SETUP  = 80
SPEED_PARK   = 75
SPEED_EXIT   = 80

SERVO_CENTER    = 570
SERVO_RIGHT_MAX = 480   # 사용자 최신값 반영
SERVO_LEFT_MAX  = 680

# ==========================================================
# [2] CAR DIM (mm) - 참고용 (실제 각도 계산엔 부족하지만, 로직 설명용)
# ==========================================================
CAR_LEN_MM = 1100
CAR_W_MM   = 600

# ==========================================================
# [3] SEARCH 임계값 (오른쪽=0~90도 기준)
# ==========================================================
CAR_EXIST_RS_TH   = 850
EMPTY_RS_TH       = 1200

# 빠르게 반응(속도 80)
PASS_GAP_CONFIRM  = 3
GAP_STABLE_TH     = 2
CAR2_CONFIRM_TH   = 2
ARM_TIME_SEC      = 0.20

# CAR2는 "우전방"에서 다시 가까워지는 것으로 감지
CAR2_RF_TH        = 950
CAR2_RS_GATE      = 0.85   # rs_detect > EMPTY_RS_TH*0.85

# ==========================================================
# [4] SETUP (좌대각 전진) 전환조건
# ==========================================================
STEER_WAIT_TIME  = 0.6
RS_SETUP_TARGET  = 1200
RS_SETUP_CONFIRM = 2

# ==========================================================
# [5] PARKING (원호 기반 유사 기하학)
# ==========================================================
# 주차: ARC1(후진 풀우) -> ARC2(후진 풀좌) -> ALIGN(센터 후진)
# 전환은 "센서 패턴"으로 결정(시간 하드코딩 최소화)

# ARC1 종료 조건: 우전방(rf)이 가까워지기 시작하면(앞차 코너가 보이면) 다음 단계로
ARC1_RF_CLOSE_TH = 700
ARC1_CONFIRM     = 2

# ARC2 종료 조건: 차량이 슬롯과 평행해졌다고 판단(rf~rr 비슷)하면 ALIGN로
PARALLEL_TH      = 220
PARALLEL_CONFIRM = 2

# 정지 조건(요구 반영): rear(90~270)에 장애물이 "안 찍히면" 정지
# *주의*: 이 조건은 물리적으로는 "뒤가 트였다"를 의미할 수 있어서,
# 실제론 "주차선 내부에서 멈춤"과 직접 대응이 어렵습니다.
STOP_CLEAR_TH = 1800
STOP_CONFIRM  = 2

# ==========================================================
# [6] EXIT (출차: 우회전해서 진입방향 반대로 나가기)
# ==========================================================
FRONT_SAFE_STOP = 350
EXIT_OPEN_FRONT_TH = 2000
EXIT_OPEN_RS_TH    = 1500
EXIT_DONE_CONFIRM  = 10

# ==========================================================
# [7] SONAR (안전용 옵션)
# ==========================================================
sonar_data = [999] * 6
IDX_LT = 2
IDX_RT = 5
US_VALID_MIN = 50
US_VALID_MAX = 5000

def valid_us(d):
    return (d is not None) and (US_VALID_MIN <= d <= US_VALID_MAX)

# ==========================================================
# LiDAR helper
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
# SERIAL helpers
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

def smooth_brake(from_speed, steps=5, dt=0.05):
    for s in np.linspace(from_speed, 0, steps):
        send_cmd(SERVO_CENTER, int(s))
        time.sleep(dt)
    send_cmd(SERVO_CENTER, 0)

# ==========================================================
# STATE
# ==========================================================
STATE_SEARCH       = 0
STATE_SETUP_LEFT   = 1
STATE_REV_ARC1     = 2
STATE_REV_ARC2     = 3
STATE_REV_ALIGN    = 4
STATE_PARK_WAIT    = 5
STATE_EXIT_RIGHT   = 6
STATE_DONE         = 7

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
        time.sleep(1.5)
    except Exception as e:
        print(f"❌ 연결 실패: {e}")
        return

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1

    # SEARCH counters
    valid_gap_count = 0
    gap_stable_count = 0
    gap_ready = False
    car2_count = 0

    gap_enter_t = None
    car2_armed = False

    # SETUP counter
    setup_t0 = time.time()
    rs_setup_cnt = 0

    # ARC transitions
    arc1_cnt = 0
    parallel_cnt = 0

    # STOP
    stop_cnt = 0

    # WAIT
    parked_t0 = None

    # EXIT done
    exit_done_cnt = 0
    done_sent = False

    prev_t = time.time()
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

            now = time.time()
            dt = now - prev_t
            prev_t = now

            read_sensors()

            # ---------------------------
            # LiDAR sectors (오른쪽=0~90)
            # ---------------------------
            # detect: 옆(오른쪽 측면) / 우전방(앞쪽 코너)
            rs_detect = get_lidar_dist_in_sector(scan, 65, 90, percentile=20)
            rf_detect = get_lidar_dist_in_sector(scan, 15, 40, percentile=20)

            # control: 우전방/우측/우후방
            rf = get_lidar_dist_in_sector(scan, 15, 60, percentile=20)
            rs = get_lidar_dist_in_sector(scan, 60, 90, percentile=20)
            rr = get_lidar_dist_in_sector(scan, 90, 130, percentile=20)

            # rear check(요구: 90~270)
            rear_all = get_lidar_dist_in_sector(scan, 90, 270, percentile=15)

            # front safety(랩어라운드)
            front_center = get_lidar_dist_in_sector(scan, 350, 10, percentile=15)

            # sonar safety(optional)
            dist_LT = sonar_data[IDX_LT]
            dist_RT = sonar_data[IDX_RT]
            if not valid_us(dist_LT): dist_LT = 9999
            if not valid_us(dist_RT): dist_RT = 9999

            # invalid guard
            rs_detect_ok = (rs_detect < 9000)
            rf_detect_ok = (rf_detect < 9000)

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            msg = ""

            sub_msg = (
                f"dt:{dt*1000:.0f}ms | "
                f"rsD:{int(rs_detect)} rfD:{int(rf_detect)} | "
                f"rf:{int(rf)} rs:{int(rs)} rr:{int(rr)} | "
                f"rearAll:{int(rear_all)} front:{int(front_center)}"
            )

            # ======================================================
            # [S0] SEARCH
            # ======================================================
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH
                cmd_servo = SERVO_CENTER

                if search_step == STEP_FIND_CAR1:
                    msg = "SEARCH CAR1 (rs_detect)"
                    if rs_detect_ok and (rs_detect < CAR_EXIST_RS_TH):
                        print("🚗 CAR1 감지 -> PASS")
                        search_step = STEP_PASS_CAR1
                        valid_gap_count = 0
                        gap_stable_count = 0
                        gap_ready = False
                        car2_count = 0
                        gap_enter_t = None
                        car2_armed = False

                elif search_step == STEP_PASS_CAR1:
                    msg = "PASS CAR1 -> WAIT GAP"
                    if rs_detect_ok and (rs_detect > EMPTY_RS_TH):
                        valid_gap_count += 1
                        if valid_gap_count >= PASS_GAP_CONFIRM:
                            print("👀 GAP 진입")
                            search_step = STEP_FIND_GAP
                            valid_gap_count = 0
                            gap_stable_count = 0
                            gap_ready = False
                            car2_count = 0
                            gap_enter_t = time.time()
                            car2_armed = False
                    else:
                        valid_gap_count = 0

                elif search_step == STEP_FIND_GAP:
                    msg = "GAP stable -> FIND CAR2"
                    if not gap_ready:
                        if rs_detect_ok and (rs_detect > EMPTY_RS_TH):
                            gap_stable_count += 1
                        else:
                            gap_stable_count = 0

                        if gap_stable_count >= GAP_STABLE_TH:
                            gap_ready = True
                            car2_count = 0
                            print("✅ GAP 안정화 -> CAR2 탐색")
                    else:
                        if (gap_enter_t is not None) and (not car2_armed):
                            if (time.time() - gap_enter_t) >= ARM_TIME_SEC:
                                car2_armed = True
                                car2_count = 0
                                print("🟩 CAR2 ARM ON")

                        if car2_armed:
                            cond_car2 = (
                                rf_detect_ok and (rf_detect < CAR2_RF_TH) and
                                rs_detect_ok and (rs_detect > (EMPTY_RS_TH * CAR2_RS_GATE))
                            )
                            car2_count = (car2_count + 1) if cond_car2 else 0
                            msg = f"FIND CAR2 cnt:{car2_count}/{CAR2_CONFIRM_TH}"

                            if car2_count >= CAR2_CONFIRM_TH:
                                print("🛑 CAR2 확정 -> STOP & SETUP")
                                smooth_brake(SPEED_SEARCH)
                                send_cmd(SERVO_CENTER, 0)
                                time.sleep(0.15)

                                state = STATE_SETUP_LEFT
                                setup_t0 = time.time()
                                rs_setup_cnt = 0
                                arc1_cnt = 0
                                parallel_cnt = 0
                                stop_cnt = 0
                                continue

            # ======================================================
            # [S1] SETUP_LEFT (좌대각 전진: 각도 만들기)
            # ======================================================
            elif state == STATE_SETUP_LEFT:
                cmd_servo = SERVO_LEFT_MAX

                if now - setup_t0 < STEER_WAIT_TIME:
                    cmd_speed = 0
                    msg = "SETUP: STEER LEFT(WAIT)"
                else:
                    cmd_speed = SPEED_SETUP
                    msg = "SETUP: FORWARD-LEFT"

                if rs > RS_SETUP_TARGET:
                    rs_setup_cnt += 1
                else:
                    rs_setup_cnt = 0

                msg += f" rs_cnt:{rs_setup_cnt}/{RS_SETUP_CONFIRM}"

                if rs_setup_cnt >= RS_SETUP_CONFIRM:
                    print("✅ SETUP 완료 -> REV ARC1")
                    send_cmd(SERVO_CENTER, 0)
                    time.sleep(0.10)
                    state = STATE_REV_ARC1
                    arc1_cnt = 0
                    continue

            # ======================================================
            # [S2] REV_ARC1 (후진 풀우: 최소회전반경 가정)
            # ======================================================
            elif state == STATE_REV_ARC1:
                msg = "REV ARC1: FULL RIGHT"
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = -SPEED_PARK

                # ARC1 종료: 우전방(rf)이 가까워지면(앞차 코너 접근) -> ARC2
                if rf < ARC1_RF_CLOSE_TH:
                    arc1_cnt += 1
                else:
                    arc1_cnt = 0

                msg += f" arc1:{arc1_cnt}/{ARC1_CONFIRM}"

                if arc1_cnt >= ARC1_CONFIRM:
                    print("➡️ ARC1 -> ARC2(펴기)")
                    state = STATE_REV_ARC2
                    parallel_cnt = 0
                    continue

            # ======================================================
            # [S3] REV_ARC2 (후진 풀좌: 차체 평행화)
            # ======================================================
            elif state == STATE_REV_ARC2:
                msg = "REV ARC2: FULL LEFT"
                cmd_servo = SERVO_LEFT_MAX
                cmd_speed = -SPEED_PARK

                # 평행화 판단: rf와 rr이 비슷해지면(차가 슬롯과 평행해지는 패턴)
                if abs(rf - rr) < PARALLEL_TH:
                    parallel_cnt += 1
                else:
                    parallel_cnt = 0

                msg += f" parallel:{parallel_cnt}/{PARALLEL_CONFIRM}"

                if parallel_cnt >= PARALLEL_CONFIRM:
                    print("✅ 평행화 -> ALIGN(센터 후진)")
                    state = STATE_REV_ALIGN
                    stop_cnt = 0
                    continue

            # ======================================================
            # [S4] REV_ALIGN (센터 후진 + rear_all '안찍히면' STOP)
            # ======================================================
            elif state == STATE_REV_ALIGN:
                msg = "REV ALIGN: CENTER"
                cmd_servo = SERVO_CENTER
                cmd_speed = -SPEED_PARK

                # 요구 반영: rear(90~270)에서 장애물 안 찍히면 STOP
                if rear_all > STOP_CLEAR_TH:
                    stop_cnt += 1
                else:
                    stop_cnt = 0

                msg += f" stop:{stop_cnt}/{STOP_CONFIRM}"

                if stop_cnt >= STOP_CONFIRM:
                    print("🛑 rear clear -> PARK WAIT")
                    send_cmd(SERVO_CENTER, 0)
                    state = STATE_PARK_WAIT
                    parked_t0 = time.time()
                    continue

            # ======================================================
            # [S5] PARK_WAIT (4초 정차)
            # ======================================================
            elif state == STATE_PARK_WAIT:
                msg = "PARK WAIT 4s"
                cmd_servo = SERVO_CENTER
                cmd_speed = 0

                if parked_t0 is not None and (time.time() - parked_t0) >= 4.0:
                    print("➡️ EXIT: FORWARD RIGHT (반대방향 출차)")
                    state = STATE_EXIT_RIGHT
                    exit_done_cnt = 0
                    continue

            # ======================================================
            # [S6] EXIT_RIGHT (전진 풀우 + 안전)
            # ======================================================
            elif state == STATE_EXIT_RIGHT:
                msg = "EXIT: FULL RIGHT"
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = SPEED_EXIT

                if front_center < FRONT_SAFE_STOP:
                    cmd_speed = 0
                    msg = "EXIT SAFETY STOP (FRONT)"

                # 종료 조건: 전방/우측이 충분히 열림 연속
                if (front_center > EXIT_OPEN_FRONT_TH) and (rs > EXIT_OPEN_RS_TH):
                    exit_done_cnt += 1
                else:
                    exit_done_cnt = 0

                msg += f" doneCnt:{exit_done_cnt}/{EXIT_DONE_CONFIRM}"

                if exit_done_cnt >= EXIT_DONE_CONFIRM:
                    state = STATE_DONE
                    continue

            # ======================================================
            # [S7] DONE
            # ======================================================
            elif state == STATE_DONE:
                msg = "DONE"
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                if not done_sent:
                    send_cmd(SERVO_CENTER, 0)
                    done_sent = True
                break

            # ---------------------------
            # optional sonar safety
            # ---------------------------
            if min(dist_LT, dist_RT) < 120:
                cmd_speed = 0
                msg = "SAFETY STOP (US)"

            send_cmd(cmd_servo, cmd_speed)

            # debug view
            img = np.zeros((380, 1100, 3), dtype=np.uint8)
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