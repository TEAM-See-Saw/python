import serial
from rplidar import RPLidar, RPLidarException
import time
import numpy as np
import cv2
from collections import deque
import threading

# ==========================================
# [1] 포트/통신 (Arduino 코드와 일치)
# ==========================================
PORT = "COM4"
LIDAR_PORT = "COM3"
SER_BAUD = 115200          # ✅ Arduino Serial.begin(115200)
SER_TIMEOUT = 0.02         # 짧게(리더 스레드가 계속 돌기 때문)

# ==========================================
# [2] 차량/주차공간 치수 (mm)
# ==========================================
CAR_W = 750
CAR_L = 1000
SLOT_W = 950
SLOT_L = 1500
SIDE_CLEAR_EACH = (SLOT_W - CAR_W) / 2  # 100mm

# ==========================================
# [3] 속도/서보
# ==========================================
SPEED_SEARCH = 70
SPEED_SETUP = 80
SPEED_REVERSE = 75
SPEED_STOP = 0

SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
SERVO_SLIGHT_RIGHT = 555

STEER_WAIT_TIME = 0.7

# ==========================================
# [4] 후방주차 시간 파라미터 (튜닝)
# ==========================================
TIME_SETUP_MOVE = 1.8
TIME_REVERSE_TURN = 2.2
TIME_REVERSE_STRAIGHT_MAX = 4.5
TIME_DELAY_STOP = 1.5

# ✅ 주차 완료 후 3초 정지
TIME_HOLD_AFTER_PARK = 3.0

# ==========================================
# [5] 라이다 처리(튐 완화)
# ==========================================
LIDAR_TEMPORAL_N = 5
LIDAR_PCTL = 10

buf_side = deque(maxlen=LIDAR_TEMPORAL_N)
buf_left = deque(maxlen=LIDAR_TEMPORAL_N)
buf_right = deque(maxlen=LIDAR_TEMPORAL_N)

def percentile_in_sector(scan, start_angle, end_angle, percentile=10) -> float:
    dists = []
    for (_, angle, dist) in scan:
        if dist <= 0:
            continue
        if start_angle <= angle <= end_angle:
            dists.append(dist)
    if not dists:
        return 9999.0
    v = np.percentile(np.array(dists, dtype=np.float32), percentile)
    return float(np.asarray(v).item())

def temporal_median(buf: deque, v: float) -> float:
    buf.append(v)
    m = np.median(np.array(buf, dtype=np.float32))
    return float(np.asarray(m).item())

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ==========================================
# [6] 탐색(차/갭) 임계값
# ==========================================
CAR_EXIST_DIST = 1700
EMPTY_SPACE_DIST = 2000
GAP_STREAK = 2

# ==========================================
# [7] 후진 주차 마무리 감지(라이다 기반)
# ==========================================
SIDE_STOP_DIST = 700

REVERSE_CENTER_KP = 0.035
REVERSE_CENTER_LIMIT = 22
REVERSE_CENTER_DEADBAND = 40

# ==========================================
# [8] 초음파 스트리밍(Arduino: US: 6개)
#     - 읽기는 계속 (버퍼 방지)
#     - "판단"은 2초 구간에서만 사용
# ==========================================

# ✅ 아래 인덱스는 "LF, LR, RF, RR"에 해당하는 센서 번호를 의미.
# 지금은 임시값(0,1,4,5)로 둠. 대회/배선에 맞춰 바꾸면 됨.
IDX_LF = 0
IDX_LR = 1
IDX_RF = 4
IDX_RR = 5

US_CONFIRM_DURATION = 2.0
US_CONFIRM_HZ = 10
US_SIDE_TH_MM = 300         # 250~450에서 튜닝
US_CONFIRM_PASS_RATIO = 0.7 # 2초 샘플 중 70% 이상 동시 감지면 통과

# 최신 초음파 값(6채널) 공유
_sonar_lock = threading.Lock()
_sonar_values = [999] * 6
_sonar_last_ts = 0.0

def get_latest_sonar():
    with _sonar_lock:
        return _sonar_values.copy(), _sonar_last_ts

def serial_reader_thread(ser: serial.Serial, stop_event: threading.Event):
    """
    Arduino가 50ms마다 보내는 'US:a,b,c,d,e,f'를 계속 읽어서 최신값만 유지.
    이 스레드가 있어야 PC/Arduino 버퍼가 안 쌓임.
    """
    global _sonar_values, _sonar_last_ts

    # 시작 시 찌꺼기 비우기(초기 동기화)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            line = ser.readline()  # timeout 짧게 설정되어 있음
            if not line:
                continue
            s = line.decode("utf-8", errors="ignore").strip()
            if not s.startswith("US:"):
                continue
            payload = s[3:]
            parts = payload.split(",")
            if len(parts) != 6:
                continue
            vals = []
            ok = True
            for p in parts:
                p = p.strip()
                if p == "":
                    ok = False
                    break
                try:
                    vals.append(int(float(p)))
                except:
                    ok = False
                    break
            if not ok:
                continue

            with _sonar_lock:
                _sonar_values = vals
                _sonar_last_ts = time.time()

        except Exception:
            # 리더 스레드는 죽지 않게만
            continue

def ultrasonic_confirm_2s_streaming():
    """
    Arduino 스트리밍으로 갱신되는 최신 sonarValues를 사용해서
    2초 동안 LF/LR/RF/RR이 동시에 임계값 이하인지 평가.
    """
    duration = US_CONFIRM_DURATION
    dt = 1.0 / US_CONFIRM_HZ
    ok = 0
    total = 0

    t0 = time.time()
    while time.time() - t0 < duration:
        vals, ts = get_latest_sonar()

        lf = vals[IDX_LF]
        lr = vals[IDX_LR]
        rf = vals[IDX_RF]
        rr = vals[IDX_RR]

        # Arduino가 pulseIn timeout 시 999를 주므로 999는 "감지 아님"으로 처리됨
        if (lf < US_SIDE_TH_MM and lr < US_SIDE_TH_MM and
            rf < US_SIDE_TH_MM and rr < US_SIDE_TH_MM):
            ok += 1
        total += 1

        time.sleep(dt)

    passed = (ok / total) >= US_CONFIRM_PASS_RATIO if total > 0 else False
    return passed, (ok, total)

# ==========================================
# [9] 상태 정의 (후방주차 FSM)
# ==========================================
STATE_SEARCH = 0
STATE_PAUSE = 99

STATE_SETUP_FORWARD = 10
STATE_REVERSE_TURN = 11
STATE_REVERSE_STRAIGHT = 12
STATE_US_CONFIRM = 13
STATE_HOLD_3S = 14
STATE_PARKED = 15

STEP_FIND_CAR1 = 0
STEP_PASS_CAR1 = 1
STEP_FIND_GAP  = 2

# ==========================================
# [10] 메인
# ==========================================
def main():
    cv2.namedWindow("Parking Monitor")

    ser = None
    lidar = None
    stop_event = threading.Event()
    reader = None

    try:
        ser = serial.Serial(PORT, SER_BAUD, timeout=SER_TIMEOUT)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ 시스템 연결 (Serial 115200 + RPLidar)")
        time.sleep(1.0)
        lidar.clean_input()

        # ✅ 초음파 스트리밍 리더 스레드 시작 (버퍼 방지)
        reader = threading.Thread(target=serial_reader_thread, args=(ser, stop_event), daemon=True)
        reader.start()

    except Exception as e:
        print(f"❌ 연결 오류: {e}")
        return

    # 초기 정렬/정지
    ser.write(f"S,{SERVO_CENTER}\n".encode())
    ser.write(b"D,0\n")
    time.sleep(1.0)

    state = STATE_SEARCH
    search_step = STEP_FIND_CAR1
    state_timer = time.time()

    valid_gap_count = 0

    pause_start = 0.0
    pause_duration = 0.0
    next_state = STATE_SEARCH
    pause_msg = ""

    side_detect_time = 0.0
    running = True

    try:
        while running:
            for scan in lidar.iter_scans():
                curr_time = time.time()

                # 탐색용 옆 거리(30~110)
                side_raw = percentile_in_sector(scan, 30, 110, LIDAR_PCTL)
                side_dist = temporal_median(buf_side, side_raw)

                # 후진 센터링/마무리용 좌/우 (90±10, 270±10)
                left_raw  = percentile_in_sector(scan, 80, 100, LIDAR_PCTL)
                right_raw = percentile_in_sector(scan, 260, 280, LIDAR_PCTL)
                dist_90   = temporal_median(buf_left, left_raw)
                dist_270  = temporal_median(buf_right, right_raw)

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
                    if curr_time - pause_start > pause_duration:
                        state = next_state
                        state_timer = curr_time
                        valid_gap_count = 0
                        side_detect_time = 0.0
                        lidar.clean_input()

                # =========================
                # [1] SEARCH: 차1 -> 갭 -> 차2 찾기
                # =========================
                elif state == STATE_SEARCH:
                    cmd_speed = SPEED_SEARCH
                    cmd_servo = SERVO_CENTER

                    if search_step == STEP_FIND_CAR1:
                        msg = f"FIND CAR1 | side={int(side_dist)}"
                        if side_dist < CAR_EXIST_DIST:
                            print(f"🚗 1번 차 감지 ({int(side_dist)}mm)")
                            search_step = STEP_PASS_CAR1

                    elif search_step == STEP_PASS_CAR1:
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"PASS CAR1 | side={int(side_dist)}"
                        if side_dist > EMPTY_SPACE_DIST:
                            valid_gap_count += 1
                            if valid_gap_count >= GAP_STREAK:
                                print(f"👀 빈공간(갭) 진입 ({int(side_dist)}mm)")
                                search_step = STEP_FIND_GAP
                        else:
                            valid_gap_count = 0

                    elif search_step == STEP_FIND_GAP:
                        cmd_servo = SERVO_SLIGHT_RIGHT
                        msg = f"FIND CAR2 | side={int(side_dist)}"
                        if side_dist < CAR_EXIST_DIST:
                            print(f"🛑 2번 차 감지 -> 후방주차 시작 ({int(side_dist)}mm)")
                            # 살짝 브레이크 느낌
                            ser.write(b"D,-150\n")
                            time.sleep(0.08)
                            ser.write(b"D,0\n")

                            state = STATE_PAUSE
                            next_state = STATE_SETUP_FORWARD
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Setup(Forward)"

                # =========================
                # [2] SETUP: 전진 공간 확보
                # =========================
                elif state == STATE_SETUP_FORWARD:
                    cmd_servo = SERVO_LEFT_MAX

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align LEFT..."
                    else:
                        cmd_speed = SPEED_SETUP
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"SETUP FWD: {driving_time:.1f}/{TIME_SETUP_MOVE}s"
                        if driving_time >= TIME_SETUP_MOVE:
                            ser.write(b"D,0\n")
                            state = STATE_PAUSE
                            next_state = STATE_REVERSE_TURN
                            pause_duration = 0.7
                            pause_start = curr_time
                            pause_msg = "Ready for Reverse Turn(Right)"

                # =========================
                # [3] REVERSE TURN: 꺾고 후진 진입
                # =========================
                elif state == STATE_REVERSE_TURN:
                    cmd_servo = SERVO_RIGHT_MAX

                    if curr_time - state_timer < STEER_WAIT_TIME:
                        cmd_speed = 0
                        msg = "Align RIGHT..."
                    else:
                        cmd_speed = -SPEED_REVERSE
                        driving_time = (curr_time - state_timer) - STEER_WAIT_TIME
                        msg = f"REV TURN: {driving_time:.1f}/{TIME_REVERSE_TURN}s"
                        if driving_time >= TIME_REVERSE_TURN:
                            ser.write(b"D,0\n")
                            state = STATE_REVERSE_STRAIGHT
                            state_timer = curr_time
                            side_detect_time = 0.0

                # =========================
                # [4] REVERSE STRAIGHT: 중앙 + 후진(센터링)
                # =========================
                elif state == STATE_REVERSE_STRAIGHT:
                    dt = curr_time - state_timer

                    if dt < STEER_WAIT_TIME:
                        cmd_speed = 0
                        cmd_servo = SERVO_CENTER
                        msg = "Align CENTER..."
                    else:
                        cmd_speed = -SPEED_REVERSE

                        # 폭이 타이트하므로 후진 중 미세 센터링
                        err = dist_90 - dist_270
                        if abs(err) < REVERSE_CENTER_DEADBAND:
                            delta = 0
                        else:
                            delta = int(clamp(REVERSE_CENTER_KP * err, -REVERSE_CENTER_LIMIT, REVERSE_CENTER_LIMIT))

                        cmd_servo = SERVO_CENTER + delta
                        msg = f"REV STRAIGHT | L={int(dist_90)} R={int(dist_270)} err={int(err)} dS={delta}"

                    # 라이다 근접 감지 -> 1.5초 더 후진 후 정지
                    detected = (dist_90 < SIDE_STOP_DIST) or (dist_270 < SIDE_STOP_DIST)

                    if detected and dt > STEER_WAIT_TIME:
                        if side_detect_time == 0.0:
                            print("✨ (라이다) 근접 감지 -> 1.5초 더 후진 후 정지")
                            side_detect_time = curr_time
                        elif (curr_time - side_detect_time) >= TIME_DELAY_STOP:
                            print("✅ 후진 주차 정지(지연 정지)")
                            ser.write(b"D,0\n")
                            state_timer = curr_time
                            state = STATE_US_CONFIRM
                            # 정지 상태로 넘어가므로 cmd_speed는 0으로 유지될 것

                    # 안전 타임아웃
                    if dt >= (STEER_WAIT_TIME + TIME_REVERSE_STRAIGHT_MAX):
                        print("✅ 후진 주차 정지(타임아웃)")
                        ser.write(b"D,0\n")
                        state_timer = curr_time
                        state = STATE_US_CONFIRM

                # =========================
                # [5] 초음파 2초 동시 인지 확인 (스트리밍 값을 사용)
                # =========================
                elif state == STATE_US_CONFIRM:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "US CONFIRM 2s (streaming US: 6ch)"

                    # 2초 동안 정지 유지
                    ser.write(b"D,0\n")

                    passed, (ok, total) = ultrasonic_confirm_2s_streaming()
                    print(f"📌 US confirm(stream): passed={passed} ok/total={ok}/{total} th={US_SIDE_TH_MM}mm "
                          f"(IDX {IDX_LF},{IDX_LR},{IDX_RF},{IDX_RR})")

                    # 초음파 확인 후 -> 3초 정지
                    state = STATE_HOLD_3S
                    state_timer = curr_time

                # =========================
                # [6] 주차 완료 후 3초 정지
                # =========================
                elif state == STATE_HOLD_3S:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    hold_t = curr_time - state_timer
                    msg = f"HOLD AFTER PARK: {hold_t:.1f}/{TIME_HOLD_AFTER_PARK:.1f}s"

                    ser.write(b"D,0\n")  # 정지 유지

                    if hold_t >= TIME_HOLD_AFTER_PARK:
                        state = STATE_PARKED
                        state_timer = curr_time

                # =========================
                # [7] PARKED
                # =========================
                elif state == STATE_PARKED:
                    cmd_speed = 0
                    cmd_servo = SERVO_CENTER
                    msg = "PARKED ✅ (press q to quit)"
                    ser.write(b"D,0\n")

                # ---- 명령 송신 ----
                # Arduino는 "S,###\n" "D,###\n"만 받음
                ser.write(f"S,{int(cmd_servo)}\n".encode())
                ser.write(f"D,{int(cmd_speed)}\n".encode())

                # ---- 디버그 UI ----
                sonar, sonar_ts = get_latest_sonar()
                debug_img = np.zeros((440, 1100, 3), dtype=np.uint8)
                cv2.putText(debug_img, f"State: {state} | {msg}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f"side(30~110)={int(side_dist)}  L(90)={int(dist_90)}  R(270)={int(dist_270)}",
                            (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img,
                            f"US6={sonar}  (t={sonar_ts:.2f})",
                            (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_img,
                            f"IDX LF/LR/RF/RR = {IDX_LF},{IDX_LR},{IDX_RF},{IDX_RR}  th={US_SIDE_TH_MM}mm",
                            (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_img,
                            f"Car {CAR_W}x{CAR_L}  Slot {SLOT_W}x{SLOT_L}  SideClearEach {int(SIDE_CLEAR_EACH)}mm",
                            (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(debug_img, "q: quit", (10, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Parking Monitor", debug_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                    break

    except RPLidarException as e:
        print(f"⚠️ 라이다 오류: {e}")
    except KeyboardInterrupt:
        print("종료(KeyboardInterrupt)")
    except Exception as e:
        print(f"시스템 오류: {e}")
    finally:
        # 스레드 종료
        stop_event.set()
        try:
            if reader is not None:
                reader.join(timeout=0.5)
        except:
            pass

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

    '''
    // ==========================================
// [Arduino Mega] Python 폴링(U) + US:4채널 출력 + 6채널 측정
//  - S,### : 조향 목표값
//  - D,### : 구동 속도
//  - U     : 초음파 4개(LF,LR,RF,RR) 1줄 응답 ("US:...")
// ==========================================

// --- 핀 설정 ---
const int R_IN1 = 3;  const int R_IN2 = 4;
const int L_IN1 = 11; const int L_IN2 = 10; const int L_PWM_POWER = 12;
const int POT_PIN = A0;
const int POT_VCC_PIN = 13;
const int STR_IN1 = 8; const int STR_IN2 = 9; const int STR_PWM = 5;

// --- 6채널 초음파(측정은 6개 모두) ---
const int SENSOR_COUNT = 6;
int trigPins[SENSOR_COUNT] = {22, 25, 26, 28, 31, 51};
int echoPins[SENSOR_COUNT] = {23, 24, 27, 29, 30, 50};

// ✅ Python이 기대하는 4채널(LF,LR,RF,RR)을 6채널 중 어떤 인덱스로 매핑할지
// TODO: 실제 센서 위치에 맞게 수정하세요.
const int IDX_LF = 0; // Left-Front
const int IDX_LR = 1; // Left-Rear
const int IDX_RF = 4; // Right-Front
const int IDX_RR = 5; // Right-Rear

long sonarValues[SENSOR_COUNT] = {999,999,999,999,999,999};
int currentSensorIdx = 0;
unsigned long lastSonarMeasureTime = 0;

// --- 제어 변수 ---
const int CENTER_VAL = 570;
const int LIMIT_MIN = 480; const int LIMIT_MAX = 680;
const int STEER_MIN_SPEED = 90; const int STEER_MAX_SPEED = 200;
const int TOLERANCE = 15;

const unsigned long FAILSAFE_TIMEOUT = 1000;
unsigned long lastCmdTime = 0;

int targetAngle = CENTER_VAL;
bool autoSteering = false;

// ★ 메가 전용 512바이트 버퍼(너희 기존 구조 유지)
char serialBuffer[512];
int bufIdx = 0;

void setup() {
  Serial.begin(115200); // ✅ Python과 동일

  pinMode(R_IN1, OUTPUT); pinMode(R_IN2, OUTPUT);
  pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT); pinMode(L_PWM_POWER, OUTPUT);
  digitalWrite(L_PWM_POWER, HIGH);

  pinMode(POT_VCC_PIN, OUTPUT);
  digitalWrite(POT_VCC_PIN, HIGH);

  pinMode(STR_IN1, OUTPUT); pinMode(STR_IN2, OUTPUT); pinMode(STR_PWM, OUTPUT);

  for (int i = 0; i < SENSOR_COUNT; i++) {
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT);
    digitalWrite(trigPins[i], LOW);
  }

  stopRearMotors();
  stopSteering();
  lastCmdTime = millis();
}

void loop() {
  unsigned long now = millis();

  // ==================================================
  // 1) 통신 수신 (512B 버퍼) - S,D,U 처리
  // ==================================================
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      serialBuffer[bufIdx] = '\0';
      parseCommand(serialBuffer);
      bufIdx = 0;
    } else {
      if (bufIdx < 511) serialBuffer[bufIdx++] = c;
      else {
        // 오버플로우 방지: 버퍼 초기화 + 잔여 제거
        bufIdx = 0;
        while (Serial.available() > 0) Serial.read();
      }
    }
  }

  // ==================================================
  // 1.5) 페일세이프
  // ==================================================
  if (now - lastCmdTime > FAILSAFE_TIMEOUT) {
    stopRearMotors();
    stopSteering();
    autoSteering = false;
  } else {
    // ==================================================
    // 2) 자동 조향
    // ==================================================
    if (autoSteering) {
      int currentPot = analogRead(POT_PIN);
      int error = targetAngle - currentPot;

      if (abs(error) <= TOLERANCE) {
        stopSteering();
      } else {
        int speed = map(abs(error), 0, 100, STEER_MIN_SPEED, STEER_MAX_SPEED);
        speed = constrain(speed, STEER_MIN_SPEED, STEER_MAX_SPEED);
        if (error > 0) moveSteering(speed, true);
        else moveSteering(speed, false);
      }
    }
  }

  // ==================================================
  // 3) 초음파 측정 (라운드로빈: 20ms마다 1개씩)
  //    ✅ 측정은 계속하지만, 절대 자동 출력하지 않음!
  // ==================================================
  if (now - lastSonarMeasureTime >= 20) {
    long dist = getDistance(trigPins[currentSensorIdx], echoPins[currentSensorIdx]);
    sonarValues[currentSensorIdx] = dist;

    currentSensorIdx++;
    if (currentSensorIdx >= SENSOR_COUNT) currentSensorIdx = 0;

    lastSonarMeasureTime = now;
  }
}

// ==================================================
// [명령 처리]
//  - S,### / D,### : lastCmdTime 갱신 (페일세이프)
//  - U             : US:LF,LR,RF,RR 1줄 응답 (lastCmdTime 갱신 X)
// ==================================================
void parseCommand(char* cmdStr) {
  if (cmdStr == NULL || cmdStr[0] == '\0') return;

  // ✅ U 폴링 지원 (Python 코드가 ser.write(b"U\n")를 보냄)
  // cmdStr 길이가 1이어도 처리해야 하므로 strlen 체크 전에 먼저 처리
  if (cmdStr[0] == 'U') {
    sendUltrasonic4();   // 1줄만 출력
    return;
  }

  // S,### / D,### 형태 최소 길이
  if (strlen(cmdStr) < 3) return;

  char cmdType = cmdStr[0];
  char* valStr = strchr(cmdStr, ',');
  if (valStr == NULL) return;
  valStr++;

  int val = atoi(valStr);

  if (cmdType == 'S') {
    targetAngle = constrain(val, LIMIT_MIN, LIMIT_MAX);
    autoSteering = true;
    lastCmdTime = millis();
  }
  else if (cmdType == 'D') {
    moveRearMotors(val);
    lastCmdTime = millis();
  }
}

// ==================================================
// [초음파 4채널 출력: Python 포맷 고정]
// Python이 기대: "US:LF,LR,RF,RR"
// ==================================================
void sendUltrasonic4() {
  Serial.print("US:");
  Serial.print(sonarValues[IDX_LF]); Serial.print(",");
  Serial.print(sonarValues[IDX_LR]); Serial.print(",");
  Serial.print(sonarValues[IDX_RF]); Serial.print(",");
  Serial.print(sonarValues[IDX_RR]);
  Serial.println();
}

// ==================================================
// [초음파 거리 측정]
// ==================================================
long getDistance(int trig, int echo) {
  digitalWrite(trig, LOW);  delayMicroseconds(2);
  digitalWrite(trig, HIGH); delayMicroseconds(10);
  digitalWrite(trig, LOW);

  // ✅ 너희 기존 코드는 4000us였는데 너무 짧아서 999가 자주 뜸
  // Python이 300mm 같은 근거리만 쓸 거라도 튐이 심해져서 20000 권장
  unsigned long duration = pulseIn(echo, HIGH, 20000);

  if (duration == 0) return 999; // 타임아웃
  return (long)(duration * 0.034 / 2.0 * 10.0); // mm
}

// ==================================================
// [모터/조향 유틸]
// ==================================================
void moveSteering(int speed, bool increaseValue) {
  analogWrite(STR_PWM, speed);
  if (increaseValue) { digitalWrite(STR_IN1, LOW);  digitalWrite(STR_IN2, HIGH); }
  else               { digitalWrite(STR_IN1, HIGH); digitalWrite(STR_IN2, LOW);  }
}

void stopSteering() {
  analogWrite(STR_PWM, 0);
  digitalWrite(STR_IN1, LOW);
  digitalWrite(STR_IN2, LOW);
}

void stopRearMotors() {
  analogWrite(R_IN1, 0); analogWrite(R_IN2, 0);
  analogWrite(L_IN1, 0); analogWrite(L_IN2, 0);
}

void moveRearMotors(int speed) {
  if (speed == 0) { stopRearMotors(); return; }

  if (speed > 0) {
    speed = constrain(speed, 0, 255);
    analogWrite(R_IN1, speed); analogWrite(R_IN2, 0);
    analogWrite(L_IN1, speed); analogWrite(L_IN2, 0);
  } else {
    int revSpeed = constrain(abs(speed), 0, 255);
    analogWrite(R_IN1, 0); analogWrite(R_IN2, revSpeed);
    analogWrite(L_IN1, 0); analogWrite(L_IN2, revSpeed);
  }
}

    '''