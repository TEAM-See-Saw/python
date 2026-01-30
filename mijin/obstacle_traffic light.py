<<<<<<< HEAD
import cv2
import numpy as np
import math
import serial
import time
import datetime
import os
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List


# =========================================================
# [0] 환경/하드웨어 설정 (여기만 튜닝하세요)
# =========================================================
IS_SUNNY = True

# Arduino
PORT = 'COM4'
BAUDRATE = 9600

# Camera
CAM_INDEX = 0

# Motor / Servo
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# LiDAR (RPLidar 가정)
# Windows면 보통 COM5/COM6 등. Linux면 /dev/ttyUSB0
LIDAR_PORT = "COM5"
ENABLE_LIDAR = True  # 장치 없으면 False로 두면 코드가 안전정지 쪽으로만 동작

# 초음파 수신 포맷 기본 가정: "U,<left_cm>,<right_cm>"
# 다르면 아래 _parse_ultrasonic_line()에서 수정하세요.


# =========================================================
# [1] 모드별(햇빛/일반) 튜닝값
# =========================================================
if IS_SUNNY:
    print("☀️ [모드: SUNNY] 강력한 햇빛 대응 설정 적용")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print("🌙 [모드: NORMAL] 저녁/실내 설정 적용")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)


# =========================================================
# [2] 미션/스케줄/임계값 설정
# =========================================================
@dataclass
class MissionConfig:
    # 제어 갱신 주기(카메라 프레임은 더 빠르게 돌더라도, D/S 송신은 주기로 제한 가능)
    CONTROL_PERIOD_SEC: float = 0.18  # 180ms 근사
    # LiDAR 섹터 각도(정면 기준)
    SECTOR_FL: Tuple[float, float] = (10.0, 35.0)      # front-left
    SECTOR_FR: Tuple[float, float] = (-35.0, -10.0)    # front-right

    # 장애물 판단 임계값(m)
    T_OBS_ENTER: float = 1.10     # 장애물 접근(회피/감속 트리거)
    T_OBS_CLEAR: float = 1.50     # 장애물 통과(클리어 판정)
    T_EMERGENCY: float = 0.45     # 충돌 임박 즉시 정지

    # 초음파 임계값(cm) - 측면 가드
    US_WARN_CM: float = 20.0
    US_STOP_CM: float = 13.0

    # 레인 목표 bias (화면 폭 기준 비율)
    # lane2(우측)로 유지하려면 +bias, lane1(좌측)로 가려면 -bias
    LANE_BIAS_RATIO: float = 0.12

    # 차선 변경 시 bias를 서서히 바꾸는 속도(프레임당 변화량)
    BIAS_SLEW_PER_FRAME: float = 0.01  # 0.01이면 약 12프레임에 0.12 이동

    # 커브에서 안정화를 위해 조향각이 커지면 속도 감속
    CURVE_SLOW_ANGLE_DEG: float = 18.0
    CURVE_SLOW_FACTOR: float = 0.65

    # 신호등 ROI 및 판정
    TL_ROI_Y1_RATIO: float = 0.05
    TL_ROI_Y2_RATIO: float = 0.25
    TL_ROI_X1_RATIO: float = 0.40
    TL_ROI_X2_RATIO: float = 0.60
    TL_MIN_PIXELS: int = 250      # 색 픽셀 최소 개수(환경 따라 튜닝)
    TL_STABLE_FRAMES: int = 5     # 연속 프레임 안정화

    # 장애물 stage 전환 안정화(연속 클리어 프레임)
    CLEAR_STABLE_FRAMES: int = 4

CFG = MissionConfig()


# =========================================================
# [3] 시리얼 연결(Arduino)
# =========================================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    print("⚠️ 주의: 영상 처리만 진행됩니다.")


# =========================================================
# [4] Lane tracing: 기존 함수 유지 + "레인 bias"만 추가 적용
# =========================================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering_angle_with_lane_bias(image, left_line, right_line, lane_bias_px: float):
    """
    기존 calculate_steering_angle에서 target_x를 잡는 방식은 유지하되,
    lane_bias_px(픽셀)을 더해 "목표 레인 중심"으로 이동시킵니다.
    """
    height, width, _ = image.shape
    car_position_x = width / 2

    if left_line is not None and right_line is not None:
        road_center_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        road_center_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        road_center_x = right_line[0][2] - (width * 0.25)
    else:
        road_center_x = car_position_x

    target_x = road_center_x + lane_bias_px

    dx = target_x - car_position_x
    dy = (height * 0.6) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# =========================================================
# [5] Ultrasonic 수신(Arduino -> PC): "U,left,right"
# =========================================================
class UltrasonicState:
    def __init__(self):
        self.left_cm: Optional[float] = None
        self.right_cm: Optional[float] = None
        self.ts: float = 0.0

US = UltrasonicState()

def _parse_ultrasonic_line(line: str) -> Optional[Tuple[float, float]]:
    """
    기본 기대 포맷: U,<left_cm>,<right_cm>
    예) U,35.2,33.8

    아두이노 출력이 다르면 여기만 맞추면 됩니다.
    """
    if not line.startswith("U,"):
        return None
    parts = line.split(",")
    if len(parts) < 3:
        return None
    try:
        l = float(parts[1])
        r = float(parts[2])
        return l, r
    except:
        return None

def poll_ultrasonic_nonblock():
    if not ser:
        return
    # 짧게 여러 줄 읽어서 최신값으로 갱신
    for _ in range(6):
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except:
            line = ""
        if not line:
            break
        parsed = _parse_ultrasonic_line(line)
        if parsed:
            US.left_cm, US.right_cm = parsed
            US.ts = time.time()


# =========================================================
# [6] LiDAR: 섹터 기반 최소거리(포인트클라우드 전체처리 X)
# =========================================================
class LidarScan:
    def __init__(self, points: List[Tuple[float, float]], ts: float):
        # points: (angle_deg, distance_m)
        self.points = points
        self.ts = ts

class LidarReader(threading.Thread):
    def __init__(self, port: str, enabled: bool = True):
        super().__init__(daemon=True)
        self.port = port
        self.enabled = enabled
        self._latest: Optional[LidarScan] = None
        self._lock = threading.Lock()
        self._stop = False

    def run(self):
        if not self.enabled:
            return
        try:
            from rplidar import RPLidar
        except Exception as e:
            print(f"[WARN] rplidar import 실패: {e} -> LiDAR 비활성화")
            self.enabled = False
            return

        lidar = None
        try:
            lidar = RPLidar(self.port)
            for scan in lidar.iter_scans():
                if self._stop:
                    break
                pts = []
                for q, angle, dist_mm in scan:
                    if dist_mm <= 0:
                        continue
                    dist_m = dist_mm / 1000.0
                    # 이상치 제거
                    if 0.05 < dist_m < 8.0:
                        pts.append((angle, dist_m))
                with self._lock:
                    self._latest = LidarScan(pts, time.time())
        except Exception as e:
            print(f"[WARN] LiDAR 런타임 오류: {e} -> LiDAR 비활성화")
            self.enabled = False
        finally:
            try:
                if lidar:
                    lidar.stop()
                    lidar.disconnect()
            except:
                pass

    def stop(self):
        self._stop = True

    def get_latest(self) -> Optional[LidarScan]:
        with self._lock:
            return self._latest

def _norm_angle(a: float) -> float:
    return a % 360.0

def _in_sector(angle: float, start: float, end: float) -> bool:
    a = _norm_angle(angle)
    s = _norm_angle(start)
    e = _norm_angle(end)
    if s <= e:
        return s <= a <= e
    else:
        return a >= s or a <= e

def sector_min_distance(scan: Optional[LidarScan], sector: Tuple[float, float]) -> Optional[float]:
    if scan is None:
        return None
    start, end = sector
    m = None
    for ang, dist in scan.points:
        if _in_sector(ang, start, end):
            if m is None or dist < m:
                m = dist
    return m


# =========================================================
# [7] 신호등 인지(카메라): HSV 기반 간단 RED/GREEN 판정
# =========================================================
class TLState(Enum):
    UNKNOWN = auto()
    RED = auto()
    GREEN = auto()

class TrafficLightDetector:
    def __init__(self):
        self.red_cnt = 0
        self.green_cnt = 0
        self.state = TLState.UNKNOWN

    def update(self, frame_bgr: np.ndarray) -> TLState:
        h, w, _ = frame_bgr.shape
        y1 = int(h * CFG.TL_ROI_Y1_RATIO)
        y2 = int(h * CFG.TL_ROI_Y2_RATIO)
        x1 = int(w * CFG.TL_ROI_X1_RATIO)
        x2 = int(w * CFG.TL_ROI_X2_RATIO)

        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # red: hue wrap-around
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([179, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # green
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = int(cv2.countNonZero(red_mask))
        green_pixels = int(cv2.countNonZero(green_mask))

        # 안정화 카운터
        if red_pixels > CFG.TL_MIN_PIXELS and red_pixels > green_pixels:
            self.red_cnt += 1
            self.green_cnt = max(0, self.green_cnt - 1)
        elif green_pixels > CFG.TL_MIN_PIXELS and green_pixels > red_pixels:
            self.green_cnt += 1
            self.red_cnt = max(0, self.red_cnt - 1)
        else:
            # 애매하면 서서히 감소
            self.red_cnt = max(0, self.red_cnt - 1)
            self.green_cnt = max(0, self.green_cnt - 1)

        if self.red_cnt >= CFG.TL_STABLE_FRAMES:
            self.state = TLState.RED
        elif self.green_cnt >= CFG.TL_STABLE_FRAMES:
            self.state = TLState.GREEN
        else:
            self.state = TLState.UNKNOWN

        return self.state


# =========================================================
# [8] 장애물&레인 변경 Stage FSM (1 -> 2 -> 1 순서 반영)
# =========================================================
class Stage(Enum):
    STAGE0_FIRST_OBS_IN_LANE1 = auto()   # 시작: 2차선 주행, 1차선 장애물은 변경 없이 통과
    STAGE1_OBS_IN_LANE2_CHANGE_TO_1 = auto()  # 2차선 장애물 감지 -> 2->1
    STAGE2_OBS_IN_LANE1_RETURN_TO_2 = auto()  # 1차선 장애물 감지 -> 1->2
    STAGE3_TRAFFIC_LIGHT = auto()        # 장애물 완료 후 신호등 미션

class MissionFSM:
    def __init__(self):
        self.stage = Stage.STAGE0_FIRST_OBS_IN_LANE1
        self.lane = 2  # 규정: 출발 2차선 고정
        self.clear_stable = 0
        self.seen_first_lane1_obs = False
        self.seen_lane2_obs = False
        self.seen_second_lane1_obs = False

        # lane bias는 "서서히" 움직임(급조향 방지)
        self.current_bias_ratio = +CFG.LANE_BIAS_RATIO  # lane2 시작(우측)
        self.target_bias_ratio = +CFG.LANE_BIAS_RATIO

        # 신호등 정지/재출발을 위한 내부 속도
        self.current_speed = 0

    def _set_lane(self, lane: int):
        self.lane = lane
        if lane == 2:
            self.target_bias_ratio = +CFG.LANE_BIAS_RATIO
        else:
            self.target_bias_ratio = -CFG.LANE_BIAS_RATIO

    def _slew_bias(self):
        # 목표 bias로 천천히 이동
        if self.current_bias_ratio < self.target_bias_ratio:
            self.current_bias_ratio = min(self.target_bias_ratio, self.current_bias_ratio + CFG.BIAS_SLEW_PER_FRAME)
        elif self.current_bias_ratio > self.target_bias_ratio:
            self.current_bias_ratio = max(self.target_bias_ratio, self.current_bias_ratio - CFG.BIAS_SLEW_PER_FRAME)

    def step(
        self,
        d_fl: Optional[float],
        d_fr: Optional[float],
        us_left: Optional[float],
        us_right: Optional[float],
        tl_state: TLState,
        angle_deg: float
    ) -> Tuple[int, int, float]:
        """
        return: (speed_cmd, servo_cmd, lane_bias_ratio)
        """
        # 1) bias 서서히 이동
        self._slew_bias()

        # 2) 라이다 기반 "어느 차선 장애물인가" 판단(좌=1차선, 우=2차선 가정)
        obs_lane1 = (d_fl is not None and d_fl < CFG.T_OBS_ENTER) and (d_fr is None or d_fl + 0.10 < d_fr)
        obs_lane2 = (d_fr is not None and d_fr < CFG.T_OBS_ENTER) and (d_fl is None or d_fr + 0.10 < d_fl)

        # 3) 긴급정지 조건
        emergency = (d_fl is not None and d_fl < CFG.T_EMERGENCY) or (d_fr is not None and d_fr < CFG.T_EMERGENCY)
        if (us_left is not None and us_left < CFG.US_STOP_CM) or (us_right is not None and us_right < CFG.US_STOP_CM):
            emergency = True

        if emergency:
            self.current_speed = 0
            return 0, SERVO_CENTER, self.current_bias_ratio

        # 4) 기본 속도 정책(커브 감속)
        base_speed = MAX_SPEED
        if abs(angle_deg) > CFG.CURVE_SLOW_ANGLE_DEG:
            base_speed = int(MAX_SPEED * CFG.CURVE_SLOW_FACTOR)

        # 5) Stage FSM
        if self.stage == Stage.STAGE0_FIRST_OBS_IN_LANE1:
            # 출발은 lane2 유지 고정
            self._set_lane(2)

            # 1차선 장애물은 "차선 변경 없이" 통과: lane2 유지 강화 + 커브면 감속
            if obs_lane1:
                self.seen_first_lane1_obs = True
                base_speed = min(base_speed, int(MAX_SPEED * 0.75))

            # 1차선 장애물 "클리어" 판정: 봤던 장애물이 사라졌다고 일정 프레임 확인되면 stage1로
            if self.seen_first_lane1_obs:
                cleared = (d_fl is None) or (d_fl > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE1_OBS_IN_LANE2_CHANGE_TO_1

        elif self.stage == Stage.STAGE1_OBS_IN_LANE2_CHANGE_TO_1:
            # 여전히 기본은 lane2로 달리다가
            # 2차선 장애물 감지되면 2->1로 변경
            if obs_lane2:
                self.seen_lane2_obs = True
                # 차선 변경 수행(좌측으로)
                # 초음파 가드: 좌측이 너무 가까우면 감속
                if us_left is not None and us_left < CFG.US_WARN_CM:
                    base_speed = int(MAX_SPEED * 0.55)
                self._set_lane(1)

            # lane1로 충분히 넘어가고 장애물이 클리어되면 다음 단계
            if self.seen_lane2_obs and self.lane == 1:
                cleared = (d_fr is None) or (d_fr > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE2_OBS_IN_LANE1_RETURN_TO_2

        elif self.stage == Stage.STAGE2_OBS_IN_LANE1_RETURN_TO_2:
            # 현재 lane1 주행 중, lane1 장애물(좌측) 나오면 1->2 복귀
            if obs_lane1:
                self.seen_second_lane1_obs = True
                # 차선 변경 수행(우측으로)
                # 초음파 가드: 우측이 가까우면 감속
                if us_right is not None and us_right < CFG.US_WARN_CM:
                    base_speed = int(MAX_SPEED * 0.55)
                self._set_lane(2)

            # lane2로 복귀하고 장애물이 클리어되면 신호등 단계로
            if self.seen_second_lane1_obs and self.lane == 2:
                cleared = (d_fl is None) or (d_fl > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE3_TRAFFIC_LIGHT

        elif self.stage == Stage.STAGE3_TRAFFIC_LIGHT:
            # 신호등 미션: RED면 정지, GREEN면 진행
            self._set_lane(2)  # 신호등 구간은 lane2 유지로 고정(규정상 안전)

            if tl_state == TLState.RED:
                # 램프다운 정지 (급정지 방지)
                self.current_speed = max(0, self.current_speed - 45)
                return self.current_speed, None, self.current_bias_ratio
            elif tl_state == TLState.GREEN:
                # 출발
                self.current_speed = min(base_speed, max(self.current_speed + 40, int(MAX_SPEED * 0.65)))
            else:
                # UNKNOWN이면 보수적으로 속도 유지/감속
                self.current_speed = min(self.current_speed, base_speed)

            return self.current_speed, None, self.current_bias_ratio

        # Stage0~2 공통: 속도는 base_speed로 따라가되 내부 speed도 갱신
        self.current_speed = base_speed
        return self.current_speed, None, self.current_bias_ratio


# =========================================================
# [9] main()
# =========================================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    # 녹화 설정(원본+마스크 붙여서 저장)
    # 저장할 폴더 생성
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # 파일명 생성 (예: dataset/autodrive_20260122_143000.mp4)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset/autodrive_{now}.mp4"

    # 코덱 설정 (H.264 추천, 실패 시 mp4v)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 20.0
    # 주의: 우리는 원본(640) + 마스크(640)를 붙여서 저장할 것이므로 너비는 1280
    out = cv2.VideoWriter(filename, fourcc, fps, (width * 2, height))

    print(f"🎥 녹화 준비 완료: {filename}")

    # LiDAR thread
    lidar = LidarReader(LIDAR_PORT, enabled=ENABLE_LIDAR)
    lidar.start()

    # FSMs
    mission = MissionFSM()
    tl = TrafficLightDetector()

    # 안전 카운트다운
    print("\n" + "=" * 30)
    print(f"🚀 출발 준비 (MAX_SPEED={MAX_SPEED})")
    print("=" * 30)
    for i in range(3, 0, -1):
        print(f"Count: {i}")
        time.sleep(1)
    print("GO!!!")

    # 출발 명령
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())
        time.sleep(0.1)

    last_control_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # 1) Lane tracing 전처리(기존과 동일)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN, 0])
        upper_white = np.array([179, 255, S_MAX])
        mask = cv2.inRange(hls, lower_white, upper_white)

        kernel = np.ones(MORPH_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(mask, 50, 150)

        roi_vertices = [
            (0, height),
            (width // 2 - 50, int(height * 0.6)),
            (width // 2 + 50, int(height * 0.6)),
            (width, height)
        ]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)

        # 2) 초음파 수신(비블로킹)
        poll_ultrasonic_nonblock()
        us_l, us_r = US.left_cm, US.right_cm

        # 3) LiDAR 섹터 거리 계산
        scan = lidar.get_latest()
        d_fl = sector_min_distance(scan, CFG.SECTOR_FL)
        d_fr = sector_min_distance(scan, CFG.SECTOR_FR)

        # 4) 신호등 상태 갱신(항상 업데이트, stage3에서만 실제로 사용)
        tl_state = tl.update(frame)

        # 5) 카메라 기반 조향: 레인 bias를 mission이 제공
        lane_bias_px = mission.current_bias_ratio * width
        angle, target_x = calculate_steering_angle_with_lane_bias(frame, left, right, lane_bias_px)

        # 조향각->서보 변환(기존과 동일)
        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 6) 제어 주기(180ms)마다 미션 FSM 업데이트 후 속도/레인 목표 반영
        now_ts = time.time()
        if now_ts - last_control_ts >= CFG.CONTROL_PERIOD_SEC:
            last_control_ts = now_ts

            speed_cmd, servo_override, bias_ratio = mission.step(
                d_fl=d_fl,
                d_fr=d_fr,
                us_left=us_l,
                us_right=us_r,
                tl_state=tl_state,
                angle_deg=angle
            )

            # bias_ratio를 mission 내부에 반영(다음 프레임부터 적용되도록)
            mission.target_bias_ratio = bias_ratio  # (slew는 step에서 진행)

            # 신호등 RED 등으로 servo를 강제로 센터로 두고 싶으면 여기서 override 가능
            if servo_override is not None:
                servo_val = servo_override

            # 시리얼 송신
            if ser:
                ser.write(f"S,{servo_val}\n".encode())
                ser.write(f"D,{speed_cmd}\n".encode())

        # 7) 디버깅 화면/녹화
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))

        # 상태 출력 텍스트
        info = (
            f"Mode:{'SUNNY' if IS_SUNNY else 'NORMAL'} "
            f"Stage:{mission.stage.name} Lane:{mission.lane} "
            f"Servo:{servo_val} "
            f"dFL:{'None' if d_fl is None else round(d_fl,2)} "
            f"dFR:{'None' if d_fr is None else round(d_fr,2)} "
            f"USL:{us_l} USR:{us_r} "
            f"TL:{tl_state.name}"
        )
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        out.write(combined)
        cv2.imshow("Autonomous Driving (Lane+Obstacle+TL)", combined)

        # q 누르면 긴급 정지
        if cv2.waitKey(1) == ord('q'):
            print("🛑 긴급 정지!")
            break

    # 종료 절차
    try:
        if ser:
            ser.write(b"D,0\n")
            time.sleep(0.1)
            ser.write(f"S,{SERVO_CENTER}\n".encode())
            ser.close()
    except:
        pass

    try:
        lidar.stop()
    except:
        pass

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("💾 녹화 완료 및 프로그램 종료")


if __name__ == "__main__":
    main()
=======
import cv2
import numpy as np
import math
import serial
import time
import datetime
import os
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List


# =========================================================
# [0] 환경/하드웨어 설정 (여기만 튜닝하세요)
# =========================================================
IS_SUNNY = True

# Arduino
PORT = 'COM4'
BAUDRATE = 9600

# Camera
CAM_INDEX = 0

# Motor / Servo
MAX_SPEED = 255
SERVO_CENTER = 570
SERVO_LEFT_MAX = 680
SERVO_RIGHT_MAX = 480

# LiDAR (RPLidar 가정)
# Windows면 보통 COM5/COM6 등. Linux면 /dev/ttyUSB0
LIDAR_PORT = "COM5"
ENABLE_LIDAR = True  # 장치 없으면 False로 두면 코드가 안전정지 쪽으로만 동작

# 초음파 수신 포맷 기본 가정: "U,<left_cm>,<right_cm>"
# 다르면 아래 _parse_ultrasonic_line()에서 수정하세요.


# =========================================================
# [1] 모드별(햇빛/일반) 튜닝값
# =========================================================
if IS_SUNNY:
    print("☀️ [모드: SUNNY] 강력한 햇빛 대응 설정 적용")
    EXPOSURE = -9
    L_MIN = 160
    S_MAX = 50
    MORPH_SIZE = (5, 5)
else:
    print("🌙 [모드: NORMAL] 저녁/실내 설정 적용")
    EXPOSURE = -4
    L_MIN = 100
    S_MAX = 60
    MORPH_SIZE = (3, 3)


# =========================================================
# [2] 미션/스케줄/임계값 설정
# =========================================================
@dataclass
class MissionConfig:
    # 제어 갱신 주기(카메라 프레임은 더 빠르게 돌더라도, D/S 송신은 주기로 제한 가능)
    CONTROL_PERIOD_SEC: float = 0.18  # 180ms 근사
    # LiDAR 섹터 각도(정면 기준)
    SECTOR_FL: Tuple[float, float] = (10.0, 35.0)      # front-left
    SECTOR_FR: Tuple[float, float] = (-35.0, -10.0)    # front-right

    # 장애물 판단 임계값(m)
    T_OBS_ENTER: float = 1.10     # 장애물 접근(회피/감속 트리거)
    T_OBS_CLEAR: float = 1.50     # 장애물 통과(클리어 판정)
    T_EMERGENCY: float = 0.45     # 충돌 임박 즉시 정지

    # 초음파 임계값(cm) - 측면 가드
    US_WARN_CM: float = 20.0
    US_STOP_CM: float = 13.0

    # 레인 목표 bias (화면 폭 기준 비율)
    # lane2(우측)로 유지하려면 +bias, lane1(좌측)로 가려면 -bias
    LANE_BIAS_RATIO: float = 0.12

    # 차선 변경 시 bias를 서서히 바꾸는 속도(프레임당 변화량)
    BIAS_SLEW_PER_FRAME: float = 0.01  # 0.01이면 약 12프레임에 0.12 이동

    # 커브에서 안정화를 위해 조향각이 커지면 속도 감속
    CURVE_SLOW_ANGLE_DEG: float = 18.0
    CURVE_SLOW_FACTOR: float = 0.65

    # 신호등 ROI 및 판정
    TL_ROI_Y1_RATIO: float = 0.05
    TL_ROI_Y2_RATIO: float = 0.25
    TL_ROI_X1_RATIO: float = 0.40
    TL_ROI_X2_RATIO: float = 0.60
    TL_MIN_PIXELS: int = 250      # 색 픽셀 최소 개수(환경 따라 튜닝)
    TL_STABLE_FRAMES: int = 5     # 연속 프레임 안정화

    # 장애물 stage 전환 안정화(연속 클리어 프레임)
    CLEAR_STABLE_FRAMES: int = 4

CFG = MissionConfig()


# =========================================================
# [3] 시리얼 연결(Arduino)
# =========================================================
ser = None
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.01)
    print(f"✅ {PORT} 포트 연결 성공! (2초 대기)")
    time.sleep(2)
except Exception as e:
    print(f"❌ 아두이노 연결 실패: {e}")
    print("⚠️ 주의: 영상 처리만 진행됩니다.")


# =========================================================
# [4] Lane tracing: 기존 함수 유지 + "레인 bias"만 추가 적용
# =========================================================
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def make_points(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    if slope == 0:
        slope = 0.001
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))
    left_line = make_points(image, np.mean(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = make_points(image, np.mean(right_fit, axis=0)) if len(right_fit) > 0 else None
    return left_line, right_line

def calculate_steering_angle_with_lane_bias(image, left_line, right_line, lane_bias_px: float):
    """
    기존 calculate_steering_angle에서 target_x를 잡는 방식은 유지하되,
    lane_bias_px(픽셀)을 더해 "목표 레인 중심"으로 이동시킵니다.
    """
    height, width, _ = image.shape
    car_position_x = width / 2

    if left_line is not None and right_line is not None:
        road_center_x = (left_line[0][2] + right_line[0][2]) / 2
    elif left_line is not None:
        road_center_x = left_line[0][2] + (width * 0.25)
    elif right_line is not None:
        road_center_x = right_line[0][2] - (width * 0.25)
    else:
        road_center_x = car_position_x

    target_x = road_center_x + lane_bias_px

    dx = target_x - car_position_x
    dy = (height * 0.6) - height
    angle_deg = math.degrees(math.atan2(dx, abs(dy)))
    return angle_deg, int(target_x)

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


# =========================================================
# [5] Ultrasonic 수신(Arduino -> PC): "U,left,right"
# =========================================================
class UltrasonicState:
    def __init__(self):
        self.left_cm: Optional[float] = None
        self.right_cm: Optional[float] = None
        self.ts: float = 0.0

US = UltrasonicState()

def _parse_ultrasonic_line(line: str) -> Optional[Tuple[float, float]]:
    """
    기본 기대 포맷: U,<left_cm>,<right_cm>
    예) U,35.2,33.8

    아두이노 출력이 다르면 여기만 맞추면 됩니다.
    """
    if not line.startswith("U,"):
        return None
    parts = line.split(",")
    if len(parts) < 3:
        return None
    try:
        l = float(parts[1])
        r = float(parts[2])
        return l, r
    except:
        return None

def poll_ultrasonic_nonblock():
    if not ser:
        return
    # 짧게 여러 줄 읽어서 최신값으로 갱신
    for _ in range(6):
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except:
            line = ""
        if not line:
            break
        parsed = _parse_ultrasonic_line(line)
        if parsed:
            US.left_cm, US.right_cm = parsed
            US.ts = time.time()


# =========================================================
# [6] LiDAR: 섹터 기반 최소거리(포인트클라우드 전체처리 X)
# =========================================================
class LidarScan:
    def __init__(self, points: List[Tuple[float, float]], ts: float):
        # points: (angle_deg, distance_m)
        self.points = points
        self.ts = ts

class LidarReader(threading.Thread):
    def __init__(self, port: str, enabled: bool = True):
        super().__init__(daemon=True)
        self.port = port
        self.enabled = enabled
        self._latest: Optional[LidarScan] = None
        self._lock = threading.Lock()
        self._stop = False

    def run(self):
        if not self.enabled:
            return
        try:
            from rplidar import RPLidar
        except Exception as e:
            print(f"[WARN] rplidar import 실패: {e} -> LiDAR 비활성화")
            self.enabled = False
            return

        lidar = None
        try:
            lidar = RPLidar(self.port)
            for scan in lidar.iter_scans():
                if self._stop:
                    break
                pts = []
                for q, angle, dist_mm in scan:
                    if dist_mm <= 0:
                        continue
                    dist_m = dist_mm / 1000.0
                    # 이상치 제거
                    if 0.05 < dist_m < 8.0:
                        pts.append((angle, dist_m))
                with self._lock:
                    self._latest = LidarScan(pts, time.time())
        except Exception as e:
            print(f"[WARN] LiDAR 런타임 오류: {e} -> LiDAR 비활성화")
            self.enabled = False
        finally:
            try:
                if lidar:
                    lidar.stop()
                    lidar.disconnect()
            except:
                pass

    def stop(self):
        self._stop = True

    def get_latest(self) -> Optional[LidarScan]:
        with self._lock:
            return self._latest

def _norm_angle(a: float) -> float:
    return a % 360.0

def _in_sector(angle: float, start: float, end: float) -> bool:
    a = _norm_angle(angle)
    s = _norm_angle(start)
    e = _norm_angle(end)
    if s <= e:
        return s <= a <= e
    else:
        return a >= s or a <= e

def sector_min_distance(scan: Optional[LidarScan], sector: Tuple[float, float]) -> Optional[float]:
    if scan is None:
        return None
    start, end = sector
    m = None
    for ang, dist in scan.points:
        if _in_sector(ang, start, end):
            if m is None or dist < m:
                m = dist
    return m


# =========================================================
# [7] 신호등 인지(카메라): HSV 기반 간단 RED/GREEN 판정
# =========================================================
class TLState(Enum):
    UNKNOWN = auto()
    RED = auto()
    GREEN = auto()

class TrafficLightDetector:
    def __init__(self):
        self.red_cnt = 0
        self.green_cnt = 0
        self.state = TLState.UNKNOWN

    def update(self, frame_bgr: np.ndarray) -> TLState:
        h, w, _ = frame_bgr.shape
        y1 = int(h * CFG.TL_ROI_Y1_RATIO)
        y2 = int(h * CFG.TL_ROI_Y2_RATIO)
        x1 = int(w * CFG.TL_ROI_X1_RATIO)
        x2 = int(w * CFG.TL_ROI_X2_RATIO)

        roi = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # red: hue wrap-around
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([179, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # green
        lower_green = np.array([40, 60, 60])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = int(cv2.countNonZero(red_mask))
        green_pixels = int(cv2.countNonZero(green_mask))

        # 안정화 카운터
        if red_pixels > CFG.TL_MIN_PIXELS and red_pixels > green_pixels:
            self.red_cnt += 1
            self.green_cnt = max(0, self.green_cnt - 1)
        elif green_pixels > CFG.TL_MIN_PIXELS and green_pixels > red_pixels:
            self.green_cnt += 1
            self.red_cnt = max(0, self.red_cnt - 1)
        else:
            # 애매하면 서서히 감소
            self.red_cnt = max(0, self.red_cnt - 1)
            self.green_cnt = max(0, self.green_cnt - 1)

        if self.red_cnt >= CFG.TL_STABLE_FRAMES:
            self.state = TLState.RED
        elif self.green_cnt >= CFG.TL_STABLE_FRAMES:
            self.state = TLState.GREEN
        else:
            self.state = TLState.UNKNOWN

        return self.state


# =========================================================
# [8] 장애물&레인 변경 Stage FSM (1 -> 2 -> 1 순서 반영)
# =========================================================
class Stage(Enum):
    STAGE0_FIRST_OBS_IN_LANE1 = auto()   # 시작: 2차선 주행, 1차선 장애물은 변경 없이 통과
    STAGE1_OBS_IN_LANE2_CHANGE_TO_1 = auto()  # 2차선 장애물 감지 -> 2->1
    STAGE2_OBS_IN_LANE1_RETURN_TO_2 = auto()  # 1차선 장애물 감지 -> 1->2
    STAGE3_TRAFFIC_LIGHT = auto()        # 장애물 완료 후 신호등 미션

class MissionFSM:
    def __init__(self):
        self.stage = Stage.STAGE0_FIRST_OBS_IN_LANE1
        self.lane = 2  # 규정: 출발 2차선 고정
        self.clear_stable = 0
        self.seen_first_lane1_obs = False
        self.seen_lane2_obs = False
        self.seen_second_lane1_obs = False

        # lane bias는 "서서히" 움직임(급조향 방지)
        self.current_bias_ratio = +CFG.LANE_BIAS_RATIO  # lane2 시작(우측)
        self.target_bias_ratio = +CFG.LANE_BIAS_RATIO

        # 신호등 정지/재출발을 위한 내부 속도
        self.current_speed = 0

    def _set_lane(self, lane: int):
        self.lane = lane
        if lane == 2:
            self.target_bias_ratio = +CFG.LANE_BIAS_RATIO
        else:
            self.target_bias_ratio = -CFG.LANE_BIAS_RATIO

    def _slew_bias(self):
        # 목표 bias로 천천히 이동
        if self.current_bias_ratio < self.target_bias_ratio:
            self.current_bias_ratio = min(self.target_bias_ratio, self.current_bias_ratio + CFG.BIAS_SLEW_PER_FRAME)
        elif self.current_bias_ratio > self.target_bias_ratio:
            self.current_bias_ratio = max(self.target_bias_ratio, self.current_bias_ratio - CFG.BIAS_SLEW_PER_FRAME)

    def step(
        self,
        d_fl: Optional[float],
        d_fr: Optional[float],
        us_left: Optional[float],
        us_right: Optional[float],
        tl_state: TLState,
        angle_deg: float
    ) -> Tuple[int, int, float]:
        """
        return: (speed_cmd, servo_cmd, lane_bias_ratio)
        """
        # 1) bias 서서히 이동
        self._slew_bias()

        # 2) 라이다 기반 "어느 차선 장애물인가" 판단(좌=1차선, 우=2차선 가정)
        obs_lane1 = (d_fl is not None and d_fl < CFG.T_OBS_ENTER) and (d_fr is None or d_fl + 0.10 < d_fr)
        obs_lane2 = (d_fr is not None and d_fr < CFG.T_OBS_ENTER) and (d_fl is None or d_fr + 0.10 < d_fl)

        # 3) 긴급정지 조건
        emergency = (d_fl is not None and d_fl < CFG.T_EMERGENCY) or (d_fr is not None and d_fr < CFG.T_EMERGENCY)
        if (us_left is not None and us_left < CFG.US_STOP_CM) or (us_right is not None and us_right < CFG.US_STOP_CM):
            emergency = True

        if emergency:
            self.current_speed = 0
            return 0, SERVO_CENTER, self.current_bias_ratio

        # 4) 기본 속도 정책(커브 감속)
        base_speed = MAX_SPEED
        if abs(angle_deg) > CFG.CURVE_SLOW_ANGLE_DEG:
            base_speed = int(MAX_SPEED * CFG.CURVE_SLOW_FACTOR)

        # 5) Stage FSM
        if self.stage == Stage.STAGE0_FIRST_OBS_IN_LANE1:
            # 출발은 lane2 유지 고정
            self._set_lane(2)

            # 1차선 장애물은 "차선 변경 없이" 통과: lane2 유지 강화 + 커브면 감속
            if obs_lane1:
                self.seen_first_lane1_obs = True
                base_speed = min(base_speed, int(MAX_SPEED * 0.75))

            # 1차선 장애물 "클리어" 판정: 봤던 장애물이 사라졌다고 일정 프레임 확인되면 stage1로
            if self.seen_first_lane1_obs:
                cleared = (d_fl is None) or (d_fl > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE1_OBS_IN_LANE2_CHANGE_TO_1

        elif self.stage == Stage.STAGE1_OBS_IN_LANE2_CHANGE_TO_1:
            # 여전히 기본은 lane2로 달리다가
            # 2차선 장애물 감지되면 2->1로 변경
            if obs_lane2:
                self.seen_lane2_obs = True
                # 차선 변경 수행(좌측으로)
                # 초음파 가드: 좌측이 너무 가까우면 감속
                if us_left is not None and us_left < CFG.US_WARN_CM:
                    base_speed = int(MAX_SPEED * 0.55)
                self._set_lane(1)

            # lane1로 충분히 넘어가고 장애물이 클리어되면 다음 단계
            if self.seen_lane2_obs and self.lane == 1:
                cleared = (d_fr is None) or (d_fr > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE2_OBS_IN_LANE1_RETURN_TO_2

        elif self.stage == Stage.STAGE2_OBS_IN_LANE1_RETURN_TO_2:
            # 현재 lane1 주행 중, lane1 장애물(좌측) 나오면 1->2 복귀
            if obs_lane1:
                self.seen_second_lane1_obs = True
                # 차선 변경 수행(우측으로)
                # 초음파 가드: 우측이 가까우면 감속
                if us_right is not None and us_right < CFG.US_WARN_CM:
                    base_speed = int(MAX_SPEED * 0.55)
                self._set_lane(2)

            # lane2로 복귀하고 장애물이 클리어되면 신호등 단계로
            if self.seen_second_lane1_obs and self.lane == 2:
                cleared = (d_fl is None) or (d_fl > CFG.T_OBS_CLEAR)
                if cleared:
                    self.clear_stable += 1
                else:
                    self.clear_stable = 0
                if self.clear_stable >= CFG.CLEAR_STABLE_FRAMES:
                    self.clear_stable = 0
                    self.stage = Stage.STAGE3_TRAFFIC_LIGHT

        elif self.stage == Stage.STAGE3_TRAFFIC_LIGHT:
            # 신호등 미션: RED면 정지, GREEN면 진행
            self._set_lane(2)  # 신호등 구간은 lane2 유지로 고정(규정상 안전)

            if tl_state == TLState.RED:
                # 램프다운 정지 (급정지 방지)
                self.current_speed = max(0, self.current_speed - 45)
                return self.current_speed, None, self.current_bias_ratio
            elif tl_state == TLState.GREEN:
                # 출발
                self.current_speed = min(base_speed, max(self.current_speed + 40, int(MAX_SPEED * 0.65)))
            else:
                # UNKNOWN이면 보수적으로 속도 유지/감속
                self.current_speed = min(self.current_speed, base_speed)

            return self.current_speed, None, self.current_bias_ratio

        # Stage0~2 공통: 속도는 base_speed로 따라가되 내부 speed도 갱신
        self.current_speed = base_speed
        return self.current_speed, None, self.current_bias_ratio


# =========================================================
# [9] main()
# =========================================================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    # 녹화 설정(원본+마스크 붙여서 저장)
    # 저장할 폴더 생성
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # 파일명 생성 (예: dataset/autodrive_20260122_143000.mp4)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset/autodrive_{now}.mp4"

    # 코덱 설정 (H.264 추천, 실패 시 mp4v)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 20.0
    # 주의: 우리는 원본(640) + 마스크(640)를 붙여서 저장할 것이므로 너비는 1280
    out = cv2.VideoWriter(filename, fourcc, fps, (width * 2, height))

    print(f"🎥 녹화 준비 완료: {filename}")

    # LiDAR thread
    lidar = LidarReader(LIDAR_PORT, enabled=ENABLE_LIDAR)
    lidar.start()

    # FSMs
    mission = MissionFSM()
    tl = TrafficLightDetector()

    # 안전 카운트다운
    print("\n" + "=" * 30)
    print(f"🚀 출발 준비 (MAX_SPEED={MAX_SPEED})")
    print("=" * 30)
    for i in range(3, 0, -1):
        print(f"Count: {i}")
        time.sleep(1)
    print("GO!!!")

    # 출발 명령
    if ser:
        ser.write(f"D,{MAX_SPEED}\n".encode())
        time.sleep(0.1)

    last_control_ts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        # 1) Lane tracing 전처리(기존과 동일)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, L_MIN, 0])
        upper_white = np.array([179, 255, S_MAX])
        mask = cv2.inRange(hls, lower_white, upper_white)

        kernel = np.ones(MORPH_SIZE, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(mask, 50, 150)

        roi_vertices = [
            (0, height),
            (width // 2 - 50, int(height * 0.6)),
            (width // 2 + 50, int(height * 0.6)),
            (width, height)
        ]
        cropped = region_of_interest(edges, np.array([roi_vertices], np.int32))

        lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
        left, right = average_slope_intercept(frame, lines)

        # 2) 초음파 수신(비블로킹)
        poll_ultrasonic_nonblock()
        us_l, us_r = US.left_cm, US.right_cm

        # 3) LiDAR 섹터 거리 계산
        scan = lidar.get_latest()
        d_fl = sector_min_distance(scan, CFG.SECTOR_FL)
        d_fr = sector_min_distance(scan, CFG.SECTOR_FR)

        # 4) 신호등 상태 갱신(항상 업데이트, stage3에서만 실제로 사용)
        tl_state = tl.update(frame)

        # 5) 카메라 기반 조향: 레인 bias를 mission이 제공
        lane_bias_px = mission.current_bias_ratio * width
        angle, target_x = calculate_steering_angle_with_lane_bias(frame, left, right, lane_bias_px)

        # 조향각->서보 변환(기존과 동일)
        servo_val = int(map_value(max(-45, min(45, angle)), -45, 45, SERVO_LEFT_MAX, SERVO_RIGHT_MAX))

        # 6) 제어 주기(180ms)마다 미션 FSM 업데이트 후 속도/레인 목표 반영
        now_ts = time.time()
        if now_ts - last_control_ts >= CFG.CONTROL_PERIOD_SEC:
            last_control_ts = now_ts

            speed_cmd, servo_override, bias_ratio = mission.step(
                d_fl=d_fl,
                d_fr=d_fr,
                us_left=us_l,
                us_right=us_r,
                tl_state=tl_state,
                angle_deg=angle
            )

            # bias_ratio를 mission 내부에 반영(다음 프레임부터 적용되도록)
            mission.target_bias_ratio = bias_ratio  # (slew는 step에서 진행)

            # 신호등 RED 등으로 servo를 강제로 센터로 두고 싶으면 여기서 override 가능
            if servo_override is not None:
                servo_val = servo_override

            # 시리얼 송신
            if ser:
                ser.write(f"S,{servo_val}\n".encode())
                ser.write(f"D,{speed_cmd}\n".encode())

        # 7) 디버깅 화면/녹화
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((frame, mask_bgr))

        # 상태 출력 텍스트
        info = (
            f"Mode:{'SUNNY' if IS_SUNNY else 'NORMAL'} "
            f"Stage:{mission.stage.name} Lane:{mission.lane} "
            f"Servo:{servo_val} "
            f"dFL:{'None' if d_fl is None else round(d_fl,2)} "
            f"dFR:{'None' if d_fr is None else round(d_fr,2)} "
            f"USL:{us_l} USR:{us_r} "
            f"TL:{tl_state.name}"
        )
        cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        out.write(combined)
        cv2.imshow("Autonomous Driving (Lane+Obstacle+TL)", combined)

        # q 누르면 긴급 정지
        if cv2.waitKey(1) == ord('q'):
            print("🛑 긴급 정지!")
            break

    # 종료 절차
    try:
        if ser:
            ser.write(b"D,0\n")
            time.sleep(0.1)
            ser.write(f"S,{SERVO_CENTER}\n".encode())
            ser.close()
    except:
        pass

    try:
        lidar.stop()
    except:
        pass

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("💾 녹화 완료 및 프로그램 종료")


if __name__ == "__main__":
    main()
>>>>>>> 48ddfbd4c872df7f1d627138120864380f23e6fc
