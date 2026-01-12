import threading
import time
import numpy as np
import cv2
import keyboard  # pip install keyboard
from Function_Library import libARDUINO, libLIDAR, libCAMERA

# ==========================================
# [0] 환경 설정 (Configuration)
# ==========================================
# 1. 통신 포트 설정 (장치 관리자 확인 필수)
PORT_ARDUINO = 'COM3'
PORT_LIDAR = 'COM4'
BAUD_RATE = 9600
CAM_ID = 0

# 2. 하드웨어 튜닝 값 (아두이노 코드와 일치시킬 것)
POT_CENTER = 570  # 중앙값
POT_LIMIT_L = 680  # 좌회전 한계 (최대값)
POT_LIMIT_R = 480  # 우회전 한계 (최소값)
STEER_GAIN = 0.5  # 조향 민감도 (픽셀 오차 * 0.5 = Pot 변화량)

# 3. 미션 상수
MISSION_IDLE = 0
MISSION_LINE_TRACE = 1
MISSION_OBSTACLE = 2
MISSION_PARKING = 3


# ==========================================
# [1] 데이터 허브 (Thread-Safe Data Hub)
# ==========================================
class DataHub:
    def __init__(self):
        self.lock = threading.Lock()

        # 입력 데이터 (Sensors)
        self.lane_error = 0  # 차선 오차 (픽셀)
        self.traffic_light = None  # "RED", "GREEN"
        self.lidar_dist = 9999  # 전방 장애물 거리

        # 출력 데이터 (Actuators)
        self.target_angle = POT_CENTER  # 초기값: 중앙
        self.target_speed = 0  # 초기값: 정지

        # 상태 플래그
        self.manual_mode = False  # 수동 모드 여부

    def update(self, key, value):
        with self.lock:
            setattr(self, key, value)

    def get(self, key):
        with self.lock:
            return getattr(self, key)


hub = DataHub()

# ==========================================
# [2] 하드웨어 객체 초기화
# ==========================================
# 1. 아두이노
ard = libARDUINO()
try:
    # 실제 연결 시 주석 해제
    serial_obj = ard.init(PORT_ARDUINO, BAUD_RATE)
    print(f"✅ Arduino Connected on {PORT_ARDUINO}")
except:
    print(f"⚠️ Arduino Connection Failed (Test Mode)")
    serial_obj = None

# 2. 라이다
try:
    lidar = libLIDAR(PORT_LIDAR)
    # lidar.init()
    print(f"✅ LiDAR Connected on {PORT_LIDAR}")
except:
    print(f"⚠️ LiDAR Connection Failed")
    lidar = None

# 3. 카메라
cam = libCAMERA()


# ==========================================
# [3] 로직 함수 (알고리즘)
# ==========================================
def logic_lane_tracing(frame_roi):
    """
    [차선 인식 및 조향각 계산]
    Return: 목표 가변저항 값 (Target Pot Value)
    """
    if frame_roi is None: return POT_CENTER

    # 1. 이미지 전처리 (라이브러리 활용)
    gray = cam.gray_conversion(frame_roi)
    blur = cam.gaussian_blurring(gray, (5, 5))
    edge = cam.canny_edge(blur, 100, 200)

    # 2. 선 검출 (Hough Transform)
    lines = cam.hough_transform(edge, 1, np.pi / 180, 50, 10, 20, mode="lineP")

    center_x = frame_roi.shape[1] // 2  # 화면 중앙 좌표
    lane_center = center_x  # 못 찾으면 중앙으로 가정

    if lines is not None:
        x_sum = 0
        count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += (x1 + x2)
            count += 2
        if count > 0:
            lane_center = x_sum // count

    # 3. 오차 계산 (화면 중앙 - 차선 중앙)
    # error > 0 : 차선이 화면 오른쪽에 있음 -> 우회전 필요
    error = lane_center - center_x
    hub.update("lane_error", error)  # 모니터링용 저장

    # 4. 가변저항 목표값 매핑 (P 제어)
    # 우회전(R)은 Pot 값이 작아지는 방향(480)이므로 (-) 연산
    target_pot = POT_CENTER - (error * STEER_GAIN)

    # 5. 값 제한 (Safety Clamping)
    # 480 ~ 680 사이를 벗어나지 않게 자름
    target_pot = max(POT_LIMIT_R, min(POT_LIMIT_L, target_pot))

    return int(target_pot)


def logic_traffic_light(frame_roi):
    """
    [신호등 인식]
    Return: "RED", "GREEN", or None
    """
    if frame_roi is None: return None
    return cam.object_detection(frame_roi, sample=5, mode="circle", print_enable=False)


# ==========================================
# [4] 쓰레드 작업 (Thread Workers)
# ==========================================
def thread_vision():
    print("📷 [Vision Thread] Start")
    cap0, _ = cam.initial_setting(cam0port=CAM_ID, capnum=1)

    while True:
        start_time = time.time()

        # 프레임 읽기
        ret_list = cam.camera_read(cap0)
        if not ret_list[0]: continue
        frame = ret_list[1]
        h, w = frame.shape[:2]

        # 1. 차선 인식 (화면 하단 50%)
        roi_lane = frame[int(h / 2):h, 0:w]
        pot_val = logic_lane_tracing(roi_lane)

        # 수동 모드가 아닐 때만 타겟 업데이트
        if not hub.get("manual_mode"):
            hub.update("target_angle", pot_val)

        # 2. 신호등 인식 (화면 상단 30%)
        # roi_traffic = frame[0:int(h/3), int(w/3):int(w*2/3)]
        # light = logic_traffic_light(roi_traffic)
        # hub.update("traffic_light", light)

        # 3. 디버깅 화면 (선택 사항)
        # cv2.imshow("Lane View", roi_lane)
        # if cv2.waitKey(1) == ord('q'): break

        # FPS 유지
        elapsed = time.time() - start_time
        if elapsed < 0.033: time.sleep(0.033 - elapsed)


def thread_arduino():
    print("🤖 [Arduino Thread] Start (Protocol: S,xxx,D,xxx)")
    last_send_time = 0

    while True:
        # 50ms 마다 전송 (너무 빠르면 아두이노 과부하)
        if time.time() - last_send_time > 0.05:
            t_angle = hub.get("target_angle")
            t_speed = hub.get("target_speed")

            # 프로토콜 포맷팅 (S:Steer, D:Drive)
            cmd = f"S,{t_angle},D,{t_speed}\n"

            if serial_obj and serial_obj.is_open:
                try:
                    serial_obj.write(cmd.encode())
                    # print(f"📤 Sent: {cmd.strip()}") # 디버깅용
                except Exception as e:
                    print(f"❌ Serial Error: {e}")

            last_send_time = time.time()

        time.sleep(0.01)


def thread_lidar():
    print("📡 [LiDAR Thread] Start")
    # 라이다 코드는 실제 장비 연결 후 활성화
    while True:
        time.sleep(1)


# ==========================================
# [5] 미션 마스터 (Main Controller)
# ==========================================
def mission_master():
    print("\n🎮 [Mission Master] Ready")
    print("   👉 'm' 키: 수동/자동 전환")
    print("   👉 화살표 키: 수동 조작")
    print("   👉 'ESC': 종료\n")

    current_mission = MISSION_LINE_TRACE

    while True:
        # ----------------------------------
        # [A] 공통 키 입력 처리 (수동/종료)
        # ----------------------------------
        if keyboard.is_pressed('esc'):
            hub.update("target_speed", 0)
            print("🛑 System Shutdown...")
            break

        if keyboard.is_pressed('m'):
            is_manual = not hub.get("manual_mode")
            hub.update("manual_mode", is_manual)
            print(f"🔄 Mode Changed: {'MANUAL' if is_manual else 'AUTO'}")
            hub.update("target_speed", 0)  # 모드 전환 시 안전 정지
            time.sleep(0.5)  # 채터링 방지

        # ----------------------------------
        # [B] 수동 제어 모드 (Manual)
        # ----------------------------------
        if hub.get("manual_mode"):
            # 조향 (Left / Right / Center)
            if keyboard.is_pressed('left'):
                hub.update("target_angle", POT_LIMIT_L)  # 680
            elif keyboard.is_pressed('right'):
                hub.update("target_angle", POT_LIMIT_R)  # 480
            else:
                hub.update("target_angle", POT_CENTER)  # 570

            # 속도 (Go / Stop)
            if keyboard.is_pressed('up'):
                hub.update("target_speed", 150)
            elif keyboard.is_pressed('down'):
                hub.update("target_speed", 0)

            time.sleep(0.05)
            continue  # 자동 로직 건너뜀

        # ----------------------------------
        # [C] 자동 주행 모드 (Auto)
        # ----------------------------------
        if current_mission == MISSION_LINE_TRACE:
            # 1. 신호등 체크
            light = hub.get("traffic_light")
            if light == "RED":
                hub.update("target_speed", 0)
                print("🚦 RED LIGHT -> STOP")
            else:
                # 2. 주행 (Vision Thread가 이미 target_angle은 계산해둠)
                hub.update("target_speed", 120)  # 기본 주행 속도

        time.sleep(0.05)


# ==========================================
# [6] 실행부
# ==========================================
if __name__ == "__main__":
    # 쓰레드 생성 및 시작
    t_vis = threading.Thread(target=thread_vision, daemon=True)
    t_ard = threading.Thread(target=thread_arduino, daemon=True)
    t_lid = threading.Thread(target=thread_lidar, daemon=True)

    t_vis.start()
    t_ard.start()
    t_lid.start()

    # 메인 루프 실행
    try:
        mission_master()
    except KeyboardInterrupt:
        pass
    finally:
        if serial_obj: serial_obj.close()
        cv2.destroyAllWindows()