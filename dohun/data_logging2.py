import serial
import time
import cv2
import csv
import os
import threading
import keyboard # pip install keyboard
from rplidar import RPLidar # pip install rplidar-roboticia

# ==========================================
# [1] 설정 (본인 환경에 맞게 수정)
# ==========================================
# 카메라 인덱스 (연결 순서에 따라 0, 1 확인 필요)
CAM_IDX_1 = 0   # 첫 번째 카메라
CAM_IDX_2 = 1   # 두 번째 카메라

# 해상도 설정 (1920 x 1080)
CAM_WIDTH = 1920
CAM_HEIGHT = 1080

LIDAR_PORT = 'COM3'     # 라이다 포트
ARDUINO_PORT = 'COM4'   # 아두이노 포트
BAUDRATE = 115200

# 제어값 설정
VAL_LEFT = 680; VAL_RIGHT = 480; VAL_CENTER = 570
SPEED_FWD = 255; SPEED_STOP = 0; SPEED_BWD = -255

# 데이터 저장 경로
BASE_SAVE_DIR = "dataset_log"

# 전역 변수 (스레드 종료용)
stop_program = False

# ==========================================
# [2] 저장소 준비
# ==========================================
def create_dataset_folder():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(BASE_SAVE_DIR, timestamp)
    
    # 카메라 2개용 폴더 각각 생성
    img_path_1 = os.path.join(save_path, "cam0") # CAM_IDX_1 저장
    img_path_2 = os.path.join(save_path, "cam1") # CAM_IDX_2 저장
    
    os.makedirs(img_path_1, exist_ok=True)
    os.makedirs(img_path_2, exist_ok=True)
    
    return save_path, img_path_1, img_path_2

# ==========================================
# [3] 스레드 1: 라이다 로깅
# ==========================================
def thread_lidar(save_path):
    global stop_program
    csv_file = os.path.join(save_path, "lidar.csv")
    
    try:
        lidar = RPLidar(LIDAR_PORT)
        print("✅ LiDAR 연결 성공")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "quality", "angle", "distance"])
            
            for scan in lidar.iter_scans():
                if stop_program: break
                now = time.time()
                for (quality, angle, distance) in scan:
                    if distance > 0:
                        writer.writerow([f"{now:.4f}", quality, f"{angle:.2f}", f"{distance:.2f}"])
    except Exception as e:
        print(f"❌ LiDAR 오류: {e}")
    finally:
        try: lidar.stop(); lidar.disconnect()
        except: pass

# ==========================================
# [4] 스레드 2: 카메라 로깅 (공통 함수)
# ==========================================
def thread_camera(cam_index, img_save_path, cam_name):
    global stop_program
    
    # 윈도우 DSHOW 백엔드 사용 (지연 시간 최소화)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    
    # ★ 요청하신 1920x1080 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    # 실제 적용된 해상도 확인 (가끔 카메라가 지원 안 하면 무시될 수 있음)
    real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"📷 {cam_name} 설정 완료: {int(real_w)}x{int(real_h)}")
    
    if not cap.isOpened():
        print(f"❌ 카메라({cam_name}) 연결 실패"); return

    print(f"✅ {cam_name} 녹화 시작")
    
    while not stop_program:
        ret, frame = cap.read()
        if ret:
            # 타임스탬프 파일명
            now = time.time()
            filename = f"{now:.4f}.jpg"
            cv2.imwrite(os.path.join(img_save_path, filename), frame)
            
            # 화면 표시 (운전용 프리뷰는 작게 축소해서 보여줌 - 렉 방지)
            preview = cv2.resize(frame, (640, 360))
            cv2.imshow(f"View {cam_name}", preview)
            
            if cv2.waitKey(1) == 27: # ESC
                stop_program = True
        else:
            time.sleep(0.01)
    
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# [5] 메인 루프: 키보드 제어 & 제어값 로깅
# ==========================================
def main_control_loop(save_path):
    global stop_program
    
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✅ 아두이노 연결 성공")
    except Exception as e:
        print(f"❌ 아두이노 연결 실패: {e}")
        return

    log_csv = os.path.join(save_path, "control.csv")
    f = open(log_csv, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["timestamp", "steering", "speed"])

    print("\n🚗 [READY] 방향키로 운전하세요. ESC로 종료.")
    
    last_steer = VAL_CENTER
    last_speed = SPEED_STOP
    last_send_time = 0
    
    try:
        while not stop_program:
            if ser.in_waiting > 0:
                ser.reset_input_buffer()

            curr_steer = VAL_CENTER
            if keyboard.is_pressed('left'): curr_steer = VAL_LEFT
            elif keyboard.is_pressed('right'): curr_steer = VAL_RIGHT

            curr_speed = SPEED_STOP
            if keyboard.is_pressed('up'): curr_speed = SPEED_FWD
            elif keyboard.is_pressed('down'): curr_speed = SPEED_BWD
            
            if keyboard.is_pressed('esc'):
                stop_program = True
                break

            current_time = time.time()

            if (curr_steer != last_steer) or (curr_speed != last_speed) or (current_time - last_send_time > 0.1):
                cmd = f"S,{curr_steer}\n"
                ser.write(cmd.encode())
                cmd = f"D,{curr_speed}\n"
                ser.write(cmd.encode())
                
                last_steer = curr_steer
                last_speed = curr_speed
                last_send_time = current_time
                
                writer.writerow([f"{current_time:.4f}", curr_steer, curr_speed])

            time.sleep(0.01)

    except KeyboardInterrupt:
        stop_program = True
    finally:
        ser.write(b"D,0\n"); ser.write(b"S,570\n")
        ser.close()
        f.close()
        print("🛑 제어 및 로깅 종료")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    save_path, img_path_1, img_path_2 = create_dataset_folder()
    
    t_lidar = threading.Thread(target=thread_lidar, args=(save_path,))
    
    # 카메라 2개 스레드 생성 (0번, 1번)
    t_cam1 = threading.Thread(target=thread_camera, args=(CAM_IDX_1, img_path_1, "CAM_FRONT"))
    t_cam2 = threading.Thread(target=thread_camera, args=(CAM_IDX_2, img_path_2, "CAM_SIDE"))
    
    t_lidar.start()
    t_cam1.start()
    t_cam2.start()
    
    main_control_loop(save_path)
    
    stop_program = True
    t_lidar.join()
    t_cam1.join()
    t_cam2.join()
    print("✅ 모든 데이터 저장 완료.")