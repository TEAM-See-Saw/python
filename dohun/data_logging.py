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
CAM_INDEX = 1
LIDAR_PORT = 'COM3'     # 라이다 포트 확인
ARDUINO_PORT = 'COM4'   # 아두이노 포트 확인
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
    img_path = os.path.join(save_path, "cam0")
    os.makedirs(img_path, exist_ok=True)
    return save_path, img_path

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
# [4] 스레드 2: 카메라 로깅 (이미지 낱장 저장)
# ==========================================
def thread_camera(img_save_path):
    global stop_program
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, 1920); cap.set(4, 1080)
    
    if not cap.isOpened():
        print("❌ 카메라 연결 실패"); stop_program = True; return

    print("✅ 카메라 녹화 시작")
    while not stop_program:
        ret, frame = cap.read()
        if ret:
            # 타임스탬프를 파일명으로 저장 (가장 중요!)
            now = time.time()
            filename = f"{now:.4f}.jpg"
            cv2.imwrite(os.path.join(img_save_path, filename), frame)
            
            # 화면 표시 (운전용)
            cv2.imshow("Driving View", frame)
            if cv2.waitKey(1) == 27: # ESC 누르면 종료
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
    
    # 아두이노 연결
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=1)
        time.sleep(2) # 아두이노 리셋 대기
        ser.reset_input_buffer()
        print("✅ 아두이노 연결 성공")
    except Exception as e:
        print(f"❌ 아두이노 연결 실패: {e}")
        return

    # 제어 로그 파일
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
            # 1. 아두이노 버퍼 비우기 (작성하신 꿀팁 유지)
            if ser.in_waiting > 0:
                ser.reset_input_buffer()

            # 2. 키보드 입력
            curr_steer = VAL_CENTER
            if keyboard.is_pressed('left'): curr_steer = VAL_LEFT
            elif keyboard.is_pressed('right'): curr_steer = VAL_RIGHT

            curr_speed = SPEED_STOP
            if keyboard.is_pressed('up'): curr_speed = SPEED_FWD
            elif keyboard.is_pressed('down'): curr_speed = SPEED_BWD
            
            # 종료 조건
            if keyboard.is_pressed('esc'):
                stop_program = True
                break

            current_time = time.time()

            # 3. 아두이노 전송 (값 변경 시 or 0.1초마다)
            if (curr_steer != last_steer) or (curr_speed != last_speed) or (current_time - last_send_time > 0.1):
                cmd = f"S,{curr_steer}\n"
                ser.write(cmd.encode())
                cmd = f"D,{curr_speed}\n"
                ser.write(cmd.encode())
                
                last_steer = curr_steer
                last_speed = curr_speed
                last_send_time = current_time
                
                # 4. 제어값 로깅 (명령을 보낸 시점 기록)
                writer.writerow([f"{current_time:.4f}", curr_steer, curr_speed])

            time.sleep(0.01) # CPU 과부하 방지

    except KeyboardInterrupt:
        stop_program = True
    finally:
        ser.write(b"D,0\n"); ser.write(b"S,570\n") # 정지
        ser.close()
        f.close()
        print("🛑 제어 및 로깅 종료")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    save_path, img_path = create_dataset_folder()
    
    # 스레드 시작 (라이다, 카메라)
    t_lidar = threading.Thread(target=thread_lidar, args=(save_path,))
    t_cam = threading.Thread(target=thread_camera, args=(img_path,))
    
    t_lidar.start()
    t_cam.start()
    
    # 메인은 운전에 집중
    main_control_loop(save_path)
    
    # 종료 대기
    stop_program = True
    t_lidar.join()
    t_cam.join()
    print("✅ 모든 데이터 저장 완료.")