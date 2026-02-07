import cv2
import pandas as pd
import numpy as np
import os
import glob

# ==========================================
# [1] 설정 (경로 수정 필수!)
# ==========================================
LIDAR_PATH = 'lidar.csv'       # 라이다 파일 경로
IMG_FOLDER = 'dataset_log/cam0' # 이미지가 저장된 폴더 경로 (수정하세요!)

# 라이다 뷰어 설정
WINDOW_SIZE = 500   # 라이다 지도 크기 (500x500 픽셀)
SCALE = 0.1         # 지도 축소 배율 (작을수록 멀리 보임)
POINT_COLOR = (0, 255, 0) # 라이다 점 색깔 (초록색)

# ==========================================
# [2] 데이터 로드 및 준비
# ==========================================
print("📂 데이터 로딩 중...")

# 1. 라이다 데이터 로드
df_lidar = pd.read_csv(LIDAR_PATH)
df_lidar = df_lidar.sort_values('timestamp') # 시간순 정렬
lidar_times = df_lidar['timestamp'].values # 검색 속도를 위해 배열로 변환

# 2. 이미지 파일 리스트 로드
img_files = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.jpg")))
if not img_files:
    print("❌ 이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

print(f"✅ 이미지 {len(img_files)}장, 라이다 포인트 {len(df_lidar)}개 로드 완료!")

# ==========================================
# [3] 뷰어 루프
# ==========================================
print("🚀 뷰어 시작! (종료: q, 일시정지: space)")

for img_path in img_files:
    # 1. 이미지 파일명에서 타임스탬프 추출
    # 파일명 예시: 'folder/1770444518.3219.jpg' -> 1770444518.3219
    filename = os.path.basename(img_path)
    try:
        current_ts = float(filename.replace(".jpg", ""))
    except ValueError:
        continue # 파일명이 숫자가 아니면 패스

    # 2. 이미지 읽기
    frame = cv2.imread(img_path)
    if frame is None: continue
    frame = cv2.resize(frame, (640, 480)) # 보기 좋게 크기 조절

    # 3. 해당 시간대의 라이다 데이터 찾기 (동기화 핵심!)
    # 현재 이미지 시간 앞뒤 0.1초(100ms) 구간의 라이다 데이터를 모두 가져옴
    # 이 구간이 모여서 '한 바퀴'를 구성함
    time_window = 0.1 
    mask = (lidar_times >= current_ts - time_window) & (lidar_times <= current_ts + time_window)
    lidar_subset = df_lidar.iloc[mask]

    # 4. 라이다 데이터 시각화 (Top-View Map 그리기)
    lidar_map = np.zeros((WINDOW_SIZE, WINDOW_SIZE, 3), dtype=np.uint8) # 검은 배경
    
    if not lidar_subset.empty:
        # 극좌표(각도, 거리) -> 직교좌표(X, Y) 변환
        angles = np.radians(lidar_subset['angle'].values)
        distances = lidar_subset['distance'].values

        # 좌표 변환 공식 (X = d*cos, Y = d*sin)
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)

        # 화면 중앙으로 이동 및 스케일 조절
        img_x = (x * SCALE) + (WINDOW_SIZE / 2)
        img_y = (y * SCALE) + (WINDOW_SIZE / 2)

        # 점 찍기
        for ix, iy in zip(img_x, img_y):
            if 0 <= ix < WINDOW_SIZE and 0 <= iy < WINDOW_SIZE:
                # 이미지 좌표는 정수여야 함
                cv2.circle(lidar_map, (int(ix), int(iy)), 1, POINT_COLOR, -1)

    # 중심점(내 차 위치) 표시 (빨간 세모)
    cv2.circle(lidar_map, (WINDOW_SIZE//2, WINDOW_SIZE//2), 5, (0, 0, 255), -1)

    # 5. 두 화면 합치기 (가로로 붙이기)
    # 높이를 맞춰야 함 (480px)
    lidar_map_resized = cv2.resize(lidar_map, (480, 480))
    combined = np.hstack((frame, lidar_map_resized))

    # 6. 화면 출력
    cv2.putText(combined, f"Time: {current_ts:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Dataset Validator (Left: Cam, Right: LiDAR)", combined)

    # 키 입력 처리
    key = cv2.waitKey(33) # 약 30fps 속도로 재생
    if key == ord('q'): # q 누르면 종료
        break
    elif key == ord(' '): # 스페이스바 누르면 일시정지
        cv2.waitKey(-1)

cv2.destroyAllWindows()