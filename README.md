# 🚗 Project See-Saw

Python과 Arduino를 연동한 자율주행 자동차 시스템입니다. 카메라 기반 차선 인식, 초음파/LiDAR 센서 장애물 감지, 모터 제어를 통해 자율주행을 구현합니다.

🏆 **2026 경기도 대학생 자율주행 경진대회 우수상 수상작**

## 📁 프로젝트 구조

```
├── python/                        # PC 측 제어 로직
│   ├── Function_Library.py        # 공통 라이브러리 (Arduino·LiDAR·Camera 클래스)
│   ├── lane_tracing_with_front_cam_no_tune.py  # 차선 인식 자율주행
│   ├── manual_control.py          # 키보드 수동 조작 + 데이터 기록
│   ├── obstacle_plusV.py          # 장애물 회피 주행
│   ├── obstacles_and_traffic.py   # 장애물 + 신호등 통합 주행
│   ├── parking_with_lidar_and_us.py  # LiDAR·초음파 자동 주차
│   ├── traffic.py                 # 신호등 인식 주행
│   ├── lidar_health_check.py      # LiDAR 센서 점검
│   └── cam_health_check.py        # 카메라 센서 점검
│
└── arduino/                       # MCU 펌웨어
    └── pwm_with_us_refactored/    # 모터 제어 + 6채널 초음파 센서
```

## ⚙️ 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 영상 처리 | OpenCV (HLS 필터링, Canny Edge, HoughLinesP) |
| 센서 | LiDAR (rplidar), 초음파 센서 6채널 |
| 통신 | Serial UART (115200 bps) |
| 하드웨어 | Arduino Mega, DC 모터, 조향 모터 + 포텐쇼미터 |

## 🔌 통신 프로토콜

| 방향 | 명령 | 예시 | 설명 |
|---|---|---|---|
| PC → Arduino | `S,<값>` | `S,570` | 조향 목표값 (570 = 중립) |
| PC → Arduino | `D,<값>` | `D,255` | 구동 속도 (255 = 최대 전진) |
| Arduino → PC | `US:<d1>,…,<d6>` | `US:120,85,999,…` | 초음파 거리(mm) |

> **Fail-safe**: 1초 이상 명령 미수신 시 자동 정지

## 🛣️ 차선 인식 파이프라인

```
카메라 입력 → Median Blur → HLS 변환 → 밝기 필터링 (L ≥ 230)
    → Canny Edge → ROI 마스킹 → HoughLinesP
    → 좌/우 차선 분류 → 목표 지점 계산 → 조향 각도 출력
```

## 🚀 실행 방법

```bash
# 1. 의존성 설치
pip install opencv-python numpy pyserial rplidar-roboticia

# 2. 센서 점검
python lidar_health_check.py
python cam_health_check.py

# 3. 수동 주행 (데이터 수집)
python manual_control.py

# 4. 자율 주행
python lane_tracing_with_front_cam_no_tune.py
```

> ⚠️ 실행 전 `COM4` 등 시리얼 포트 번호를 각 스크립트 상단에서 환경에 맞게 수정할 것
