import cv2
import numpy as np

# ===========================
# [설정] 튜닝할 카메라 번호 설정
CAM_INDEX = 0

# [설정] 목표 해상도 (FHD)
TARGET_W = 1920
TARGET_H = 1080


# ===========================

def nothing(x):
    """트랙바 콜백용 더미 함수"""
    pass


def main():
    print("========================================")
    print(f"   [FHD 1920x1080 ROI 튜닝 모드]")
    print(f"   Camera Index: {CAM_INDEX}")
    print("========================================")

    # 1. 카메라 열기
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Error: 카메라 #{CAM_INDEX}를 열 수 없습니다.")
        return

    # 2. 해상도 강제 설정 (1920 x 1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

    # 실제 설정된 해상도 확인
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"👉 카메라 해상도 설정: {real_w}x{real_h}")
    print(f"👉 목표 해상도: {TARGET_W}x{TARGET_H}")

    window_name = "FHD ROI Tuning (View Scaled 0.5x)"
    cv2.namedWindow(window_name)

    # 3. 트랙바 설정 (FHD 해상도에 맞춰 범위 확장)
    # Top Width: 화면 중앙에서 좌우로 벌어지는 너비 (최대값: 1920/2 = 960)
    cv2.createTrackbar("ROI Top Width", window_name, 450, TARGET_W // 2, nothing)

    # Top Y: 사다리꼴 윗변의 Y좌표 (최대값: 1080)
    cv2.createTrackbar("ROI Top Y", window_name, 540, TARGET_H, nothing)

    # Bottom Y: 사다리꼴 밑변의 Y좌표 (최대값: 1080)
    cv2.createTrackbar("ROI Bottom Y", window_name, TARGET_H, TARGET_H, nothing)

    # 흰색 감도
    cv2.createTrackbar("L Min (White)", window_name, 180, 255, nothing)

    print("\n🚀 튜닝 시작! (화면은 모니터 크기에 맞게 50% 축소되어 표시됩니다)")
    print("   하지만 출력되는 결과값은 1920x1080 기준입니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 강제 리사이즈 (카메라가 1080p를 미지원할 경우를 대비하여 강제로 늘림)
        if frame.shape[1] != TARGET_W or frame.shape[0] != TARGET_H:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        h, w = frame.shape[:2]  # w=1920, h=1080

        # ----------------------------------------------------
        # 트랙바 값 읽기
        # ----------------------------------------------------
        roi_top_w = cv2.getTrackbarPos("ROI Top Width", window_name)
        roi_top_y = cv2.getTrackbarPos("ROI Top Y", window_name)
        roi_bot_y = cv2.getTrackbarPos("ROI Bottom Y", window_name)
        l_min = cv2.getTrackbarPos("L Min (White)", window_name)

        # ----------------------------------------------------
        # ROI 계산 (1920x1080 기준)
        # ----------------------------------------------------
        center_x = w // 2
        pts = np.array([
            [0, roi_bot_y],  # 좌하단 (보통 0, 1080)
            [center_x - roi_top_w, roi_top_y],  # 좌상단
            [center_x + roi_top_w, roi_top_y],  # 우상단
            [w, roi_bot_y]  # 우하단 (보통 1920, 1080)
        ], np.int32)

        # 1) 원본에 박스 그리기
        display_frame = frame.copy()
        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)  # 선 두께 3

        # 2) 마스킹
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        roi_img = cv2.bitwise_and(frame, mask)

        # 3) 흰색 검출
        hls = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, l_min, 0])
        upper_white = np.array([179, 255, 255])
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        mask_3ch = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)

        # ----------------------------------------------------
        # 화면 출력 (View Only)
        # ----------------------------------------------------
        # 1920 + 1920 = 3840 픽셀이므로 모니터에 다 안 들어갑니다.
        # 따라서 시각화용으로만 0.5배 줄여서 보여줍니다.

        result_full = np.hstack((display_frame, mask_3ch))
        result_view = cv2.resize(result_full, (0, 0), fx=0.5, fy=0.5)  # 50% 축소

        cv2.imshow(window_name, result_view)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("🎉 [1920x1080 기준] 최종 튜닝 결과")
    print("=" * 50)
    print(f"ROI Top Width  : {roi_top_w}")
    print(f"ROI Top Y      : {roi_top_y}")
    print(f"ROI Bottom Y   : {roi_bot_y}")
    print(f"L Min Threshold: {l_min}")
    print("=" * 50)


if __name__ == "__main__":
    main()