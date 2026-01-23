import cv2

# ===========================
# [설정] 카메라 번호
CAM0_INDEX = 0
CAM1_INDEX = 1


# ===========================

def main():
    print("========================================")
    print(f"   [카메라 단독 점검] Index: {CAM0_INDEX}, {CAM1_INDEX}")
    print("========================================")

    # 카메라 0번 열기 (Windows용 DSHOW 옵션 포함)
    cap0 = cv2.VideoCapture(CAM0_INDEX, cv2.CAP_DSHOW)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 카메라 1번 열기
    cap1 = cv2.VideoCapture(CAM1_INDEX, cv2.CAP_DSHOW)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 연결 상태 확인
    if cap0.isOpened():
        print(f"✅ 카메라 #{CAM0_INDEX} 연결 성공")
    else:
        print(f"❌ 카메라 #{CAM0_INDEX} 연결 실패")

    if cap1.isOpened():
        print(f"✅ 카메라 #{CAM1_INDEX} 연결 성공")
    else:
        print(f"❌ 카메라 #{CAM1_INDEX} 연결 실패")

    print("\n화면이 나오면 'q'를 눌러 종료하세요.")

    while True:
        # 프레임 읽기
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        # 화면 띄우기 (연결된 것만)
        if ret0:
            cv2.imshow(f'Camera #{CAM0_INDEX}', frame0)

        if ret1:
            cv2.imshow(f'Camera #{CAM1_INDEX}', frame1)

        # 둘 다 안 나오면 루프 종료
        if not ret0 and not ret1:
            print("⚠️ 모든 카메라 신호 없음. 종료합니다.")
            break

        # 'q' 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("프로그램 종료.")


if __name__ == "__main__":
    main()