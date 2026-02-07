import time
import numpy as np
import Function_Library as LiDAR_Lib

# ===========================
# [설정] LiDAR 포트 번호
LIDAR_PORT = 'COM3'


# ===========================

def main():
    print("========================================")
    print(f"   [LiDAR 단독 점검] 포트: {LIDAR_PORT}")
    print("========================================")

    try:
        env = LiDAR_Lib.libLIDAR(LIDAR_PORT)
        env.init()
        print("✅ 연결 성공! 모터가 회전합니다...")
    except Exception as e:
        print(f"❌ 연결 실패! 포트나 케이블을 확인하세요.\n에러 내용: {e}")
        return

    print("\n[측정 시작] 1초마다 상태를 출력합니다. (종료: Ctrl + C)")

    start_time = time.time()
    scan_count = 0

    try:
        # 데이터 스캔 루프
        for scan in env.scanning():
            scan_count += 1
            current_time = time.time()
            elapsed = current_time - start_time

            # 1초마다 출력
            if elapsed >= 1.0:
                hz = scan_count / elapsed

                # [수정됨] 라이브러리 특성에 맞춰 인덱스 변경 (2 -> 1)
                # 데이터 구조: [Angle, Distance] 이므로 Distance는 인덱스 1입니다.
                valid_points = len(scan[scan[:, 1] > 0])

                print(f"📡 속도: {hz:.2f} Hz | 감지된 포인트: {valid_points} 개")

                start_time = current_time
                scan_count = 0

    except KeyboardInterrupt:
        print("\n🛑 사용자 종료 요청.")
    except Exception as e:
        print(f"\n❌ 실행 중 에러 발생: {e}")
    finally:
        print("장치를 정지합니다...")
        env.stop()


if __name__ == "__main__":
    main()