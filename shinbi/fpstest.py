import cv2
import time

# OpenCV 4.5+ 버전부터 가능한 '매개변수 주입' 방식입니다.
# 카메라를 켬과 동시에 설정을 적용하여 YUY2로 진입하는 것을 원천 차단합니다.
params = [
    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'),
    cv2.CAP_PROP_FRAME_WIDTH, 1920,
    cv2.CAP_PROP_FRAME_HEIGHT, 1080,
    cv2.CAP_PROP_FPS, 30
]

# 1. DSHOW(빠른 실행) + params(설정 강제 주입)
# 주의: index(0) 다음 인자로 apiPreference와 params를 순서대로 넣어야 합니다.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW, params)

# -----------------------------------------------------------

# 디버깅: 실제로 적용된 코덱 확인
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

print(f"Current Codec: {codec}")
print(f"Resolution   : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"FPS Setting  : {cap.get(cv2.CAP_PROP_FPS)}")

if codec != "MJPG":
    print("!!! 경고: 여전히 YUY2 모드입니다. USB 포트를 바꿔보거나 드라이버를 확인해야 합니다. !!!")

prev = time.time()

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - prev
    if dt == 0: dt = 0.001
    fps = 1.0 / dt
    prev = now

    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("raw", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()