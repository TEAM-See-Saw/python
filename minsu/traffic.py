# run_traffic_stop.py
import serial
import time
import cv2

from common_lane_base import (
    PORT, BAUDRATE, SERIAL_DELAY, SPEED_REFRESH_DELAY,
    CAM_INDEX, CAM_INDEX_TRAFFIC, MAX_SPEED,
    configure_camera_auto, compute_lane,
    base_target_from_lines, angle_from_target, servo_from_angle,
    ROI_HEIGHT_RATIO,
    CROSSWALK_RATIO_MIN, CROSSWALK_MAX_WAIT, CROSSWALK_COOLDOWN, CROSSWALK_CONFIRM_TIME,
    detect_stop_line,
    TRAFFIC_BOX_X1, TRAFFIC_BOX_X2, TRAFFIC_BOX_Y1, TRAFFIC_BOX_Y2,
    TRAFFIC_HIGHLIGHT_PCTL, TRAFFIC_TH_MIN
)

# ====== 신호등 인식 함수 (너 코드 그대로) ======
import numpy as np

def detect_traffic_lr_robust(frame_bgr):
    H, W = frame_bgr.shape[:2]

    x1 = int(W * TRAFFIC_BOX_X1)
    x2 = int(W * TRAFFIC_BOX_X2)
    y1 = int(H * TRAFFIC_BOX_Y1)
    y2 = int(H * TRAFFIC_BOX_Y2)

    x1 = max(0, min(W - 2, x1))
    x2 = max(x1 + 1, min(W - 1, x2))
    y1 = max(0, min(H - 2, y1))
    y2 = max(y1 + 1, min(H - 1, y2))

    roi = frame_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 5 or rw < 5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "ROI_TOO_SMALL"}

    third = max(1, rw // 3)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    S_GATE = 60
    V_GATE = 80
    COLOR_TH = 0.003

    sat_mask = cv2.inRange(hsv, (0, S_GATE, 0), (180, 255, 255))
    red1 = cv2.inRange(hsv, (0, S_GATE, V_GATE), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, S_GATE, V_GATE), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, (35, S_GATE, V_GATE), (85, 255, 255))

    red_mask = cv2.bitwise_and(red_mask, sat_mask)
    green_mask = cv2.bitwise_and(green_mask, sat_mask)

    k = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k)

    red_left = red_mask[:, 0:third]
    green_right = green_mask[:, 2 * third:rw] if (2 * third) < rw else green_mask[:, third:rw]

    red_left_ratio = cv2.countNonZero(red_left) / float(red_left.size)
    green_right_ratio = cv2.countNonZero(green_right) / float(green_right.size)

    if red_left_ratio > COLOR_TH and green_right_ratio < COLOR_TH:
        return "LEFT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR",
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    if green_right_ratio > COLOR_TH and red_left_ratio < COLOR_TH:
        return "RIGHT", {"box": (x1, y1, x2, y2), "mode": "HSV_COLOR",
                         "red_left": red_left_ratio, "green_right": green_right_ratio}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    p = np.percentile(gray, TRAFFIC_HIGHLIGHT_PCTL)
    thr = int(max(TRAFFIC_TH_MIN, p))
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_NONE", "thr": thr,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < 30:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_SMALL", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    roi_area = float(rh * rw)
    if area > roi_area * 0.25:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_TOO_BIG", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    x, y, ww, hh = cv2.boundingRect(c)
    if ww > hh * 2.5:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_WIDE", "thr": thr, "area": area,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    M = cv2.moments(c)
    if M["m00"] == 0:
        return "NONE", {"box": (x1, y1, x2, y2), "mode": "FALLBACK_BADMOM", "thr": thr,
                        "red_left": red_left_ratio, "green_right": green_right_ratio}

    cx = int(M["m10"] / M["m00"])
    if cx < third:
        state = "LEFT"
    elif cx > 2 * third:
        state = "RIGHT"
    else:
        state = "NONE"

    return state, {"box": (x1, y1, x2, y2), "mode": "BRIGHT_BLOB", "thr": thr,
                   "cx": cx, "area": area, "red_left": red_left_ratio, "green_right": green_right_ratio}


def main():
    ser = None
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)
        print(f"✅ {PORT} 포트 연결 성공! (1초 대기)")
        time.sleep(1)
    except Exception as e:
        print(f"❌ 시리얼 연결 실패: {e}")
        ser = None

    cap_lane = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap_traffic = cv2.VideoCapture(CAM_INDEX_TRAFFIC, cv2.CAP_DSHOW)
    width, height = 640, 480
    cap_lane.set(3, width); cap_lane.set(4, height)
    cap_traffic.set(3, width); cap_traffic.set(4, height)

    if not cap_lane.isOpened():
        print("❌ 차선 카메라 오류")
        return
    if not cap_traffic.isOpened():
        print("❌ 신호등 카메라 오류")
        return

    configure_camera_auto(cap_lane)
    configure_camera_auto(cap_traffic)

    is_crosswalk_stop = False
    crosswalk_start_time = 0.0
    crosswalk_cooldown_timer = 0.0
    crosswalk_detect_timer = 0.0

    last_serial_time = 0.0
    last_speed_time = 0.0

    try:
        if ser:
            ser.write(f"D,{MAX_SPEED}\n".encode())

        print("🚀 주행 시작 (Traffic Stop Mode)")

        while True:
            if ser:
                try:
                    if ser.in_waiting > 0:
                        ser.read(ser.in_waiting)
                except:
                    pass

            now = time.time()

            ret_l, frame_lane = cap_lane.read()
            ret_t, frame_traffic = cap_traffic.read()
            if not ret_l or not ret_t:
                print("❌ 카메라 신호 끊김")
                break

            if frame_lane.shape[1] != width:
                frame_lane = cv2.resize(frame_lane, (width, height))
            if frame_traffic.shape[1] != width:
                frame_traffic = cv2.resize(frame_traffic, (width, height))

            lane = compute_lane(frame_lane)
            mask = lane["mask"]
            mask_bgr = lane["mask_bgr"]
            roi_points = lane["roi_points"]
            ratio = lane["ratio"]
            left, right = lane["left"], lane["right"]

            # ===== 신호등 =====
            traffic_state, tdbg = detect_traffic_lr_robust(frame_traffic)

            # ===== 라인트레이싱 기본 조향 =====
            target = base_target_from_lines(width, left, right)
            angle = angle_from_target(width, height, target)
            servo_val = servo_from_angle(angle)

            status_msg = "NORMAL"
            status_color = (0, 255, 0)
            final_speed = MAX_SPEED

            # ===== 정지선 + 신호등 로직 =====
            if is_crosswalk_stop:
                final_speed = 0
                elapsed = now - crosswalk_start_time

                if traffic_state == "RIGHT":
                    is_crosswalk_stop = False
                    crosswalk_cooldown_timer = now
                    crosswalk_detect_timer = 0
                    status_msg = "GO (RIGHT ON)"
                    status_color = (0, 255, 0)

                elif elapsed > CROSSWALK_MAX_WAIT:
                    is_crosswalk_stop = False
                    crosswalk_cooldown_timer = now
                    crosswalk_detect_timer = 0
                    status_msg = "GO (TIMEOUT)"
                    status_color = (0, 255, 0)

                else:
                    status_msg = f"WAIT RIGHT.. ({elapsed:.1f}s)"
                    status_color = (0, 0, 255)

            else:
                if (ratio > CROSSWALK_RATIO_MIN) and (now - crosswalk_cooldown_timer > CROSSWALK_COOLDOWN):
                    if detect_stop_line(mask, mask_bgr, ROI_HEIGHT_RATIO):
                        if traffic_state == "LEFT":
                            if crosswalk_detect_timer == 0:
                                crosswalk_detect_timer = now
                            elif now - crosswalk_detect_timer > CROSSWALK_CONFIRM_TIME:
                                is_crosswalk_stop = True
                                crosswalk_start_time = now
                                final_speed = 0
                                status_msg = "STOP! (Line+LEFT ON)"
                                status_color = (0, 0, 255)
                                crosswalk_detect_timer = 0
                            else:
                                status_msg = "Checking Line+LEFT..."
                                status_color = (0, 255, 255)
                        else:
                            crosswalk_detect_timer = 0
                            status_msg = f"Line Detected ({traffic_state})"
                            status_color = (0, 255, 255)
                    else:
                        crosswalk_detect_timer = 0
                else:
                    crosswalk_detect_timer = 0

            # ===== 통신 =====
            if ser:
                if now - last_serial_time > SERIAL_DELAY:
                    ser.write(f"S,{servo_val}\n".encode())
                    last_serial_time = now
                if now - last_speed_time > SPEED_REFRESH_DELAY:
                    ser.write(f"D,{final_speed}\n".encode())
                    last_speed_time = now

            # ===== 디스플레이 =====
            box = tdbg.get("box", None)
            mode = tdbg.get("mode", "-")
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame_traffic, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(frame_traffic, f"Traffic:{traffic_state} mode:{mode}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            cv2.polylines(mask_bgr, [roi_points], True, (0, 255, 255), 2)
            cv2.circle(mask_bgr, (int(target), int(height * ROI_HEIGHT_RATIO)), 10, (0, 0, 255), -1)

            cv2.putText(mask_bgr, status_msg, (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

            combined = cv2.hconcat([frame_traffic, mask_bgr])
            cv2.imshow("Traffic + Lane", combined)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        print("\n🛑 안전 정지")
        if ser:
            try:
                for _ in range(3):
                    ser.write(b"D,0\n")
                    ser.write(b"S,570\n")
                    time.sleep(0.05)
                ser.close()
            except:
                pass
        cap_lane.release()
        cap_traffic.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
