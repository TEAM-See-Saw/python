import serial
from rplidar import RPLidar
import time
import numpy as np
import cv2

# ==========================================
# [1] Hardware Config
# ==========================================
PORT = 'COM4'
LIDAR_PORT = 'COM3'

# Speed
SPEED_SEARCH = 80
SPEED_SWING = 80
SPEED_PARK = 75
SPEED_STOP = 0

# Servo
SERVO_CENTER = 570
SERVO_RIGHT_MAX = 440
SERVO_LEFT_MAX = 680
STEER_WAIT_TIME = 0.8

# Tuning
ALIGN_THRES = 50
VALID_DIST = 800
PARKING_DEPTH_TIME = 1.5
FORWARD_OFFSET_TIME = 0.6

# LiDAR Thresholds
# Ensure these values match your environment
EMPTY_DIST_MIN = 1000  # Gap > 1m
OBSTACLE_DIST_MAX = 600  # Car < 60cm

# Sonar Indices
IDX_RM = 4
IDX_RT = 5

# States
STATE_SEARCH = 0
STATE_FORWARD_OFFSET = 1
STATE_SWING_OUT = 2
STATE_PREP_REVERSE = 3
STATE_REVERSE_ENTRY = 4
STATE_PREP_STRAIGHT = 5
STATE_REVERSE_FINISH = 6
STATE_DONE = 7

# ★ Updated Search Steps
STEP_FIND_FIRST_CAR = 0  # NEW: Find the first car
STEP_PASS_CAR1 = 1  # Wait for the first car to end
STEP_GAP = 2  # Wait for the second car (wall)

# ==========================================
# [2] Helper Functions
# ==========================================
sonar_data = [999] * 6


def read_sensors():
    global sonar_data
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("US:"):
                parts = line.replace("US:", "").split(",")
                if len(parts) >= 6:
                    sonar_data = [int(p) for p in parts]
        except:
            pass


def get_lidar_dist_at_angle(scan, target_angle, angle_range=5):
    dists = []
    min_a = target_angle - angle_range
    max_a = target_angle + angle_range
    for (_, angle, dist) in scan:
        if dist > 0 and (min_a <= angle <= max_a):
            dists.append(dist)
    if len(dists) > 0: return np.mean(dists)
    return 9999


def draw_dashboard(state, rm, rt, lidar_dist, diff, search_step):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cx, cy = 320, 240
    cw, ch = 100, 180
    cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), (200, 200, 200), 2)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 100), (0, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Sensor Values
    cv2.putText(img, f"RM: {rm}", (cx + 60, cy), font, 0.7, (0, 255, 0) if rm < VALID_DIST else (0, 0, 255), 2)
    cv2.putText(img, f"RT: {rt}", (cx + 60, cy + 80), font, 0.7, (0, 255, 0) if rt < VALID_DIST else (0, 0, 255), 2)
    cv2.putText(img, f"Lidar R: {int(lidar_dist)}", (cx + 60, cy - 80), font, 0.6, (255, 255, 0), 1)

    # State Info
    state_str = ["SEARCH", "OFFSET", "SWING", "PREP REV", "ENTRY", "PREP STR", "FINISH", "DONE"]
    curr_state = state_str[state] if state < len(state_str) else "UNK"
    cv2.putText(img, f"STATE: {curr_state}", (20, 50), font, 1.0, (0, 255, 255), 2)

    # Search Step Info
    if state == STATE_SEARCH:
        step_str = ["FIND CAR1", "PASS CAR1", "FIND GAP"]
        curr_step = step_str[search_step] if search_step < len(step_str) else "UNK"
        cv2.putText(img, f"STEP: {curr_step}", (20, 100), font, 0.8, (255, 0, 255), 2)

    if state == STATE_REVERSE_ENTRY:
        cv2.putText(img, f"DIFF: {diff}", (20, 400), font, 1.2, (0, 255, 0) if diff < ALIGN_THRES else (0, 0, 255), 3)

    return img


# ==========================================
# [3] Main Loop
# ==========================================
def main():
    global ser, lidar

    try:
        ser = serial.Serial(PORT, 9600, timeout=0.1)
        lidar = RPLidar(LIDAR_PORT)
        print("✅ Hardware Connected")
        time.sleep(2)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    state = STATE_SEARCH
    # ★ Start from finding the first car
    search_step = STEP_FIND_FIRST_CAR
    timer = 0

    car1_count = 0
    gap_count = 0
    obs_count = 0

    print("🚀 Parking System Started")

    try:
        for scan in lidar.iter_scans():
            read_sensors()

            rm_dist = sonar_data[IDX_RM]
            rt_dist = sonar_data[IDX_RT]
            lidar_right = get_lidar_dist_at_angle(scan, 90, 5)
            diff = abs(rm_dist - rt_dist)

            cmd_speed = 0
            cmd_servo = SERVO_CENTER
            curr_time = time.time()

            # ---------------------------------------------------
            # [1] SEARCH STATE
            # ---------------------------------------------------
            if state == STATE_SEARCH:
                cmd_speed = SPEED_SEARCH

                # Step 0: Find 1st Car (Obstacle)
                if search_step == STEP_FIND_FIRST_CAR:
                    # Check if something is close (Car 1)
                    if lidar_right < OBSTACLE_DIST_MAX:
                        car1_count += 1
                    else:
                        car1_count = 0

                    if car1_count > 2:
                        print("🚗 1st Car Detected (Start Passing)")
                        search_step = STEP_PASS_CAR1
                        car1_count = 0

                # Step 1: Pass 1st Car -> Find Gap
                elif search_step == STEP_PASS_CAR1:
                    # Check if space opens up (Gap)
                    if lidar_right > EMPTY_DIST_MIN:
                        gap_count += 1
                    else:
                        gap_count = 0

                    if gap_count > 3:
                        print("👀 Gap Detected (Entering Empty Space)")
                        search_step = STEP_GAP
                        gap_count = 0

                # Step 2: Pass Gap -> Find 2nd Car
                elif search_step == STEP_GAP:
                    # Check if something appears again (Car 2)
                    if lidar_right < OBSTACLE_DIST_MAX:
                        obs_count += 1
                    else:
                        obs_count = 0

                    if obs_count > 2:
                        print("🛑 2nd Car Detected (Trigger)")
                        state = STATE_FORWARD_OFFSET
                        timer = curr_time

            # ---------------------------------------------------
            # [Rest of the logic remains the same]
            # ---------------------------------------------------
            elif state == STATE_FORWARD_OFFSET:
                cmd_speed = SPEED_SEARCH
                if curr_time - timer > FORWARD_OFFSET_TIME:
                    print("🛑 Stop & Prep Swing")
                    ser.write(b"D,-100\n");
                    time.sleep(0.1)
                    ser.write(b"D,0\n");
                    time.sleep(1.0)
                    state = STATE_SWING_OUT
                    timer = curr_time

            elif state == STATE_SWING_OUT:
                cmd_servo = SERVO_LEFT_MAX
                if curr_time - timer < STEER_WAIT_TIME:
                    cmd_speed = 0
                else:
                    cmd_speed = SPEED_SWING

                if curr_time - timer > (STEER_WAIT_TIME + 1.2):
                    print("🛑 Swing Done -> Prep Reverse")
                    ser.write(b"D,0\n");
                    time.sleep(0.5)
                    state = STATE_PREP_REVERSE
                    timer = curr_time

            elif state == STATE_PREP_REVERSE:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = 0
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_ENTRY
                    timer = curr_time

            elif state == STATE_REVERSE_ENTRY:
                cmd_servo = SERVO_RIGHT_MAX
                cmd_speed = -SPEED_PARK

                valid = (rm_dist < VALID_DIST and rt_dist < VALID_DIST)
                parallel = (diff < ALIGN_THRES)
                time_ok = (curr_time - timer > 1.0)

                if valid and parallel and time_ok:
                    print(f"✅ Parallel! (Diff: {diff})")
                    ser.write(b"D,0\n");
                    time.sleep(0.5)
                    state = STATE_PREP_STRAIGHT
                    timer = curr_time
                elif curr_time - timer > 4.0:
                    print("⚠️ Timeout -> Force Straight")
                    state = STATE_PREP_STRAIGHT
                    timer = curr_time

            elif state == STATE_PREP_STRAIGHT:
                cmd_servo = SERVO_CENTER
                cmd_speed = 0
                if curr_time - timer > STEER_WAIT_TIME:
                    state = STATE_REVERSE_FINISH
                    timer = curr_time

            elif state == STATE_REVERSE_FINISH:
                cmd_servo = SERVO_CENTER
                cmd_speed = -SPEED_PARK
                if curr_time - timer > PARKING_DEPTH_TIME:
                    print("🅿️ Done")
                    state = STATE_DONE

            elif state == STATE_DONE:
                cmd_speed = 0
                ser.write(b"D,0\n")

            ser.write(f"S,{cmd_servo}\n".encode())
            ser.write(f"D,{cmd_speed}\n".encode())

            dashboard = draw_dashboard(state, rm_dist, rt_dist, lidar_right, diff, search_step)
            cv2.imshow("Dashboard", dashboard)
            if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        if ser: ser.close()
        if lidar: lidar.stop(); lidar.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()