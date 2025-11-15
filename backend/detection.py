import cv2
import time
import numpy as np
from ultralytics import YOLO
import math
from collections import OrderedDict
import requests
from datetime import datetime

# -------------------- CONFIGURATION & CONSTANTS --------------------
# Model and Detection Settings
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.6
MIN_BOX_AREA = 4000

# Tracking Settings
MAX_DIST = 120
MAX_LOST = 50

# Frame and Room Geometry
FRAME_W, FRAME_H = 960, 540
ROOM_PADDING = 30
ROOM_X1, ROOM_Y1 = ROOM_PADDING, ROOM_PADDING
ROOM_X2, ROOM_Y2 = FRAME_W - ROOM_PADDING, FRAME_H - ROOM_PADDING

# Crossing Lines (Logic: Right -> Left)
ENTRY_LINE_X = ROOM_X2 - 80 # Green line (Right)
EXIT_LINE_X = ROOM_X1 + 80 # Red line (Left)

# Colors (B-G-R)
COL_ENTRY = (0, 255, 0) # Green
COL_EXIT = (0, 0, 255) # Red
COL_ROOM = (60, 60, 60) # Dark Grey
COL_INSIDE = (255, 255, 0) # Yellow/Cyan
COL_DEFAULT = (255, 255, 255) # White
COL_ALERT = (0, 0, 255) # Red for alert

# **[NEW FEATURE]** Crowd Limit Constant
MAX_PEOPLE = 10 # Do not detect or assign ID to more than 10 people

# Stats Storage
waiting_times = [] # Individual stay times. This list remains global.

# API Configuration
API_URL = "http://localhost:8000/updateData/"  # FastAPI backend URL


# -------------------- UTILITIES --------------------

def calculate_distance(p1: tuple, p2: tuple) -> float:
    """Calculates Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_centroid(box: tuple) -> tuple:
    """Calculates the center point (centroid) of a bounding box."""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_box_area(box: tuple) -> int:
    """Calculates the area of a bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def send_exit_data_to_db(person_id: int, entry_time: float, exit_time: float, wait_time: float, alert_popped: bool):
    """
    Sends person exit data to the database via POST request.
    
    Args:
        person_id: ID of the person
        entry_time: Unix timestamp of entry
        exit_time: Unix timestamp of exit
        wait_time: Calculated wait time in seconds
        alert_popped: Boolean indicating if alert was active (is_crowded)
    """
    try:
        # Convert Unix timestamps to datetime objects
        entry_datetime = datetime.fromtimestamp(entry_time)
        exit_datetime = datetime.fromtimestamp(exit_time)
        
        # Prepare data payload
        payload = {
            "id": person_id,
            "entryTime": entry_datetime.isoformat(),
            "exitTime": exit_datetime.isoformat(),
            "waitTime": wait_time,
            "alert": 1 if alert_popped else 0
        }
        
        # Send POST request to FastAPI backend
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Data saved to database for Person ID {person_id}")
        else:
            print(f"⚠️ Failed to save data for Person ID {person_id}: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending data to database for Person ID {person_id}: {e}")

def draw_stats_panel(panel: np.ndarray, entered: int, exited: int, total_wait: float, avg_wait: float):
    """Draws the statistics panel content."""
    # Note: total_wait and avg_wait are passed in as arguments now.
    cv2.putText(panel, "People Tracker", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_DEFAULT, 2)

    cv2.putText(panel, f"Entered: {entered}", (14, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_ENTRY, 2)
    cv2.putText(panel, f"Exited: {exited}", (14, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_EXIT, 2)
    cv2.putText(panel, f"Inside: {entered - exited}", (14, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_INSIDE, 2)

    # Waiting Time Stats
    cv2.putText(panel, "WAIT TIME (s)", (14, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_DEFAULT, 1)
    cv2.putText(panel, f"Total: {int(total_wait)}", (14, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_INSIDE, 2)
    cv2.putText(panel, f"Average: {avg_wait:.1f}", (14, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_INSIDE, 2)
    
    # **[NEW FEATURE]** Display MAX_PEOPLE limit
    cv2.putText(panel, f"Max Limit: {MAX_PEOPLE}", (14, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_DEFAULT, 1)

    cv2.putText(panel, "Press Q to Quit", (14, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)


# -------------------- INITIALIZATION --------------------

# Load Model
model = YOLO(MODEL_PATH)

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

# Tracking Variables
next_id = 0
tracks = OrderedDict()
entered, exited = 0, 0

print("✅ Press 'q' to quit.")


# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    results = model(frame, verbose=False, classes=0, conf=CONF_THRESHOLD) # Filter in inference step
    detections = results[0].boxes.data.cpu().numpy()

    # **[NEW FEATURE]** Check for Crowd Limit
    is_crowded = len(tracks) >= MAX_PEOPLE
    
    # 1. Prepare Valid Detections
    boxes = []
    for x1, y1, x2, y2, conf, cls in detections:
        box = (int(x1), int(y1), int(x2), int(y2))
        # Filter by minimum box area
        if get_box_area(box) > MIN_BOX_AREA:
            boxes.append(box)

    det_centroids = [get_centroid(b) for b in boxes]
    
    # **[NEW FEATURE]** If crowded, ignore new detections (to not create new IDs)
    # Only track existing people, but don't add new ones.
    if is_crowded:
        used_det_indices = [True] * len(boxes) # Treat all detections as "used" so no new tracks are created
    else:
        used_det_indices = [False] * len(boxes)

    # 2. Track Matching & Update (Standard logic unchanged)
    ids = list(tracks.keys())
    tr_centroids = [tracks[i]['centroid'] for i in ids]

    for i, c in enumerate(det_centroids):
        best_tid, best_d = None, MAX_DIST + 1

        # Find the closest existing track
        for j, tid in enumerate(ids):
            d = calculate_distance(c, tr_centroids[j])
            if d < best_d and d <= MAX_DIST:
                best_tid, best_d = tid, d

        # Update the matched track
        if best_tid is not None:
            t = tracks[best_tid]
            t['centroid'] = c
            t['box'] = boxes[i]
            t['lost'] = 0
            t['history'].append(c)
            # **[NEW FEATURE]** Only mark as used if not crowded, OR if it's an existing track being updated
            used_det_indices[i] = True 
    
    # 3. Create New Tracks (Modified to check for crowd limit)
    # **[NEW FEATURE]** New tracks are only created if the limit is NOT reached.
    if not is_crowded:
        for i, is_used in enumerate(used_det_indices):
            if not is_used:
                # **[NEW FEATURE]** Second check to ensure we don't exceed MAX_PEOPLE even mid-loop
                if len(tracks) < MAX_PEOPLE:
                    tracks[next_id] = {
                        'centroid': det_centroids[i],
                        'box': boxes[i],
                        'history': [det_centroids[i]],
                        'lost': 0,
                        'entered': False,
                        'exited': False,
                        'entry_time': None,
                        'waiting_time': None
                    }
                    next_id += 1
                else:
                    # Break loop if adding this track hits the limit
                    break


    # 4. Remove Lost Tracks (Standard logic unchanged)
    for tid in list(tracks.keys()):
        tracks[tid]['lost'] += 1
        if tracks[tid]['lost'] > MAX_LOST:
            # OPTIONAL: Log unaccounted tracks before deletion
            if tracks[tid]['entered'] and not tracks[tid]['exited']:
                print(f"⚠️ ID {tid} LOST without exiting!")
            del tracks[tid]

    # 5. ENTRY/EXIT LOGIC & Statistics Update (Standard logic unchanged)
    for tid, t in tracks.items():
        if len(t['history']) < 2:
            continue

        p_prev = t['history'][-2] # Previous centroid position
        p_curr = t['history'][-1] # Current centroid position

        # Crossing check (movement right -> left: p_prev.x > Line && p_curr.x < Line)

        # ENTRY crossing (Line: ENTRY_LINE_X - Right side)
        if (
            not t['entered'] and
            p_prev[0] > ENTRY_LINE_X and p_curr[0] < ENTRY_LINE_X
        ):
            t['entered'] = True
            t['entry_time'] = time.time()
            entered += 1
            print(f"🟢 ID {tid} ENTERED at {time.strftime('%H:%M:%S')}")

        # EXIT crossing (Line: EXIT_LINE_X - Left side)
        elif (
            t['entered'] and not t['exited'] and
            p_prev[0] > EXIT_LINE_X and p_curr[0] < EXIT_LINE_X
        ):
            t['exited'] = True
            exit_time = time.time()
            exited += 1

            # Calculate and store waiting time (no global needed here)
            stay = exit_time - t['entry_time']
            t['waiting_time'] = stay
            waiting_times.append(stay) # Only append to the global list

            print(f"🔴 ID {tid} EXITED | Stay: {stay:.1f}s")

            # Print individual's detailed wait info
            print(f"🕒 Individual Wait Details:")
            print(f"    ↳ Person ID: {tid}")
            print(f"    ↳ Entry Time: {time.strftime('%H:%M:%S', time.localtime(t['entry_time']))}")
            print(f"    ↳ Exit Time:  {time.strftime('%H:%M:%S', time.localtime(exit_time))}")
            print(f"    ↳ Wait Duration: {stay:.2f} seconds")
            
            # Send exit data to database with alert status
            # Alert is 1 if crowd limit was reached (is_crowded) at exit time, 0 otherwise
            send_exit_data_to_db(
                person_id=tid,
                entry_time=t['entry_time'],
                exit_time=exit_time,
                wait_time=stay,
                alert_popped=is_crowded
            )
            print()


    # 6. DRAWING SECTION
    # Room Boundary
    cv2.rectangle(frame, (ROOM_X1, ROOM_Y1), (ROOM_X2, ROOM_Y2), COL_ROOM, 2)

    # ENTRY line (Right)
    cv2.line(frame, (ENTRY_LINE_X, ROOM_Y1), (ENTRY_LINE_X, ROOM_Y2), COL_ENTRY, 3)
    cv2.putText(frame, "ENTRY", (ENTRY_LINE_X - 70, ROOM_Y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_ENTRY, 2)

    # EXIT line (Left)
    cv2.line(frame, (EXIT_LINE_X, ROOM_Y1), (EXIT_LINE_X, ROOM_Y2), COL_EXIT, 3)
    cv2.putText(frame, "EXIT", (EXIT_LINE_X + 10, ROOM_Y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_EXIT, 2)

    # Draw tracked persons
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t['box']

        # Set color based on state
        if t['entered'] and not t['exited']:
            color = COL_INSIDE # Inside (Yellow/Cyan)
        elif t['exited']:
            color = COL_EXIT  # Exited (Red)
        else:
            color = COL_DEFAULT # Default (White)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Prepare label text
        label = f"ID:{tid}"
        if t['entered'] and not t['exited']:
            stay = int(time.time() - t['entry_time'])
            label += f" | {stay}s (in)"
        elif t['exited']:
            stay = int(t['waiting_time'])
            label += f" | {stay}s (wait)"

        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # **[NEW FEATURE]** Alert Message for Crowd Limit
    if is_crowded:
        alert_text = "CROWD ALERT: OVER LIMIT!"
        (text_w, text_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_x = int((FRAME_W - text_w) / 2)
        text_y = int(FRAME_H / 2)
        
        # Draw background for alert
        cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10), (text_x + text_w + 10, text_y + 10), (0, 0, 255), -1)
        # Draw alert text
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        print(f"🚨 CROWD ALERT: Limit of {MAX_PEOPLE} reached or exceeded. New people are NOT being tracked.")

    # 7. SIDE PANEL
    panel = np.full((FRAME_H, 300, 3), (25, 25, 25), np.uint8) # Dark background

    # Calculate statistics right before drawing them
    current_total_waiting = sum(waiting_times)
    current_average_waiting = current_total_waiting / len(waiting_times) if waiting_times else 0.0
    
    draw_stats_panel(panel, entered, exited, current_total_waiting, current_average_waiting)

    # Combine view + panel
    cv2.imshow("Smart Entry-Exit Tracker", np.hstack((frame, panel)))

    # 8. Quit Handler
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

final_total_waiting = sum(waiting_times)
final_average_waiting = final_total_waiting / len(waiting_times) if waiting_times else 0.0

print(f"\n✅ FINAL SUMMARY")
print(f"  → Total Entered: {entered}")
print(f"  → Total Exited: {exited}")
print(f"  → Current Inside (Estimated): {entered - exited}")
print(f"  → Total Wait Time Recorded: {final_total_waiting:.1f}s")
print(f"  → Average Wait Time: {final_average_waiting:.1f}s")