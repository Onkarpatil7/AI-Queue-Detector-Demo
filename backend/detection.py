import cv2
import time
import numpy as np
from ultralytics import YOLO
import requests
from datetime import datetime

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

person_entry_time = {}
person_wait_time = {}
previous_boxes = {}
next_id = 0
API_URL = "http://127.0.0.1:8000/update_data/"
last_sent_time = 0

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    a1, b1, a2, b2 = box2
    inter_x1, inter_y1 = max(x1, a1), max(y1, b1)
    inter_x2, inter_y2 = min(x2, a2), min(y2, b2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (a2 - a1) * (b2 - b1)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area else 0

while True:
    ret, frame = cap.read()
    if not ret:
        break


    frame = cv2.resize(frame, (960, 540))
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    current_boxes = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0 and conf > 0.5:
            current_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    new_boxes = {}
    used_ids = set()

    for box in current_boxes:
        matched_id = None
        best_iou = 0
        for pid, prev_box in previous_boxes.items():
            iou_score = iou(box, prev_box)
            if iou_score > 0.3 and iou_score > best_iou and pid not in used_ids:
                matched_id = pid
                best_iou = iou_score
        if matched_id is None:
            matched_id = next_id
            next_id += 1
            person_entry_time[matched_id] = time.time()
        used_ids.add(matched_id)
        new_boxes[matched_id] = box
        person_wait_time[matched_id] = time.time() - person_entry_time[matched_id]

    previous_boxes = new_boxes
    person_count = len(new_boxes)
    avg_wait_time = sum(person_wait_time.values()) / person_count if person_count > 0 else 0

    for pid, box in new_boxes.items():  
        x1, y1, x2, y2 = box
        wt = person_wait_time[pid]
        if wt < 20:
            color = (0, 255, 0)
        elif wt < 40:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    panel_width = 300
    panel = np.ones((frame.shape[0], panel_width, 3), dtype=np.uint8) * 30
    cv2.putText(panel, "ID   | Time(s)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(panel, (20, 50), (panel_width - 20, 50), (255, 255, 255), 1)

    y_offset = 80
    for pid, wt in sorted(person_wait_time.items(), key=lambda x: x[1], reverse=True):
        if wt < 20:
            c = (0, 255, 0)
        elif wt < 40:
            c = (0, 255, 255)
        else:
            c = (0, 0, 255)
        cv2.putText(panel, f"{pid:<4} | {wt:5.1f}", (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
        y_offset += 35

    cv2.putText(panel, f"Total: {person_count}", (30, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f"Avg Wait: {avg_wait_time:.1f}s", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    combined = np.hstack((frame, panel))
    cv2.imshow("Queue Detector (YOLOv8)", combined)

    if time.time() - last_sent_time > 10:
        for pid, wt in person_wait_time.items():
            try:
                payload = {"id": pid, "timeStamp": datetime.now().isoformat(), "waitTime": wt}
                requests.post(API_URL, json=payload)
                print(f"Sent -> ID: {pid}, Wait: {wt:.1f}s")
            except Exception as e:
                print("Send failed:", e)
        last_sent_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for pid, wt in sorted(person_wait_time.items()):
    print(f"Person {pid}: {wt:.1f} seconds")
