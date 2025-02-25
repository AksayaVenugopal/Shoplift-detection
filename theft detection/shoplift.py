import cv2
import torch
import numpy as np
import pandas as pd
import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

print("Libraries Imported")

# Load YOLOv8 model (Replace with custom model if trained)
model = YOLO("yolov8s.pt")
print("YOLO Model Loaded")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50)  # Increased age to handle temporary occlusions

# Define product classes (adjust based on dataset)
PRODUCT_CLASSES = ["apple", "bottle", "cup", "box", "handbag", "backpack"]

# Store tracking info
object_tracking = {}

# Initialize CSV logging
CSV_FILE = "theft_log.csv"
if not pd.io.common.file_exists(CSV_FILE):
    df = pd.DataFrame(columns=["Timestamp", "Person_ID", "Stolen_Item"])
    df.to_csv(CSV_FILE, index=False)

# Function to log theft events in CSV
def log_theft_event(person_id, item):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, person_id, item]], columns=["Timestamp", "Person_ID", "Stolen_Item"])
    df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    print(f"📝 Theft Logged: {timestamp}, Person {person_id}, Item: {item}")

# Function to save theft snapshot
def save_theft_snapshot(frame, person_id, item):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"theft_images/person_{person_id}_{item}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Theft snapshot saved: {filename}")

print("🎥 Opening Camera...")
cap = cv2.VideoCapture(0)  # Open webcam

# Ensure the window appears
cv2.namedWindow("Retail Theft Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Retail Theft Detection", 800, 600)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read frame. Exiting...")
        break

    frame_count += 1

    # Run YOLO only every 3rd frame to improve speed
    if frame_count % 3 == 0:
        results = model(frame)[0]

        # Extract detections (bounding boxes, confidence, class IDs)
        detections = []
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, class_id = box.tolist()
            class_name = model.names[int(class_id)]

            if class_name in PRODUCT_CLASSES or class_name == "person":
                detections.append([[x1, y1, x2, y2], conf, int(class_id)])

        # Run tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # Process each tracked object
        person_bboxes = []  # Store all detected persons

        for track in tracks:
            if not track.is_confirmed():  # Ignore unconfirmed tracks
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()  # Bounding box (Top-Left-Bottom-Right format)
            class_id = track.det_class
            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, bbox)

            # Track person separately for logic
            if class_name == "person":
                person_bboxes.append((x1, y1, x2, y2))

            # Store object tracking information
            if track_id not in object_tracking:
                object_tracking[track_id] = {"class": class_name, "status": "on_shelf"}

            # Check if the object is near any detected person
            for px1, py1, px2, py2 in person_bboxes:
                if x1 > px1 and x2 < px2 and y1 > py1 and y2 < py2:
                    object_tracking[track_id]["status"] = "in_hand"

            # Check if the product disappears (not detected for 5+ frames)
            if track_id in object_tracking and object_tracking[track_id]["status"] == "in_hand" and track.time_since_update > 5:
                print(f"🚨 Theft Alert! Object '{class_name}' stolen by person {track_id}!")
                log_theft_event(track_id, class_name)
                save_theft_snapshot(frame, track_id, class_name)
                object_tracking[track_id]["status"] = "stolen"

            # Draw bounding box & label
            color = (0, 255, 0) if object_tracking[track_id]["status"] == "on_shelf" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} ({track_id})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow("Retail Theft Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program Exited Cleanly")
