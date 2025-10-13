import numpy as np

import cv2
import torch
from ultralytics import YOLO
import time





# --- STEP 1: Load calibration file ---
# Load calibration data
calib_data = np.load("calibration.npz")
mtx = calib_data["cameraMatrix"]
dist = calib_data["dist"]

print("Camera Matrix:\n", mtx)
print("Distortion Coeffs:\n", dist)


# --- STEP 2: Open webcam ---
cap = cv2.VideoCapture(1)  # Change index if multiple cameras

# Get first frame to set up undistortion map
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")
h, w = frame.shape[:2]

# Compute undistortion map + ROI
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
x, y, crop_w, crop_h = roi  # ROI rectangle

# --- STEP 3: Load YOLO model ---
model = YOLO("yolov8n.pt")  # nano model = fastest for real-time
model.to("cuda")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", model.device)

# --- STEP 4: Real-time loop ---
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort + crop
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    cropped = undistorted[y:y+crop_h, x:x+crop_w]

    # Run YOLO inference
    results = model.predict(cropped, conf=0.5, verbose=False)

    person_coords = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                x_c, y_c, w_box, h_box = box.xywh[0]
                cx, cy = int(x_c), int(y_c)
                person_coords.append((cx, cy))

                # Draw bounding box
                x1 = int(cx - w_box / 2)
                y1 = int(cy - h_box / 2)
                x2 = int(cx + w_box / 2)
                y2 = int(cy + h_box / 2)

                cv2.rectangle(cropped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(cropped, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(cropped, f"({cx}, {cy})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- FPS Calculation ---
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # Print detected coordinates + FPS
    if person_coords:
        print(f"Detected person(s): {person_coords} | FPS: {fps:.2f}")

    # Overlay FPS on video
    cv2.putText(cropped, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show video
    cv2.imshow("Real-Time Detection (Cropped)", cropped)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()