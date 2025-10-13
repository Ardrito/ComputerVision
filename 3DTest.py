import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# -----------------------------
# Camera Thread Class
# -----------------------------
class CameraThread(threading.Thread):
    def __init__(self, cam_id, width=640, height=480, fps=30):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame = None
        self.timestamp = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
                    self.timestamp = time.time()

    def get_frame(self):
        with self.lock:
            return self.frame, self.timestamp

    def stop(self):
        self.running = False
        self.cap.release()


# -----------------------------
# Load stereo calibration
# -----------------------------
calib = np.load("stereo_calibration.npz")
mtx_r, dist_r = calib["camera_matrix_right"], calib["dist_right"]
mtx_l, dist_l = calib["camera_matrix_left"], calib["dist_left"]
R, T = calib["R"], calib["T"]
#R, T = calib["R"], calib["T"]
# R = np.eye(3)  # identity rotation if cameras are parallel
# T = np.array([0.133, 0, 0])  # baseline = 133 mm = 0.133 m


# Projection matrices from stereo rectification
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, (640, 480), R, T
)

# Init undistortion maps
map_lx, map_ly = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (640,480), 5)
map_rx, map_ry = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (640,480), 5)

# -----------------------------
# Start cameras (adjust indices!)
# -----------------------------
camL = CameraThread(4)   # /dev/video0
camR = CameraThread(1)   # /dev/video2
camL.start()
camR.start()

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        frameL, tsL = camL.get_frame()
        frameR, tsR = camR.get_frame()

        if frameL is None or frameR is None:
            continue

        # Ensure near-synchronous
        if abs(tsL - tsR) * 1000 > 15:  # skip if >15 ms apart
            continue

        # Rectify frames
        rectL = cv2.remap(frameL, map_lx, map_ly, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map_rx, map_ry, cv2.INTER_LINEAR)

        # Run YOLO on both
        resultsL = model.predict(rectL, conf=0.5, device="cuda:0", verbose=False)
        resultsR = model.predict(rectR, conf=0.5, device="cuda:0", verbose=False)

        # Extract detections
        boxesL = [b for b in resultsL[0].boxes if model.names[int(b.cls[0])] == "person"]
        boxesR = [b for b in resultsR[0].boxes if model.names[int(b.cls[0])] == "person"]

        #Simplest case: assume same number of people, matched in order
        for boxL, boxR in zip(boxesL, boxesR):
            # Get centers
            uL, vL, _, _ = boxL.xywh[0].cpu().numpy()
            uR, vR, _, _ = boxR.xywh[0].cpu().numpy()

            # Triangulate
            ptsL = np.array([[uL], [vL]])
            ptsR = np.array([[uR], [vR]])
            point_4d = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
            point_3d = point_4d / point_4d[3]

            X, Y, Z = point_3d[:3].ravel()
            print(f"Person 3D position: X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")

            # Draw on left image
            cv2.circle(rectL, (int(uL), int(vL)), 5, (0,0,255), -1)
            cv2.putText(rectL, f"Z={Z:.2f}m", (int(uL), int(vL)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Show
        stereo_view = cv2.hconcat([rectL, rectR])
        cv2.imshow("Stereo YOLO", stereo_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camL.stop()
    camR.stop()
    camL.join()
    camR.join()
    cv2.destroyAllWindows()
