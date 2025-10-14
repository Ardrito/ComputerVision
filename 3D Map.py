import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# Use interactive GUI backend for Matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# -----------------------------
# Camera Thread Class
# -----------------------------
class CameraThread(threading.Thread):
    def __init__(self, cam_id, width=1280, height=720, fps=30):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.frame = None
        self.timestamp = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            ts = time.time()
            with self.lock:
                self.frame = frame
                self.timestamp = ts

    def get_frame(self):
        with self.lock:
            return self.frame, self.timestamp

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# -----------------------------
# Live 3D Viewer (fixed 5x5x5m) with triangulation rays
# -----------------------------
class Live3D:
    def __init__(self, T_vec):
        self.T = np.asarray(T_vec).reshape(3)
        self.baseline = float(np.linalg.norm(self.T))

        plt.ion()
        self.fig = plt.figure("3D Positions (Fixed 5x5x5 m space + Rays)", figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m) [down is positive]")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=20, azim=-60)

        # Fixed 5x5x5 m space centered between cameras
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(2.5, -2.5)  # flipped Y-axis
        self.ax.set_zlim(0.0, 5.0)

        # Camera positions (in midpoint frame)
        self.B2 = self.baseline / 2.0
        self.camL = np.array([-self.B2, 0.0, 0.0])
        self.camR = np.array([+self.B2, 0.0, 0.0])
        self.ax.scatter([self.camL[0], self.camR[0]],
                        [self.camL[1], self.camR[1]],
                        [self.camL[2], self.camR[2]],
                        s=80, marker='^', label='Cameras', c='blue')

        # Scatter for detected points
        self.scatter_pts = self.ax.scatter([], [], [], s=60, marker='o', label='Detections', c='red')

        # Initialize Line3DCollections with dummy data (needed for 3D add)
        dummy_seg = np.array([[[0, 0, 0], [0, 0, 0]]])
        self.rays_left = Line3DCollection(dummy_seg, colors='green', linewidths=1.5, alpha=0.9, label='Left rays')
        self.rays_right = Line3DCollection(dummy_seg, colors='orange', linewidths=1.5, alpha=0.9, label='Right rays')

        self.ax.add_collection3d(self.rays_left)
        self.ax.add_collection3d(self.rays_right)
        self.ax.legend(loc='upper left')

        plt.show(block=False)
        self._flush()

    def update(self, points_midframe):
        if points_midframe.size == 0:
            self.scatter_pts._offsets3d = ([], [], [])
            self.rays_left.set_segments(np.array([[[0, 0, 0], [0, 0, 0]]]))
            self.rays_right.set_segments(np.array([[[0, 0, 0], [0, 0, 0]]]))
            self.fig.canvas.draw_idle()
            self._flush()
            return

        P = np.asarray(points_midframe)
        # Flip Y axis before plotting
        P[:, 1] *= -1

        xs, ys, zs = P[:, 0], P[:, 1], P[:, 2]
        self.scatter_pts._offsets3d = (xs, ys, zs)

        # Build ray segments
        segs_left = np.stack([np.tile(self.camL, (P.shape[0], 1)), P], axis=1)
        segs_right = np.stack([np.tile(self.camR, (P.shape[0], 1)), P], axis=1)
        self.rays_left.set_segments(segs_left)
        self.rays_right.set_segments(segs_right)

        self.fig.canvas.draw_idle()
        self._flush()

    def _flush(self):
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass
        try:
            self.fig.canvas.start_event_loop(0.001)
        except Exception:
            plt.pause(0.001)

# -----------------------------
# Load stereo calibration
# -----------------------------
calib = np.load("stereo_calibration.npz")
mtx_r, dist_r = calib["camera_matrix_right"], calib["dist_right"]
mtx_l, dist_l = calib["camera_matrix_left"], calib["dist_left"]
R, T = calib["R"], calib["T"]

if np.linalg.norm(T) > 1.0:
    T = T / 1000.0

viz = Live3D(T)

# Stereo rectification
#img_size = (640, 480)
img_size = (1280,720)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T)
map_lx, map_ly = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, 5)
map_rx, map_ry = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, 5)

# -----------------------------
# Start cameras (adjust indices!)
# -----------------------------
camL = CameraThread(2)
camR = CameraThread(0)
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
            viz.update(np.zeros((0, 3)))
            continue

        if tsL is None or tsR is None or abs(tsL - tsR) * 1000.0 > 40.0:
            viz.update(np.zeros((0, 3)))
            continue

        rectL = cv2.remap(frameL, map_lx, map_ly, cv2.INTER_LINEAR)
        rectR = cv2.remap(frameR, map_rx, map_ry, cv2.INTER_LINEAR)

        resultsL = model.predict(rectL, conf=0.4, device="cuda:0", verbose=False)
        resultsR = model.predict(rectR, conf=0.4, device="cuda:0", verbose=False)

        peopleL = [b for b in resultsL[0].boxes if model.names[int(b.cls[0])] == "person"]
        peopleR = [b for b in resultsR[0].boxes if model.names[int(b.cls[0])] == "person"]

        midframe_points = []

        for bL, bR in zip(peopleL, peopleR):
            uL, vL, _, _ = bL.xywh[0].cpu().numpy()
            uR, vR, _, _ = bR.xywh[0].cpu().numpy()

            ptsL = np.array([[uL], [vL]], dtype=np.float32)
            ptsR = np.array([[uR], [vR]], dtype=np.float32)

            p4 = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
            p3 = (p4 / p4[3])[:3].ravel()

            point_mid = p3 - (T.reshape(3) / 2.0)
            midframe_points.append(point_mid)

            cv2.circle(rectL, (int(uL), int(vL)), 6, (0, 0, 255), -1)
            cv2.putText(rectL, f"Z={p3[2]:.2f}m", (int(uL), int(vL) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if midframe_points:
            P = np.vstack(midframe_points)
            viz.update(P)
        else:
            viz.update(np.zeros((0, 3)))

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
