import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO


class CameraThread(threading.Thread):
    def __init__(self, cam_id, resolution=(1280, 720), fps=30):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.frame = None
        self.timestamp = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
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
        self.cap.release()

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / (n + eps)

def closest_approach_triangulation(C1, d1, C2, d2, clamp_front=True, eps=1e-12):
    """All vectors in SAME frame (here: LEFT camera frame)."""
    d1 = normalize(d1, eps)
    d2 = normalize(d2, eps)
    w0 = C1 - C2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))
    den = a * c - b * b
    if abs(den) < 1e-14:
        s = 0.0
        t = e / (c + eps)
    else:
        s = (b * e - c * d) / den
        t = (a * e - b * d) / den
    if clamp_front:
        s = max(0.0, s); t = max(0.0, t)
    P1 = C1 + s * d1
    P2 = C2 + t * d2
    P  = 0.5 * (P1 + P2)
    gap = float(np.linalg.norm(P1 - P2))
    return P, P1, P2, gap

def main():
    LEFT_CAM_ID  = 0
    RIGHT_CAM_ID = 2
    RESOLUTION   = (1280, 720)
    FPS_REQ      = 30
    YOLO_MODEL   = "yolov8n.pt"
    CONF_THRES   = 0.5
    CALIB_FILE   = "stereo_calibration.npz"

    cal = np.load(CALIB_FILE)

    K_L   = cal["camera_matrix_left"].astype(np.float64)
    K_R   = cal["camera_matrix_right"].astype(np.float64)
    distL = cal["dist_left"].astype(np.float64)
    distR = cal["dist_right"].astype(np.float64)
    R_lr  = cal["R"].astype(np.float64)             # maps LEFT->RIGHT
    T_lr  = cal["T"].astype(np.float64).reshape(3, 1)

    camL = CameraThread(LEFT_CAM_ID, RESOLUTION, FPS_REQ)
    camR = CameraThread(RIGHT_CAM_ID, RESOLUTION, FPS_REQ)
    camL.start(); camR.start()

    frameL, _ = None, None
    frameR, _ = None, None
    for _ in range(100):
        frameL, _ = camL.get_frame()
        frameR, _ = camR.get_frame()
        if frameL is not None and frameR is not None:
            break
        time.sleep(0.01)
    if frameL is None or frameR is None:
        camL.stop(); camR.stop()
        camL.join(); camR.join()
        raise RuntimeError("Could not grab initial frames from both cameras.")

    hL, wL = frameL.shape[:2]
    hR, wR = frameR.shape[:2]

    newK_L, _ = cv2.getOptimalNewCameraMatrix(K_L, distL, (wL, hL), alpha=0)
    newK_R, _ = cv2.getOptimalNewCameraMatrix(K_R, distR, (wR, hR), alpha=0)

    mapxL, mapyL = cv2.initUndistortRectifyMap(K_L, distL, np.eye(3), newK_L, (wL, hL), cv2.CV_32FC1)
    mapxR, mapyR = cv2.initUndistortRectifyMap(K_R, distR, np.eye(3), newK_R, (wR, hR), cv2.CV_32FC1)


    Kinv_L = np.linalg.inv(newK_L)
    Kinv_R = np.linalg.inv(newK_R)

    C_L = np.zeros(3, dtype=np.float64)
    C_R = (-R_lr.T @ T_lr).ravel()    



    fxL, fyL = float(newK_L[0,0]), float(newK_L[1,1])
    cxL, cyL = float(newK_L[0,2]), float(newK_L[1,2])
    fxR, fyR = float(newK_R[0,0]), float(newK_R[1,1])
    cxR, cyR = float(newK_R[0,2]), float(newK_R[1,2])



    P1 = newK_L @ np.hstack([np.eye(3), np.zeros((3,1))])         # K'L [I|0]
    P2 = newK_R @ np.hstack([R_lr, T_lr])                         # K'R [R|T]


    model = YOLO(YOLO_MODEL).to("cuda")

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255)
    org1 = (20, 40); org2 = (20, 75)
    prev = time.time()

    try:
        while True:
            fL, tsL = camL.get_frame()
            fR, tsR = camR.get_frame()
            if fL is None or fR is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Undistort only (no rectification)
            imgL = cv2.remap(fL, mapxL, mapyL, cv2.INTER_LINEAR)
            imgR = cv2.remap(fR, mapxR, mapyR, cv2.INTER_LINEAR)
            

            # Detect
            resL = model.predict(imgL, conf=0.5, device="cuda:0", verbose=False)
            resR = model.predict(imgR, conf=0.5, device="cuda:0", verbose=False)

            boxesL = [b for b in resL[0].boxes if resL[0].names[int(b.cls[0])] == "person"]
            boxesR = [b for b in resR[0].boxes if resR[0].names[int(b.cls[0])] == "person"]

            overlay1, overlay2 = "No pairs", ""
            if boxesL and boxesR:
                for boxL, boxR in zip(boxesL, boxesR):
                    uL, vL, wL, hL = boxL.xywh[0].detach().cpu().numpy()
                    uR, vR, wR, hR = boxR.xywh[0].detach().cpu().numpy()
                    # choose a stable point (slightly above center)
                    uL = float(uL); vL = float(vL - hL/5.0)
                    uR = float(uR); vR = float(vR - hR/5.0)

                    #Method A: RAY triangulation in LEFT frame
                    pL = np.array([uL, vL, 1.0], dtype=np.float64)
                    pR = np.array([uR, vR, 1.0], dtype=np.float64)
                    dL_cam = normalize(Kinv_L @ pL)
                    dR_cam = normalize(Kinv_R @ pR)
                    dR_left = R_lr.T @ dR_cam                 

                    P_ray, _, _, gap = closest_approach_triangulation(C_L, dL_cam, C_R, dR_left)
                    Z_ray = float(P_ray[2])

                    #Method B: cv2.triangulatePoints with P1, P2 (explicit R & T)
                    pL2 = np.array([[uL], [vL]], dtype=np.float64)
                    pR2 = np.array([[uR], [vR]], dtype=np.float64)
                    X4 = cv2.triangulatePoints(P1, P2, pL2, pR2)
                    X  = (X4[:3] / X4[3]).ravel()              
                    Z_tr = float(X[2])

                    print(f"Ray: {Z_ray:.2f}    openCV: {Z_tr:.2f}")

                    diff = abs(Z_ray - Z_tr)

                    overlay1 = f"Z_ray: {Z_ray:.2f} m   Z_triang: {Z_tr:.2f} m   |Δ|: {diff:.2f} m"
                    overlay2 = f"gap: {gap:.03f} m   (cxL={cxL:.1f}, cxR={cxR:.1f})"

                    # debug points
                    cv2.circle(imgL, (int(round(uL)), int(round(vL))), 5, (0,0,255), -1)
                    cv2.circle(imgR, (int(round(uR)), int(round(vR))), 5, (0,0,255), -1)
                    cv2.circle(imgL, (int(cxL), int(cyL)), 6, (255,255,0), 1)
                    cv2.circle(imgR, (int(cxR), int(cyR)), 6, (255,255,0), 1)
                    break

            # Show
            view = cv2.hconcat([imgL, imgR])
            cv2.putText(view, overlay1, org1, font, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(view, overlay2, org2, font, 0.8, color, 2, cv2.LINE_AA)

            now = time.time()
            fps = 1.0 / (now - prev + 1e-9)
            prev = now
            cv2.putText(view, f"FPS: {fps:.1f}", (20, 110), font, 0.7, (0,200,0), 2, cv2.LINE_AA)
            cv2.imshow("left",imgL)
            cv2.imshow("right", imgR)
            cv2.imshow("Stereo (Undistorted, Unrectified) — uses full R & T", view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camL.stop(); camR.stop()
        camL.join(); camR.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
