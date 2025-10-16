import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO("yolov8n.pt")

class CameraThread(threading.Thread):
    def __init__(self, cam_id, resolution: tuple[int,int] = (1280,720), fps:int = 30):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.frame = None
        self.timestamp = None
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
                self.timestamp = time.time()

    def get_frame(self):
        with self.lock:
            return self.frame, self.timestamp

    def stop(self):
        self.running = False
        self.cap.release()


plt.ion()  # Turn on interactive mode

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X (right)')
ax.set_ylabel('Y (down)')
ax.set_zlabel('Z (forward)')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)
ax.set_title('Live 3D Vector Visualization')

# Initialize the quiver (arrow)
quiver_l = ax.quiver(-0.0665, 0, 0, 0, 0, 1, color='r', length=1.0, normalize=True,arrow_length_ratio=0)
quiver_r = ax.quiver(0.0665, 0, 0, 0, 0, 1, color='r', length=1.0, normalize=True,arrow_length_ratio=0)
plt.show(block=False)

calib = np.load("camera_calibration_data.npz")
mapx_l, mapy_l, roi_l, newMtx_l = calib["mapx_one"], calib["mapy_one"], calib["roi_one"], calib["newMtx_one"]
mapx_r, mapy_r, roi_r, newMtx_r = calib["mapx_two"], calib["mapy_two"], calib["roi_two"], calib["newMtx_two"]
x_l, y_l, w_l, h_l = roi_l
x_r, y_r, w_r, h_r = roi_r

cp_l = (int(newMtx_l[0][2]),int(newMtx_l[1][2]))
cp_r = (int(newMtx_r[0][2]), int(newMtx_r[1][2]))


#Approx width and height visible in each camera for current calibration @ 1280,720
w_viz = 600 
h_viz = 325


camL = CameraThread(0)  # left camera
camR = CameraThread(2)  # right camera
camL.start()
camR.start()

quivers = {}

# def update_ray(id, ray_l, ray_r):
#     if id in quivers:
#         quivers[id].remove()
#     quiver_l.remove()  # remove the old arrow
#     quiver_l = ax.quiver(-0.0665, 0, 0, ray_l[0], ray_l[1], ray_l[2],
#                     length=1.0, normalize=True, color='r')
    
#     quiver_r.remove()  # remove the old arrow
#     quiver_r = ax.quiver(0.0665, 0, 0, ray_r[0], ray_r[1], ray_r[2],
#                     length=1.0, normalize=True, color='b')


prev_time = time.time()
try:
    while True:
        frameL, tsL = camL.get_frame()
        frameR, tsR = camR.get_frame()

        if frameL is not None and frameR is not None:
            # undist_frameL = cv2.remap(frameL, mapx_l, mapy_l, cv2.INTER_LINEAR)
            # undist_frameL = undist_frameL[cp_l[1]-h_viz: cp_l[1] + h_viz, cp_l[0]-w_viz:cp_l[0] + w_viz]

            # undist_frameR = cv2.remap(frameR, mapx_r, mapy_r, cv2.INTER_LINEAR)
            # undist_frameR = undist_frameR[cp_r[1]-h_viz: cp_r[1] + h_viz, cp_r[0]-w_viz:cp_r[0] + w_viz]

            undist_frameL = frameL
            undist_frameR = frameR

            # Run YOLO on both
            resultsL = model.predict(undist_frameL, conf=0.5, device="cuda:0", verbose=False)
            resultsR = model.predict(undist_frameR, conf=0.5, device="cuda:0", verbose=False)

            # Extract detections
            boxesL = [b for b in resultsL[0].boxes if model.names[int(b.cls[0])] == "person"]
            boxesR = [b for b in resultsR[0].boxes if model.names[int(b.cls[0])] == "person"]

            box_id = 0

            for boxL, boxR in zip(boxesL, boxesR):
            # Get centers
                uL, vL, _, h_boxL = boxL.xywh[0].cpu().numpy()
                uR, vR, _, h_boxR = boxR.xywh[0].cpu().numpy()
                vL = vL - (h_boxL/5)
                vR = vR - (h_boxR/5)

                # pixel_h_l = np.array([uL, vL, 1.0])
                # ray_l = np.linalg.inv(newMtx_l) @ pixel_h_l
                # ray_l /= np.linalg.norm(ray_l)

                # pixel_h_r = np.array([uR, vR, 1.0])
                # ray_r = np.linalg.inv(newMtx_r) @ pixel_h_r
                # ray_r /= np.linalg.norm(ray_r)

                # print (ray_l)
                # print (ray_r)
                print (box_id)

                # quiver_l.remove()  # remove the old arrow
                # quiver_l = ax.quiver(-0.0665, 0, 0, ray_l[0], ray_l[1], ray_l[2],
                #                 length=1.0, normalize=True, color='r')
                
                # quiver_r.remove()  # remove the old arrow
                # quiver_r = ax.quiver(0.0665, 0, 0, ray_r[0], ray_r[1], ray_r[2],
                #                 length=1.0, normalize=True, color='b')
                #update_ray(box_id,ray_l,ray_r)
                
                

                cv2.circle(undist_frameL, (int(uL), int(vL)), 5, (0,0,255), -1)
                cv2.circle(undist_frameR, (int(uR), int(vR)), 5, (0,0,255), -1)

                cv2.circle(undist_frameL, (int(newMtx_l[0][2]),int(newMtx_l[1][2])), 6, (255,255,0), 1)
                cv2.circle(undist_frameR, (int(newMtx_r[0][2]),int(newMtx_r[1][2])), 6, (255,255,0), 1)

                box_id += 1

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Timestamp difference (ms)
            delta = abs(tsL - tsR) * 1000
            curr_time = time.time()
            cap_fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            print(f"Timestamp difference: {delta:.2f} ms   FPS: {cap_fps:.2f}")

            # Show side by side
            # cv2.imshow("L", undist_frameL)
            # cv2.imshow("R", undist_frameR)
            stereo = cv2.hconcat([undist_frameL, undist_frameR])
            cv2.imshow("Stereo Pair", stereo)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camL.stop()
    camR.stop()
    camL.join()
    camR.join()
    cv2.destroyAllWindows()