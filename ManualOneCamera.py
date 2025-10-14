import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
import torch

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import time

model = YOLO("yolov8n.pt")  # nano model = fastest for real-time
model.to("cuda")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", model.device)
prev_time = time.time()

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
quiver = ax.quiver(0, 0, 0, 0, 0, 1, color='r', length=1.0, normalize=True)
plt.show(block=False)

calib = np.load("camera_calibration_data.npz")
mtx_r, dist_r = calib["camera_matrix_one"], calib["dist_one"]
#mtx_l, dist_l = calib["camera_matrix_left"], calib["dist_left"]
R, T = calib["R"], calib["T"]

cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from camera.")
h_frame, w_frame = frame.shape[:2]
newMtx_r, roi = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(w_frame,h_frame),1,(w_frame,h_frame))
mapx, mapy = cv2.initUndistortRectifyMap(mtx_r, dist_r, None, newMtx_r, (w_frame, h_frame), 5)
x,y,w,h = roi


while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistImg = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    undistImg = undistImg[y:y+h, x:x+w]

    results = model.predict(undistImg, conf=0.5, verbose=False)

    person_coords = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                x_c, y_c, _, h_box = box.xywh[0]
                cx, cy = int(x_c), int(y_c-(h_box/5))
                #print (cx,cy)
                pixel_h = np.array([cx, cy, 1.0])
                ray = np.linalg.inv(newMtx_r) @ pixel_h
                ray /= np.linalg.norm(ray)

                print (newMtx_r)
                print (w,h, w_frame, h_frame)
                print (ray)
                quiver.remove()  # remove the old arrow
                quiver = ax.quiver(0, 0, 0, ray[0], ray[1], ray[2],
                                length=1.0, normalize=True, color='r')

                fig.canvas.draw()
                fig.canvas.flush_events()


                person_coords.append((cx,cy))
                cv2.circle(undistImg, (cx, cy), 3, (0,0,255), -1)
                cv2.circle(undistImg, (int(w/2),int(h/2)), 6, (255,0,255), 1)
                cv2.circle(undistImg, (int(w_frame/2),int(h_frame/2)), 6, (255,0,0), 1)
                cv2.circle(undistImg, (int(655),int(337)), 6, (0,255,0), 1)
                cv2.circle(undistImg, (int(newMtx_r[0][2]),int(newMtx_r[1][2])), 6, (255,255,0), 1)


    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (10, 30)

    # fontScale
    fontScale = 0.5
    
    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2
    
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    
    # Using cv2.putText() method
    cv2.putText(undistImg, f'Undistorted | FPS: {fps:.2f}', org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    # Show video
    cv2.imshow("Undistorted Image", undistImg)

    # cv2.putText(frame, 'Frame', org, font, 
    #                 fontScale, color, thickness, cv2.LINE_AA)
    # cv2.imshow("Frame", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()