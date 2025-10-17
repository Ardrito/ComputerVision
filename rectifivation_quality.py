import cv2
import numpy as np

data = np.load("stereo_calibration.npz")

mapxL, mapyL = data["mapx_left"], data["mapy_left"]
mapxR, mapyR = data["mapx_right"], data["mapy_right"]
res = (1280,720)

capL = cv2.VideoCapture(0)
capR = cv2.VideoCapture(2)
capL.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
#capL.set(cv2.CAP_PROP_FPS,          fps)
capR.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR):
        break

    rectL = cv2.remap(frameL, mapxL, mapyL, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapxR, mapyR, cv2.INTER_LINEAR)

    # Draw horizontal lines for visual alignment check
    h, w = rectL.shape[:2]
    step = 40
    for y in range(0, h, step):
        cv2.line(rectL, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(rectR, (0, y), (w, y), (0, 255, 0), 1)

    cv2.imshow("Left Rectified", rectL)
    cv2.imshow("Right Rectified", rectR)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
