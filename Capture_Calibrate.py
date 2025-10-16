import cv2
import numpy as np
import time
import tkinter as tk

#   v4l2-ctl --list-devices
#   Use above command on linux to find camera numbers

def getCalibration(camera_number: int = 0, chessboardSize: tuple[int,int] = (10,7), square_size:float = 0.018, snap_time:float = 2, num_images = 10, resolution: tuple[int,int] = (1280,720)):
    cap = cv2.VideoCapture(camera_number)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera.")
    h_frame, w_frame = frame.shape[:2]


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 30)
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 2


    worldP = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    worldP[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    worldP *= square_size

    t_prev = time.time()
    t_cap = time.time()


    worldPoints = []
    imgPoints = []
    timer = 0
    images_taken = 0

    while True:
        ret_frame, frame = cap.read()
        taken = False
        if (not ret_frame) or (len(imgPoints) > num_images-1):
            break
        
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        if ret == True:
            cornersSubPix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            cv2.drawChessboardCorners(img, chessboardSize, cornersSubPix, ret)

            timer = (t_prev - t_cap)
            if (timer > snap_time):
                worldPoints.append(worldP)
                imgPoints.append(corners)
                t_cap = time.time()
                taken = True
                images_taken += 1


        t_curr = time.time()
        delta_t = t_curr - t_prev
        t_prev = t_curr
        fps = 1.0/delta_t
        if not taken:
            cv2.putText(img, f'Undistorted | Res: {w_frame}, {h_frame} | FPS: {fps:.2f} | Images Taken {images_taken:02d} | Timer: {snap_time-timer:.2f}', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)
        if taken:
            cv2.putText(img, f'Undistorted | Res: {w_frame}, {h_frame} | FPS: {fps:.2f} | Images Taken {images_taken:02d} |  Timer: {snap_time-timer:.2f} | Taken', org, font, 
                            fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break

    repError, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(worldPoints, imgPoints, (w_frame,h_frame),None,None)
    newMtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix,dist,(w_frame,h_frame),1,(w_frame,h_frame))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newMtx, (w_frame, h_frame), 5)
    cap.release()
    cv2.destroyAllWindows()

    meanError = 0

    for i in range(len(worldPoints)):
        newImgPoints, _ = cv2.projectPoints(worldPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgPoints[i], newImgPoints, cv2.NORM_L2)/len(newImgPoints)
        meanError += error

    error = meanError/len(worldPoints)

    print ("Error is", error)

    return (cameraMatrix, dist, newMtx, roi, mapx, mapy, repError)

if __name__ == "__main__":
    camera_number_one = 0
    camera_number_two = 2 # Set to -1 to skip
    chessboard_size = (10,7)
    snap_time = 3
    num_images = 10
    square_size = 0.018
    R = np.eye(3)
    T = np.array([0.133,0,0])
    resolution = (640,480)


    print ("Start:", camera_number_one)
    camera_matrix_one, dist_one, newMtx_one, roi_one, mapx_one, mapy_one, error_one = getCalibration(
        camera_number=camera_number_one,chessboardSize=(chessboard_size),
        square_size=square_size,snap_time=snap_time, num_images=num_images, resolution=resolution)
    print ("Camera One Error", error_one)
    print ("Camera One Matrix:", camera_matrix_one)
    
    if camera_number_two > -1:
        camera_matrix_two, dist_two, newMtx_two, roi_two, mapx_two, mapy_two, error_two = getCalibration(
        camera_number=camera_number_two,chessboardSize=(chessboard_size),
        square_size=square_size,snap_time=snap_time, num_images=num_images, resolution=resolution)
    else:
        camera_matrix_two, dist_two, newMtx_two, roi_two, mapx_two, mapy_two, error_two = None , None, None, None, None, None, None


    print ("Camera Two Error", error_two)
    print ("Camera Two Matrix:", camera_matrix_two)

    np.savez(
        "camera_calibration_data.npz",
        camera_matrix_one = camera_matrix_one,
        dist_one = dist_one,
        newMtx_one = newMtx_one,
        roi_one = roi_one,
        mapx_one = mapx_one,
        mapy_one = mapy_one,
        error_one = error_one,
        camera_matrix_two = camera_matrix_two,
        dist_two = dist_two,
        newMtx_two = newMtx_two,
        roi_two = roi_two,
        mapx_two = mapx_two,
        mapy_two = mapy_two,
        error_two = error_two,
        R=R,
        T=T
    )
