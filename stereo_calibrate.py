import cv2
import numpy as np
import time
from pathlib import Path


LEFT_CAM_ID  = 0
RIGHT_CAM_ID = 2
RESOLUTION   = (1280, 720)
FPS_REQ      = 30

# Chessboard settings
PATTERN_SIZE   = (10, 7)          # (cols, rows) inner corners
SQUARE_SIZE_M  = 0.018           # meters per square (SET THIS!)

# Output file
OUT_FILE = "stereo_calibration.npz"

# Corner subpixel & calibrate criteria
CRIT_SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
CRIT_CAL    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)


def make_object_points(pattern_size, square_size_m):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= float(square_size_m)
    return objp

def draw_corners(img, corners, pattern_size, found):
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    return vis

def ensure_capture(id, res, fps):
    cap = cv2.VideoCapture(id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS,          fps)
    ok, _ = cap.read()
    if not ok:
        raise RuntimeError(f"Camera {id} failed to open/read.")
    return cap


def main():
    capL = ensure_capture(LEFT_CAM_ID, RESOLUTION, FPS_REQ)
    capR = ensure_capture(RIGHT_CAM_ID, RESOLUTION, FPS_REQ)

    objp_template = make_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    objpoints = []   # 3D board points (shared)
    imgpointsL = []  # 2D corners in left
    imgpointsR = []  # 2D corners in right

    captured = 0
    last_info = "Press [SPACE]=capture, [S]=stereo calibrate+save, [C]=clear, [Q]=quit"

    while True:
        okL, frameL = capL.read()
        okR, frameR = capR.read()
        if not (okL and okR):
            print("Failed to read from one of the cameras.")
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, PATTERN_SIZE)
        foundR, cornersR = cv2.findChessboardCorners(grayR, PATTERN_SIZE)

        dispL = frameL.copy()
        dispR = frameR.copy()

        if foundL:
            cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), CRIT_SUBPIX)
            dispL = draw_corners(dispL, cornersL, PATTERN_SIZE, True)
        if foundR:
            cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), CRIT_SUBPIX)
            dispR = draw_corners(dispR, cornersR, PATTERN_SIZE, True)

        # HUD
        hudL = f"L corners: {'OK' if foundL else '--'}   Captured pairs: {captured}"
        hudR = f"R corners: {'OK' if foundR else '--'}"
        cv2.putText(dispL, hudL, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(dispR, hudR, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(dispL, last_info, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        stereo = cv2.hconcat([dispL, dispR])
        cv2.imshow("Stereo Chessboard", stereo)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        elif key == ord('c'):
            objpoints.clear(); imgpointsL.clear(); imgpointsR.clear()
            captured = 0
            last_info = "Cleared all captures."

        elif key == ord(' '):
            if foundL and foundR:
                objpoints.append(objp_template.copy())
                imgpointsL.append(cornersL)
                imgpointsR.append(cornersR)
                captured += 1
                last_info = f"Captured pair #{captured}"
            else:
                last_info = "Both boards must be detected to capture."

        elif key == ord('s'):
            if captured < 8:
                last_info = "Need at least ~8–15 good pairs; capture more!"
                continue

            imsize = grayL.shape[::-1]  # (w, h)

            # Calibrate each camera (or set CALIB_FIX_INTRINSIC if you already know K,dist)
            print("[*] Calibrating left camera...")
            retL, K_L, distL, rvecsL, tvecsL = cv2.calibrateCamera(
                objpoints, imgpointsL, imsize, None, None, criteria=CRIT_CAL)
            print(f"Left RMS: {retL:.6f}")

            print("[*] Calibrating right camera...")
            retR, K_R, distR, rvecsR, tvecsR = cv2.calibrateCamera(
                objpoints, imgpointsR, imsize, None, None, criteria=CRIT_CAL)
            print(f"Right RMS: {retR:.6f}")

            # Stereo calibration
            print("[*] Stereo calibration...")
            flags = cv2.CALIB_FIX_INTRINSIC  # keep single-cam K/dist fixed during stereo
            retval, K_L, distL, K_R, distR, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpointsL, imgpointsR,
                K_L, distL, K_R, distR,
                imsize,
                criteria=CRIT_CAL,
                flags=flags
            )
            print(f"Stereo RMS: {retval:.6f}")
            print("R:\n", R)
            print("T (meters):", T.ravel())

            # Rectification
            print("[*] Rectifying...")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                K_L, distL, K_R, distR, imsize, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
            )

            # Undistort/rectify maps
            mapx1, mapy1 = cv2.initUndistortRectifyMap(K_L, distL, R1, P1, imsize, cv2.CV_32FC1)
            mapx2, mapy2 = cv2.initUndistortRectifyMap(K_R, distR, R2, P2, imsize, cv2.CV_32FC1)

            # Save
            np.savez(
                OUT_FILE,
                image_size=np.array(imsize),
                camera_matrix_left=K_L,  dist_left=distL,
                camera_matrix_right=K_R, dist_right=distR,
                R=R, T=T, E=E, F=F,
                R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                roi_left=np.array(roi1), roi_right=np.array(roi2),
                mapx_left=mapx1, mapy_left=mapy1,
                mapx_right=mapx2, mapy_right=mapy2,
                stereo_rms=retval, left_rms=retL, right_rms=retR,
            )

            last_info = f"Saved to {OUT_FILE}. Baseline ≈ {np.linalg.norm(T):.4f} m"

    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
