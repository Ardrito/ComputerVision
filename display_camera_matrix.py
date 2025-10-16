import numpy as np

calib = np.load("camera_calibration_data.npz")

camera_matrix_one = calib["camera_matrix_one"]
dist_one = calib["dist_one"]
newMtx_one = calib["newMtx_one"]
roi_one = calib["roi_one"]
mapx_one = calib["mapx_one"]
mapy_one = calib["mapy_one"]
error_one = calib["error_one"]
camera_matrix_two = calib["camera_matrix_two"]
dist_two = calib["dist_two"]
newMtx_two = calib["newMtx_two"]
roi_two = calib["roi_two"]
mapx_two = calib["mapx_two"]
mapy_two = calib["mapy_two"]
error_two = calib["error_two"]
R= calib["R"]
T= calib["T"]


# Pretty-print all values
print("\n=== CAMERA ONE ===")
print("camera_matrix_one:\n", camera_matrix_one)
print("dist_one:", dist_one)
print("newMtx_one:\n", newMtx_one)
print("roi_one:", roi_one)
print("mapx_one shape:", mapx_one.shape)
print("mapy_one shape:", mapy_one.shape)
print("error_one:", error_one)

print("\n=== CAMERA TWO ===")
print("camera_matrix_two:\n", camera_matrix_two)
print("dist_two:", dist_two)
print("newMtx_two:\n", newMtx_two)
print("roi_two:", roi_two)
print("mapx_two shape:", mapx_two.shape)
print("mapy_two shape:", mapy_two.shape)
print("error_two:", error_two)

print("\n=== STEREO PARAMETERS ===")
print("R (rotation matrix):\n", R)
print("T (translation vector):", T)

print("\n=== CALIBRATION SUMMARY ===")

def print_cam_summary(name, K, dist, newK, roi):
    x, y, w, h = roi
    print(f"\n{name}:")
    print(f"  fx = {K[0,0]:.2f}, fy = {K[1,1]:.2f}")
    print(f"  cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
    print(f"  Distortion coeffs: {dist.ravel()}")
    print(f"  Optimized principal point: ({newK[0,2]:.2f}, {newK[1,2]:.2f})")
    print(f"  ROI -> x:{int(x)}, y:{int(y)}, w:{int(w)}, h:{int(h)}")

print_cam_summary("Camera One", camera_matrix_one, dist_one, newMtx_one, roi_one)
print_cam_summary("Camera Two", camera_matrix_two, dist_two, newMtx_two, roi_two)