import numpy as np

data = np.load("stereo_calibration.npz")

print("Keys saved:")
print(list(data.keys()))

# Example: print a few key matrices
print("\nLeft Camera Matrix:\n", data["camera_matrix_left"])
print("Left Distortion Coefficients:\n", data["dist_left"])

print("\nRight Camera Matrix:\n", data["camera_matrix_right"])
print("Right Distortion Coefficients:\n", data["dist_right"])

print("\nRotation (R):\n", data["R"])
print("Translation (T):\n", data["T"].ravel(), "  (Baseline ≈ %.4f m)" % np.linalg.norm(data["T"]))

print("\nRectification Matrices:")
print("R1:\n", data["R1"])
print("R2:\n", data["R2"])

print("\nProjection Matrices:")
print("P1:\n", data["P1"])
print("P2:\n", data["P2"])

print("\nDisparity-to-Depth (Q):\n", data["Q"])
