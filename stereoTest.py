import cv2
import threading
import time

class CameraThread(threading.Thread):
    def __init__(self, cam_id):
        super().__init__()
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

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


# --- Main stereo loop ---
if __name__ == "__main__":
    camL = CameraThread(0)  # left camera
    camR = CameraThread(2)  # right camera
    camL.start()
    camR.start()

    try:
        while True:
            frameL, tsL = camL.get_frame()
            frameR, tsR = camR.get_frame()

            if frameL is not None and frameR is not None:
                # Timestamp difference (ms)
                delta = abs(tsL - tsR) * 1000
                print(f"Timestamp difference: {delta:.2f} ms")

                # Show side by side
                stereo = cv2.hconcat([frameL, frameR])
                cv2.imshow("Stereo Pair", stereo)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camL.stop()
        camR.stop()
        camL.join()
        camR.join()
        cv2.destroyAllWindows()
