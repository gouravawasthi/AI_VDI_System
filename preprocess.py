import cv2
import numpy as np
import time

class FramePreprocessor:
    def __init__(self, camera_index=0, duration=1, blur_kernel=(5, 5)):
        """
        Initializes the preprocessor with camera and blur settings.
        
        Parameters:
            camera_index (int): The index of the camera to use.
            duration (float): Time in seconds to capture frames.
            blur_kernel (tuple): Kernel size for Gaussian blur.
        """
        self.camera_index = camera_index
        self.duration = duration
        self.blur_kernel = blur_kernel

    def capture_frames(self):
        """Capture frames for the given duration."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print("❌ Camera not accessible.")
            return []

        frames = []
        start_time = time.time()

        while time.time() - start_time < self.duration:
            ret, frame = cap.read()
            if not ret:
                continue

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(frame, self.blur_kernel, 0)
            frames.append(blurred)

        cap.release()
        return frames

    def average_frames(self, frames):
        """Compute average of frames to stabilize the image."""
        if not frames:
            print("⚠️ No frames to average.")
            return None

        avg_frame = np.mean(np.array(frames, dtype=np.float32), axis=0)
        avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)
        return avg_frame

    def preprocess(self):
        """
        Perform full preprocessing:
        1. Capture frames
        2. Apply Gaussian blur
        3. Average into one stabilized frame
        """
        frames = self.capture_frames()
        if not frames:
            return None
        return self.average_frames(frames)
