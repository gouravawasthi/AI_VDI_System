import cv2
import numpy as np
from Image_reg import ImageRegistration


class SobelBottom:
    """
    A modular Sobel-based edge presence detector for image inspection,
    using image registration before analysis.
    """

    def __init__(self, grad_threshold=40, ksize_x=3, ksize_y=3, blur_kernel=(3,3)):
        """
        Initialize the Sobel edge detector.

        Parameters:
            grad_threshold : float - threshold for mean gradient magnitude to mark presence
            ksize_x : int - kernel size for Sobel X
            ksize_y : int - kernel size for Sobel Y
            blur_kernel : tuple - Gaussian blur kernel size
        """
        self.grad_threshold = grad_threshold
        self.ksize_x = ksize_x
        self.ksize_y = ksize_y
        self.blur_kernel = blur_kernel
        self.registrar = ImageRegistration()  # uses your registration module


    # -------------------------------------------------------
    # Gradient computation
    # -------------------------------------------------------
    def compute_gradient_magnitude(self, roi_img: np.ndarray) -> np.ndarray:
        """Compute the Sobel gradient magnitude map for a given ROI image."""
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.ksize_x)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=self.ksize_y)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        return grad_mag


    # -------------------------------------------------------
    # ROI Analysis
    # -------------------------------------------------------
    def analyze_roi(self, ref_img, inp_img, roi_definitions):
       # Step 1: Register image
        registered = self.registrar.register(ref_img, inp_img)
        if isinstance(registered, tuple):
            registered = registered[0]

        annotated = registered.copy()

        # Step 2: Analyze Plate ROI
        if "Plate" not in roi_definitions:
            raise ValueError("ROI definition for 'Plate' missing.")

        x, y, w, h = roi_definitions["Plate"]
        roi = registered[y:y+h, x:x+w]
        grad_mag = self.compute_gradient_magnitude(roi)
        mean_grad = np.mean(grad_mag)

        # Step 3: Determine presence
        plate_status = "PASS" if mean_grad > self.grad_threshold else "FAIL"
        screw_status = plate_status  # mirror behavior

        color = (0, 255, 0) if plate_status == "PASS" else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"Plate: {plate_status} ({mean_grad:.1f})",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(f"[SOBEL] Plate mean_grad={mean_grad:.2f} → {plate_status}")

        # Step 4: Return clean results
        results = {
            "Screw": screw_status,
            "Plate": plate_status
        }

        return {
            "annotated": annotated,
            "results": results
        }
