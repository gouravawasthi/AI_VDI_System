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
    def analyze_roi(self, ref_path: str, inp_path: str, roi_detect: tuple, roi_display: tuple):
        """
        Run the full Sobel inspection pipeline:
        1. Register input image with reference.
        2. Compute Sobel gradients in ROI.
        3. Mark result on display ROI.

        Parameters:
            ref_path : str - reference image path
            inp_path : str - input image path
            roi_detect : tuple (x, y, w, h) - ROI for gradient analysis
            roi_display : tuple (x, y, w, h) - ROI to display result

        Returns:
            result_dict : {
                'status': 'PRESENT' or 'ABSENT',
                'mean_gradient': float,
                'grad_display': np.ndarray,
                'annotated_img': np.ndarray
            }
        """
        # Load images
        ref = cv2.imread(ref_path)
        inp = cv2.imread(inp_path)
        if ref is None or inp is None:
            raise ValueError("Could not load input or reference image.")

        # Step 1: Register image
        registered, H = self.registrar.register(ref, inp)

        # Step 2: Extract ROI for gradient analysis
        x, y, w, h = roi_detect
        roi_img = registered[y:y+h, x:x+w].copy()

        # Step 3: Compute gradient magnitude
        grad_mag = self.compute_gradient_magnitude(roi_img)
        mean_grad = np.mean(grad_mag)

        # Step 4: Determine result
        status = "PRESENT" if mean_grad > self.grad_threshold else "ABSENT"
        color = (0, 255, 0) if status == "PRESENT" else (0, 0, 255)

        # Visualization-friendly image
        grad_display = cv2.convertScaleAbs(grad_mag)

        # Step 5: Annotate display ROI on registered image
        xd, yd, wd, hd = roi_display
        annotated = registered.copy()
        cv2.rectangle(annotated, (xd, yd), (xd+wd, yd+hd), color, 2)
        cv2.putText(annotated, f"{status}", (xd, yd - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(f"[SOBEL] ROI mean gradient: {mean_grad:.2f}  -->  {status}")

        return {
            "status": status,
            "annotated_img": annotated,
        
        }
