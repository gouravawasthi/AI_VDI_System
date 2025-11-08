import cv2
import numpy as np
import easyocr
from Image_reg import ImageRegistration


class ImageInspectionPipeline:
    """
    Main inspection pipeline with modular registration + ROI analysis.
    - roi1, roi2 → OCR detection
    - roi3 → Circle detection (perfect circular object)
    """

    def __init__(self, ocr_langs=['en']):
        self.reader = easyocr.Reader(ocr_langs)
        self.registrar = ImageRegistration()   # use modular registration

    # ---------------------------------------------------
    # (1) ROI Analysis Methods
    # ---------------------------------------------------
    def analyze_ocr(self, roi):
        """Perform OCR detection within the ROI."""
        results = self.reader.readtext(roi)
        return "PASS" if len(results) > 0 else "FAIL"

    def analyze_circle(self, roi):

        """Detect presence of near-perfect circular shapes using HoughCircles + roundness check."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=10,
            param1=40,
            param2=20,
            minRadius=3,
            maxRadius=40
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (cx, cy, r) in circles[0, :]:
                # Extract circle region
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(gray.shape[1], cx + r), min(gray.shape[0], cy + r)
                crop = gray[y1:y2, x1:x2]

                # Detect edges instead of Otsu threshold
                edges = cv2.Canny(crop, 30, 100)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                cnt = max(contours, key=cv2.contourArea)
                if len(cnt) < 5 or cv2.contourArea(cnt) < 10:
                    continue

                # Fit ellipse for roundness
                ellipse = cv2.fitEllipse(cnt)
                (_, axes, _) = ellipse
                major, minor = max(axes), min(axes)
                ratio = minor / major

                # Check roundness with tolerance
                if 0.90 <= ratio <= 1.05:
                    cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
                    cv2.putText(roi, f"Roundness={ratio:.2f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return "PASS"

        return "FAIL"

    

    # ---------------------------------------------------
    # (2) Run Pipeline
    # ---------------------------------------------------
    def run(self, ref_img, input_img, roi_definitions):
        """Main execution pipeline."""
        ref = ref_img
        inp = input_img

        if ref is None or inp is None:
            raise ValueError("Error loading images.")

        # Register input image to reference
        registered = self.registrar.register(ref, inp)

        annotated = registered.copy()
        results = {}

        # Iterate through ROIs
        for roi_name, (x, y, w, h) in roi_definitions.items():
            roi = registered[y:y+h, x:x+w]
            color = (0, 0, 255)
            status = "FAIL"

            if roi_name in ['Antenna', 'Capacitor']:
                status = self.analyze_ocr(roi)
            elif roi_name == 'Speaker':
                status = self.analyze_circle(roi)

            color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, f"{roi_name}: {status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            results[roi_name] = status

        return {
            "annotated": annotated,
            "results": results
        }
