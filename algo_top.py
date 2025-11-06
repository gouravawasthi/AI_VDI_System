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
        """Detect presence of perfect circular shapes using HoughCircles."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=200
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (cx, cy, r) in circles[0, :]:
                # optional: check roundness or area if needed
                cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
            return "PASS"

        return "FAIL"

    # ---------------------------------------------------
    # (2) Run Pipeline
    # ---------------------------------------------------
    def run(self, reference_img_path, input_img_path, roi_definitions):
        """Main execution pipeline."""
        ref = cv2.imread(reference_img_path)
        inp = cv2.imread(input_img_path)

        if ref is None or inp is None:
            raise ValueError("Error loading images.")

        # Register input image to reference
        registered, H = self.registrar.register(ref, inp)

        annotated = registered.copy()
        results = {}

        # Iterate through ROIs
        for roi_name, (x, y, w, h) in roi_definitions.items():
            roi = registered[y:y+h, x:x+w]
            color = (0, 0, 255)
            status = "FAIL"

            if roi_name in ['roi1', 'roi2']:
                status = self.analyze_ocr(roi)
            elif roi_name == 'roi3':
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
