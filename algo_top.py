import cv2
import numpy as np
import easyocr
from Image_reg import ImageRegistration


class ImageInspectionPipeline:
    """
    Main inspection pipeline with modular registration + ROI analysis.
    """

    def __init__(self, ocr_langs=['en']):
        self.reader = easyocr.Reader(ocr_langs)
        self.registrar = ImageRegistration()   # Import and use the new registration module


    # ---------------------------------------------------
    # (1) ROI Analysis Methods
    # ---------------------------------------------------
    def analyze_ocr(self, roi):
        results = self.reader.readtext(roi)
        return "PASS" if len(results) > 0 else "FAIL"

    def analyze_rectangle(self, roi, min_area=200):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                return "PASS"
        return "FAIL"


    # ---------------------------------------------------
    # (2) Run Pipeline
    # ---------------------------------------------------
    def run(self, reference_img_path, input_img_path, roi_definitions):
        ref = cv2.imread(reference_img_path)
        inp = cv2.imread(input_img_path)
        if ref is None or inp is None:
            raise ValueError("Error loading images.")

        # Register using separate module
        registered, H = self.registrar.register(ref, inp)

        annotated = registered.copy()
        results = {}

        for roi_name, roi_data in roi_definitions.items():
            x, y, w, h = roi_data['coords']
            roi_type = roi_data.get('type', 'ocr')
            roi = registered[y:y+h, x:x+w]

            status = "FAIL"
            if roi_type == "ocr":
                status = self.analyze_ocr(roi)
            elif roi_type == "rectangle":
                status = self.analyze_rectangle(roi)

            color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, f"{roi_name}: {status}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            results[roi_name] = status

        return {
            "annotated": annotated,
            "results": results
        }
