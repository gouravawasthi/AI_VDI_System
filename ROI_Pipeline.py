import cv2
import numpy as np
import pytesseract
import easyocr

# Initialize OCR reader
reader = easyocr.Reader(['en'])

def run_pipeline(reference_img_path, input_img_path, roi_definitions):
    """
    Pipeline to register, analyze, and generate results from images.

    Parameters:
        reference_img_path : str - Path to reference image
        input_img_path     : str - Path to input image
        roi_definitions    : dict - {
              'roi1': (x1, y1, w1, h1),
              'roi2': (x2, y2, w2, h2),
              'roi3': (x3, y3, w3, h3)
        }
    Returns:
        result_images : dict containing:
            - 'registered' : registered image
            - 'annotated'  : registered image with ROI highlights (Pass/Fail)
    """

    # ---------------------------------------------------
    # (1) Load Images
    # ---------------------------------------------------
    ref = cv2.imread(reference_img_path)
    inp = cv2.imread(input_img_path)

    # ---------------------------------------------------
    # (2) Register Input Image to Reference Image
    # ---------------------------------------------------
    # Convert to grayscale
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_inp = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)

    # ORB feature detector and matcher
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_inp, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Minimum good matches
    good_matches = matches[:50]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    h, w, _ = ref.shape
    registered = cv2.warpPerspective(inp, H, (w, h))

    # ---------------------------------------------------
    # (3) ROI Analysis
    # ---------------------------------------------------
    annotated = registered.copy()
    results = {}

    for i, (roi_name, (x, y, w, h)) in enumerate(roi_definitions.items(), 1):
        roi = registered[y:y+h, x:x+w]
        color = (0, 0, 255)  # default red = fail

        if roi_name in ['roi1', 'roi2']:
            # OCR detection
            ocr_result = reader.readtext(roi)
            if len(ocr_result) > 0:  # some text found
                status = "PASS"
                color = (0, 255, 0)
            else:
                status = "FAIL"

        elif roi_name == 'roi3':
            # Circle detection with roundness filtering
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)

            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                param1=50, param2=30, minRadius=5, maxRadius=50
            )

            roundness_threshold = 1
            found_round_circle = False

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (cx, cy, r) in circles[0, :]:
                    # Extract region around detected circle
                    x1, y1 = max(0, cx - r), max(0, cy - r)
                    x2, y2 = min(gray.shape[1], cx + r), min(gray.shape[0], cy + r)
                    crop = gray[y1:y2, x1:x2]

                    # Binary threshold + contour detection
                    _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        if len(cnt) >= 5:
                            ellipse = cv2.fitEllipse(cnt)
                            (center, axes, angle) = ellipse
                            major, minor = max(axes), min(axes)
                            ratio = minor / major

                            if ratio >= roundness_threshold:
                                found_round_circle = True
                                cv2.circle(roi, (cx, cy), r, (0, 255, 0), 2)
                            else:
                                cv2.circle(roi, (cx, cy), r, (0, 255,0), 2)

            if found_round_circle:
                status = "PASS"
                color = (0, 255, 0)
            # Draw ROI and status
        cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated, f"{roi_name}: {status}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results[roi_name] = status

    # ---------------------------------------------------
    # (4) Outputs
    # ---------------------------------------------------
    result_images = {
        "registered": registered,
        "annotated": annotated,
        "results": results
    }

    return result_images


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":
    roi_definitions = {
    'roi1': (177, 115, 105, 56),
    'roi2': (561, 144, 193, 150),
    'roi3': (795, 205, 180, 217),
}

    results = run_pipeline("OCR5.png", "OCR5.png", roi_definitions)

    cv2.imwrite("registered_output.jpg", results['registered'])
    cv2.imwrite("annotated_output.jpg", results['annotated'])
    print("Results:", results['results'])

