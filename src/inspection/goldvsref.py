import cv2
import numpy as np

def inspect_image(gold_img, gold_mask, new_img, position_hint):
    """
    Compare new image with gold standard inside ROI after registration.
    
    Args:
        gold_img (np.ndarray): Gold standard image (BGR or grayscale)
        gold_mask (np.ndarray): Mask of ROI (same size as gold_img, binary 0/255)
        new_img (np.ndarray): New input image (BGR or grayscale)
        position_hint (str): Instruction string if registration fails

    Returns:
        dict: {
            "Status": 1 or 0,
            "Message": status_message,
            "OutputImage": annotated_image (only if Status=1)
        }
    """

    try:
        # --- Step (a) Register new image with gold image ---
        # Convert to grayscale if needed
        gray_gold = cv2.cvtColor(gold_img, cv2.COLOR_BGR2GRAY) if len(gold_img.shape) == 3 else gold_img
        gray_new = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY) if len(new_img.shape) == 3 else new_img

        # ORB keypoints + descriptors
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray_gold, None)
        kp2, des2 = orb.detectAndCompute(gray_new, None)

        if des1 is None or des2 is None:
            return {"Status": 0, "Message": f"Please keep the inspection object in {position_hint}"}

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 10:
            return {"Status": 0, "Message": f"Please keep the inspection object in {position_hint}"}

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            return {"Status": 0, "Message": f"Please keep the inspection object in {position_hint}"}

        # Warp new image to align with gold
        h, w = gray_gold.shape
        aligned_new = cv2.warpPerspective(new_img, H, (w, h))

        # --- Step (b) Apply ROI mask ---
        roi_new = cv2.bitwise_and(aligned_new, aligned_new, mask=gold_mask)
        
        # --- Step (c) Apply ROI mask on Gold standard ---
        roi_gold = cv2.bitwise_and(gold_img, gold_img, mask=gold_mask)

        # --- Step (d) Gradient comparison ---
        def get_edges(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return edges

        edges_gold = get_edges(roi_gold)
        edges_new = get_edges(roi_new)

        # Identify differences
        only_new = cv2.subtract(edges_new, edges_gold)
        only_gold = cv2.subtract(edges_gold, edges_new)
        common = cv2.bitwise_and(edges_gold, edges_new)

        # Create color overlay
        result = np.zeros_like(gold_img)
        result[common > 0] = (255, 255, 255)  # white common edges
        result[only_new > 0] = (0, 0, 255)    # red new edges
        result[only_gold > 0] = (255, 0, 0)   # blue missing edges

        # Apply mask again to restrict to ROI
        result = cv2.bitwise_and(result, result, mask=gold_mask)

        return {
            "Status": 1,
            "Message": "Inspection completed successfully",
            "OutputImage": result
        }

    except Exception as e:
        return {"Status": 0, "Message": str(e)}
if __name__ == "__main__":
    # Test the function with sample images
    gold_image = cv2.imread("/Users/gourav/Desktop/Taisys/AI_VDI_System/reference_image.jpg")
    gold_mask = cv2.imread("/Users/gourav/Desktop/Taisys/AI_VDI_System/reference_mask.png", cv2.IMREAD_GRAYSCALE)
    new_image = cv2.imread("/Users/gourav/Desktop/Taisys/AI_VDI_System/averaged_capture.jpg")

    result = inspect_image(gold_image, gold_mask, new_image, "the indicated position")
    cv2.imshow("Inspection Result", result["OutputImage"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()