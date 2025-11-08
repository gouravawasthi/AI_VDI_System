import cv2
import numpy as np


class ImageRegistration:
    """
    Handles feature-based registration between two images using ORB + Homography,
    with built-in histogram equalization for improved feature matching.
    """

    def __init__(self, max_features=5000, good_match_limit=50):
        """
        Parameters:
            max_features : int - number of ORB keypoints to extract
            good_match_limit : int - number of best matches used for homography
        """
        self.orb = cv2.ORB_create(max_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.good_match_limit = good_match_limit

    # ------------------------------------------------------------------
    # Internal helper: Histogram Equalization (Y-channel of YCrCb)
    # ------------------------------------------------------------------
    def _equalize_histogram_color(self, img):
        """
        Apply histogram equalization on a color image via YCrCb luminance.
        Keeps color balance intact while enhancing contrast.
        """
        if img is None or img.size == 0:
            raise ValueError("Empty image passed for histogram equalization.")
        
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize luminance channel only
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return equalized

    # ------------------------------------------------------------------
    # Homography computation
    # ------------------------------------------------------------------
    def compute_homography(self, ref_img, inp_img):
        """
        Compute the homography matrix aligning inp_img → ref_img.
        Histogram equalization is applied to both images before ORB.
        """
        # ✅ Apply histogram equalization to improve matching robustness
        ref_eq = self._equalize_histogram_color(ref_img)
        inp_eq = self._equalize_histogram_color(inp_img)

        # Convert to grayscale
        gray_ref = cv2.cvtColor(ref_eq, cv2.COLOR_BGR2GRAY)
        gray_inp = cv2.cvtColor(inp_eq, cv2.COLOR_BGR2GRAY)

        # ORB feature detection and descriptor computation
        kp1, des1 = self.orb.detectAndCompute(gray_ref, None)
        kp2, des2 = self.orb.detectAndCompute(gray_inp, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            print("[WARN] Insufficient ORB keypoints for matching.")
            return None

        # Match features using Hamming distance
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:self.good_match_limit]

        if len(matches) < 10:
            print(f"[WARN] Low feature matches ({len(matches)}). Registration may be unreliable.")
            return None

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Compute homography using RANSAC
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("[WARN] Homography computation failed (matrix is None).")
            return None

        return H

    # ------------------------------------------------------------------
    # Registration wrapper
    # ------------------------------------------------------------------
    def register(self, ref_img, inp_img):
        """
        Warp input image to match reference using histogram-equalized ORB registration.
        Returns registered image.
        """
        try:
            H = self.compute_homography(ref_img, inp_img)

            if H is None:
                black = np.zeros_like(inp_img)
                cv2.putText(black, "Registration Failed", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return black

            h, w, _ = ref_img.shape
            registered = cv2.warpPerspective(inp_img, H, (w, h))
            return registered

        except Exception as e:
            print(f"[ERROR] Registration exception: {e}")
            black = np.zeros_like(inp_img)
            cv2.putText(black, "Registration Error", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return black
