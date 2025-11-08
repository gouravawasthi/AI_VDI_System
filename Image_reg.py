import cv2
import numpy as np


class ImageRegistration:
    """
    Handles feature-based registration between two images using ORB + Homography.
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

    def compute_homography(self, ref_img, inp_img):
        """
        Compute the homography matrix aligning inp_img → ref_img.
        """
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_inp = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray_ref, None)
        kp2, des2 = self.orb.detectAndCompute(gray_inp, None)

        if des1 is None or des2 is None:
            raise ValueError("Failed to extract features from one or both images.")
        
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:self.good_match_limit]
        if len(matches)<500:
            print(f"[WARN] Low feature matches: {len(matches)}. Returning black frame.")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            print("[WARN] Homography computation failed (matrix is None).")
            return None

        return H
    def register(self, ref_img, inp_img):
        """
        Warp input image to match reference.
        Returns registered image and homography matrix.
        """
        try:
            H = self.compute_homography(ref_img, inp_img)

            # Return black image if homography fails
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
    
