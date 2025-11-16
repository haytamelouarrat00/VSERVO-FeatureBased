import cv2
import numpy as np

class SIFTManager:
    def __init__(self):
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Brute Force Matcher with L2 norm (standard for SIFT)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

        self.kp_des = None  # Keypoints in desired image
        self.des_des = None  # Descriptors in desired image

    def set_desired_view(self, img):
        """ Process the 'Goal' image once """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.kp_des, self.des_des = self.sift.detectAndCompute(gray, None)
        return img

    def process_current_view(self, img):
        """
        Detect features in current view and match with desired.
        Returns: matched_current_pts, matched_desired_pts
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = self.sift.detectAndCompute(gray, None)

        if des_curr is None or len(kp_curr) < 4:
            return None, None, img

        # KNN Match (k=2) for Ratio Test
        matches = self.bf.knnMatch(des_curr, self.des_des, k=2)

        # Apply Ratio Test (Lowe's paper)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # We need at least 4 points to solve matrices (though 3 is min, 4+ is safer)
        if len(good_matches) < 4:
            return None, None, img

        # Extract coordinates
        # QueryIdx = Current Image, TrainIdx = Desired Image
        pts_curr = np.float32([kp_curr[m.queryIdx].pt for m in good_matches])
        pts_des = np.float32([self.kp_des[m.trainIdx].pt for m in good_matches])

        # Visualization: Draw matches
        vis_img = cv2.drawMatches(img, kp_curr, np.zeros_like(img), self.kp_des, good_matches, None, flags=2)

        return pts_curr.T, pts_des.T, vis_img