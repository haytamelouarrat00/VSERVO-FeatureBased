import numpy as np
import cv2


class TexturePlanarSimulator:
    def __init__(self, image_path=None, width=800, height=600):
        self.W, self.H = width, height

        # Generate a synthetic texture if no image provided
        if image_path is None:
            self.texture = np.zeros((600, 600, 3), dtype=np.uint8)
            # Draw a grid and some shapes so SIFT has something to find
            cv2.rectangle(self.texture, (0, 0), (600, 600), (200, 200, 200), -1)
            for i in range(0, 600, 60):
                cv2.line(self.texture, (i, 0), (i, 600), (0, 0, 0), 2)
                cv2.line(self.texture, (0, i), (600, i), (0, 0, 0), 2)
            cv2.circle(self.texture, (300, 300), 100, (0, 0, 255), 5)
            cv2.putText(self.texture, "VS TARGET", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        else:
            self.texture = cv2.imread(image_path)
            self.texture = cv2.resize(self.texture, (600, 600))

        # Define the physical size of this texture in the world (e.g., 1 meter x 1 meter)
        # Centered at World (0,0,0)
        s = 0.5  # Half size
        self.obj_points = np.array([
            [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]
        ]).T

        # Texture corners (pixel coordinates in the source image)
        h, w = self.texture.shape[:2]
        self.src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    def get_view(self, cam):
        """
        Simulates the camera view by warping the texture based on cam pose.
        """
        # 1. Project the 4 corners of the object into the camera
        proj_corners = cam.project(self.obj_points)  # 2x4
        dst_corners = proj_corners.T.astype(np.float32)

        # 2. Compute Homography from Source Texture -> Camera Screen
        H_mat, _ = cv2.findHomography(self.src_corners, dst_corners)

        # 3. Warp the texture
        if H_mat is not None:
            warped_img = cv2.warpPerspective(self.texture, H_mat, (cam.width, cam.height))
        else:
            warped_img = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)

        return warped_img