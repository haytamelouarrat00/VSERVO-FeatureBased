"""
image_features.py
=================
Image-based feature detection and management for visual servoing.
Uses Harris corner detection to extract features from images.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
import cv2


class HarrisCornerDetector:
    """
    Harris corner detector for extracting features from images.
    """

    def __init__(
        self, k=0.04, threshold=0.01, window_size=3, sigma=1.0, min_distance=10
    ):
        """
        Initialize Harris corner detector.

        Args:
            k: Harris detector free parameter (typically 0.04-0.06)
            threshold: Threshold for corner response (fraction of max response)
            window_size: Size of Gaussian window for computing gradients
            sigma: Standard deviation for Gaussian smoothing
            min_distance: Minimum distance between detected corners (pixels)
        """
        self.k = k
        self.threshold = threshold
        self.window_size = window_size
        self.sigma = sigma
        self.min_distance = min_distance

    def detect_corners(self, image, max_corners=None, min_distance=None):
        """
        Detect Harris corners in an image.

        Args:
            image: Grayscale image (H x W)
            max_corners: Maximum number of corners to return
            min_distance: Minimum distance between corners (pixels)

        Returns:
            Nx2 array of corner coordinates (u, v)
        """
        if min_distance is None:
            min_distance = self.min_distance

        if len(image.shape) == 3:
            # Convert to grayscale if needed
            image = np.mean(image, axis=2)

        # Normalize image to [0, 1]
        image = image.astype(float)
        if image.max() > 1.0:
            image = image / 255.0

        # Compute image gradients
        Ix, Iy = self._compute_gradients(image)

        # Compute structure tensor elements
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Apply Gaussian smoothing to structure tensor
        window = self._gaussian_window(self.window_size, self.sigma)

        Sxx = signal.convolve2d(Ixx, window, mode="same", boundary="symm")
        Syy = signal.convolve2d(Iyy, window, mode="same", boundary="symm")
        Sxy = signal.convolve2d(Ixy, window, mode="same", boundary="symm")

        # Compute Harris corner response
        # R = det(M) - k * trace(M)^2
        # where M is the structure tensor
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M**2)

        # Threshold
        threshold_val = self.threshold * R.max()
        R_thresholded = R * (R > threshold_val)

        # Non-maximum suppression
        corners = self._non_maximum_suppression(R_thresholded, min_distance)

        # Limit number of corners
        if max_corners is not None and len(corners) > max_corners:
            # Sort by corner response and keep top corners
            responses = R[corners[:, 1], corners[:, 0]]
            top_indices = np.argsort(responses)[-max_corners:]
            corners = corners[top_indices]

        return corners

    def _compute_gradients(self, image):
        """
        Compute image gradients using Sobel operators.

        Args:
            image: Input image

        Returns:
            tuple: (Ix, Iy) gradient images
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0

        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

        Ix = signal.convolve2d(image, sobel_x, mode="same", boundary="symm")
        Iy = signal.convolve2d(image, sobel_y, mode="same", boundary="symm")

        return Ix, Iy

    def _gaussian_window(self, size, sigma):
        """
        Create a Gaussian window.

        Args:
            size: Window size
            sigma: Standard deviation

        Returns:
            2D Gaussian kernel
        """
        ax = np.arange(-size // 2 + 1, size // 2 + 1)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        return kernel

    def _non_maximum_suppression(self, response, min_distance):
        """
        Apply non-maximum suppression to corner response.

        Args:
            response: Harris corner response map
            min_distance: Minimum distance between corners

        Returns:
            Nx2 array of corner coordinates
        """
        # Find local maxima
        local_max = maximum_filter(response, size=min_distance)
        maxima_mask = (response == local_max) & (response > 0)

        # Get coordinates of corners
        corners = np.column_stack(np.where(maxima_mask))

        # Convert from (row, col) to (u, v) = (x, y)
        corners = corners[:, [1, 0]]

        return corners

    def visualize_corners(self, image, corners, title="Harris Corners"):
        """
        Visualize detected corners on image.

        Args:
            image: Input image
            corners: Nx2 array of corner coordinates
            title: Plot title
        """
        plt.figure(figsize=(10, 8))

        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)

        if len(corners) > 0:
            plt.scatter(
                corners[:, 0], corners[:, 1], c="red", marker="x", s=100, linewidths=2
            )

            # Number the corners
            for i, corner in enumerate(corners):
                plt.text(
                    corner[0] + 5,
                    corner[1] + 5,
                    str(i),
                    color="yellow",
                    fontsize=12,
                    fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
                )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()


class ImageFeatureScene:
    """
    Creates a virtual scene from image features.
    Uses Harris corners from a reference image to generate 3D points.
    """

    def __init__(
        self,
        image_path=None,
        image_array=None,
        max_features=20,
        plane_depth=0.0,
        plane_size=1.0,
    ):
        """
        Initialize image feature scene.

        Args:
            image_path: Path to reference image file
            image_array: Or provide image as numpy array
            max_features: Maximum number of features to extract
            plane_depth: Z-coordinate of the planar scene
            plane_size: Physical size of the image plane in 3D (meters)
        """
        # Load image
        if image_path is not None:
            self.image = self._load_image(image_path)
        elif image_array is not None:
            self.image = image_array
        else:
            # Create a synthetic test pattern
            self.image = self._create_test_pattern()

        self.max_features = max_features
        self.plane_depth = plane_depth
        self.plane_size = plane_size

        # Detect corners
        self.detector = HarrisCornerDetector(k=0.04, threshold=0.01, min_distance=20)

        self.image_corners = self.detector.detect_corners(
            self.image, max_corners=max_features, min_distance=20
        )

        # Convert to 3D points
        self.points_3d = self._corners_to_3d_points()

        print(f"Detected {len(self.image_corners)} corners")

    def _load_image(self, path):
        """Load image from file."""
        try:
            import cv2

            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image from {path}")
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except ImportError:
            # Fallback to matplotlib
            image = plt.imread(path)
            return image

    def _create_test_pattern(self, size=512):
        """
        Create a synthetic test pattern with corners.

        Args:
            size: Image size

        Returns:
            Test pattern image
        """
        image = np.ones((size, size)) * 255

        # Draw rectangles
        rectangles = [
            (100, 100, 150, 150),
            (300, 100, 150, 150),
            (100, 300, 150, 150),
            (300, 300, 150, 150),
            (200, 200, 100, 100),
        ]

        for x, y, w, h in rectangles:
            image[y : y + h, x : x + w] = 0

        # Add some circles
        for cx, cy, r in [(150, 400), (400, 400), (256, 450)]:
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
            image[mask] = 128

        return image.astype(np.uint8)

    def _corners_to_3d_points(self):
        """
        Convert 2D image corners to 3D points on a plane.

        The image plane is mapped to a physical plane in 3D space.
        Image coordinates are normalized to [-size/2, size/2].

        Returns:
            Nx3 array of 3D points
        """
        h, w = self.image.shape[:2]

        # Normalize corners to [-plane_size/2, plane_size/2]
        # Image coordinates: (0, 0) at top-left, (w, h) at bottom-right
        # 3D coordinates: centered at origin

        corners_normalized = self.image_corners.copy().astype(float)

        # Convert to centered coordinates
        corners_normalized[:, 0] = (
            (corners_normalized[:, 0] - w / 2) / w * self.plane_size
        )
        corners_normalized[:, 1] = (
            -(corners_normalized[:, 1] - h / 2) / h * self.plane_size
        )  # Flip Y

        # Create 3D points (all at same depth)
        points_3d = np.zeros((len(corners_normalized), 3))
        points_3d[:, 0] = corners_normalized[:, 0]  # X
        points_3d[:, 1] = corners_normalized[:, 1]  # Y
        points_3d[:, 2] = self.plane_depth  # Z

        return points_3d

    def get_scene_info(self):
        """Get information about the scene."""
        return {
            "n_features": len(self.image_corners),
            "image_shape": self.image.shape,
            "image_corners": self.image_corners,
            "points_3d": self.points_3d,
            "plane_size": self.plane_size,
            "plane_depth": self.plane_depth,
        }

    def visualize(self, save_path=None):
        """
        Visualize the detected corners on the image.

        Args:
            save_path: Optional path to save the visualization
        """
        self.detector.visualize_corners(
            self.image,
            self.image_corners,
            title=f"Harris Corners Detected ({len(self.image_corners)} features)",
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def to_virtual_scene(self):
        """
        Convert to VirtualScene object for use with simulator.

        Returns:
            VirtualScene object
        """
        from scene import VirtualScene

        scene = VirtualScene(points_3d=self.points_3d, scene_type="custom")

        return scene


def create_checkerboard_pattern(square_size=50, n_squares=8):
    """
    Create a checkerboard pattern for testing.

    Args:
        square_size: Size of each square in pixels
        n_squares: Number of squares per side

    Returns:
        Checkerboard image
    """
    size = square_size * n_squares
    image = np.zeros((size, size), dtype=np.uint8)

    for i in range(n_squares):
        for j in range(n_squares):
            if (i + j) % 2 == 0:
                y1, y2 = i * square_size, (i + 1) * square_size
                x1, x2 = j * square_size, (j + 1) * square_size
                image[y1:y2, x1:x2] = 255

    return image


def create_star_pattern(size=512):
    """
    Create a star pattern with strong corners.

    Args:
        size: Image size

    Returns:
        Star pattern image
    """
    image = np.ones((size, size)) * 255

    # Draw star shape
    center = size // 2
    outer_radius = size // 3
    inner_radius = size // 6
    n_points = 5

    points = []
    for i in range(n_points * 2):
        angle = i * np.pi / n_points
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius

        x = int(center + r * np.cos(angle - np.pi / 2))
        y = int(center + r * np.sin(angle - np.pi / 2))
        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    # Fill star
    from matplotlib.path import Path

    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    coords = np.c_[xx.ravel(), yy.ravel()]
    path = Path(points)
    mask = path.contains_points(coords).reshape(size, size)
    image[mask] = 0

    return image.astype(np.uint8)


# Testing
if __name__ == "__main__":
    print("=== Testing Harris Corner Detection ===\n")

    # Test 1: Synthetic test pattern
    print("1. Testing with synthetic pattern...")
    scene = ImageFeatureScene(max_features=15)
    print(f"   Detected {len(scene.image_corners)} corners")
    scene.visualize()

    # Test 2: Checkerboard
    print("\n2. Testing with checkerboard...")
    checkerboard = create_checkerboard_pattern(square_size=64, n_squares=8)
    scene_checker = ImageFeatureScene(
        image_array=checkerboard, max_features=20, plane_size=1.0
    )
    print(f"   Detected {len(scene_checker.image_corners)} corners")
    scene_checker.visualize()

    # Test 3: Star pattern
    print("\n3. Testing with star pattern...")
    star = create_star_pattern(size=512)
    scene_star = ImageFeatureScene(image_array=star, max_features=10, plane_size=0.8)
    print(f"   Detected {len(scene_star.image_corners)} corners")
    scene_star.visualize()

    # Print 3D points
    print("\n4. 3D points for first scene:")
    print(scene.points_3d)

    print("\n5. Scene info:")
    info = scene.get_scene_info()
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            print(f"   {key}: shape {value.shape}")
        else:
            print(f"   {key}: {value}")
