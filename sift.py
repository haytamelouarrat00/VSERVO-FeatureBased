"""
sift_features.py
================
SIFT-based feature detection and tracking for visual servoing.
Maintains fixed feature correspondence throughout the motion.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


class SIFTFeatureTracker:
    """
    SIFT feature detector with fixed feature correspondence.
    Features are detected once at desired pose and tracked throughout motion.
    """

    def __init__(self,
                 n_features=20,
                 contrast_threshold=0.04,
                 edge_threshold=10):
        """
        Initialize SIFT feature tracker.

        Args:
            n_features: Maximum number of features to detect
            contrast_threshold: Contrast threshold for feature detection
            edge_threshold: Edge threshold for feature detection
        """
        self.n_features = n_features
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

        # Create SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold
        )

        # Store reference features (detected at desired pose)
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.reference_image = None

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def extract_reference_features(self, image):
        """
        Extract reference features from desired view.
        This is done ONCE at initialization.

        Args:
            image: Reference image (desired view)

        Returns:
            Nx2 array of feature coordinates
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Detect keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        if keypoints is None or len(keypoints) == 0:
            raise ValueError("No SIFT features detected in reference image!")

        # Store reference features
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        self.reference_image = gray.copy()

        # Extract coordinates
        coords = np.array([kp.pt for kp in keypoints])

        print(f"Extracted {len(keypoints)} reference features")

        return coords

    def track_features(self, current_image, match_ratio=0.75):
        """
        Track reference features in current image.
        Maintains correspondence with reference features.

        Args:
            current_image: Current camera view
            match_ratio: Lowe's ratio test threshold

        Returns:
            tuple: (current_coords, reference_coords, valid_mask)
                - current_coords: Nx2 matched positions in current image
                - reference_coords: Nx2 corresponding reference positions
                - valid_mask: N boolean array indicating good matches
        """
        if self.reference_keypoints is None:
            raise ValueError("Reference features not set! Call extract_reference_features first.")

        if len(current_image.shape) == 3:
            gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = current_image.astype(np.uint8)

        # Detect features in current image
        current_keypoints, current_descriptors = self.sift.detectAndCompute(gray, None)

        if current_keypoints is None or len(current_keypoints) == 0:
            # No features detected - return empty
            n_ref = len(self.reference_keypoints)
            return (np.zeros((n_ref, 2)),
                    np.array([kp.pt for kp in self.reference_keypoints]),
                    np.zeros(n_ref, dtype=bool))

        # Match descriptors using KNN (k=2 for ratio test)
        matches = self.matcher.knnMatch(
            self.reference_descriptors,
            current_descriptors,
            k=2
        )

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < match_ratio * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Only one match found, accept it
                good_matches.append(match_pair[0])

        # Extract matched coordinates
        n_ref = len(self.reference_keypoints)
        current_coords = np.zeros((n_ref, 2))
        reference_coords = np.array([kp.pt for kp in self.reference_keypoints])
        valid_mask = np.zeros(n_ref, dtype=bool)

        for match in good_matches:
            ref_idx = match.queryIdx
            curr_idx = match.trainIdx

            current_coords[ref_idx] = current_keypoints[curr_idx].pt
            valid_mask[ref_idx] = True

        print(f"Tracked {np.sum(valid_mask)}/{n_ref} features")

        return current_coords, reference_coords, valid_mask

    def get_reference_coordinates(self):
        """
        Get reference feature coordinates.

        Returns:
            Nx2 array of reference coordinates
        """
        if self.reference_keypoints is None:
            return None

        return np.array([kp.pt for kp in self.reference_keypoints])

    def visualize_reference(self, save_path=None):
        """Visualize reference features."""
        if self.reference_image is None:
            print("No reference image available")
            return

        img_with_keypoints = cv2.drawKeypoints(
            self.reference_image,
            self.reference_keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        plt.figure(figsize=(12, 8))
        plt.imshow(img_with_keypoints)
        plt.title(f'Reference SIFT Features ({len(self.reference_keypoints)} features)',
                  fontsize=14, fontweight='bold')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def visualize_tracking(self, current_image, save_path=None):
        """
        Visualize feature tracking with matches.

        Args:
            current_image: Current camera view
            save_path: Optional path to save visualization
        """
        if self.reference_image is None:
            print("No reference features available")
            return

        current_coords, reference_coords, valid_mask = self.track_features(current_image)

        if len(current_image.shape) == 3:
            current_gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
        else:
            current_gray = current_image.astype(np.uint8)

        # Create side-by-side visualization
        h1, w1 = self.reference_image.shape
        h2, w2 = current_gray.shape

        h = max(h1, h2)
        w = w1 + w2

        combined = np.zeros((h, w), dtype=np.uint8)
        combined[:h1, :w1] = self.reference_image
        combined[:h2, w1:] = current_gray

        # Convert to color for drawing
        combined_color = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)

        # Draw matches
        n_matched = 0
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                pt1 = tuple(reference_coords[i].astype(int))
                pt2 = tuple((current_coords[i] + np.array([w1, 0])).astype(int))

                # Draw line
                cv2.line(combined_color, pt1, pt2, (0, 255, 0), 1)

                # Draw circles
                cv2.circle(combined_color, pt1, 5, (255, 0, 0), 2)
                cv2.circle(combined_color, pt2, 5, (0, 0, 255), 2)

                n_matched += 1

        plt.figure(figsize=(16, 8))
        plt.imshow(combined_color)
        plt.title(f'Feature Tracking: {n_matched}/{len(valid_mask)} matches',
                  fontsize=14, fontweight='bold')
        plt.axis('off')

        # Add text
        plt.text(w1 / 2, 30, 'Reference (Desired)',
                 ha='center', color='white', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        plt.text(w1 + w2 / 2, 30, 'Current',
                 ha='center', color='white', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


class SIFTImageScene:
    """
    Creates a virtual scene from SIFT features with fixed correspondence.
    """

    def __init__(self,
                 reference_image_path=None,
                 reference_image_array=None,
                 max_features=20,
                 plane_depth=0.0,
                 plane_size=1.0):
        """
        Initialize SIFT-based image scene.

        Args:
            reference_image_path: Path to reference image (desired view)
            reference_image_array: Or provide as numpy array
            max_features: Maximum features to extract
            plane_depth: Z-coordinate of planar scene
            plane_size: Physical size in meters
        """
        # Load reference image
        if reference_image_path is not None:
            self.reference_image = cv2.imread(reference_image_path)
            if self.reference_image is None:
                raise ValueError(f"Could not load image: {reference_image_path}")
            self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2RGB)
        elif reference_image_array is not None:
            self.reference_image = reference_image_array
        else:
            # Use test pattern
            from features import create_checkerboard_pattern
            self.reference_image = create_checkerboard_pattern(64, 8)

        self.max_features = max_features
        self.plane_depth = plane_depth
        self.plane_size = plane_size

        # Create SIFT tracker
        self.tracker = SIFTFeatureTracker(n_features=max_features)

        # Extract reference features
        self.reference_coords = self.tracker.extract_reference_features(
            self.reference_image
        )

        # Convert to 3D points
        self.points_3d = self._coords_to_3d_points(self.reference_coords)

        print(f"Created scene with {len(self.points_3d)} SIFT features")

    def _coords_to_3d_points(self, coords):
        """Convert 2D coordinates to 3D points on plane."""
        h, w = self.reference_image.shape[:2]

        # Normalize to [-plane_size/2, plane_size/2]
        coords_normalized = coords.copy().astype(float)
        coords_normalized[:, 0] = (coords_normalized[:, 0] - w / 2) / w * self.plane_size
        coords_normalized[:, 1] = -(coords_normalized[:, 1] - h / 2) / h * self.plane_size

        # Create 3D points
        points_3d = np.zeros((len(coords), 3))
        points_3d[:, 0] = coords_normalized[:, 0]
        points_3d[:, 1] = coords_normalized[:, 1]
        points_3d[:, 2] = self.plane_depth

        return points_3d

    def track_features_in_view(self, current_image):
        """
        Track reference features in current view.

        Args:
            current_image: Current camera view (rendered image)

        Returns:
            tuple: (current_coords, valid_mask)
        """
        current_coords, _, valid_mask = self.tracker.track_features(current_image)
        return current_coords, valid_mask

    def visualize_reference(self, save_path=None):
        """Visualize reference features."""
        self.tracker.visualize_reference(save_path)

    def to_virtual_scene(self):
        """Convert to VirtualScene for simulator."""
        from scene import VirtualScene
        return VirtualScene(points_3d=self.points_3d, scene_type='sift')


class SIFTBasedVSSimulator:
    """
    Visual servoing simulator using SIFT features with fixed correspondence.

    KEY CONCEPT: Features are extracted ONCE at desired pose, then TRACKED
    throughout the motion. This maintains feature correspondence which is
    essential for IBVS.
    """

    def __init__(self,
                 sift_scene,
                 initial_camera,
                 desired_camera,
                 controller_params=None,
                 simulation_params=None):
        """
        Initialize SIFT-based simulator.

        Args:
            sift_scene: SIFTImageScene with reference features
            initial_camera: Initial camera pose
            desired_camera: Desired camera pose
            controller_params: Controller parameters
            simulation_params: Simulation parameters
        """
        from simulator import VisualServoingSimulator

        self.sift_scene = sift_scene

        # Create base simulator
        scene = sift_scene.to_virtual_scene()

        # Default parameters
        if controller_params is None:
            controller_params = {
                'gain': 0.5,
                'control_law': 'classic',
                'depth_estimation': 'desired',
                'velocity_limits': {'linear': 0.5, 'angular': 0.5}
            }

        if simulation_params is None:
            simulation_params = {
                'dt': 0.01,
                'max_iterations': 1500,
                'convergence_threshold': 1e-3,
                'check_visibility': True,
                'stop_if_features_lost': True
            }

        # Note: We don't actually use feature tracking during simulation
        # because we're working with perfect 3D->2D projection
        # In a real system, you would use sift_scene.track_features_in_view()

        # Create simulator
        self.simulator = VisualServoingSimulator(
            scene,
            initial_camera,
            desired_camera,
            controller_params,
            simulation_params
        )

    def run(self, verbose=True, callback=None, **kwargs):
        """
        Run simulation with optional visualization callback.

        Extra keyword arguments are forwarded to the base simulator.
        """
        return self.simulator.run(verbose=verbose, callback=callback, **kwargs)

    def __getattr__(self, name):
        """Delegate to base simulator."""
        return getattr(self.simulator, name)


def create_sift_simulator(image_path=None,
                          image_array=None,
                          max_features=20,
                          gain=0.5,
                          displacement='medium'):
    """
    Factory function for SIFT-based simulator.

    Args:
        image_path: Path to reference image
        image_array: Or provide as array
        max_features: Maximum features
        gain: Control gain
        displacement: Initial displacement size

    Returns:
        SIFTBasedVSSimulator
    """
    from camera import Camera

    # Create SIFT scene
    sift_scene = SIFTImageScene(
        reference_image_path=image_path,
        reference_image_array=image_array,
        max_features=max_features,
        plane_depth=0.0,
        plane_size=1.0
    )

    # Create cameras
    if displacement == 'small':
        initial_pos = [0.15, 0.10, -1.5]
        desired_pos = [0, 0, -1.8]
    elif displacement == 'medium':
        initial_pos = [0.3, 0.25, -1.3]
        desired_pos = [0, 0, -1.8]
    elif displacement == 'large':
        initial_pos = [0.5, 0.4, -1.2]
        desired_pos = [0, 0, -2.0]
    else:
        initial_pos = [0.15, 0.10, -1.5]
        desired_pos = [0, 0, -1.8]

    initial_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=initial_pos,
        orientation=np.eye(3)
    )

    desired_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=desired_pos,
        orientation=np.eye(3)
    )

    # Orient cameras
    centroid = np.mean(sift_scene.points_3d, axis=0)
    initial_camera.look_at(centroid)
    desired_camera.look_at(centroid)

    # Controller parameters
    controller_params = {
        'gain': gain,
        'control_law': 'classic',
        'depth_estimation': 'desired',
        'velocity_limits': {'linear': 0.5, 'angular': 0.5}
    }

    return SIFTBasedVSSimulator(
        sift_scene,
        initial_camera,
        desired_camera,
        controller_params
    )


# Testing
if __name__ == "__main__":
    print("=== Testing SIFT Feature Tracking ===\n")

    # Create test image
    from features import create_checkerboard_pattern

    test_image = create_checkerboard_pattern(64, 8)

    # Test 1: Feature extraction
    print("1. Extracting SIFT features...")
    tracker = SIFTFeatureTracker(n_features=20)
    coords = tracker.extract_reference_features(test_image)
    print(f"   Extracted {len(coords)} features")
    tracker.visualize_reference()

    # Test 2: Feature tracking (simulate slight movement)
    print("\n2. Testing feature tracking...")
    # Create slightly modified image (simulate camera movement)
    import cv2

    M = np.float32([[1, 0, 20], [0, 1, 10]])  # Translation
    moved_image = cv2.warpAffine(test_image, M, (test_image.shape[1], test_image.shape[0]))

    current, reference, valid = tracker.track_features(moved_image)
    print(f"   Tracked {np.sum(valid)}/{len(valid)} features")
    tracker.visualize_tracking(moved_image)

    # Test 3: Full simulator
    print("\n3. Creating SIFT-based simulator...")
    sim = create_sift_simulator(
        image_array=test_image,
        max_features=15,
        gain=0.5,
        displacement='small'
    )

    sim.sift_scene.visualize_reference()

    print("\n4. Running simulation...")
    results = sim.run(verbose=True)

    print(f"\nFinal results:")
    print(f"   Converged: {results['converged']}")
    print(f"   Iterations: {results['iterations']}")
    print(f"   Final error: {results['final_error']:.6f}")
