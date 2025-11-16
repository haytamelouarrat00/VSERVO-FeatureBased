"""
image_based_simulator.py
========================
Visual servoing simulator using image-based Harris corner features.
"""

import numpy as np
from camera import Camera
from features import ImageFeatureScene, HarrisCornerDetector
from simulator import VisualServoingSimulator
from plot_manager import LiveVisualizer


class ImageBasedVSSimulator(VisualServoingSimulator):
    """
    Visual servoing simulator using features detected from an image.
    Extends base simulator to work with Harris corners.
    """

    def __init__(
        self,
        image_scene,
        initial_camera,
        desired_camera,
        controller_params=None,
        simulation_params=None,
    ):
        """
        Initialize image-based visual servoing simulator.

        Args:
            image_scene: ImageFeatureScene object
            initial_camera: Initial camera pose
            desired_camera: Desired camera pose
            controller_params: Controller parameters
            simulation_params: Simulation parameters
        """
        # Convert ImageFeatureScene to VirtualScene
        scene = image_scene.to_virtual_scene()

        # Store the image scene for visualization
        self.image_scene = image_scene

        # Call parent constructor
        super().__init__(
            scene, initial_camera, desired_camera, controller_params, simulation_params
        )


def create_image_based_simulator(
    image_path=None, image_array=None, max_features=20, gain=0.5, displacement="small"
):
    """
    Factory function to create image-based simulator.

    Args:
        image_path: Path to reference image
        image_array: Or provide image as array
        max_features: Maximum features to detect
        gain: Control gain
        displacement: 'small', 'medium', or 'large'

    Returns:
        ImageBasedVSSimulator instance
    """
    # Create image scene
    image_scene = ImageFeatureScene(
        image_path=image_path,
        image_array=image_array,
        max_features=max_features,
        plane_depth=0.0,
        plane_size=1.0,
    )

    # Create cameras based on displacement
    if displacement == "small":
        initial_pos = [0.15, 0.10, -1.5]
        desired_pos = [0, 0, -1.8]
    elif displacement == "medium":
        initial_pos = [0.3, 0.25, -1.3]
        desired_pos = [0, 0, -1.8]
    elif displacement == "large":
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
        orientation=np.eye(3),
    )

    desired_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=desired_pos,
        orientation=np.eye(3),
    )

    # Orient cameras to look at scene
    centroid = np.mean(image_scene.points_3d, axis=0)
    initial_camera.look_at(centroid)
    desired_camera.look_at(centroid)

    # Controller parameters
    controller_params = {
        "gain": gain,
        "control_law": "classic",
        "depth_estimation": "desired",
        "velocity_limits": {"linear": 0.5, "angular": 0.5},
    }

    simulation_params = {
        "dt": 0.01,
        "max_iterations": 1500,
        "convergence_threshold": 1e-3,
        "check_visibility": True,
        "stop_if_features_lost": True,
    }

    return ImageBasedVSSimulator(
        image_scene,
        initial_camera,
        desired_camera,
        controller_params,
        simulation_params,
    )


# Testing and examples
if __name__ == "__main__":
    from features import create_checkerboard_pattern, create_star_pattern

    print("=== Testing Image-Based Visual Servoing ===\n")

    # Example 1: Checkerboard pattern
    print("1. Creating simulator with checkerboard pattern...")
    checkerboard = create_checkerboard_pattern(square_size=64, n_squares=8)

    sim_checker = create_image_based_simulator(
        image_array=checkerboard, max_features=16, gain=0.5, displacement="small"
    )

    print(f"   Features detected: {len(sim_checker.scene.points_3d)}")

    # Visualize the features
    sim_checker.image_scene.visualize()

    print("\n2. Running simulation...")
    results = sim_checker.run(verbose=True)

    sim_checker.print_results(results)

    # Example 2: Star pattern with live visualization
    print("\n3. Testing with star pattern and live visualization...")
    star = create_star_pattern(size=512)

    sim_star = create_image_based_simulator(
        image_array=star, max_features=10, gain=0.6, displacement="medium"
    )

    sim_star.image_scene.visualize()

    print("\n4. Running with live visualization...")
    live_vis = LiveVisualizer(sim_star)
    results_star = live_vis.run_with_visualization(verbose=True)

    live_vis.save_all(prefix="image_based_star")
