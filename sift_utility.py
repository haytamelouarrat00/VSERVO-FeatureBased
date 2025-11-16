

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
from camera import Camera
from sift import SIFTImageScene, create_sift_simulator
from plot_manager import LiveVisualizer, PlotManager, Visualizer3D


def load_user_image(image_path):
    """
    Load and validate user image.

    Args:
        image_path: Path to image file

    Returns:
        Image array or None if failed
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None

    # Try loading with OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"✓ Image loaded successfully")
    print(f"  Size: {image.shape[1]}x{image.shape[0]}")
    print(f"  Channels: {image.shape[2] if len(image.shape) == 3 else 1}")

    return image


def preview_image_with_features(image, max_features=20):
    """
    Preview image with detected SIFT features.

    Args:
        image: Input image
        max_features: Maximum features to detect

    Returns:
        Number of features detected
    """
    from sift import SIFTFeatureTracker

    print("\nDetecting SIFT features for preview...")
    tracker = SIFTFeatureTracker(n_features=max_features)

    try:
        coords = tracker.extract_reference_features(image)

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Image with features
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        img_with_kp = cv2.drawKeypoints(
            gray,
            tracker.reference_keypoints,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        axes[1].imshow(img_with_kp, cmap='gray' if len(img_with_kp.shape) == 2 else None)
        axes[1].set_title(f'Detected SIFT Features ({len(coords)} features)',
                          fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        return len(coords)

    except Exception as e:
        print(f"Error detecting features: {e}")
        return 0


def create_user_image_simulator(image_path=None,
                                image_array=None,
                                max_features=20,
                                gain=0.5,
                                displacement='medium',
                                feature_quality='medium'):
    """
    Create visual servoing simulator from user image.

    Args:
        image_path: Path to user's image
        image_array: Or provide image as array
        max_features: Maximum features to detect
        gain: Control gain
        displacement: Initial camera displacement ('small', 'medium', 'large')
        feature_quality: SIFT quality ('low', 'medium', 'high')

    Returns:
        Configured simulator
    """
    # Load image if path provided
    if image_path is not None and image_array is None:
        image_array = load_user_image(image_path)
        if image_array is None:
            raise ValueError("Failed to load image")

    if image_array is None:
        raise ValueError("No image provided")

    # Adjust SIFT parameters based on quality
    if feature_quality == 'low':
        contrast_threshold = 0.06
        edge_threshold = 15
    elif feature_quality == 'high':
        contrast_threshold = 0.02
        edge_threshold = 8
    else:  # medium
        contrast_threshold = 0.04
        edge_threshold = 10

    print(f"\nCreating simulator with:")
    print(f"  Max features: {max_features}")
    print(f"  Feature quality: {feature_quality}")
    print(f"  Control gain: {gain}")
    print(f"  Initial displacement: {displacement}")

    # Create SIFT scene
    scene = SIFTImageScene(
        reference_image_array=image_array,
        max_features=max_features,
        plane_depth=0.0,
        plane_size=1.0
    )

    # Adjust detection parameters
    scene.tracker.contrast_threshold = contrast_threshold
    scene.tracker.edge_threshold = edge_threshold
    scene.tracker.sift = cv2.SIFT_create(
        nfeatures=max_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold
    )

    # Re-extract with new parameters
    scene.reference_coords = scene.tracker.extract_reference_features(image_array)
    scene.points_3d = scene._coords_to_3d_points(scene.reference_coords)

    # Create cameras based on displacement
    displacement_configs = {
        'small': {
            'initial_pos': [0.15, 0.10, -1.5],
            'desired_pos': [0, 0, -1.8]
        },
        'medium': {
            'initial_pos': [0.3, 0.25, -1.3],
            'desired_pos': [0, 0, -1.8]
        },
        'large': {
            'initial_pos': [0.5, 0.4, -1.2],
            'desired_pos': [0, 0, -2.0]
        }
    }

    config = displacement_configs.get(displacement, displacement_configs['medium'])

    initial_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=config['initial_pos'],
        orientation=np.eye(3)
    )

    desired_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=config['desired_pos'],
        orientation=np.eye(3)
    )

    # Orient cameras to look at scene
    centroid = np.mean(scene.points_3d, axis=0)
    initial_camera.look_at(centroid)
    desired_camera.look_at(centroid)

    # Controller parameters
    controller_params = {
        'gain': gain,
        'control_law': 'classic',
        'depth_estimation': 'desired',
        'velocity_limits': {'linear': 0.5, 'angular': 0.5}
    }

    simulation_params = {
        'dt': 0.01,
        'max_iterations': 1500,
        'convergence_threshold': 1e-3,
        'check_visibility': True,
        'stop_if_features_lost': True
    }

    from sift import SIFTBasedVSSimulator

    sim = SIFTBasedVSSimulator(
        scene,
        initial_camera,
        desired_camera,
        controller_params,
        simulation_params
    )

    return sim


def run_user_image_vs_interactive():
    """
    Interactive visual servoing with user's image.
    Prompts user for all parameters.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "VISUAL SERVOING WITH YOUR IMAGE")
    print(" " * 20 + "SIFT Feature Tracking")
    print("=" * 70)

    # Get image path
    print("\n📁 Image Selection")
    print("-" * 70)
    print("Enter the path to your desired image:")
    print("(Supported formats: .jpg, .jpeg, .png, .bmp)")
    print("Or press Enter to use a sample image")

    image_path = input("\nImage path: ").strip()

    # Handle default/sample images
    if not image_path:
        print("\nNo image provided. Using sample checkerboard pattern...")
        from features import create_checkerboard_pattern
        image = create_checkerboard_pattern(64, 8)
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_name = "sample_checkerboard"
    else:
        # Remove quotes if user copied path with quotes
        image_path = image_path.strip('"').strip("'")

        image = load_user_image(image_path)
        if image is None:
            print("\n❌ Failed to load image. Exiting.")
            return None

        image_name = Path(image_path).stem

    # Preview with features
    print("\n👁️  Feature Preview")
    print("-" * 70)
    preview_choice = input("Preview image with detected features? (y/n) [default: y]: ").strip().lower()

    if preview_choice != 'n':
        n_preview_features = preview_image_with_features(image, max_features=30)
        if n_preview_features < 4:
            print("\n⚠️  Warning: Very few features detected!")
            print("   This image may not be suitable for visual servoing.")
            cont = input("   Continue anyway? (y/n): ").strip().lower()
            if cont != 'y':
                return None

    # Get parameters
    print("\n⚙️  Configuration Parameters")
    print("-" * 70)

    # Number of features
    print("\n1. Number of features (4-50)")
    print("   Recommended: 15-25 for good performance")
    n_features_input = input("   Max features [default: 20]: ").strip()
    try:
        n_features = int(n_features_input)
        n_features = np.clip(n_features, 4, 50)
    except:
        n_features = 20

    # Feature quality
    print("\n2. Feature detection quality")
    print("   low    - Detect only very strong features (fewer, more reliable)")
    print("   medium - Balanced (recommended)")
    print("   high   - Detect more features (may include weaker ones)")
    quality = input("   Quality (low/medium/high) [default: medium]: ").strip().lower()
    if quality not in ['low', 'medium', 'high']:
        quality = 'medium'

    # Control gain
    print("\n3. Control gain (0.1-2.0)")
    print("   Lower (0.2-0.4): Slower but more stable")
    print("   Medium (0.4-0.7): Balanced (recommended)")
    print("   Higher (0.7-1.5): Faster but may oscillate")
    gain_input = input("   Gain [default: 0.5]: ").strip()
    try:
        gain = float(gain_input)
        gain = np.clip(gain, 0.1, 2.0)
    except:
        gain = 0.5

    # Displacement
    print("\n4. Initial camera displacement")
    print("   small  - Small displacement, fast convergence")
    print("   medium - Moderate displacement (recommended)")
    print("   large  - Large displacement, tests robustness")
    displacement = input("   Displacement (small/medium/large) [default: medium]: ").strip().lower()
    if displacement not in ['small', 'medium', 'large']:
        displacement = 'medium'

    # Visualization
    print("\n5. Visualization options")
    live_viz = input("   Show live visualization? (y/n) [default: y]: ").strip().lower()
    show_live = live_viz != 'n'

    # Summary
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Image: {image_name}")
    print(f"  Max features: {n_features}")
    print(f"  Feature quality: {quality}")
    print(f"  Control gain: {gain}")
    print(f"  Displacement: {displacement}")
    print(f"  Live visualization: {show_live}")
    print("=" * 70)

    confirm = input("\nProceed with simulation? (y/n) [default: y]: ").strip().lower()
    if confirm == 'n':
        print("Simulation cancelled.")
        return None

    # Create simulator
    print("\n🔧 Creating simulator...")
    try:
        sim = create_user_image_simulator(
            image_array=image,
            max_features=n_features,
            gain=gain,
            displacement=displacement,
            feature_quality=quality
        )
    except Exception as e:
        print(f"\n❌ Error creating simulator: {e}")
        return None

    if len(sim.scene.points_3d) < 4:
        print(f"\n❌ Error: Only {len(sim.scene.points_3d)} features detected!")
        print("   Need at least 4 features for 6-DOF control.")
        print("   Try:")
        print("   - Using an image with more texture/corners")
        print("   - Increasing max features")
        print("   - Using 'high' quality setting")
        return None

    # Show detected features
    print("\n📊 Feature Detection Results")
    print("-" * 70)
    sim.sift_scene.visualize_reference(save_path=f'{image_name}_features.png')

    # Run simulation
    print("\n🚀 Running Visual Servoing Simulation")
    print("=" * 70)

    try:
        if show_live:
            live_vis = LiveVisualizer(sim)
            results = live_vis.run_with_visualization(verbose=True)
            live_vis.save_all(prefix=f'user_image_{image_name}')
        else:
            results = sim.run(verbose=True)

            # Create visualizations
            plot_manager = PlotManager()
            plot_manager.update_from_controller(sim.controller)

            vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
            vis_3d.update_visualization(sim.current_camera)

            plot_manager.save(f'user_image_{image_name}_plots.png')
            vis_3d.save(f'user_image_{image_name}_3d.png')

            plt.show()

        # Print results
        sim.print_results(results)

        print("\n📁 Results saved:")
        print(f"   - {image_name}_features.png (detected features)")
        if show_live:
            print(f"   - user_image_{image_name}_plots.png")
            print(f"   - user_image_{image_name}_3d.png")

        return results

    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_user_image_vs_batch(image_path,
                            max_features=20,
                            gain=0.5,
                            displacement='medium',
                            show_viz=True):
    """
    Run visual servoing with user image (non-interactive).

    Args:
        image_path: Path to image file
        max_features: Maximum features to detect
        gain: Control gain
        displacement: Initial displacement
        show_viz: Show visualization

    Returns:
        Simulation results
    """
    print(f"\n🖼️  Loading image: {image_path}")

    image = load_user_image(image_path)
    if image is None:
        return None

    print("\n🔧 Creating simulator...")
    sim = create_user_image_simulator(
        image_array=image,
        max_features=max_features,
        gain=gain,
        displacement=displacement
    )

    if len(sim.scene.points_3d) < 4:
        print(f"❌ Only {len(sim.scene.points_3d)} features detected. Need at least 4.")
        return None

    print(f"✓ Detected {len(sim.scene.points_3d)} features")

    # Show features
    image_name = Path(image_path).stem
    sim.sift_scene.visualize_reference(save_path=f'{image_name}_features.png')

    # Run simulation
    print("\n🚀 Running simulation...")

    if show_viz:
        live_vis = LiveVisualizer(sim)
        results = live_vis.run_with_visualization(verbose=True)
        live_vis.save_all(prefix=f'user_image_{image_name}')
    else:
        results = sim.run(verbose=True)

    sim.print_results(results)

    return results