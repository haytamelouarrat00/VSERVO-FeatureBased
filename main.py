"""
Visual Servoing Simulation - Main Script
=========================================

Complete implementation of Image-Based Visual Servoing (IBVS) based on
Chaumette's work. This script demonstrates various scenarios and configurations.

Author: El Ouarrat Haytam
Date: 2025
"""

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Import all components
from camera import Camera
from scene import VirtualScene, SceneConfiguration
from interaction_matrix import InteractionMatrix
from controller import VSController, AdvancedVSController
from simulator import VisualServoingSimulator, SimulatorFactory
from plot_manager import PlotManager, Visualizer3D, LiveVisualizer


def example_8_harris_corners_checkerboard():
    """
    Example 8: Visual servoing with Harris corners on checkerboard.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Harris Corners - Checkerboard Pattern")
    print("=" * 70)

    from features import create_checkerboard_pattern
    from ibvs import create_image_based_simulator
    from plot_manager import LiveVisualizer

    # Create checkerboard
    print("\nCreating checkerboard pattern...")
    checkerboard = create_checkerboard_pattern(square_size=64, n_squares=8)

    # Create simulator
    print("Detecting Harris corners...")
    sim = create_image_based_simulator(
        image_array=checkerboard, max_features=16, gain=0.5, displacement="small"
    )

    print(f"Features detected: {len(sim.scene.points_3d)}")

    # Show detected corners
    sim.image_scene.visualize(save_path="example8_corners.png")

    print("\nRunning visual servoing simulation...")

    # Run with live visualization
    live_vis = LiveVisualizer(sim)
    results = live_vis.run_with_visualization(verbose=True)

    live_vis.save_all(prefix="example8_checkerboard")

    sim.print_results(results)

    return results


def example_9_harris_corners_custom_image():
    """
    Example 9: Visual servoing with Harris corners from custom image.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 9: Harris Corners - Custom Image")
    print("=" * 70)

    from features import create_star_pattern
    from ibvs import create_image_based_simulator
    from plot_manager import PlotManager, Visualizer3D

    # Create star pattern
    print("\nCreating star pattern...")
    star = create_star_pattern(size=512)

    # Create simulator
    print("Detecting Harris corners...")
    sim = create_image_based_simulator(
        image_array=star, max_features=12, gain=0.6, displacement="medium"
    )

    print(f"Features detected: {len(sim.scene.points_3d)}")

    # Show detected corners
    sim.image_scene.visualize(save_path="example9_corners.png")

    print("\nRunning visual servoing simulation...")
    results = sim.run(verbose=True)

    # Visualize
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example9_plots.png")
    vis_3d.save("example9_3d.png")

    sim.print_results(results)

    plt.show()

    return results


def example_10_load_custom_image():
    """
    Example 10: Load and use your own image.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 10: Visual Servoing with Your Own Image")
    print("=" * 70)

    from ibvs import create_image_based_simulator
    from plot_manager import LiveVisualizer

    # Ask for image path
    print("\nEnter path to your image (or press Enter for test pattern):")
    image_path = input("Image path: ").strip()

    if not image_path:
        print("Using default test pattern...")
        image_path = None

    print("\nHow many features to detect? (5-30)")
    n_features_input = input("Number of features [default: 15]: ").strip()
    try:
        n_features = int(n_features_input)
        n_features = np.clip(n_features, 5, 30)
    except:
        n_features = 15

    print("\nControl gain? (0.1-2.0)")
    gain_input = input("Gain [default: 0.5]: ").strip()
    try:
        gain = float(gain_input)
        gain = np.clip(gain, 0.1, 2.0)
    except:
        gain = 0.5

    print("\nInitial displacement? (small/medium/large)")
    disp_input = input("Displacement [default: medium]: ").strip().lower()
    if disp_input not in ["small", "medium", "large"]:
        disp_input = "medium"

    print("\n" + "-" * 70)
    print("CONFIGURATION:")
    print(f"  Image: {image_path if image_path else 'Test pattern'}")
    print(f"  Max features: {n_features}")
    print(f"  Control gain: {gain}")
    print(f"  Displacement: {disp_input}")
    print("-" * 70 + "\n")

    # Create simulator
    try:
        sim = create_image_based_simulator(
            image_path=image_path,
            max_features=n_features,
            gain=gain,
            displacement=disp_input,
        )

        print(f"Successfully detected {len(sim.scene.points_3d)} features!")

        # Show detected corners
        sim.image_scene.visualize()

        # Run with live visualization
        print("\nRunning simulation with live visualization...")
        live_vis = LiveVisualizer(sim)
        results = live_vis.run_with_visualization(verbose=True)

        live_vis.save_all(prefix="example10_custom")

        sim.print_results(results)

        return results

    except Exception as e:
        print(f"\nError: {e}")
        print("Could not load or process the image.")
        return None


def example_1_basic_simulation():
    """
    Example 1: Basic visual servoing with planar features.
    Simple scenario with small displacement.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Visual Servoing Simulation")
    print("=" * 70)

    # Create a standard simulator
    sim = SimulatorFactory.create_standard_simulator(
        scene_type="planar", gain=0.5, control_law="classic"
    )

    print("\nConfiguration:")
    print(f"  Scene: Planar (4 corner points)")
    print(f"  Control Law: Classic IBVS")
    print(f"  Gain: 0.5")
    print(f"  Initial displacement: Small")

    # Run simulation
    results = sim.run(verbose=True)

    # Visualize results
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    # Save results
    plot_manager.save("example1_plots.png")
    vis_3d.save("example1_3d.png")

    sim.print_results(results)

    plt.show()

    return results


def example_2_large_displacement():
    """
    Example 2: Visual servoing with large displacement.
    Tests controller stability with significant initial error.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Large Displacement Scenario")
    print("=" * 70)

    # Create large displacement simulator
    sim = SimulatorFactory.create_large_displacement_simulator(gain=0.3)

    print("\nConfiguration:")
    print(f"  Scene: Planar")
    print(f"  Control Law: Adaptive gain")
    print(f"  Initial gain: 0.3 (conservative for stability)")
    print(f"  Initial displacement: Large")

    # Run without live visualization first
    results = sim.run(verbose=True)

    # Create visualizations
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example2_plots.png")
    vis_3d.save("example2_3d.png")

    sim.print_results(results)

    plt.show()

    return results


def example_3_different_scenes():
    """
    Example 3: Compare performance on different scene types.
    Tests with planar, cube, and grid configurations.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Comparison of Different Scene Types")
    print("=" * 70)

    scene_types = ["planar", "cube", "grid"]
    results_all = {}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Error Evolution for Different Scene Types", fontsize=14, fontweight="bold"
    )

    for idx, scene_type in enumerate(scene_types):
        print(f"\n--- Testing {scene_type.upper()} scene ---")

        # Create simulator
        sim = SimulatorFactory.create_standard_simulator(
            scene_type=scene_type, gain=0.5, control_law="classic"
        )

        # Run simulation
        results = sim.run(verbose=False)
        results_all[scene_type] = results

        # Plot error evolution
        ax = axes[idx]
        iterations = np.arange(len(sim.controller.error_norm_history))
        ax.plot(
            iterations, sim.controller.error_norm_history, linewidth=2, color="blue"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error Norm")
        ax.set_title(f"{scene_type.capitalize()} Scene")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Print summary
        print(f"  Converged: {results['converged']}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Final error: {results['final_error']:.6f}")

    plt.tight_layout()
    plt.savefig("example3_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results_all


def example_4_gain_comparison():
    """
    Example 4: Compare different control gains.
    Shows effect of gain on convergence speed and stability.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Control Gain Comparison")
    print("=" * 70)

    gains = [0.2, 0.5, 0.8, 1.2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Effect of Control Gain on Convergence", fontsize=14, fontweight="bold"
    )
    axes = axes.flatten()

    for idx, gain in enumerate(gains):
        print(f"\n--- Testing gain = {gain} ---")

        # Create scene and cameras
        scene, initial_cam, desired_cam = SceneConfiguration.create_standard_setup(
            "planar"
        )

        # Create simulator with specific gain
        controller_params = {
            "gain": gain,
            "control_law": "classic",
            "depth_estimation": "desired",
            "velocity_limits": {"linear": 1.0, "angular": 1.0},
        }

        simulation_params = {
            "dt": 0.01,
            "max_iterations": 50000,
            "convergence_threshold": 1e-3,
            "check_visibility": True,
            "stop_if_features_lost": True,
        }

        sim = VisualServoingSimulator(
            scene, initial_cam, desired_cam, controller_params, simulation_params
        )

        # Run simulation
        results = sim.run(verbose=False)

        # Plot
        ax = axes[idx]
        iterations = np.arange(len(sim.controller.error_norm_history))
        ax.plot(
            iterations, sim.controller.error_norm_history, linewidth=2, color="blue"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Error Norm")
        ax.set_title(f"Gain λ = {gain}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Add convergence info
        if results["converged"]:
            ax.text(
                0.6,
                0.9,
                f"Converged\nIter: {results['iterations']}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
                fontsize=9,
            )
        else:
            ax.text(
                0.6,
                0.9,
                f"Not converged\nFinal error: {results['final_error']:.4f}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
                fontsize=9,
            )

        print(f"  Converged: {results['converged']}")
        print(f"  Iterations: {results['iterations']}")
        print(f"  Final error: {results['final_error']:.6f}")

    plt.tight_layout()
    plt.savefig("example4_gain_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def example_5_live_visualization():
    """
    Example 5: Run simulation with live visualization.
    Real-time plotting of error, velocity, and 3D motion.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Live Visualization")
    print("=" * 70)
    print("\nThis example will show real-time visualization during simulation.")
    print("Close the plot windows when done to continue.\n")

    # Create simulator
    sim = SimulatorFactory.create_standard_simulator(
        scene_type="planar", gain=0.5, control_law="classic"
    )

    # Create live visualizer
    live_vis = LiveVisualizer(sim)

    # Run with live visualization
    results = live_vis.run_with_visualization(verbose=True)

    # Save results
    live_vis.save_all(prefix="example5_live")

    sim.print_results(results)

    return results


def example_6_custom_scenario():
    """
    Example 6: Custom scenario - pure rotation task.
    Camera position stays same, only orientation changes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Pure Rotation Task")
    print("=" * 70)

    # Create scene
    scene = VirtualScene(scene_type="planar", size=1.0, z_plane=0.0)

    # Create cameras at same position, different orientations
    position = [0, 0, -2.0]

    initial_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=position,
        orientation=np.eye(3),
    )

    desired_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=position,
        orientation=np.eye(3),
    )

    # Rotate initial camera by 20 degrees around Y axis
    rotation = R.from_euler("y", 25, degrees=True).as_matrix()
    initial_camera.rotation = rotation @ initial_camera.rotation

    # Make cameras look at scene
    centroid = scene.get_centroid()
    desired_camera.look_at(centroid)

    print("\nConfiguration:")
    print(f"  Task: Pure rotation (no translation)")
    print(f"  Rotation angle: 25 degrees around Y axis")
    print(f"  Scene: Planar")

    # Create controller parameters
    controller_params = {
        "gain": 0.6,
        "control_law": "classic",
        "depth_estimation": "desired",
        "velocity_limits": {"linear": 0.5, "angular": 0.5},
    }

    simulation_params = {
        "dt": 0.01,
        "max_iterations": 50000,
        "convergence_threshold": 1e-3,
        "check_visibility": True,
        "stop_if_features_lost": True,
    }

    # Create simulator
    sim = VisualServoingSimulator(
        scene, initial_camera, desired_camera, controller_params, simulation_params
    )

    # Run
    results = sim.run(verbose=True)

    # Visualize
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example6_plots.png")
    vis_3d.save("example6_3d.png")

    sim.print_results(results)

    plt.show()

    return results


def example_7_sphere_scene():
    """
    Example 7: Visual servoing with 3D sphere point cloud.
    Tests with non-planar features distributed in 3D.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: 3D Sphere Scene")
    print("=" * 70)

    # Create sphere scene
    scene = VirtualScene(scene_type="sphere", n_points=12, radius=0.5)

    # Create cameras
    initial_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=[0.5, 0.4, -1.8],
        orientation=np.eye(3),
    )

    desired_camera = Camera(
        focal_length=800,
        image_width=640,
        image_height=480,
        position=[0, 0, -2.5],
        orientation=np.eye(3),
    )

    # Orient cameras
    centroid = scene.get_centroid()
    initial_camera.look_at(centroid)
    desired_camera.look_at(centroid)

    print("\nConfiguration:")
    print(f"  Scene: Sphere with 12 points")
    print(f"  Radius: 0.5 m")
    print(f"  Features distributed in 3D")

    # Create simulator
    controller_params = {
        "gain": 0.5,
        "control_law": "classic",
        "depth_estimation": "desired",
        "velocity_limits": {"linear": 0.5, "angular": 0.5},
    }

    simulation_params = {
        "dt": 0.01,
        "max_iterations": 50000,
        "convergence_threshold": 1e-3,
        "check_visibility": True,
        "stop_if_features_lost": True,
    }

    sim = VisualServoingSimulator(
        scene, initial_camera, desired_camera, controller_params, simulation_params
    )

    # Run
    results = sim.run(verbose=True)

    # Visualize
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example7_plots.png")
    vis_3d.save("example7_3d.png")

    sim.print_results(results)

    plt.show()

    return results


def interactive_demo():
    """
    Interactive demonstration where user can adjust parameters.
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE DEMO")
    print("=" * 70)

    print("\nSelect scene type:")
    print("  1. Planar (4 corners)")
    print("  2. Cube (8 corners)")
    print("  3. Grid (4x4)")
    print("  4. Sphere (12 points)")

    scene_choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    scene_map = {"1": "planar", "2": "cube", "3": "grid", "4": "sphere"}
    scene_type = scene_map.get(scene_choice, "planar")

    print("\nSelect control gain:")
    gain_input = input("Enter gain (0.1-2.0) [default: 0.5]: ").strip()
    try:
        gain = float(gain_input)
        gain = np.clip(gain, 0.1, 2.0)
    except:
        gain = 0.5

    print("\nSelect control law:")
    print("  1. Classic IBVS")
    print("  2. Adaptive gain")
    law_choice = input("\nEnter choice (1-2) [default: 1]: ").strip()
    control_law = "adaptive" if law_choice == "2" else "classic"

    print("\nShow live visualization?")
    viz_choice = input("(y/n) [default: y]: ").strip().lower()
    show_live = viz_choice != "n"

    print("\n" + "-" * 70)
    print("CONFIGURATION:")
    print(f"  Scene type: {scene_type}")
    print(f"  Control gain: {gain}")
    print(f"  Control law: {control_law}")
    print(f"  Live visualization: {show_live}")
    print("-" * 70 + "\n")

    # Create simulator
    sim = SimulatorFactory.create_standard_simulator(
        scene_type=scene_type, gain=gain, control_law=control_law
    )

    if show_live:
        live_vis = LiveVisualizer(sim)
        results = live_vis.run_with_visualization(verbose=True)
        live_vis.save_all(prefix="interactive_demo")
    else:
        results = sim.run(verbose=True)

        plot_manager = PlotManager()
        plot_manager.update_from_controller(sim.controller)

        vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
        vis_3d.update_visualization(sim.current_camera)

        plot_manager.save("interactive_demo_plots.png")
        vis_3d.save("interactive_demo_3d.png")

        plt.show()

    sim.print_results(results)

    return results


def example_11_user_image_interactive():
    """
    Example 11: Visual servoing with user's own image (interactive).
    Complete interactive setup with parameter selection.
    """
    from sift_utility import run_user_image_vs_interactive

    print("\n" + "=" * 70)
    print("EXAMPLE 11: Visual Servoing with Your Own Image")
    print("=" * 70)

    results = run_user_image_vs_interactive()

    return results


def example_12_user_image_batch():
    """
    Example 12: Visual servoing with user image (batch mode).
    Quick setup for users who know their parameters.
    """
    from sift_utility import run_user_image_vs_batch

    print("\n" + "=" * 70)
    print("EXAMPLE 12: User Image - Batch Mode")
    print("=" * 70)

    print("\nEnter path to your image:")
    image_path = input("Image path: ").strip().strip('"').strip("'")

    if not image_path:
        print("No image provided. Exiting.")
        return None

    print("\nUsing default parameters:")
    print("  Max features: 20")
    print("  Gain: 0.5")
    print("  Displacement: medium")
    print("  Visualization: enabled")

    results = run_user_image_vs_batch(
        image_path=image_path,
        max_features=20,
        gain=0.5,
        displacement="medium",
        show_viz=True,
    )

    return results


def example_13_compare_multiple_images():
    """
    Example 13: Compare visual servoing performance on multiple user images.
    """
    from sift_utility import create_user_image_simulator

    print("\n" + "=" * 70)
    print("EXAMPLE 13: Compare Multiple Images")
    print("=" * 70)

    print("\nHow many images do you want to compare? (2-4)")
    n_images_input = input("Number of images [default: 2]: ").strip()
    try:
        n_images = int(n_images_input)
        n_images = np.clip(n_images, 2, 4)
    except:
        n_images = 2
    image_paths = []
    for i in range(n_images):
        print(f"\nImage {i + 1}:")
        path = input("  Path: ").strip().strip('"').strip("'")
        if path and os.path.exists(path):
            image_paths.append(path)
        else:
            print(f"  Skipping invalid path")

    if len(image_paths) < 2:
        print("\n Need at least 2 valid images for comparison")
        return None

    print(f"\n Running simulations on {len(image_paths)} images...")

    results_all = {}

    fig, axes = plt.subplots(len(image_paths), 2, figsize=(14, 6 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)

    for idx, image_path in enumerate(image_paths):
        image_name = Path(image_path).stem
        print(f"\n{'=' * 70}")
        print(f"Processing: {image_name}")
        print(f"{'=' * 70}")

        # Load image
        from sift_utility import load_user_image

        image = load_user_image(image_path)
        if image is None:
            continue
        try:
            sim = create_user_image_simulator(
                image_array=image, max_features=20, gain=0.5, displacement="medium"
            )

            print(f"Features detected: {len(sim.scene.points_3d)}")
            results = sim.run(verbose=False)
            results_all[image_name] = results

            print(f"Converged: {results['converged']}")
            print(f"Iterations: {results['iterations']}")
            print(f"Final error: {results['final_error']:.6f}")

            # Plot image with features
            ax_img = axes[idx, 0]
            ax_img.imshow(image)

            # Draw detected features
            coords = sim.sift_scene.reference_coords
            ax_img.scatter(
                coords[:, 0], coords[:, 1], c="red", marker="x", s=100, linewidths=2
            )
            ax_img.set_title(
                f"{image_name}\n({len(coords)} features)",
                fontsize=12,
                fontweight="bold",
            )
            ax_img.axis("off")

            # Plot error evolution
            ax_error = axes[idx, 1]
            iterations = np.arange(len(sim.controller.error_norm_history))
            ax_error.plot(
                iterations, sim.controller.error_norm_history, linewidth=2, color="blue"
            )
            ax_error.set_xlabel("Iteration", fontsize=11)
            ax_error.set_ylabel("Error Norm", fontsize=11)
            ax_error.set_title(f"Error Evolution - {image_name}", fontsize=12)
            ax_error.set_yscale("log")
            ax_error.grid(True, alpha=0.3)

            if results["converged"]:
                ax_error.axvline(
                    results["iterations"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label="Converged",
                )
                ax_error.legend()

        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            continue

    plt.tight_layout()
    plt.savefig("user_images_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(
        f"{'Image':<25} {'Features':<10} {'Converged':<12} {'Iterations':<12} {'Final Error':<15}"
    )
    print("-" * 70)

    for name, results in results_all.items():
        converged = "✓ Yes" if results["converged"] else "✗ No"
        iters = results["iterations"]
        error = results["final_error"]
        n_features = results.get("n_features", "N/A")
        print(f"{name:<25} {n_features:<10} {converged:<12} {iters:<12} {error:<15.6f}")

    print("=" * 70)

    return results_all


# Update the main menu
def main():
    """
    Main function - run all examples or select specific ones.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "VISUAL SERVOING SIMULATOR")
    print(" " * 10 + "Image-Based Visual Servoing (IBVS)")
    print("=" * 70)

    print("\nAvailable examples:")
    print("  1. Basic simulation (planar scene, small displacement)")
    print("  2. Large displacement scenario")
    print("  3. Comparison of different scene types")
    print("  4. Control gain comparison")
    print("  5. Live visualization demo")
    print("  6. Pure rotation task")
    print("  7. 3D sphere scene")
    print("  8. Interactive demo (customize parameters)")
    print("  9. Harris corners - Checkerboard")
    print(" 10. Harris corners - Custom patterns")
    print(" 11. Your own image - Interactive (SIFT)")
    print(" 12. Your own image - Batch mode (SIFT)")
    print(" 13. Compare multiple images (SIFT)")
    print(" 14. Run all basic examples (1-8)")
    print("  0. Exit")

    while True:
        choice = input("\nEnter choice (0-14): ").strip()

        if choice == "0":
            print("\nExiting. Goodbye!")
            break
        elif choice == "1":
            example_1_basic_simulation()
        elif choice == "2":
            example_2_large_displacement()
        elif choice == "3":
            example_3_different_scenes()
        elif choice == "4":
            example_4_gain_comparison()
        elif choice == "5":
            example_5_live_visualization()
        elif choice == "6":
            example_6_custom_scenario()
        elif choice == "7":
            example_7_sphere_scene()
        elif choice == "8":
            interactive_demo()
        elif choice == "9":
            example_8_harris_corners_checkerboard()
        elif choice == "10":
            example_9_harris_corners_custom_image()
        elif choice == "11":
            example_11_user_image_interactive()
        elif choice == "12":
            example_12_user_image_batch()
        elif choice == "13":
            example_13_compare_multiple_images()
        elif choice == "14":
            print("\nRunning all basic examples (1-8)...\n")
            example_1_basic_simulation()
            example_2_large_displacement()
            example_3_different_scenes()
            example_4_gain_comparison()
            example_5_live_visualization()
            example_6_custom_scenario()
            example_7_sphere_scene()
            interactive_demo()
            print("\nAll basic examples completed!")
        else:
            print("Invalid choice. Please enter 0-14.")

        cont = input("\nRun another example? (y/n): ").strip().lower()
        if cont != "y":
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run main menu
    main()
