"""
Visual Servoing Simulation - Main Script
=========================================

Lightweight menu with the core visual servoing demos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from camera import Camera
from scene import VirtualScene
from simulator import VisualServoingSimulator, SimulatorFactory
from plot_manager import PlotManager, Visualizer3D, LiveVisualizer


def example_1_live_visualization():
    """
    Example 1: Run simulation with live visualization.
    Real-time plotting of error, velocity, and 3D motion.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Live Visualization")
    print("=" * 70)
    print("\nThis example will show real-time visualization during simulation.")
    print("Close the plot windows when done to continue.\n")

    sim = SimulatorFactory.create_standard_simulator(
        scene_type="planar", gain=0.5, control_law="classic"
    )
    live_vis = LiveVisualizer(sim)

    results = live_vis.run_with_visualization(verbose=True)
    live_vis.save_all(prefix="example1_live")
    sim.print_results(results)
    return results


def example_2_pure_rotation():
    """
    Example 2: Custom scenario - pure rotation task.
    Camera position stays same, only orientation changes.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Pure Rotation Task")
    print("=" * 70)

    scene = VirtualScene(scene_type="planar", size=1.0, z_plane=0.0)
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

    rotation = R.from_euler("y", 25, degrees=True).as_matrix()
    initial_camera.rotation = rotation @ initial_camera.rotation

    centroid = scene.get_centroid()
    desired_camera.look_at(centroid)

    print("\nConfiguration:")
    print("  Task: Pure rotation (no translation)")
    print("  Rotation angle: 25 degrees around Y axis")
    print("  Scene: Planar")

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

    sim = VisualServoingSimulator(
        scene, initial_camera, desired_camera, controller_params, simulation_params
    )

    results = sim.run(verbose=True)

    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example2_plots.png")
    vis_3d.save("example2_3d.png")
    sim.print_results(results)
    plt.show()
    return results


def example_3_sphere_scene():
    """
    Example 3: Visual servoing with 3D sphere point cloud.
    Tests with non-planar features distributed in 3D.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: 3D Sphere Scene")
    print("=" * 70)

    scene = VirtualScene(scene_type="sphere", n_points=12, radius=0.5)

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

    centroid = scene.get_centroid()
    initial_camera.look_at(centroid)
    desired_camera.look_at(centroid)

    print("\nConfiguration:")
    print("  Scene: Sphere with 12 points")
    print("  Radius: 0.5 m")
    print("  Features distributed in 3D")

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

    results = sim.run(verbose=True)

    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    plot_manager.save("example3_plots.png")
    vis_3d.save("example3_3d.png")
    sim.print_results(results)
    plt.show()
    return results


def example_4_interactive_demo():
    """Interactive demonstration where the user can adjust parameters."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Interactive Demo")
    print("=" * 70)

    print("\nSelect scene type:")
    print("  1. Planar (4 corners)")
    print("  2. Cube (8 corners)")
    print("  3. Grid (4x4)")
    print("  4. Sphere (12 points)")

    scene_choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    scene_map = {"1": "planar", "2": "cube", "3": "grid", "4": "sphere"}
    scene_type = scene_map.get(scene_choice, "planar")

    gain_input = input("Enter gain (0.1-2.0) [default: 0.5]: ").strip()
    try:
        gain = float(gain_input)
        gain = np.clip(gain, 0.1, 2.0)
    except Exception:
        gain = 0.5

    print("\nSelect control law:")
    print("  1. Classic IBVS")
    print("  2. Adaptive gain")
    law_choice = input("\nEnter choice (1-2) [default: 1]: ").strip()
    control_law = "adaptive" if law_choice == "2" else "classic"

    viz_choice = input("Show live visualization? (y/n) [default: y]: ").strip().lower()
    show_live = viz_choice != "n"

    print("\n" + "-" * 70)
    print("CONFIGURATION:")
    print(f"  Scene type: {scene_type}")
    print(f"  Control gain: {gain}")
    print(f"  Control law: {control_law}")
    print(f"  Live visualization: {show_live}")
    print("-" * 70 + "\n")

    sim = SimulatorFactory.create_standard_simulator(
        scene_type=scene_type, gain=gain, control_law=control_law
    )

    if show_live:
        live_vis = LiveVisualizer(sim)
        results = live_vis.run_with_visualization(verbose=True)
        live_vis.save_all(prefix="example4_interactive")
    else:
        results = sim.run(verbose=True)
        plot_manager = PlotManager()
        plot_manager.update_from_controller(sim.controller)
        vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
        vis_3d.update_visualization(sim.current_camera)
        plot_manager.save("example4_plots.png")
        vis_3d.save("example4_3d.png")
        plt.show()

    sim.print_results(results)
    return results


def example_5_harris_checkerboard():
    """Example 5: Visual servoing with Harris corners on a checkerboard."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Harris Corners - Checkerboard Pattern")
    print("=" * 70)

    from features import create_checkerboard_pattern
    from ibvs import create_image_based_simulator

    checkerboard = create_checkerboard_pattern(square_size=64, n_squares=8)
    sim = create_image_based_simulator(
        image_array=checkerboard, max_features=16, gain=0.5, displacement="small"
    )

    print(f"Features detected: {len(sim.scene.points_3d)}")
    sim.image_scene.visualize(save_path="example5_corners.png")

    print("\nRunning visual servoing simulation...")
    live_vis = LiveVisualizer(sim)
    results = live_vis.run_with_visualization(verbose=True)
    live_vis.save_all(prefix="example5_checkerboard")
    sim.print_results(results)
    return results


def example_6_user_image_interactive():
    """Example 6: Visual servoing with a user-supplied image (interactive)."""
    from sift_utility import run_user_image_vs_interactive

    print("\n" + "=" * 70)
    print("EXAMPLE 6: Visual Servoing with Your Own Image")
    print("=" * 70)

    results = run_user_image_vs_interactive()
    return results


def main():
    """Main function - run examples from the trimmed menu."""
    print("\n" + "=" * 70)
    print(" " * 15 + "VISUAL SERVOING SIMULATOR")
    print(" " * 10 + "Image-Based Visual Servoing (IBVS)")
    print("=" * 70)

    print("\nAvailable examples:")
    print("  1. Live visualization demo")
    print("  2. Pure rotation task")
    print("  3. 3D sphere scene")
    print("  4. Interactive parameters demo")
    print("  5. Harris corners - Checkerboard")
    print("  6. Your own image - Interactive (SIFT)")
    print("  7. Run all non-image examples (1-5)")
    print("  0. Exit")

    valid_choices = {"0", "1", "2", "3", "4", "5", "6", "7"}

    while True:
        choice = input("\nEnter choice (0-7): ").strip()

        if choice == "0":
            print("\nExiting. Goodbye!")
            break
        elif choice == "1":
            example_1_live_visualization()
        elif choice == "2":
            example_2_pure_rotation()
        elif choice == "3":
            example_3_sphere_scene()
        elif choice == "4":
            example_4_interactive_demo()
        elif choice == "5":
            example_5_harris_checkerboard()
        elif choice == "6":
            example_6_user_image_interactive()
        elif choice == "7":
            print("\nRunning examples 1-5...\n")
            example_1_live_visualization()
            example_2_pure_rotation()
            example_3_sphere_scene()
            example_4_interactive_demo()
            example_5_harris_checkerboard()
            print("\nAll non-image examples completed!")
        else:
            print("Invalid choice. Please enter one of: " + ", ".join(sorted(valid_choices)))

        cont = input("\nRun another example? (y/n): ").strip().lower()
        if cont != "y":
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    np.random.seed(42)
    main()
