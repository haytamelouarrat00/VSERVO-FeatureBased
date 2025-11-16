import numpy as np
import matplotlib.pyplot as plt
from features import (ImageFeatureScene, HarrisCornerDetector,
                            create_checkerboard_pattern, create_star_pattern)
from ibvs import create_image_based_simulator
from plot_manager import LiveVisualizer


def demo_harris_detection():
    """Demonstrate Harris corner detection on various patterns."""
    print("=" * 70)
    print("DEMO: Harris Corner Detection")
    print("=" * 70)

    patterns = {
        'Checkerboard': create_checkerboard_pattern(64, 8),
        'Star': create_star_pattern(512),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Harris Corner Detection on Different Patterns',
                 fontsize=14, fontweight='bold')

    for idx, (name, pattern) in enumerate(patterns.items()):
        print(f"\nDetecting corners in {name} pattern...")

        detector = HarrisCornerDetector(k=0.04, threshold=0.01)
        corners = detector.detect_corners(pattern, max_corners=20, min_distance=20)

        print(f"  Found {len(corners)} corners")

        ax = axes[idx]
        ax.imshow(pattern, cmap='gray')
        ax.scatter(corners[:, 0], corners[:, 1],
                   c='red', marker='x', s=100, linewidths=2)
        ax.set_title(f'{name} ({len(corners)} corners)')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('harris_detection_demo.png', dpi=150, bbox_inches='tight')
    plt.show()


def demo_full_pipeline_checkerboard():
    """Full visual servoing pipeline with checkerboard."""
    print("\n" + "=" * 70)
    print("DEMO: Full Visual Servoing Pipeline - Checkerboard")
    print("=" * 70)

    # Create pattern
    print("\n1. Creating checkerboard pattern...")
    checkerboard = create_checkerboard_pattern(square_size=64, n_squares=8)

    # Create simulator
    print("2. Detecting Harris corners and creating simulator...")
    sim = create_image_based_simulator(
        image_array=checkerboard,
        max_features=16,
        gain=0.5,
        displacement='medium'
    )

    print(f"   Detected {len(sim.scene.points_3d)} features")
    print(f"   Initial camera position: {sim.initial_camera.position}")
    print(f"   Desired camera position: {sim.desired_camera.position}")

    # Show detected corners
    print("\n3. Visualizing detected corners...")
    sim.image_scene.visualize(save_path='demo_checkerboard_corners.png')

    # Run simulation
    print("\n4. Running visual servoing with live visualization...")
    live_vis = LiveVisualizer(sim)
    results = live_vis.run_with_visualization(verbose=True)

    # Save results
    live_vis.save_all(prefix='demo_checkerboard')

    print("\n5. Results Summary:")
    sim.print_results(results)

    return results


def demo_full_pipeline_star():
    """Full visual servoing pipeline with star pattern."""
    print("\n" + "="*70)
    print("DEMO: Full Visual Servoing Pipeline - Star Pattern")
    print("="*70)

    # Create pattern
    print("\n1. Creating star pattern...")
    star = create_star_pattern(size=512)

    # Create simulator
    print("2. Detecting Harris corners and creating simulator...")
    sim = create_image_based_simulator(
        image_array=star,
        max_features=10,
        gain=0.6,
        displacement='medium'
    )

    print(f"   Detected {len(sim.scene.points_3d)} features")

    # Show detected corners
    print("\n3. Visualizing detected corners...")
    sim.image_scene.visualize(save_path='demo_star_corners.png')

    # Run simulation
    print("\n4. Running visual servoing with live visualization...")
    live_vis = LiveVisualizer(sim)
    results = live_vis.run_with_visualization(verbose=True)

    # Save results
    live_vis.save_all(prefix='demo_star')

    print("\n5. Results Summary:")
    sim.print_results(results)

    return results


def demo_comparison_patterns():
    """Compare visual servoing performance on different patterns."""
    print("\n" + "="*70)
    print("DEMO: Pattern Comparison")
    print("="*70)

    patterns = {
        'Checkerboard': create_checkerboard_pattern(64, 8),
        'Star': create_star_pattern(512)
    }

    results_all = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Visual Servoing with Different Patterns',
                 fontsize=16, fontweight='bold')

    for idx, (name, pattern) in enumerate(patterns.items()):
        print(f"\n{'='*70}")
        print(f"Testing with {name} pattern")
        print(f"{'='*70}")

        # Create simulator
        sim = create_image_based_simulator(
            image_array=pattern,
            max_features=12,
            gain=0.5,
            displacement='medium'
        )

        print(f"Detected {len(sim.scene.points_3d)} features")

        # Run simulation
        results = sim.run(verbose=False)
        results_all[name] = results

        print(f"Converged: {results['converged']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Final error: {results['final_error']:.6f}")

        # Plot pattern with corners
        ax_pattern = axes[idx, 0]
        ax_pattern.imshow(pattern, cmap='gray')
        ax_pattern.scatter(sim.image_scene.image_corners[:, 0],
                          sim.image_scene.image_corners[:, 1],
                          c='red', marker='x', s=100, linewidths=2)
        ax_pattern.set_title(f'{name} Pattern\n({len(sim.scene.points_3d)} corners)')
        ax_pattern.axis('off')

        # Plot error evolution
        ax_error = axes[idx, 1]
        iterations = np.arange(len(sim.controller.error_norm_history))
        ax_error.plot(iterations, sim.controller.error_norm_history,
                     linewidth=2, color='blue')
        ax_error.set_xlabel('Iteration', fontsize=11)
        ax_error.set_ylabel('Error Norm', fontsize=11)
        ax_error.set_title(f'{name} - Error Evolution')
        ax_error.set_yscale('log')
        ax_error.grid(True, alpha=0.3)

        if results['converged']:
            ax_error.axvline(results['iterations'], color='green',
                           linestyle='--', alpha=0.7, label='Converged')
            ax_error.legend()

    plt.tight_layout()
    plt.savefig('demo_pattern_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Pattern':<15} {'Features':<10} {'Converged':<12} {'Iterations':<12} {'Final Error':<15}")
    print("-"*70)

    for name, results in results_all.items():
        sim_name = name
        n_features = results_all[name]['iterations']  # This needs to be fixed, but for display
        converged = "Yes" if results['converged'] else "No"
        iters = results['iterations']
        error = results['final_error']
        print(f"{name:<15} {12:<10} {converged:<12} {iters:<12} {error:<15.6f}")

    return results_all


def demo_interactive():
    """Interactive demo with user choices."""
    print("\n" + "="*70)
    print("INTERACTIVE DEMO: Visual Servoing with Image Features")
    print("="*70)

    print("\nSelect pattern:")
    print("  1. Checkerboard (8x8)")
    print("  2. Star")
    print("  3. Custom test pattern")

    pattern_choice = input("\nEnter choice (1-3) [default: 1]: ").strip()

    if pattern_choice == '2':
        pattern = create_star_pattern(512)
        pattern_name = "Star"
    elif pattern_choice == '3':
        # Create custom pattern
        from features import ImageFeatureScene
        scene_temp = ImageFeatureScene()
        pattern = scene_temp.image
        pattern_name = "Custom"
    else:
        pattern = create_checkerboard_pattern(64, 8)
        pattern_name = "Checkerboard"

    print(f"\nSelected pattern: {pattern_name}")

    # Number of features
    n_features_input = input("\nMax features to detect (5-30) [default: 15]: ").strip()
    try:
        n_features = int(n_features_input)
        n_features = np.clip(n_features, 5, 30)
    except:
        n_features = 15

    # Control gain
    gain_input = input("\nControl gain (0.1-2.0) [default: 0.5]: ").strip()
    try:
        gain = float(gain_input)
        gain = np.clip(gain, 0.1, 2.0)
    except:
        gain = 0.5

    # Displacement
    print("\nInitial displacement:")
    print("  1. Small")
    print("  2. Medium")
    print("  3. Large")
    disp_choice = input("\nEnter choice (1-3) [default: 2]: ").strip()
    disp_map = {'1': 'small', '2': 'medium', '3': 'large'}
    displacement = disp_map.get(disp_choice, 'medium')

    # Live visualization
    viz_choice = input("\nShow live visualization? (y/n) [default: y]: ").strip().lower()
    show_live = viz_choice != 'n'

    print("\n" + "-"*70)
    print("CONFIGURATION:")
    print(f"  Pattern: {pattern_name}")
    print(f"  Max features: {n_features}")
    print(f"  Control gain: {gain}")
    print(f"  Displacement: {displacement}")
    print(f"  Live visualization: {show_live}")
    print("-"*70 + "\n")

    # Create simulator
    print("Creating simulator and detecting corners...")
    sim = create_image_based_simulator(
        image_array=pattern,
        max_features=n_features,
        gain=gain,
        displacement=displacement
    )

    print(f"Detected {len(sim.scene.points_3d)} features")

    # Show detected corners
    sim.image_scene.visualize(save_path='demo_interactive_corners.png')

    # Run simulation
    if show_live:
        print("\nRunning with live visualization...")
        live_vis = LiveVisualizer(sim)
        results = live_vis.run_with_visualization(verbose=True)
        live_vis.save_all(prefix='demo_interactive')
    else:
        print("\nRunning simulation...")
        results = sim.run(verbose=True)

        from plot_manager import PlotManager, Visualizer3D

        plot_manager = PlotManager()
        plot_manager.update_from_controller(sim.controller)

        vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
        vis_3d.update_visualization(sim.current_camera)

        plot_manager.save('demo_interactive_plots.png')
        vis_3d.save('demo_interactive_3d.png')

        plt.show()

    sim.print_results(results)

    return results


def main():
    """Main menu for image-based demos."""
    print("\n" + "="*70)
    print(" "*10 + "IMAGE-BASED VISUAL SERVOING DEMO")
    print(" "*15 + "Harris Corner Detection")
    print("="*70)

    print("\nAvailable demos:")
    print("  1. Harris corner detection demo")
    print("  2. Full pipeline - Checkerboard")
    print("  3. Full pipeline - Star pattern")
    print("  4. Pattern comparison")
    print("  5. Interactive demo")
    print("  6. Run all demos")
    print("  0. Exit")

    while True:
        choice = input("\nEnter choice (0-6): ").strip()

        if choice == '0':
            print("\nExiting. Goodbye!")
            break
        elif choice == '1':
            demo_harris_detection()
        elif choice == '2':
            demo_full_pipeline_checkerboard()
        elif choice == '3':
            demo_full_pipeline_star()
        elif choice == '4':
            demo_comparison_patterns()
        elif choice == '5':
            demo_interactive()
        elif choice == '6':
            print("\nRunning all demos...\n")
            demo_harris_detection()
            input("\nPress Enter to continue to next demo...")
            demo_full_pipeline_checkerboard()
            input("\nPress Enter to continue to next demo...")
            demo_full_pipeline_star()
            input("\nPress Enter to continue to next demo...")
            demo_comparison_patterns()
            print("\nAll demos completed!")
        else:
            print("Invalid choice. Please enter 0-6.")

        cont = input("\nRun another demo? (y/n): ").strip().lower()
        if cont != 'y':
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    np.random.seed(42)
    main()