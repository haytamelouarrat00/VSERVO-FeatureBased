import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class PlotManager:
    """
    Manages real-time plotting of error and velocity evolution.
    """

    def __init__(self, figsize=(12, 8)):
        """
        Initialize plot manager.

        Args:
            figsize: Figure size tuple
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Visual Servoing - Error and Velocity Evolution',
                          fontsize=14, fontweight='bold')

        # Flatten axes for easier access
        self.ax_error_norm = self.axes[0, 0]
        self.ax_error_components = self.axes[0, 1]
        self.ax_velocity_linear = self.axes[1, 0]
        self.ax_velocity_angular = self.axes[1, 1]

        # Initialize empty line objects
        self.lines = {}
        self._setup_axes()

    def _setup_axes(self):
        """Setup axes labels and properties."""
        # Error norm plot
        self.ax_error_norm.set_xlabel('Iteration')
        self.ax_error_norm.set_ylabel('Error Norm')
        self.ax_error_norm.set_title('Feature Error Norm')
        self.ax_error_norm.grid(True, alpha=0.3)
        self.ax_error_norm.set_yscale('log')

        # Error components plot
        self.ax_error_components.set_xlabel('Iteration')
        self.ax_error_components.set_ylabel('Error Component')
        self.ax_error_components.set_title('Error Components (x, y)')
        self.ax_error_components.grid(True, alpha=0.3)

        # Linear velocity plot
        self.ax_velocity_linear.set_xlabel('Iteration')
        self.ax_velocity_linear.set_ylabel('Linear Velocity (m/s)')
        self.ax_velocity_linear.set_title('Linear Velocities')
        self.ax_velocity_linear.grid(True, alpha=0.3)

        # Angular velocity plot
        self.ax_velocity_angular.set_xlabel('Iteration')
        self.ax_velocity_angular.set_ylabel('Angular Velocity (rad/s)')
        self.ax_velocity_angular.set_title('Angular Velocities')
        self.ax_velocity_angular.grid(True, alpha=0.3)

        plt.tight_layout()

    def update_from_controller(self, controller):
        """
        Update plots from controller history.

        Args:
            controller: VSController instance with history
        """
        if len(controller.error_history) == 0:
            return

        iterations = np.arange(len(controller.error_history))

        # Plot error norm
        self.ax_error_norm.clear()
        self._setup_error_norm_axis()
        self.ax_error_norm.plot(iterations, controller.error_norm_history,
                                'b-', linewidth=2)

        # Plot all error components (pairs for each feature)
        self.ax_error_components.clear()
        self._setup_error_components_axis()

        error_array = np.array(controller.error_history)
        n_components = error_array.shape[1]

        cmap = plt.get_cmap('tab20')
        labels = [f"e{i + 1}" for i in range(n_components)]

        for i in range(n_components):
            color = cmap(i % cmap.N)
            self.ax_error_components.plot(iterations, error_array[:, i],
                                          color=color, label=labels[i],
                                          linewidth=1.2, alpha=0.8)

        self.ax_error_components.legend(loc='upper right', ncol=2, fontsize=8)

        # Plot velocities
        if len(controller.velocity_history) > 0:
            velocity_array = np.array(controller.velocity_history)

            # Linear velocities
            self.ax_velocity_linear.clear()
            self._setup_velocity_linear_axis()
            self.ax_velocity_linear.plot(iterations, velocity_array[:, 0],
                                         'r-', label='vₓ', linewidth=1.5)
            self.ax_velocity_linear.plot(iterations, velocity_array[:, 1],
                                         'g-', label='vᵧ', linewidth=1.5)
            self.ax_velocity_linear.plot(iterations, velocity_array[:, 2],
                                         'b-', label='vᵤ', linewidth=1.5)
            self.ax_velocity_linear.legend(loc='upper right')

            # Angular velocities
            self.ax_velocity_angular.clear()
            self._setup_velocity_angular_axis()
            self.ax_velocity_angular.plot(iterations, velocity_array[:, 3],
                                          'r-', label='ωₓ', linewidth=1.5)
            self.ax_velocity_angular.plot(iterations, velocity_array[:, 4],
                                          'g-', label='ωᵧ', linewidth=1.5)
            self.ax_velocity_angular.plot(iterations, velocity_array[:, 5],
                                          'b-', label='ωᵤ', linewidth=1.5)
            self.ax_velocity_angular.legend(loc='upper right')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _setup_error_norm_axis(self):
        """Setup error norm axis."""
        self.ax_error_norm.set_xlabel('Iteration')
        self.ax_error_norm.set_ylabel('Error Norm')
        self.ax_error_norm.set_title('Feature Error Norm')
        self.ax_error_norm.grid(True, alpha=0.3)
        self.ax_error_norm.set_yscale('log')

    def _setup_error_components_axis(self):
        """Setup error components axis."""
        self.ax_error_components.set_xlabel('Iteration')
        self.ax_error_components.set_ylabel('Error Component')
        self.ax_error_components.set_title('Error Components (all)')
        self.ax_error_components.grid(True, alpha=0.3)

    def _setup_velocity_linear_axis(self):
        """Setup linear velocity axis."""
        self.ax_velocity_linear.set_xlabel('Iteration')
        self.ax_velocity_linear.set_ylabel('Linear Velocity (m/s)')
        self.ax_velocity_linear.set_title('Linear Velocities')
        self.ax_velocity_linear.grid(True, alpha=0.3)

    def _setup_velocity_angular_axis(self):
        """Setup angular velocity axis."""
        self.ax_velocity_angular.set_xlabel('Iteration')
        self.ax_velocity_angular.set_ylabel('Angular Velocity (rad/s)')
        self.ax_velocity_angular.set_title('Angular Velocities')
        self.ax_velocity_angular.grid(True, alpha=0.3)

    def save(self, filename='vs_plots.png', dpi=150):
        """Save plots to file."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plots saved to {filename}")



class Visualizer3D:
    """
    3D visualization of camera motion and scene.
    """

    def __init__(self, scene, initial_camera, desired_camera, controller=None, figsize=(12, 10)):
        """
        Initialize 3D visualizer.

        Args:
            scene: VirtualScene object
            initial_camera: Initial camera pose
            desired_camera: Desired camera pose
            controller: Optional controller for displaying current error
            figsize: Figure size
        """
        self.scene = scene
        self.initial_camera = initial_camera
        self.desired_camera = desired_camera
        self.controller = controller

        # Create figure with 3D subplot and 2 image subplots
        self.fig = plt.figure(figsize=figsize)

        # 3D scene view (large, left side)
        self.ax_3d = self.fig.add_subplot(1, 2, 1, projection='3d')

        # Right side: current and desired image views
        self.ax_current = self.fig.add_subplot(2, 2, 2)
        self.ax_desired = self.fig.add_subplot(2, 2, 4)

        self._setup_3d_axis()
        self._setup_image_axes()

    def _setup_3d_axis(self):
        """Setup 3D axis properties."""
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Scene and Camera Motion', fontweight='bold')

        # Set equal aspect ratio
        points = self.scene.points_3d
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        self.ax_3d.set_xlim(mid_x - max_range * 1.5, mid_x + max_range * 1.5)
        self.ax_3d.set_ylim(mid_y - max_range * 1.5, mid_y + max_range * 1.5)
        self.ax_3d.set_zlim(mid_z - max_range * 1.5, mid_z + max_range * 1.5)

    def _setup_image_axes(self):
        """Setup image view axes."""
        self.ax_current.set_title('Current View', fontweight='bold')
        self.ax_current.set_xlim(0, 640)
        self.ax_current.set_ylim(480, 0)
        self.ax_current.set_xlabel('u (pixels)')
        self.ax_current.set_ylabel('v (pixels)')
        self.ax_current.grid(True, alpha=0.3)

        self.ax_desired.set_title('Desired View', fontweight='bold')
        self.ax_desired.set_xlim(0, 640)
        self.ax_desired.set_ylim(480, 0)
        self.ax_desired.set_xlabel('u (pixels)')
        self.ax_desired.set_ylabel('v (pixels)')
        self.ax_desired.grid(True, alpha=0.3)

    def draw_camera_frustum(self, camera, color='blue', alpha=0.3,
                            label='Camera', depth=0.5, ax=None):
        """
        Draw camera frustum in 3D.

        Args:
            camera: Camera object
            color: Color of frustum
            alpha: Transparency
            label: Label for legend
            depth: Depth of frustum
        """
        ax = self.ax_3d if ax is None else ax

        frustum = camera.get_frustum_corners(depth=depth)

        # Draw camera position
        ax.scatter(camera.position[0], camera.position[1],
                   camera.position[2], c=color, s=100,
                   marker='o', label=label)

        # Draw frustum edges
        cam_pos = frustum[0]
        corners = frustum[1:]

        # Lines from camera to corners
        for corner in corners:
            ax.plot([cam_pos[0], corner[0]],
                    [cam_pos[1], corner[1]],
                    [cam_pos[2], corner[2]],
                    color=color, alpha=alpha, linewidth=1)

        # Draw frustum rectangle
        corners_closed = np.vstack([corners, corners[0]])
        ax.plot(corners_closed[:, 0], corners_closed[:, 1],
                corners_closed[:, 2], color=color, alpha=alpha,
                linewidth=2)

        # Draw camera axes
        axis_length = 0.3
        x_axis, y_axis, z_axis = camera.get_camera_axes()

        origin = camera.position
        ax.quiver(origin[0], origin[1], origin[2],
                  x_axis[0], x_axis[1], x_axis[2],
                  color='red', length=axis_length,
                  arrow_length_ratio=0.3, alpha=0.8)
        ax.quiver(origin[0], origin[1], origin[2],
                  y_axis[0], y_axis[1], y_axis[2],
                  color='green', length=axis_length,
                  arrow_length_ratio=0.3, alpha=0.8)
        ax.quiver(origin[0], origin[1], origin[2],
                  z_axis[0], z_axis[1], z_axis[2],
                  color='blue', length=axis_length,
                  arrow_length_ratio=0.3, alpha=0.8)

    def draw_scene_points(self, color='black', size=50, ax=None):
        """Draw 3D scene points."""
        ax = self.ax_3d if ax is None else ax
        points = self.scene.points_3d
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, s=size, marker='o', alpha=0.6,
                   label='Features')

    def draw_trajectory(self, camera, color='orange', linewidth=2, ax=None):
        """
        Draw camera trajectory.

        Args:
            camera: Camera with position_history
            color: Color of trajectory
            linewidth: Line width
        """
        ax = self.ax_3d if ax is None else ax
        if len(camera.position_history) > 1:
            trajectory = np.array(camera.position_history)
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                    trajectory[:, 2], color=color,
                    linewidth=linewidth, label='Trajectory', alpha=0.7)

    def draw_image_features(self, camera, ax, color='red', size=100):
        """
        Draw features as seen by camera in image plane.

        Args:
            camera: Camera object
            ax: Matplotlib axis for image view
            color: Color of feature points
            size: Size of markers
        """
        img_points, depths, valid = self.scene.project_to_camera(camera)

        if np.any(valid):
            valid_points = img_points[valid]
            ax.scatter(valid_points[:, 0], valid_points[:, 1],
                       c=color, s=size, marker='+', linewidths=2)

            # Number the points
            for i, point in enumerate(valid_points):
                ax.text(point[0] + 10, point[1] + 10, str(i),
                        fontsize=9, color=color)

    def update_visualization(self, current_camera):
        """
        Update complete visualization.

        Args:
            current_camera: Current camera pose
        """
        # Clear axes
        self.ax_3d.clear()
        self.ax_current.clear()
        self.ax_desired.clear()

        # Re-setup
        self._setup_3d_axis()
        self._setup_image_axes()

        # Draw 3D scene
        self.draw_scene_points(color='black', size=80)
        self.draw_camera_frustum(self.initial_camera, color='gray',
                                 alpha=0.2, label='Initial', depth=0.4)
        self.draw_camera_frustum(self.desired_camera, color='green',
                                 alpha=0.3, label='Desired', depth=0.5)
        self.draw_camera_frustum(current_camera, color='blue',
                                 alpha=0.5, label='Current', depth=0.5)
        self.draw_trajectory(current_camera, color='orange', linewidth=2)

        self.ax_3d.legend(loc='upper right')

        # Draw image views
        self.draw_image_features(current_camera, self.ax_current,
                                 color='blue', size=120)
        self.draw_image_features(self.desired_camera, self.ax_desired,
                                 color='green', size=120)

        # Add convergence info to current view
        if hasattr(self, 'controller') and self.controller and getattr(self.controller, 'error_norm_history', None):
            err = self.controller.error_norm_history[-1]
        else:
            img_current, _, valid_current = self.scene.project_to_camera(current_camera)
            img_desired, _, valid_desired = self.scene.project_to_camera(self.desired_camera)
            valid_both = valid_current & valid_desired
            err = np.linalg.norm(img_current[valid_both] - img_desired[valid_both]) if np.any(valid_both) else 0.0

        self.ax_current.text(10, 30, f'Error: {err:.4f}',
                             fontsize=10, bbox=dict(boxstyle='round',
                                                    facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filename='vs_3d_view.png', dpi=150):
        """Save visualization to file."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"3D visualization saved to {filename}")



class LiveVisualizer:
    """
    Combined live visualization during simulation.
    """

    def __init__(self, simulator):
        """
        Initialize live visualizer.

        Args:
            simulator: VisualServoingSimulator instance
        """
        self.simulator = simulator

        # Create plot manager and 3D visualizer
        self.plot_manager = PlotManager(figsize=(14, 8))
        self.visualizer_3d = Visualizer3D(
            simulator.scene,
            simulator.initial_camera,
            simulator.desired_camera,
            controller=simulator.controller,
            figsize=(14, 10)
        )

        plt.ion()  # Enable interactive mode

    def callback(self, simulator, step_info):
        """
        Callback function for simulator to update visualization.

        Args:
            simulator: Simulator instance
            step_info: Step information dictionary
        """
        # Update every N iterations to avoid slowdown
        update_frequency = 10

        if simulator.iteration % update_frequency == 0:
            # Update plots
            self.plot_manager.update_from_controller(simulator.controller)

            # Update 3D visualization
            self.visualizer_3d.update_visualization(simulator.current_camera)

            plt.pause(0.01)

    def run_with_visualization(self, verbose=True):
        """
        Run simulation with live visualization.

        Args:
            verbose: Print progress

        Returns:
            Simulation results
        """
        # Run simulation with callback
        results = self.simulator.run(verbose=verbose, callback=self.callback)

        # Final update
        self.plot_manager.update_from_controller(self.simulator.controller)
        self.visualizer_3d.update_visualization(self.simulator.current_camera)

        plt.ioff()  # Disable interactive mode
        plt.show()

        return results

    def save_all(self, prefix='vs_simulation'):
        """
        Save plots and 3D visualization.

        Args:
            prefix: Output filename prefix
        """
        self.plot_manager.save(f'{prefix}_plots.png')
        self.visualizer_3d.save(f'{prefix}_3d.png')


# Testing
if __name__ == "__main__":
    from simulator import SimulatorFactory

    print("=== Testing Visualizers ===\n")

    # Create simulator
    print("Creating simulator...")
    sim = SimulatorFactory.create_standard_simulator(
        scene_type='planar',
        gain=0.5,
        control_law='classic'
    )

    # Option 1: Run without visualization, then plot
    print("\n1. Running simulation without live visualization...")
    results = sim.run(verbose=True)

    # Create plots after simulation
    plot_manager = PlotManager()
    plot_manager.update_from_controller(sim.controller)

    # Create 3D visualization
    vis_3d = Visualizer3D(sim.scene, sim.initial_camera, sim.desired_camera)
    vis_3d.update_visualization(sim.current_camera)

    # Save
    plot_manager.save('test_plots.png')
    vis_3d.save('test_3d.png')

    print("\nPress Enter to continue to live visualization test...")
    input()

    # Option 2: Run with live visualization
    print("\n2. Running simulation WITH live visualization...")
    sim.reset()

    live_vis = LiveVisualizer(sim)
    results = live_vis.run_with_visualization(verbose=True)

    # Save everything
    live_vis.save_all(prefix='live_test')

    print("\nVisualization complete!")
