import numpy as np
from camera import Camera
from scene import VirtualScene
from interaction_matrix import InteractionMatrix
from controller import VSController


class VisualServoingSimulator:
    """
    Main simulator class that orchestrates the visual servoing control loop.
    """

    def __init__(
        self,
        scene,
        initial_camera,
        desired_camera,
        controller_params=None,
        simulation_params=None,
    ):
        """
        Initialize visual servoing simulator.

        Args:
            scene: VirtualScene object with 3D points
            initial_camera: Camera object at initial pose
            desired_camera: Camera object at desired pose
            controller_params: Dictionary with controller parameters
            simulation_params: Dictionary with simulation parameters
        """
        self.scene = scene
        self.initial_camera = initial_camera.copy()
        self.desired_camera = desired_camera.copy()
        self.current_camera = initial_camera.copy()

        # Controller parameters
        if controller_params is None:
            controller_params = {
                "gain": 0.5,
                "control_law": "classic",
                "depth_estimation": "desired",
                "velocity_limits": {"linear": 0.5, "angular": 0.5},
            }
        self.controller_params = controller_params

        # Create controller
        self.controller = VSController(**controller_params)

        # Simulation parameters
        if simulation_params is None:
            simulation_params = {
                "dt": 0.01,  # Time step (seconds)
                "max_iterations": 1000,
                "convergence_threshold": 1e-3,
                "check_visibility": True,
                "stop_if_features_lost": True,
            }
        self.simulation_params = simulation_params

        # Get desired features
        self.desired_features, self.desired_depths, self.desired_valid = (
            self.scene.project_to_normalized(self.desired_camera)
        )

        # Simulation state
        self.iteration = 0
        self.converged = False
        self.features_lost = False
        self.time_elapsed = 0.0

        # History storage
        self.camera_poses_history = []
        self.feature_history = []
        self.error_history = []
        self.velocity_history = []
        self.time_history = []

    def reset(self):
        """Reset simulation to initial state."""
        self.current_camera = self.initial_camera.copy()
        self.current_camera.reset_history()

        self.controller.reset_history()

        self.iteration = 0
        self.converged = False
        self.features_lost = False
        self.time_elapsed = 0.0

        self.camera_poses_history = []
        self.feature_history = []
        self.error_history = []
        self.velocity_history = []
        self.time_history = []

    def get_current_features(self):
        """
        Get current feature positions and depths.

        Returns:
            tuple: (features, depths, valid_mask)
        """
        features, depths, valid = self.scene.project_to_normalized(self.current_camera)
        return features, depths, valid

    def check_feature_visibility(self, features, valid_mask):
        """
        Check if enough features are visible.

        Args:
            features: Nx2 feature coordinates
            valid_mask: N boolean array

        Returns:
            Boolean indicating if configuration is valid
        """
        n_visible = np.sum(valid_mask)

        # Need at least 3 points for 6-DOF control
        if n_visible < 3:
            return False

        # Check if features are well distributed (not all collinear)
        if n_visible >= 3:
            visible_features = features[valid_mask]
            # Compute covariance of feature positions
            if len(visible_features) >= 3:
                cov = np.cov(visible_features.T)
                # Check if covariance matrix is well-conditioned
                eigenvalues = np.linalg.eigvals(cov)
                if np.min(np.abs(eigenvalues)) < 1e-6:
                    return False

        return True

    def step(self):
        """
        Execute one iteration of the visual servoing control loop.

        Returns:
            Dictionary with step information
        """
        dt = self.simulation_params["dt"]

        # Get current features
        current_features, current_depths, current_valid = self.get_current_features()

        # Check visibility if required
        if self.simulation_params["check_visibility"]:
            if not self.check_feature_visibility(current_features, current_valid):
                self.features_lost = True
                return {
                    "success": False,
                    "reason": "features_lost",
                    "iteration": self.iteration,
                }

        # Use only valid features (both in current and desired views)
        valid_both = current_valid & self.desired_valid

        if np.sum(valid_both) < 3:
            self.features_lost = True
            return {
                "success": False,
                "reason": "insufficient_features",
                "iteration": self.iteration,
            }

        # Extract valid features
        current_features_valid = current_features[valid_both]
        current_depths_valid = current_depths[valid_both]
        desired_features_valid = self.desired_features[valid_both]
        desired_depths_valid = self.desired_depths[valid_both]

        # Compute control velocity
        velocity = self.controller.compute_control(
            current_features_valid,
            desired_features_valid,
            current_depths_valid,
            desired_depths_valid,
        )

        # Update camera pose
        self.current_camera.update_pose_with_velocity(velocity, dt)

        # Update state
        self.iteration += 1
        self.time_elapsed += dt

        # Store history
        self.camera_poses_history.append(
            {
                "position": self.current_camera.position.copy(),
                "rotation": self.current_camera.rotation.copy(),
            }
        )
        self.feature_history.append(current_features_valid.copy())
        self.time_history.append(self.time_elapsed)

        # Check convergence
        error_norm = self.controller.get_current_error_norm()
        threshold = self.simulation_params["convergence_threshold"]

        if self.controller.has_converged(threshold=threshold, window=5):
            self.converged = True
            return {
                "success": True,
                "reason": "converged",
                "iteration": self.iteration,
                "error_norm": error_norm,
            }

        return {
            "success": True,
            "reason": "running",
            "iteration": self.iteration,
            "error_norm": error_norm,
            "velocity_norm": np.linalg.norm(velocity),
        }

    def run(self, verbose=True, callback=None):
        """
        Run the complete visual servoing simulation.

        Args:
            verbose: Print progress information
            callback: Optional callback function called each iteration
                     callback(simulator, step_info)

        Returns:
            Dictionary with simulation results
        """
        self.reset()

        max_iter = self.simulation_params["max_iterations"]

        if verbose:
            print("=" * 60)
            print("Starting Visual Servoing Simulation")
            print("=" * 60)
            print(f"Initial position: {self.initial_camera.position}")
            print(f"Desired position: {self.desired_camera.position}")
            print(f"Number of features: {self.scene.n_points}")
            print(f"Control gain: {self.controller.gain}")
            print(f"Max iterations: {max_iter}")
            print(
                f"Convergence threshold: {self.simulation_params['convergence_threshold']}"
            )
            print("-" * 60)

        # Run simulation loop
        for i in range(max_iter):
            step_info = self.step()

            # Call callback if provided
            if callback is not None:
                callback(self, step_info)

            # Print progress
            if verbose and i % 50 == 0:
                error_norm = step_info.get("error_norm", 0)
                print(f"Iteration {i:4d}: Error = {error_norm:.6f}")

            # Check termination conditions
            if not step_info["success"]:
                if verbose:
                    print(f"\nSimulation stopped: {step_info['reason']}")
                    print(f"Final iteration: {step_info['iteration']}")

                return self.get_results(success=False, reason=step_info["reason"])

            if self.converged:
                if verbose:
                    print(f"\nConverged at iteration {i}")
                    print(f"Final error: {step_info['error_norm']:.8f}")
                    print(f"Time elapsed: {self.time_elapsed:.3f} seconds")

                return self.get_results(success=True, reason="converged")

        # Max iterations reached
        if verbose:
            print(f"\nMax iterations ({max_iter}) reached without convergence")
            error_norm = self.controller.get_current_error_norm()
            print(f"Final error: {error_norm:.6f}")

        return self.get_results(success=False, reason="max_iterations")

    def get_results(self, success=True, reason=""):
        """
        Compile simulation results.

        Returns:
            Dictionary with comprehensive results
        """
        results = {
            "success": success,
            "reason": reason,
            "converged": self.converged,
            "iterations": self.iteration,
            "time_elapsed": self.time_elapsed,
            "final_error": self.controller.get_current_error_norm(),
            # Camera poses
            "initial_position": self.initial_camera.position.copy(),
            "desired_position": self.desired_camera.position.copy(),
            "final_position": self.current_camera.position.copy(),
            "initial_rotation": self.initial_camera.rotation.copy(),
            "desired_rotation": self.desired_camera.rotation.copy(),
            "final_rotation": self.current_camera.rotation.copy(),
            # Errors
            "position_error": np.linalg.norm(
                self.current_camera.position - self.desired_camera.position
            ),
            # History
            "error_history": np.array(self.controller.error_norm_history),
            "velocity_history": np.array(self.controller.velocity_history),
            "time_history": np.array(self.time_history),
            "camera_trajectory": self.current_camera.position_history,
            # Controller info
            "controller_params": self.controller_params,
            "simulation_params": self.simulation_params,
        }

        # Compute rotation error (Frobenius norm of rotation difference)
        R_error = self.current_camera.rotation @ self.desired_camera.rotation.T
        rotation_error_angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        results["rotation_error_angle"] = rotation_error_angle

        return results

    def print_results(self, results):
        """Print formatted simulation results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"Success: {results['success']}")
        print(f"Reason: {results['reason']}")
        print(f"Iterations: {results['iterations']}")
        print(f"Time elapsed: {results['time_elapsed']:.3f} seconds")
        print(f"\nFinal Errors:")
        print(f"  Feature error: {results['final_error']:.8f}")
        print(f"  Position error: {results['position_error']:.6f} m")
        print(
            f"  Rotation error: {np.degrees(results['rotation_error_angle']):.4f} degrees"
        )
        print(f"\nPositions:")
        print(f"  Initial:  {results['initial_position']}")
        print(f"  Desired:  {results['desired_position']}")
        print(f"  Final:    {results['final_position']}")
        print("=" * 60)

    def get_feature_trajectories(self):
        """
        Get trajectories of features in image plane over time.

        Returns:
            List of Nx2 arrays (one per iteration)
        """
        return self.feature_history

    def get_velocity_components(self):
        """
        Get velocity components separated into linear and angular.

        Returns:
            tuple: (linear_velocities, angular_velocities) - each is Nx3 array
        """
        velocities = np.array(self.controller.velocity_history)

        if len(velocities) == 0:
            return np.array([]), np.array([])

        linear = velocities[:, :3]
        angular = velocities[:, 3:]

        return linear, angular


class SimulatorFactory:
    """
    Factory class for creating common simulator configurations.
    """

    @staticmethod
    def create_standard_simulator(scene_type="planar", gain=0.5, control_law="classic"):
        """
        Create a standard simulator setup.

        Args:
            scene_type: Type of scene
            gain: Controller gain
            control_law: Type of control law

        Returns:
            VisualServoingSimulator instance
        """
        from scene import SceneConfiguration

        scene, initial_cam, desired_cam = SceneConfiguration.create_standard_setup(
            scene_type
        )

        controller_params = {
            "gain": gain,
            "control_law": control_law,
            "depth_estimation": "desired",
            "velocity_limits": {"linear": 0.5, "angular": 0.5},
        }

        simulation_params = {
            "dt": 0.01,
            "max_iterations": 1000,
            "convergence_threshold": 1e-3,
            "check_visibility": True,
            "stop_if_features_lost": True,
        }

        return VisualServoingSimulator(
            scene, initial_cam, desired_cam, controller_params, simulation_params
        )

    @staticmethod
    def create_large_displacement_simulator(gain=0.3):
        """
        Create simulator with large initial displacement.
        """
        from scene import SceneConfiguration

        scene, initial_cam, desired_cam = (
            SceneConfiguration.create_large_displacement_setup()
        )

        controller_params = {
            "gain": gain,  # Lower gain for stability
            "control_law": "adaptive",
            "depth_estimation": "desired",
            "velocity_limits": {"linear": 0.3, "angular": 0.3},
        }

        simulation_params = {
            "dt": 0.01,
            "max_iterations": 2000,
            "convergence_threshold": 1e-3,
            "check_visibility": True,
            "stop_if_features_lost": True,
        }

        return VisualServoingSimulator(
            scene, initial_cam, desired_cam, controller_params, simulation_params
        )

    @staticmethod
    def create_fast_simulator(scene_type="planar"):
        """
        Create simulator optimized for fast convergence.
        """
        from scene import SceneConfiguration

        scene, initial_cam, desired_cam = SceneConfiguration.create_standard_setup(
            scene_type
        )

        controller_params = {
            "gain": 0.8,  # High gain
            "control_law": "classic",
            "depth_estimation": "desired",
            "velocity_limits": {"linear": 0.8, "angular": 0.8},
        }

        simulation_params = {
            "dt": 0.005,  # Smaller time step
            "max_iterations": 500,
            "convergence_threshold": 1e-3,
            "check_visibility": True,
            "stop_if_features_lost": True,
        }

        return VisualServoingSimulator(
            scene, initial_cam, desired_cam, controller_params, simulation_params
        )
