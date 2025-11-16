import numpy as np
from interaction_matrix import InteractionMatrix


class VSController:
    """
    Visual Servoing Controller implementing IBVS control law.
    Computes camera velocity from feature errors.
    """

    def __init__(
        self,
        gain=0.5,
        control_law="classic",
        depth_estimation="desired",
        velocity_limits=None,
    ):
        """
        Initialize visual servoing controller.

        Args:
            gain: Control gain lambda (positive scalar)
            control_law: 'classic', 'adaptive', or 'second_order'
            depth_estimation: Strategy for depth estimation
            velocity_limits: Dictionary with 'linear' and 'angular' max velocities
        """
        self.gain = gain
        self.control_law = control_law
        self.depth_estimation = depth_estimation

        # Velocity limits for safety
        if velocity_limits is None:
            self.velocity_limits = {"linear": 0.5, "angular": 0.5}  # m/s  # rad/s
        else:
            self.velocity_limits = velocity_limits

        # Interaction matrix computer
        self.L_computer = InteractionMatrix()

        # History for plotting
        self.error_history = []
        self.velocity_history = []
        self.error_norm_history = []
        self.gain_history = []

    def compute_feature_error(self, current_features, desired_features):
        """
        Compute feature error vector.

        Args:
            current_features: Nx2 array of current feature coordinates
            desired_features: Nx2 array of desired feature coordinates

        Returns:
            2N array of stacked errors [e_x1, e_y1, e_x2, e_y2, ...]
        """
        current_features = np.atleast_2d(current_features)
        desired_features = np.atleast_2d(desired_features)

        # Error: e = s - s*
        error = (current_features - desired_features).flatten()

        return error

    def compute_velocity_classic(self, error, L):
        """
        Compute velocity using classic IBVS control law.

        v_c = -lambda * L+ * e

        Args:
            error: Feature error vector (2N)
            L: Interaction matrix (2N x 6)

        Returns:
            6-element velocity vector [v_x, v_y, v_z, omega_x, omega_y, omega_z]
        """
        # Compute pseudo-inverse
        L_pinv = self.L_computer.compute_pseudoinverse(L)

        # Control law
        velocity = -self.gain * L_pinv @ error

        return velocity

    def compute_velocity_adaptive(self, error, L, error_norm_prev=None):
        """
        Compute velocity using adaptive gain.

        Gain adapts based on error magnitude to ensure smooth convergence.

        Args:
            error: Feature error vector
            L: Interaction matrix
            error_norm_prev: Previous error norm for gain adaptation

        Returns:
            6-element velocity vector
        """
        error_norm = np.linalg.norm(error)

        # Adaptive gain: smaller gain when error is large, larger when small
        # This helps avoid overshooting
        if error_norm > 0.1:
            adaptive_gain = self.gain * 0.5  # Reduce gain for large errors
        elif error_norm < 0.01:
            adaptive_gain = self.gain * 1.5  # Increase gain for small errors
        else:
            adaptive_gain = self.gain

        # Store for history
        self.gain_history.append(adaptive_gain)

        # Compute velocity
        L_pinv = self.L_computer.compute_pseudoinverse(L)
        velocity = -adaptive_gain * L_pinv @ error

        return velocity

    def compute_velocity_second_order(self, error, L, dt=0.01):
        """
        Compute velocity using second-order control (with acceleration term).

        Args:
            error: Feature error vector
            L: Interaction matrix
            dt: Time step

        Returns:
            6-element velocity vector
        """
        # Classic term
        L_pinv = self.L_computer.compute_pseudoinverse(L)
        velocity_classic = -self.gain * L_pinv @ error

        # Add damping based on previous velocity if available
        if len(self.velocity_history) > 0:
            velocity_prev = self.velocity_history[-1]
            damping = 0.1  # Damping coefficient
            velocity = velocity_classic - damping * velocity_prev
        else:
            velocity = velocity_classic

        return velocity

    def limit_velocity(self, velocity):
        """
        Apply velocity limits for safety and stability.

        Args:
            velocity: 6-element velocity vector

        Returns:
            Limited velocity vector
        """
        velocity_limited = velocity.copy()

        # Limit linear velocities
        linear_vel = velocity[:3]
        linear_norm = np.linalg.norm(linear_vel)
        if linear_norm > self.velocity_limits["linear"]:
            velocity_limited[:3] = linear_vel * (
                self.velocity_limits["linear"] / linear_norm
            )

        # Limit angular velocities
        angular_vel = velocity[3:]
        angular_norm = np.linalg.norm(angular_vel)
        if angular_norm > self.velocity_limits["angular"]:
            velocity_limited[3:] = angular_vel * (
                self.velocity_limits["angular"] / angular_norm
            )

        return velocity_limited

    def compute_control(
        self, current_features, desired_features, current_depths, desired_depths
    ):
        """
        Main control function - compute velocity command from features.

        Args:
            current_features: Nx2 current normalized feature coordinates
            desired_features: Nx2 desired normalized feature coordinates
            current_depths: N array of current depths
            desired_depths: N array of desired depths

        Returns:
            6-element velocity command
        """
        # Compute feature error
        error = self.compute_feature_error(current_features, desired_features)

        # Compute interaction matrix with depth estimation
        L = self.L_computer.compute_with_depth_estimation(
            current_features,
            current_depths,
            desired_features,
            desired_depths,
            depth_estimation=self.depth_estimation,
        )

        # Compute velocity based on control law
        if self.control_law == "classic":
            velocity = self.compute_velocity_classic(error, L)
        elif self.control_law == "adaptive":
            error_norm_prev = (
                self.error_norm_history[-1] if self.error_norm_history else None
            )
            velocity = self.compute_velocity_adaptive(error, L, error_norm_prev)
        elif self.control_law == "second_order":
            velocity = self.compute_velocity_second_order(error, L)
        else:
            raise ValueError(f"Unknown control law: {self.control_law}")

        # Apply velocity limits
        velocity = self.limit_velocity(velocity)

        # Store history
        self.error_history.append(error.copy())
        self.velocity_history.append(velocity.copy())
        self.error_norm_history.append(np.linalg.norm(error))

        return velocity

    def has_converged(self, threshold=1e-3, window=5):
        """
        Check if the system has converged.

        Args:
            threshold: Error threshold for convergence
            window: Number of iterations to check for stability

        Returns:
            Boolean indicating convergence
        """
        if len(self.error_norm_history) < window:
            return False

        # Check if error has been below threshold for 'window' iterations
        recent_errors = self.error_norm_history[-window:]
        return all(e < threshold for e in recent_errors)

    def get_current_error_norm(self):
        """Get the most recent error norm."""
        if self.error_norm_history:
            return self.error_norm_history[-1]
        return float("inf")

    def reset_history(self):
        """Reset all history arrays."""
        self.error_history = []
        self.velocity_history = []
        self.error_norm_history = []
        self.gain_history = []

    def set_gain(self, gain):
        """Update control gain."""
        self.gain = gain

    def set_velocity_limits(self, linear=None, angular=None):
        """Update velocity limits."""
        if linear is not None:
            self.velocity_limits["linear"] = linear
        if angular is not None:
            self.velocity_limits["angular"] = angular


class AdvancedVSController(VSController):
    """
    Advanced visual servoing controller with additional features.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.singularity_detected = False
        self.redundancy_task = None

    def compute_velocity_with_singularity_handling(self, error, L, damping=0.01):
        """
        Compute velocity with damped least squares to handle singularities.

        Args:
            error: Feature error vector
            L: Interaction matrix
            damping: Damping factor

        Returns:
            6-element velocity vector
        """
        # Check condition number
        cond = self.L_computer.compute_condition_number(L)

        if cond > 100:
            self.singularity_detected = True
            # Use damped least squares
            L_pinv = self.L_computer.compute_pseudoinverse(
                L, method="damped", damping=damping
            )
        else:
            self.singularity_detected = False
            L_pinv = self.L_computer.compute_pseudoinverse(L)

        velocity = -self.gain * L_pinv @ error

        return velocity

    def compute_velocity_with_redundancy(self, error, L, secondary_task_velocity):
        """
        Compute velocity with redundancy resolution (secondary task).

        Uses null space of interaction matrix to achieve secondary objectives
        while still performing visual servoing.

        Args:
            error: Feature error vector
            L: Interaction matrix
            secondary_task_velocity: 6-element desired velocity for secondary task

        Returns:
            6-element velocity vector
        """
        # Primary task velocity
        L_pinv = self.L_computer.compute_pseudoinverse(L)
        v_primary = -self.gain * L_pinv @ error

        # Compute null space projector: P = I - L+ L
        I = np.eye(6)
        P = I - L_pinv @ L

        # Project secondary task into null space
        v_secondary = P @ secondary_task_velocity

        # Combine
        velocity = v_primary + v_secondary

        return velocity

    def compute_optimal_gain(self, error, L, max_velocity=0.5):
        """
        Compute optimal gain to ensure velocity doesn't exceed limits.

        Args:
            error: Feature error vector
            L: Interaction matrix
            max_velocity: Maximum allowed velocity magnitude

        Returns:
            Optimal gain value
        """
        L_pinv = self.L_computer.compute_pseudoinverse(L)

        # Compute velocity with unit gain
        v_unit = L_pinv @ error
        v_norm = np.linalg.norm(v_unit)

        if v_norm < 1e-6:
            return self.gain

        # Scale gain to respect velocity limit
        max_gain = max_velocity / v_norm
        optimal_gain = min(self.gain, max_gain)

        return optimal_gain
