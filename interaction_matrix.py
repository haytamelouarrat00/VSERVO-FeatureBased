import numpy as np


class InteractionMatrix:
    """
    Computes the interaction matrix (image Jacobian) for visual servoing.
    Relates the time derivative of image features to camera velocity.
    """

    def __init__(self, feature_type="point"):
        """
        Initialize interaction matrix computer.

        Args:
            feature_type: Type of features ('point', 'normalized_point')
        """
        self.feature_type = feature_type

    def compute_for_point(self, x, y, Z):
        """
        Compute interaction matrix for a single 2D point feature in normalized coordinates.

        For a point (x, y) with depth Z, the interaction matrix is:
        L = [ -1/Z    0    x/Z   xy        -(1+x²)   y  ]
            [  0    -1/Z   y/Z   1+y²      -xy      -x  ]

        Args:
            x: Normalized x coordinate (X/Z)
            y: Normalized y coordinate (Y/Z)
            Z: Depth of the point in camera frame

        Returns:
            2x6 interaction matrix for this point
        """
        if Z <= 0:
            raise ValueError(f"Depth Z must be positive, got {Z}")

        L = np.array(
            [
                [-1 / Z, 0, x / Z, x * y, -(1 + x**2), y],
                [0, -1 / Z, y / Z, (1 + y**2), -x * y, -x],
            ]
        )

        return L

    def compute_for_point_pixel(self, u, v, Z, focal_length):
        """
        Compute interaction matrix for a point in pixel coordinates.

        Args:
            u: Pixel coordinate u
            v: Pixel coordinate v
            Z: Depth of the point
            focal_length: Camera focal length in pixels

        Returns:
            2x6 interaction matrix
        """
        # Convert to normalized coordinates
        x = u / focal_length
        y = v / focal_length

        # Compute interaction matrix in normalized coordinates
        L_norm = self.compute_for_point(x, y, Z)

        # Scale by focal length to get pixel velocities
        L_pixel = focal_length * L_norm
        # Note: The rotational part doesn't scale, only translational
        L_pixel[:, :3] = focal_length * L_norm[:, :3]
        L_pixel[:, 3:] = L_norm[:, 3:]

        return L_pixel

    def compute_for_points(self, normalized_points, depths):
        """
        Compute stacked interaction matrix for multiple points.

        Args:
            normalized_points: Nx2 array of normalized coordinates (x, y)
            depths: N array of depths (Z values)

        Returns:
            (2N)x6 interaction matrix stacked for all points
        """
        normalized_points = np.atleast_2d(normalized_points)
        depths = np.atleast_1d(depths)

        n_points = len(normalized_points)
        L = np.zeros((2 * n_points, 6))

        for i in range(n_points):
            x, y = normalized_points[i]
            Z = depths[i]

            if Z > 0:
                L[2 * i : 2 * i + 2, :] = self.compute_for_point(x, y, Z)
            else:
                # Handle invalid depth - use large depth as approximation
                L[2 * i : 2 * i + 2, :] = self.compute_for_point(x, y, 10.0)

        return L

    def compute_for_points_pixel(self, image_points, depths, focal_length):
        """
        Compute stacked interaction matrix for points in pixel coordinates.

        Args:
            image_points: Nx2 array of pixel coordinates (u, v)
            depths: N array of depths
            focal_length: Camera focal length

        Returns:
            (2N)x6 interaction matrix
        """
        # Convert to normalized coordinates
        normalized_points = image_points / focal_length

        return self.compute_for_points(normalized_points, depths)

    def compute_pseudoinverse(self, L, method="svd", damping=0.0):
        """
        Compute pseudo-inverse of interaction matrix.

        Args:
            L: Interaction matrix
            method: 'svd' or 'damped' for damped least squares
            damping: Damping factor for damped least squares (lambda²)

        Returns:
            Pseudo-inverse of L
        """
        if method == "svd":
            # Standard pseudo-inverse using SVD
            L_pinv = np.linalg.pinv(L)

        elif method == "damped":
            # Damped least squares: L+ = L^T (LL^T + λ²I)^-1
            n = L.shape[0]
            L_pinv = L.T @ np.linalg.inv(L @ L.T + damping * np.eye(n))

        else:
            raise ValueError(f"Unknown method: {method}")

        return L_pinv

    def estimate_depth_from_desired(
        self, current_points, desired_points, desired_depths, method="mean"
    ):
        """
        Estimate current depth from desired depths (common approximation in IBVS).

        Args:
            current_points: Nx2 array of current normalized points
            desired_points: Nx2 array of desired normalized points
            desired_depths: N array of desired depths
            method: 'mean', 'desired', or 'adaptive'

        Returns:
            N array of estimated depths
        """
        if method == "desired":
            # Use desired depth (standard approximation)
            return desired_depths.copy()

        elif method == "mean":
            # Use mean of desired depths
            mean_depth = np.mean(desired_depths)
            return np.full_like(desired_depths, mean_depth)

        elif method == "adaptive":
            # Scale desired depth based on feature displacement
            # Rough approximation: closer features move faster
            norms_current = np.linalg.norm(current_points, axis=1)
            norms_desired = np.linalg.norm(desired_points, axis=1)

            # Avoid division by zero
            scale = np.where(norms_current > 1e-6, norms_desired / norms_current, 1.0)

            return desired_depths * scale

        else:
            raise ValueError(f"Unknown method: {method}")

    def compute_with_depth_estimation(
        self,
        current_points,
        current_depths,
        desired_points,
        desired_depths,
        depth_estimation="desired",
    ):
        """
        Compute interaction matrix with depth estimation strategy.

        Args:
            current_points: Nx2 current normalized coordinates
            current_depths: Nx1 current depths (may be unknown/inaccurate)
            desired_points: Nx2 desired normalized coordinates
            desired_depths: Nx1 desired depths
            depth_estimation: Strategy for depth ('current', 'desired', 'mean', 'adaptive')

        Returns:
            (2N)x6 interaction matrix
        """
        if depth_estimation == "current":
            depths_to_use = current_depths
        else:
            depths_to_use = self.estimate_depth_from_desired(
                current_points, desired_points, desired_depths, method=depth_estimation
            )

        return self.compute_for_points(current_points, depths_to_use)

    def compute_condition_number(self, L):
        """
        Compute condition number of interaction matrix.
        Useful for detecting singularities.

        Args:
            L: Interaction matrix

        Returns:
            Condition number (ratio of largest to smallest singular value)
        """
        U, s, Vt = np.linalg.svd(L)

        if s[-1] < 1e-10:
            return np.inf

        return s[0] / s[-1]

    def is_well_conditioned(self, L, threshold=100):
        """
        Check if interaction matrix is well-conditioned.

        Args:
            L: Interaction matrix
            threshold: Maximum acceptable condition number

        Returns:
            Boolean indicating if matrix is well-conditioned
        """
        cond = self.compute_condition_number(L)
        return cond < threshold

    def compute_rank(self, L, tolerance=1e-10):
        """
        Compute numerical rank of interaction matrix.

        Args:
            L: Interaction matrix
            tolerance: Tolerance for singular value cutoff

        Returns:
            Numerical rank
        """
        s = np.linalg.svd(L, compute_uv=False)
        return np.sum(s > tolerance)


class InteractionMatrixAnalyzer:
    """
    Utility class for analyzing interaction matrix properties during servoing.
    """

    def __init__(self):
        self.condition_history = []
        self.rank_history = []
        self.smallest_sv_history = []

    def analyze(self, L):
        """
        Analyze interaction matrix and store metrics.

        Args:
            L: Interaction matrix

        Returns:
            Dictionary with analysis results
        """
        U, s, Vt = np.linalg.svd(L)

        condition = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
        rank = np.sum(s > 1e-10)
        smallest_sv = s[-1]

        self.condition_history.append(condition)
        self.rank_history.append(rank)
        self.smallest_sv_history.append(smallest_sv)

        analysis = {
            "condition_number": condition,
            "rank": rank,
            "singular_values": s,
            "smallest_sv": smallest_sv,
            "largest_sv": s[0],
            "is_full_rank": rank == min(L.shape),
            "is_well_conditioned": condition < 100,
        }

        return analysis

    def plot_history(self, ax=None):
        """
        Plot analysis history.

        Args:
            ax: Matplotlib axis (optional)
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        else:
            axes = ax

        iterations = range(len(self.condition_history))

        axes[0].plot(iterations, self.condition_history)
        axes[0].set_ylabel("Condition Number")
        axes[0].set_yscale("log")
        axes[0].grid(True)
        axes[0].set_title("Interaction Matrix Analysis")

        axes[1].plot(iterations, self.rank_history)
        axes[1].set_ylabel("Rank")
        axes[1].grid(True)

        axes[2].plot(iterations, self.smallest_sv_history)
        axes[2].set_ylabel("Smallest Singular Value")
        axes[2].set_xlabel("Iteration")
        axes[2].set_yscale("log")
        axes[2].grid(True)

        plt.tight_layout()

        return axes
