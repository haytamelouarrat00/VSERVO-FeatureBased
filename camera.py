import numpy as np
from scipy.spatial.transform import Rotation as R


class Camera:
    """
    Virtual camera class for visual servoing simulation.
    Handles camera pose, projections, and velocity-based updates.
    """

    def __init__(self,
                 focal_length=800.0,
                 image_width=640,
                 image_height=480,
                 position=None,
                 orientation=None):
        """
        Initialize camera with intrinsic and extrinsic parameters.

        Args:
            focal_length: Focal length in pixels (assuming fx = fy)
            image_width: Image width in pixels
            image_height: Image height in pixels
            position: 3D position [x, y, z] in world frame
            orientation: Rotation matrix (3x3) or None for identity
        """
        # Intrinsic parameters
        self.focal_length = focal_length
        self.image_width = image_width
        self.image_height = image_height

        # Principal point (image center)
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Camera intrinsic matrix K
        self.K = np.array([
            [focal_length, 0, self.cx],
            [0, focal_length, self.cy],
            [0, 0, 1]
        ])

        # Extrinsic parameters (camera pose in world frame)
        self.position = np.array([0, 0, 0], dtype=float) if position is None else np.array(position, dtype=float)
        self.rotation = np.eye(3) if orientation is None else np.array(orientation, dtype=float)

        # History for trajectory plotting
        self.position_history = [self.position.copy()]
        self.rotation_history = [self.rotation.copy()]

    def set_pose(self, position, rotation):
        """
        Set camera pose directly.

        Args:
            position: 3D position [x, y, z]
            rotation: 3x3 rotation matrix
        """
        self.position = np.array(position, dtype=float)
        self.rotation = np.array(rotation, dtype=float)

    def set_pose_from_transform(self, T):
        """
        Set camera pose from homogeneous transformation matrix.

        Args:
            T: 4x4 homogeneous transformation matrix
        """
        self.rotation = T[:3, :3]
        self.position = T[:3, 3]

    def get_transform_matrix(self):
        """
        Get homogeneous transformation matrix from world to camera frame.

        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T

    def get_inverse_transform(self):
        """
        Get transformation from camera to world frame.

        Returns:
            4x4 transformation matrix
        """
        T_inv = np.eye(4)
        T_inv[:3, :3] = self.rotation.T
        T_inv[:3, 3] = -self.rotation.T @ self.position
        return T_inv

    def world_to_camera_frame(self, points_world):
        """
        Transform 3D points from world frame to camera frame.

        Args:
            points_world: Nx3 array of 3D points in world coordinates

        Returns:
            Nx3 array of 3D points in camera coordinates
        """
        points_world = np.atleast_2d(points_world)

        # Transform: P_cam = R^T * (P_world - t)
        points_cam = (self.rotation.T @ (points_world - self.position).T).T

        return points_cam

    def camera_to_world_frame(self, points_cam):
        """
        Transform 3D points from camera frame to world frame.

        Args:
            points_cam: Nx3 array of 3D points in camera coordinates

        Returns:
            Nx3 array of 3D points in world coordinates
        """
        points_cam = np.atleast_2d(points_cam)

        # Transform: P_world = R * P_cam + t
        points_world = (self.rotation @ points_cam.T).T + self.position

        return points_world

    def project_to_image(self, points_3d, frame='world'):
        """
        Project 3D points to image plane.

        Args:
            points_3d: Nx3 array of 3D points
            frame: 'world' or 'camera' indicating the coordinate frame of input points

        Returns:
            tuple: (image_points, depths, valid_mask)
                - image_points: Nx2 array of 2D image coordinates (u, v)
                - depths: N array of depths (Z values in camera frame)
                - valid_mask: N boolean array indicating points in front of camera
        """
        points_3d = np.atleast_2d(points_3d)

        # Transform to camera frame if needed
        if frame == 'world':
            points_cam = self.world_to_camera_frame(points_3d)
        else:
            points_cam = points_3d.copy()

        # Check which points are in front of camera
        depths = points_cam[:, 2]
        valid_mask = depths > 0.01  # Small epsilon to avoid division issues

        # Perspective projection
        image_points = np.zeros((len(points_3d), 2))

        if np.any(valid_mask):
            X, Y, Z = points_cam[valid_mask].T

            # Normalized image coordinates
            x_norm = X / Z
            y_norm = Y / Z

            # Apply intrinsic matrix
            u = self.focal_length * x_norm + self.cx
            v = self.focal_length * y_norm + self.cy

            image_points[valid_mask, 0] = u
            image_points[valid_mask, 1] = v

        return image_points, depths, valid_mask

    def project_to_normalized_plane(self, points_3d, frame='world'):
        """
        Project 3D points to normalized image plane (metric coordinates).

        Args:
            points_3d: Nx3 array of 3D points
            frame: 'world' or 'camera' indicating coordinate frame

        Returns:
            tuple: (normalized_points, depths, valid_mask)
                - normalized_points: Nx2 array of normalized coordinates (x, y)
                - depths: N array of depths
                - valid_mask: N boolean array
        """
        points_3d = np.atleast_2d(points_3d)

        # Transform to camera frame if needed
        if frame == 'world':
            points_cam = self.world_to_camera_frame(points_3d)
        else:
            points_cam = points_3d.copy()

        depths = points_cam[:, 2]
        valid_mask = depths > 0.01

        normalized_points = np.zeros((len(points_3d), 2))

        if np.any(valid_mask):
            X, Y, Z = points_cam[valid_mask].T
            normalized_points[valid_mask, 0] = X / Z
            normalized_points[valid_mask, 1] = Y / Z

        return normalized_points, depths, valid_mask

    def is_in_field_of_view(self, image_points, margin=0):
        """
        Check if image points are within the camera's field of view.

        Args:
            image_points: Nx2 array of image coordinates
            margin: Margin in pixels from image borders

        Returns:
            N boolean array indicating points within FOV
        """
        image_points = np.atleast_2d(image_points)

        u, v = image_points[:, 0], image_points[:, 1]

        in_fov = (
                (u >= margin) & (u < self.image_width - margin) &
                (v >= margin) & (v < self.image_height - margin)
        )

        return in_fov

    def update_pose_with_velocity(self, velocity, dt):
        """
        Update camera pose using velocity command (exponential map).

        Args:
            velocity: 6-element array [v_x, v_y, v_z, omega_x, omega_y, omega_z]
                     Linear velocities (m/s) and angular velocities (rad/s) in camera frame
            dt: Time step in seconds
        """
        velocity = np.array(velocity)

        # Extract linear and angular velocities
        v_linear = velocity[:3]  # In camera frame
        omega = velocity[3:]  # Angular velocity in camera frame

        # Update translation (convert from camera frame to world frame)
        v_world = self.rotation @ v_linear
        self.position = self.position + v_world * dt

        # Update rotation using exponential map
        theta = np.linalg.norm(omega)

        if theta > 1e-8:
            # Rodrigues' formula
            omega_normalized = omega / theta
            omega_skew = self.skew_symmetric(omega_normalized)

            # Exponential map: exp(theta * [omega]_x)
            delta_R = (np.eye(3) +
                       np.sin(theta * dt) * omega_skew +
                       (1 - np.cos(theta * dt)) * (omega_skew @ omega_skew))

            # Update rotation: R_new = R_old * delta_R (rotation in camera frame)
            self.rotation = self.rotation @ delta_R

        # Store history
        self.position_history.append(self.position.copy())
        self.rotation_history.append(self.rotation.copy())

    @staticmethod
    def skew_symmetric(v):
        """
        Create skew-symmetric matrix from vector.

        Args:
            v: 3-element vector

        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def get_optical_axis(self):
        """
        Get the optical axis direction (Z-axis of camera frame in world coordinates).

        Returns:
            3-element unit vector
        """
        z_cam = np.array([0, 0, 1])
        z_world = self.rotation @ z_cam
        return z_world

    def get_up_vector(self):
        """
        Get the up direction (negative Y-axis of camera frame in world coordinates).

        Returns:
            3-element unit vector
        """
        y_cam = np.array([0, -1, 0])  # Image Y points down, so -Y is up
        y_world = self.rotation @ y_cam
        return y_world

    def get_camera_axes(self):
        """
        Get all camera axes in world frame.

        Returns:
            tuple: (x_axis, y_axis, z_axis) - each is a 3-element vector
        """
        axes_cam = np.eye(3)
        axes_world = self.rotation @ axes_cam
        return axes_world[:, 0], axes_world[:, 1], axes_world[:, 2]

    def look_at(self, target_point, up_world=None):
        """
        Orient camera to look at a target point.

        Args:
            target_point: 3D point to look at
            up_world: Desired up direction in world frame (default: [0, 0, 1])
        """
        if up_world is None:
            up_world = np.array([0.0, 0.0, 1.0])

        target_point = np.array(target_point, dtype=float)
        up_world = np.array(up_world, dtype=float)

        # Camera Z-axis points towards target
        z_cam = target_point - self.position
        norm_z = np.linalg.norm(z_cam)
        if norm_z < 1e-9:
            raise ValueError("Camera look_at target coincides with camera position.")
        z_cam = z_cam / norm_z

        # Ensure up vector is not parallel to viewing direction
        def ensure_valid_up(up_guess, forward):
            up_guess = up_guess / np.linalg.norm(up_guess)
            if np.linalg.norm(np.cross(up_guess, forward)) < 1e-6:
                return None
            return up_guess

        candidate_ups = [
            up_world,
            np.array([0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0])
        ]

        valid_up = None
        for cand in candidate_ups:
            result = ensure_valid_up(cand, z_cam)
            if result is not None:
                valid_up = result
                break

        if valid_up is None:
            raise ValueError("Could not find a valid up direction for look_at.")

        # Camera X-axis is perpendicular to Z and chosen up
        x_cam = np.cross(valid_up, z_cam)
        x_cam = x_cam / np.linalg.norm(x_cam)

        # Camera Y-axis completes the right-handed system
        y_cam = np.cross(z_cam, x_cam)

        # Build rotation matrix
        self.rotation = np.column_stack([x_cam, y_cam, z_cam])

    def copy(self):
        """
        Create a deep copy of the camera.

        Returns:
            New Camera instance with same parameters
        """
        cam = Camera(
            focal_length=self.focal_length,
            image_width=self.image_width,
            image_height=self.image_height,
            position=self.position.copy(),
            orientation=self.rotation.copy()
        )
        return cam

    def reset_history(self):
        """Reset the position and rotation history."""
        self.position_history = [self.position.copy()]
        self.rotation_history = [self.rotation.copy()]

    def get_frustum_corners(self, depth=1.0):
        """
        Get the corners of the camera frustum at a given depth.

        Args:
            depth: Distance from camera center

        Returns:
            5x3 array: camera position + 4 corners in world frame
        """
        # Compute frustum dimensions at given depth
        half_width = (self.image_width / 2.0) * depth / self.focal_length
        half_height = (self.image_height / 2.0) * depth / self.focal_length

        # Four corners in camera frame
        corners_cam = np.array([
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth]
        ])

        # Transform to world frame
        corners_world = self.camera_to_world_frame(corners_cam)

        # Include camera position
        frustum = np.vstack([self.position, corners_world])

        return frustum

    def __repr__(self):
        return (f"Camera(position={self.position}, "
                f"focal_length={self.focal_length}, "
                f"resolution={self.image_width}x{self.image_height})")


# Example usage and testing
if __name__ == "__main__":
    # Create a camera
    cam = Camera(focal_length=800, image_width=640, image_height=480)

    # Set initial pose
    cam.set_pose(position=[0, 0, -2], rotation=np.eye(3))

    # Create some 3D points
    points_3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Project to image
    img_points, depths, valid = cam.project_to_image(points_3d, frame='world')
    print("Image points:\n", img_points)
    print("Depths:", depths)
    print("Valid:", valid)

    # Test velocity update
    velocity = np.array([0.1, 0, 0, 0, 0, 0.1])  # Move right and rotate
    cam.update_pose_with_velocity(velocity, dt=0.1)
    print("\nNew position:", cam.position)
    print("New rotation:\n", cam.rotation)
