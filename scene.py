import numpy as np
from camera import Camera


class VirtualScene:
    """
    Manages 3D points and scene configuration for visual servoing simulation.
    """

    def __init__(self, points_3d=None, scene_type="planar", **scene_kwargs):
        """
        Initialize virtual scene.

        Args:
            points_3d: Nx3 array of 3D points in world frame
            scene_type: Type of scene ('planar', 'cube', 'random', 'custom')
            **scene_kwargs: Additional generation parameters (size, z_plane, etc.)
        """
        self.scene_type = scene_type
        self.scene_params = dict(scene_kwargs)

        if points_3d is not None:
            self.points_3d = np.array(points_3d, dtype=float)
        else:
            self.points_3d = self.generate_scene(scene_type, **scene_kwargs)

        self.n_points = len(self.points_3d)

    def generate_scene(self, scene_type="planar", **kwargs):
        """
        Generate 3D points for different scene types.

        Args:
            scene_type: Type of scene to generate
            **kwargs: Additional parameters for scene generation

        Returns:
            Nx3 array of 3D points
        """
        if scene_type == "planar":
            return self.generate_planar_points(**kwargs)
        elif scene_type == "cube":
            return self.generate_cube_points(**kwargs)
        elif scene_type == "sphere":
            return self.generate_sphere_points(**kwargs)
        elif scene_type == "random":
            return self.generate_random_points(**kwargs)
        elif scene_type == "grid":
            return self.generate_grid_points(**kwargs)
        else:
            raise ValueError(f"Unknown scene type: {scene_type}")

    def generate_planar_points(self, n_points=4, size=1.0, z_plane=0.0):
        """
        Generate points on a plane (default: XY plane).

        Args:
            n_points: Number of points (4 for corners, or more)
            size: Size of the planar region
            z_plane: Z coordinate of the plane

        Returns:
            Nx3 array of points
        """
        if n_points == 4:
            # Four corners of a square
            half_size = size / 2
            points = np.array(
                [
                    [half_size, half_size, z_plane],
                    [-half_size, half_size, z_plane],
                    [-half_size, -half_size, z_plane],
                    [half_size, -half_size, z_plane],
                ]
            )
        else:
            # Random points on plane
            xy = np.random.uniform(-size / 2, size / 2, (n_points, 2))
            z = np.full((n_points, 1), z_plane)
            points = np.hstack([xy, z])

        return points

    def generate_cube_points(self, size=1.0, center=None):
        """
        Generate 8 corner points of a cube.

        Args:
            size: Side length of cube
            center: Center position of cube (default: origin)

        Returns:
            8x3 array of corner points
        """
        if center is None:
            center = np.array([0, 0, 0])
        else:
            center = np.array(center)

        half_size = size / 2

        # 8 corners of a cube
        corners = (
            np.array(
                [
                    [1, 1, 1],
                    [-1, 1, 1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, -1],
                    [1, -1, -1],
                ]
            )
            * half_size
        )

        points = corners + center

        return points

    def generate_sphere_points(self, n_points=20, radius=1.0, center=None):
        """
        Generate points on a sphere surface.

        Args:
            n_points: Number of points
            radius: Sphere radius
            center: Center position (default: origin)

        Returns:
            Nx3 array of points on sphere
        """
        if center is None:
            center = np.array([0, 0, 0])
        else:
            center = np.array(center)

        # Fibonacci sphere algorithm for uniform distribution
        indices = np.arange(0, n_points, dtype=float) + 0.5

        phi = np.arccos(1 - 2 * indices / n_points)
        theta = np.pi * (1 + 5**0.5) * indices

        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)

        points = np.column_stack([x, y, z]) + center

        return points

    def generate_grid_points(self, grid_size=(3, 3), spacing=0.5, z_plane=0.0):
        """
        Generate points on a regular grid.

        Args:
            grid_size: (rows, cols) tuple
            spacing: Distance between points
            z_plane: Z coordinate of the grid

        Returns:
            Nx3 array of grid points
        """
        rows, cols = grid_size

        x = np.linspace(-(cols - 1) * spacing / 2, (cols - 1) * spacing / 2, cols)
        y = np.linspace(-(rows - 1) * spacing / 2, (rows - 1) * spacing / 2, rows)

        xv, yv = np.meshgrid(x, y)

        points = np.column_stack(
            [xv.flatten(), yv.flatten(), np.full(rows * cols, z_plane)]
        )

        return points

    def generate_random_points(self, n_points=10, bounds=None):
        """
        Generate random 3D points within bounds.

        Args:
            n_points: Number of points
            bounds: Dictionary with 'x', 'y', 'z' ranges, or None for default

        Returns:
            Nx3 array of random points
        """
        if bounds is None:
            bounds = {"x": [-1, 1], "y": [-1, 1], "z": [-0.5, 0.5]}

        x = np.random.uniform(bounds["x"][0], bounds["x"][1], n_points)
        y = np.random.uniform(bounds["y"][0], bounds["y"][1], n_points)
        z = np.random.uniform(bounds["z"][0], bounds["z"][1], n_points)

        points = np.column_stack([x, y, z])

        return points

    def transform_points(self, transformation):
        """
        Apply homogeneous transformation to all points.

        Args:
            transformation: 4x4 homogeneous transformation matrix
        """
        # Convert to homogeneous coordinates
        points_hom = np.hstack([self.points_3d, np.ones((self.n_points, 1))])

        # Apply transformation
        points_transformed = (transformation @ points_hom.T).T

        # Convert back to 3D
        self.points_3d = points_transformed[:, :3]

    def rotate_points(self, rotation_matrix, center=None):
        """
        Rotate points around a center.

        Args:
            rotation_matrix: 3x3 rotation matrix
            center: Center of rotation (default: origin)
        """
        if center is None:
            center = np.zeros(3)
        else:
            center = np.array(center)

        # Translate to origin, rotate, translate back
        self.points_3d = (rotation_matrix @ (self.points_3d - center).T).T + center

    def translate_points(self, translation):
        """
        Translate all points.

        Args:
            translation: 3-element translation vector
        """
        self.points_3d += np.array(translation)

    def scale_points(self, scale_factor, center=None):
        """
        Scale points around a center.

        Args:
            scale_factor: Scalar or 3-element array
            center: Center of scaling (default: origin)
        """
        if center is None:
            center = np.zeros(3)
        else:
            center = np.array(center)

        self.points_3d = (self.points_3d - center) * scale_factor + center

    def get_points(self):
        """Get all 3D points."""
        return self.points_3d.copy()

    def get_point(self, index):
        """Get a specific point."""
        return self.points_3d[index].copy()

    def get_centroid(self):
        """Compute centroid of all points."""
        return np.mean(self.points_3d, axis=0)

    def get_bounding_box(self):
        """
        Get axis-aligned bounding box of points.

        Returns:
            Dictionary with 'min' and 'max' coordinates
        """
        return {
            "min": np.min(self.points_3d, axis=0),
            "max": np.max(self.points_3d, axis=0),
        }

    def add_noise(self, noise_std=0.01):
        """
        Add Gaussian noise to point positions.

        Args:
            noise_std: Standard deviation of noise
        """
        noise = np.random.normal(0, noise_std, self.points_3d.shape)
        self.points_3d += noise

    def project_to_camera(self, camera, return_depths=True):
        """
        Project all scene points to a camera view.

        Args:
            camera: Camera object
            return_depths: Whether to return depth values

        Returns:
            If return_depths: (image_points, depths, valid_mask)
            Otherwise: image_points
        """
        image_points, depths, valid_mask = camera.project_to_image(
            self.points_3d, frame="world"
        )

        if return_depths:
            return image_points, depths, valid_mask
        else:
            return image_points

    def project_to_normalized(self, camera, return_depths=True):
        """
        Project all scene points to normalized image plane.

        Args:
            camera: Camera object
            return_depths: Whether to return depth values

        Returns:
            If return_depths: (normalized_points, depths, valid_mask)
            Otherwise: normalized_points
        """
        normalized_points, depths, valid_mask = camera.project_to_normalized_plane(
            self.points_3d, frame="world"
        )

        if return_depths:
            return normalized_points, depths, valid_mask
        else:
            return normalized_points

    def are_points_visible(self, camera, margin=20):
        """
        Check which points are visible in camera view.

        Args:
            camera: Camera object
            margin: Margin in pixels from image border

        Returns:
            Boolean array indicating visible points
        """
        image_points, depths, valid_depth = camera.project_to_image(
            self.points_3d, frame="world"
        )

        in_fov = camera.is_in_field_of_view(image_points, margin=margin)

        visible = valid_depth & in_fov

        return visible

    def filter_visible_points(self, camera, margin=20):
        """
        Get only the points visible in camera view.

        Args:
            camera: Camera object
            margin: Margin in pixels

        Returns:
            tuple: (visible_3d_points, visible_indices)
        """
        visible_mask = self.are_points_visible(camera, margin)
        visible_points = self.points_3d[visible_mask]
        visible_indices = np.where(visible_mask)[0]

        return visible_points, visible_indices


class SceneConfiguration:
    """
    Helper class for setting up common visual servoing scenarios.
    """

    @staticmethod
    def create_standard_setup(scene_type="planar"):
        """
        Create a standard visual servoing setup.

        Returns:
            tuple: (scene, initial_camera, desired_camera)
        """
        # Create scene
        if scene_type == "planar":
            scene = VirtualScene(scene_type="planar", size=0.6, z_plane=0.0)
        elif scene_type == "cube":
            scene = VirtualScene(scene_type="cube", size=0.8, center=[0, 0, 0])
        else:
            scene = VirtualScene(scene_type=scene_type)

        # Create cameras
        initial_camera = Camera(
            focal_length=800,
            image_width=640,
            image_height=480,
            position=[0.5, 0.3, -1.5],
            orientation=np.eye(3),
        )

        desired_camera = Camera(
            focal_length=800,
            image_width=640,
            image_height=480,
            position=[0, 0, -2.0],
            orientation=np.eye(3),
        )

        # Make cameras look at scene centroid
        centroid = scene.get_centroid()
        initial_camera.look_at(centroid)
        desired_camera.look_at(centroid)

        return scene, initial_camera, desired_camera

    @staticmethod
    def create_large_displacement_setup():
        """
        Create setup with large initial displacement.

        Returns:
            tuple: (scene, initial_camera, desired_camera)
        """
        scene = VirtualScene(scene_type="planar", size=0.8, z_plane=0.0)

        initial_camera = Camera(
            focal_length=800,
            image_width=640,
            image_height=480,
            position=[0.8, 0.5, -1.2],
            orientation=np.eye(3),
        )

        desired_camera = Camera(
            focal_length=800,
            image_width=640,
            image_height=480,
            position=[0, 0, -2.0],
            orientation=np.eye(3),
        )

        centroid = scene.get_centroid()
        initial_camera.look_at(centroid)
        desired_camera.look_at(centroid)

        return scene, initial_camera, desired_camera

    @staticmethod
    def create_rotation_setup():
        """
        Create setup with primarily rotational displacement.

        Returns:
            tuple: (scene, initial_camera, desired_camera)
        """
        scene = VirtualScene(scene_type="planar", size=0.8, z_plane=0.0)

        # Both cameras at same position, different orientations
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

        # Rotate initial camera
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_euler("xyz", [15, 20, 10], degrees=True).as_matrix()
        initial_camera.rotation = rotation @ initial_camera.rotation

        centroid = scene.get_centroid()
        desired_camera.look_at(centroid)

        return scene, initial_camera, desired_camera
