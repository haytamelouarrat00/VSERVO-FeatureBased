import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from camera import Camera  # Replace 'your_module' with actual filename (e.g., camera)


class TestCamera(unittest.TestCase):

    def setUp(self):
        self.camera = Camera(
            focal_length=800.0,
            image_width=640,
            image_height=480,
            position=[1.0, 2.0, 3.0],
            orientation=np.eye(3)
        )

    def test_initialization_defaults(self):
        cam = Camera()
        np.testing.assert_array_equal(cam.position, [0, 0, 0])
        np.testing.assert_array_equal(cam.rotation, np.eye(3))
        self.assertEqual(cam.focal_length, 800.0)
        self.assertEqual(cam.image_width, 640)
        self.assertEqual(cam.image_height, 480)
        np.testing.assert_array_equal(cam.K, np.array([
            [800.0, 0, 320.0],
            [0, 800.0, 240.0],
            [0, 0, 1]
        ]))

    def test_set_pose(self):
        new_pos = [0.5, -1.0, 2.5]
        new_rot = R.from_euler('xyz', [0.1, 0.2, 0.3]).as_matrix()
        self.camera.set_pose(new_pos, new_rot)
        np.testing.assert_array_almost_equal(self.camera.position, new_pos)
        np.testing.assert_array_almost_equal(self.camera.rotation, new_rot)

    def test_set_pose_from_transform(self):
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', [0.2, -0.1, 0.4]).as_matrix()
        T[:3, 3] = [10, -5, 2]
        self.camera.set_pose_from_transform(T)
        np.testing.assert_array_almost_equal(self.camera.rotation, T[:3, :3])
        np.testing.assert_array_almost_equal(self.camera.position, T[:3, 3])

    def test_get_transform_matrix(self):
        T = self.camera.get_transform_matrix()
        np.testing.assert_array_equal(T[:3, :3], self.camera.rotation)
        np.testing.assert_array_equal(T[:3, 3], self.camera.position)
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    def test_get_inverse_transform(self):
        T_inv = self.camera.get_inverse_transform()
        T = self.camera.get_transform_matrix()
        np.testing.assert_array_almost_equal(T_inv @ T, np.eye(4), decimal=6)

    def test_world_to_camera_frame(self):
        world_pts = np.array([[1, 2, 3], [0, 0, 0]])
        cam_pts = self.camera.world_to_camera_frame(world_pts)
        expected = (self.camera.rotation.T @ (world_pts - self.camera.position).T).T
        np.testing.assert_array_almost_equal(cam_pts, expected)

    def test_camera_to_world_frame(self):
        cam_pts = np.array([[0, 0, 1], [1, 0, 0]])
        world_pts = self.camera.camera_to_world_frame(cam_pts)
        expected = (self.camera.rotation @ cam_pts.T).T + self.camera.position
        np.testing.assert_array_almost_equal(world_pts, expected)

    def test_project_to_image_camera_frame(self):
        cam_pts = np.array([[0.1, 0.2, 2.0], [-0.1, -0.2, 1.0]])
        img_pts, depths, valid = self.camera.project_to_image(cam_pts, frame='camera')
        u = self.camera.focal_length * (cam_pts[:, 0] / cam_pts[:, 2]) + self.camera.cx
        v = self.camera.focal_length * (cam_pts[:, 1] / cam_pts[:, 2]) + self.camera.cy
        expected_img = np.column_stack([u, v])
        np.testing.assert_array_almost_equal(img_pts[valid], expected_img[valid])
        np.testing.assert_array_equal(depths, cam_pts[:, 2])
        np.testing.assert_array_equal(valid, cam_pts[:, 2] > 0.01)

    def test_project_to_image_world_frame(self):
        world_pts = np.array([[1, 2, 5], [1.1, 2.2, 5.5]])
        cam_pts = self.camera.world_to_camera_frame(world_pts)
        img_pts1, _, _ = self.camera.project_to_image(world_pts, frame='world')
        img_pts2, _, _ = self.camera.project_to_image(cam_pts, frame='camera')
        np.testing.assert_array_almost_equal(img_pts1, img_pts2)

    def test_project_to_normalized_plane(self):
        cam_pts = np.array([[0.2, -0.3, 3.0], [0, 0, -1.0]])
        norm_pts, depths, valid = self.camera.project_to_normalized_plane(cam_pts, frame='camera')
        expected_norm = np.zeros_like(norm_pts)
        mask = cam_pts[:, 2] > 0.01
        expected_norm[mask, 0] = cam_pts[mask, 0] / cam_pts[mask, 2]
        expected_norm[mask, 1] = cam_pts[mask, 1] / cam_pts[mask, 2]
        np.testing.assert_array_almost_equal(norm_pts, expected_norm)
        np.testing.assert_array_equal(depths, cam_pts[:, 2])
        np.testing.assert_array_equal(valid, mask)

    def test_is_in_field_of_view(self):
        img_pts = np.array([
            [0, 0],
            [320, 240],
            [639, 479],
            [640, 480],
            [-1, 200]
        ])
        fov = self.camera.is_in_field_of_view(img_pts)
        expected = np.array([True, True, True, False, False])
        np.testing.assert_array_equal(fov, expected)

        # Test with margin
        fov_margin = self.camera.is_in_field_of_view(img_pts, margin=10)
        expected_margin = np.array([False, True, False, False, False])
        np.testing.assert_array_equal(fov_margin, expected_margin)

    def test_update_pose_with_velocity_translation_only(self):
        cam = Camera(position=[0, 0, 0], orientation=np.eye(3))
        velocity = [0.1, 0.2, 0.3, 0, 0, 0]
        dt = 2.0
        cam.update_pose_with_velocity(velocity, dt)
        expected_pos = np.array([0.2, 0.4, 0.6])
        np.testing.assert_array_almost_equal(cam.position, expected_pos)
        np.testing.assert_array_equal(cam.rotation, np.eye(3))

    def test_update_pose_with_velocity_rotation_only(self):
        cam = Camera(position=[0, 0, 0], orientation=np.eye(3))
        omega = [0, 0, np.pi/2]  # 90 deg around z
        velocity = [0, 0, 0] + omega
        dt = 1.0
        cam.update_pose_with_velocity(velocity, dt)
        expected_rot = R.from_euler('z', np.pi/2).as_matrix()
        np.testing.assert_array_almost_equal(cam.rotation, expected_rot, decimal=6)

    def test_skew_symmetric(self):
        v = [1, 2, 3]
        S = Camera.skew_symmetric(v)
        expected = np.array([
            [0, -3, 2],
            [3, 0, -1],
            [-2, 1, 0]
        ])
        np.testing.assert_array_equal(S, expected)

    def test_get_optical_axis(self):
        cam = Camera(orientation=R.from_euler('x', np.pi/2).as_matrix())
        axis = cam.get_optical_axis()
        expected = np.array([0, -1, 0])  # rotated Z-axis
        np.testing.assert_array_almost_equal(axis, expected)

    def test_get_up_vector(self):
        cam = Camera(orientation=R.from_euler('x', np.pi/2).as_matrix())
        up = cam.get_up_vector()
        expected = np.array([0, 0, 1])  # -Y in cam becomes +Z in world after Rx(90)
        np.testing.assert_array_almost_equal(up, expected)

    def test_look_at_simple(self):
        cam = Camera(position=[0, 0, 0])
        cam.look_at([0, 0, 1])  # Look +Z
        np.testing.assert_array_almost_equal(cam.rotation, np.eye(3))

    def test_look_at_complex(self):
        cam = Camera(position=[1, 1, 1])
        target = [2, 2, 2]
        cam.look_at(target)
        # Z should point toward target
        expected_z = np.array([1, 1, 1]) / np.sqrt(3)
        actual_z = cam.rotation[:, 2]
        np.testing.assert_array_almost_equal(actual_z, expected_z)

    def test_copy(self):
        cam2 = self.camera.copy()
        np.testing.assert_array_equal(cam2.position, self.camera.position)
        np.testing.assert_array_equal(cam2.rotation, self.camera.rotation)
        self.assertEqual(cam2.focal_length, self.camera.focal_length)
        self.assertIsNot(cam2.position, self.camera.position)  # deep copy

    def test_reset_history(self):
        self.camera.update_pose_with_velocity([0.1, 0, 0, 0, 0, 0], 0.1)
        self.camera.reset_history()
        self.assertEqual(len(self.camera.position_history), 1)
        self.assertEqual(len(self.camera.rotation_history), 1)
        np.testing.assert_array_equal(self.camera.position_history[0], self.camera.position)

    def test_get_frustum_corners(self):
        cam = Camera(position=[0, 0, 0], orientation=np.eye(3))
        depth = 2.0
        corners = cam.get_frustum_corners(depth)
        self.assertEqual(corners.shape, (5, 3))
        np.testing.assert_array_equal(corners[0], [0, 0, 0])  # camera center

        half_w = (640 / 2) * depth / 800  # = 0.8
        half_h = (480 / 2) * depth / 800  # = 0.6
        expected_corners = np.array([
            [0, 0, 0],
            [-0.8, -0.6, 2.0],
            [0.8, -0.6, 2.0],
            [0.8, 0.6, 2.0],
            [-0.8, 0.6, 2.0]
        ])
        np.testing.assert_array_almost_equal(corners, expected_corners)

    def test_pose_history_tracking(self):
        initial_pos = self.camera.position.copy()
        initial_rot = self.camera.rotation.copy()
        self.camera.update_pose_with_velocity([0.1, 0, 0, 0, 0, 0], 0.1)
        self.assertEqual(len(self.camera.position_history), 2)
        self.assertEqual(len(self.camera.rotation_history), 2)
        np.testing.assert_array_equal(self.camera.position_history[0], initial_pos)
        np.testing.assert_array_equal(self.camera.rotation_history[0], initial_rot)


if __name__ == '__main__':
    unittest.main()