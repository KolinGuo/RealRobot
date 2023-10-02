import numpy as np
import open3d as o3d

from ..camera import transform_points, transform_points_batch


class XArmGripper:
    """An object representing a UFactory XArm gripper."""

    def __init__(self):
        self.load_gripper_points()

        self.joint_limits = [0.0, 0.0453556139430441]

    def load_gripper_points(self):
        """Load gripper points"""
        # finger_pts are the 4 corners of the square finger contact pad
        self.finger_l_pts = np.array([[0.01475, -0.026003, 0.022253],
                                      [-0.01475, -0.026003, 0.022253],
                                      [-0.01475, -0.026003, 0.059753],
                                      [0.01475, -0.026003, 0.059753]])
        self.finger_r_pts = self.finger_l_pts * [1, -1, 1]  # invert y coords
        T_gripper_finger_l = np.eye(4)
        T_gripper_finger_l[:3, -1] = [0, 0.02682323, 0.11348719]
        self.finger_l_pts = transform_points(self.finger_l_pts, T_gripper_finger_l)
        self.finger_l_joint_axis = np.array([0, 0.96221329, -0.27229686])
        T_gripper_finger_r = np.eye(4)
        T_gripper_finger_r[:3, -1] = [0, -0.02682323, 0.11348719]
        self.finger_r_pts = transform_points(self.finger_r_pts, T_gripper_finger_r)
        self.finger_r_joint_axis = np.array([0, -0.96221329, -0.27229686])

        self.gripper_opening_to_q_val = 2 * np.cos(
            np.arctan2(self.finger_l_joint_axis[2], self.finger_l_joint_axis[1])
        )
        self.gripper_width = 0.086

    def get_control_points(self, q=0.0, pose=np.eye(4), symmetric=False) -> np.ndarray:
        """Return the 5 control points of gripper representation
        Control point order indices are shown below (symmetric=True is in parentheses)
                  * 0 (0)
                  |            y <---*
        1 (2) *-------* 2 (1)        |
              |       |              v
        3 (4) *       * 4 (3)         z
            left    right

        :param q: gripper finger joint value
        :param pose: gripper pose from world frame to xarm_gripper_base_link
        :return pts: [batch_size, 5, 3] np.floating np.ndarray
        """
        q = np.array([q]).reshape(-1)
        pose = np.array([pose]).reshape(-1, 4, 4)
        assert len(q) == len(pose), f"{q.shape = } {pose.shape = }"

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_l_joint_axis * q[:, None]
        finger_l_pts = transform_points_batch(
            self.finger_l_pts.reshape(2, 2, 3).mean(1), T
        )

        T = np.tile(np.eye(4), (len(q), 1, 1))
        T[..., :3, -1] = self.finger_r_joint_axis * q[:, None]
        finger_r_pts = transform_points_batch(
            self.finger_r_pts.reshape(2, 2, 3).mean(1), T
        )

        if symmetric:
            control_points = np.stack(
                [finger_r_pts, finger_l_pts], axis=-2
            ).reshape(-1, 4, 3)
        else:
            control_points = np.stack(
                [finger_l_pts, finger_r_pts], axis=-2
            ).reshape(-1, 4, 3)
        control_points = np.concatenate(
            [np.zeros((len(q), 1, 3)), control_points], axis=-2
        )
        control_points = transform_points_batch(control_points, pose)

        return control_points

    def get_control_points_lineset(self,
                                   control_points: np.ndarray) -> o3d.geometry.LineSet:
        """Contruct a LineSet for visualizing control_points
        :param control_points: [batch_size, 5, 3] np.floating np.ndarray
        """
        control_points = np.array(control_points).reshape(-1, 5, 3)
        batch_size = len(control_points)

        # Add mid point
        control_points = np.concatenate([
            control_points[:, 1:3].mean(1, keepdims=True), control_points
        ], axis=-2)

        lines = np.array([[0, 1], [2, 3], [2, 4], [3, 5]])
        lines = np.tile(lines, (batch_size, 1, 1))
        lines += np.arange(batch_size)[:, None, None] * 6

        lineset = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(control_points.reshape(-1, 3)),
            o3d.utility.Vector2iVector(lines.reshape(-1, 2))
        )
        return lineset
