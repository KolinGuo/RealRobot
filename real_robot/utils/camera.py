from typing import Tuple
import numpy as np
import cv2
from sapien.core import Pose

# Convert between camera frame conventions
#   OpenCV frame convention: right(x), down(y), forward(z)
#   OpenGL frame convention: right(x), up(y), backwards(z)
#   ROS/Sapien frame convention: forward(x), left(y) and up(z)
# For ROS frame conventions, see https://www.ros.org/reps/rep-0103.html#axis-orientation
T_CV_GL = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]], dtype=np.float32)
pose_CV_GL = Pose.from_transformation_matrix(T_CV_GL)
pose_GL_CV = pose_CV_GL.inv()
T_GL_CV = pose_GL_CV.to_transformation_matrix()
T_CV_ROS = np.array([[0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
pose_CV_ROS = Pose.from_transformation_matrix(T_CV_ROS)
pose_ROS_CV = pose_CV_ROS.inv()
T_ROS_CV = pose_ROS_CV.to_transformation_matrix()
T_GL_ROS = np.array([[0, -1, 0, 0],
                     [0, 0, 1, 0],
                     [-1, 0, 0, 0],
                     [0, 0, 0, 1]], dtype=np.float32)
pose_GL_ROS = Pose.from_transformation_matrix(T_GL_ROS)
pose_ROS_GL = pose_GL_ROS.inv()
T_ROS_GL = pose_ROS_GL.to_transformation_matrix()


def depth2xyz(depth_image, intrinsics, depth_scale=1000.0) -> np.ndarray:
    """Use camera intrinsics to convert depth_image to xyz_image
    :param depth_image: [H, W] or [H, W, 1] np.uint16 np.ndarray
    :param intrinsics: [3, 3] camera intrinsics matrix
    :return xyz_image: [H, W, 3] np.float64 np.ndarray
    """
    if intrinsics.size == 4:
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    if depth_image.ndim == 3:
        assert depth_image.shape[-1] == 1, \
            f"Wrong number of channels: {depth_image.shape}"
        depth_image = depth_image[..., 0]

    height, width = depth_image.shape[:2]
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))

    z = depth_image / depth_scale
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz_image = np.stack([x, y, z], axis=-1)
    return xyz_image


def transform_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Transform points by 4x4 transformation matrix H
    :return out: same shape as pts
    """
    assert H.shape == (4, 4), H.shape
    assert pts.shape[-1] == 3, pts.shape

    return pts @ H[:3, :3].T + H[:3, 3]


def resize_obs_image(rgb_image, depth_image, intr_params: Tuple, new_size,
                     interpolation=cv2.INTER_NEAREST_EXACT):
    """Resize rgb/depth images into shape=(width, height)
    :param rgb_image: [H, W, 3] np.uint8 np.ndarray
    :param depth_image: [H, W] np.uint16 np.ndarray
    :param intr_params: (fx, fy, cx, cy) tuple
    :param new_size: (width, height) tuple
    """
    new_width, new_height = new_size
    fx, fy, cx, cy = intr_params

    # Update intrinsics
    old_height, old_width = rgb_image.shape[:2]
    u_ratio = new_width / old_width
    v_ratio = new_height / old_height
    intr_params = fx * u_ratio, fy * v_ratio, cx * u_ratio, cy * v_ratio

    # Resize images
    rgb_image = cv2.resize(rgb_image, new_size, interpolation=interpolation)
    depth_image = cv2.resize(depth_image, new_size, interpolation=interpolation)
    return rgb_image, depth_image, intr_params
