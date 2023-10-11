from pathlib import Path
from collections import OrderedDict
from typing import Dict, Union, Any, Optional

import numpy as np
from sapien.core import Pose
from gym import spaces

from ..utils.camera import pose_CV_ROS, pose_ROS_CV
from ..utils.realsense import RSDevice
from ..utils.multiprocessing import ctx, SharedObject, start_and_wait_for_process


CALIB_CAMERA_POSE_DIR = Path(__file__).resolve().parents[1] / "assets/hec_camera_poses"
CALIB_CAMERA_POSES = {
    "front_camera": Pose.from_transformation_matrix(np.load(
        CALIB_CAMERA_POSE_DIR / "Tb_b2c_20230918_CSE4144_front_jiacheng.npy"
    )) * pose_CV_ROS
}


class CameraConfig:
    def __init__(
        self,
        uid: str,
        device_sn: str,
        pose: Pose = pose_CV_ROS,
        width: int = 848,
        height: int = 480,
        fps: int = 30, *,
        preset="Default",
        depth_option_kwargs={},
        color_option_kwargs={},
        parent_pose_so_name: Optional[str] = None,
    ):
        """Camera configuration.

        :param uid: unique id of the camera
        :param device_sn: unique serial number of the camera
        :param pose: camera pose in world frame, following ROS frame conventions.
                     Format is forward(x), left(y) and up(z)
                     If parent_pose_so_name is not None, this is pose relative to
                     parent link.
        :param width: width of the camera
        :param height: height of the camera
        :param fps: camera streaming fps
        :param preset: depth sensor presets
        :param depth_option_kwargs: depth sensor optional keywords
        :param color_option_kwargs: color sensor optional keywords
        :param parent_pose_so_name: SharedObject name of camera's parent link pose
                                    in world frame. Defaults to None (camera has no
                                    mounting parent link).
        """
        self.uid = uid
        self.device_sn = device_sn
        self.pose = pose
        self.width = width
        self.height = height
        self.fps = fps

        self.preset = preset
        self.depth_option_kwargs = depth_option_kwargs
        self.color_option_kwargs = color_option_kwargs
        self.parent_pose_so_name = parent_pose_so_name

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + str(self.__dict__) + ")"


def parse_camera_cfgs(camera_cfgs):
    if isinstance(camera_cfgs, (tuple, list)):
        return OrderedDict([(cfg.uid, cfg) for cfg in camera_cfgs])
    elif isinstance(camera_cfgs, dict):
        return OrderedDict(camera_cfgs)
    elif isinstance(camera_cfgs, CameraConfig):
        return OrderedDict([(camera_cfgs.uid, camera_cfgs)])
    else:
        raise TypeError(type(camera_cfgs))


def update_camera_cfgs_from_dict(camera_cfgs: Dict[str, CameraConfig],
                                 cfg_dict: Dict[str, Union[Any, Dict[str, Any]]]):
    # First, apply global configuration
    for k, v in cfg_dict.items():
        if k in camera_cfgs:  # camera_name, camera-specific config
            continue
        for cfg in camera_cfgs.values():
            if not hasattr(cfg, k):
                raise AttributeError(f"{k} is not a valid attribute of CameraConfig")
            else:
                setattr(cfg, k, v)

    # Then, apply camera-specific configuration
    for name, v in cfg_dict.items():
        if name not in camera_cfgs:  # not camera_name, global config
            continue
        cfg = camera_cfgs[name]
        for k in v:
            assert hasattr(cfg, k), f"{k} is not a valid attribute of CameraConfig"
        cfg.__dict__.update(v)


class Camera:
    """Wrapper for RealSense camera (RSDevice)"""

    def __init__(self, camera_cfg: CameraConfig, *, record_bag_path=None):
        """
        :param record_bag_path: path to save bag recording if not None.
                                Must end with ".bag" if it's a file
        """
        self.camera_cfg = camera_cfg
        self.uid = camera_cfg.uid
        self.device_sn = camera_cfg.device_sn
        self.local_pose = camera_cfg.pose
        self.width = camera_cfg.width
        self.height = camera_cfg.height
        self.fps = camera_cfg.fps
        self.parent_pose_so_name = camera_cfg.parent_pose_so_name

        self.record_bag_path = record_bag_path

        config = (self.width, self.height, self.fps)
        self.device_proc = ctx.Process(
            target=RSDevice, name=f"RSDevice_{self.uid}",
            args=(self.device_sn, self.uid),
            kwargs=dict(
                color_config=config,
                depth_config=config,
                preset=camera_cfg.preset,
                color_option_kwargs=camera_cfg.color_option_kwargs,
                depth_option_kwargs=camera_cfg.depth_option_kwargs,
                record_bag_path=record_bag_path,
                run_as_process=True,
                parent_pose_so_name=self.parent_pose_so_name,
                local_pose=self.local_pose,
            )
        )
        start_and_wait_for_process(self.device_proc, timeout=30)

        # Create SharedObject to control RSDevice and fetch data
        self.so_joined = SharedObject(f"join_rs_{self.uid}")
        self.so_sync = SharedObject(f"sync_rs_{self.uid}")
        self.so_start = SharedObject(f"start_rs_{self.uid}")
        self.so_color = SharedObject(f"rs_{self.uid}_color")
        self.so_depth = SharedObject(f"rs_{self.uid}_depth")
        self.so_intr = SharedObject(f"rs_{self.uid}_intr")
        self.so_pose = SharedObject(f"rs_{self.uid}_pose")

        self.so_start.assign(True)  # start the RSDevice
        # Wait for intrinsic matrix
        while not self.so_intr.modified:
            pass
        self.intrinsic_matrix = self.so_intr.fetch()

        self.camera_buffer = None  # (pose, rgb, depth) for self.take_picture()

    def take_picture(self):
        """Fetch images and camera pose from the camera"""
        assert self.device_proc.is_alive(), f"RSDevice for {self!r} has died"

        # Trigger camera capture in other processes
        self.so_sync.trigger()

        self.camera_buffer = (
            self.so_pose.fetch(),
            self.so_color.fetch(),
            self.so_depth.fetch(lambda d: d[..., None].astype(np.float32)) / 1000.0
        )

    def get_images(self, take_picture=False) -> Dict[str, np.ndarray]:
        """Get (raw) images from the camera. Takes ~300 us for 848x480 @ 60fps
        :return rgb: color image, [H, W, 3] np.uint8 array
        :return depth: depth image, [H, W, 1] np.float32 array
        """
        if take_picture:
            self.take_picture()

        return {
            "rgb": self.camera_buffer[1],
            "depth": self.camera_buffer[2],
        }

    @property
    def pose(self) -> Pose:
        """Camera pose in world frame, following ROS frame conventions
        Format is forward(x), left(y) and up(z)
        """
        if self.parent_pose_so_name is not None:  # dynamic camera pose
            return self.so_pose.fetch()
        else:  # static camera pose
            return self.local_pose

    def get_extrinsic_matrix(self, pose: Pose = None) -> np.ndarray:
        """Returns a 4x4 extrinsic camera matrix in OpenCV format
        right(x), down(y), forward(z)
        """
        if pose is not None:
            return (pose_CV_ROS * pose.inv()).to_transformation_matrix()
        else:
            return (pose_CV_ROS * self.pose.inv()).to_transformation_matrix()

    def get_model_matrix(self, pose: Pose = None) -> np.ndarray:
        """Returns a 4x4 camera model matrix in OpenCV format
        right(x), down(y), forward(z)
        Note: this impl is different from sapien where the format is
              right(x), up(y), back(z)
        """
        if pose is not None:
            return (pose * pose_ROS_CV).to_transformation_matrix()
        else:
            return (self.pose * pose_ROS_CV).to_transformation_matrix()

    def get_params(self):
        """Get camera parameters."""
        pose = self.camera_buffer[0]
        return dict(
            extrinsic_cv=self.get_extrinsic_matrix(pose),
            cam2world_cv=self.get_model_matrix(pose),
            intrinsic_cv=self.intrinsic_matrix,
        )

    @property
    def observation_space(self) -> spaces.Dict:
        height, width = self.height, self.width
        obs_spaces = OrderedDict(
            rgb=spaces.Box(
                low=0, high=255, shape=(height, width, 3), dtype=np.uint8
            ),
            depth=spaces.Box(
                low=0, high=np.inf, shape=(height, width, 1), dtype=np.float32
            ),
        )
        return spaces.Dict(obs_spaces)

    def __del__(self):
        self.so_joined.trigger()
        self.device_proc.join()

    def __repr__(self):
        return (f"<{self.__class__.__name__}: {self.uid} (S/N: {self.device_sn}) "
                f"{self.width}x{self.height} @ {self.fps}fps>")
