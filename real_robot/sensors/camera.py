from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyrealsense2 as rs
from gymnasium import spaces
from sapien import Pose

from ..utils.camera import pose_CV_ROS, pose_ROS_CV
from ..utils.multiprocessing import SharedObject, ctx, start_and_wait_for_process
from ..utils.realsense import (
    RSDevice,
    get_connected_rs_devices,
    get_default_stream_config,
)

CALIB_CAMERA_POSE_DIR = Path(__file__).resolve().parents[1] / "assets/hec_camera_poses"
CALIB_CAMERA_POSES = {
    "front_camera": (
        Pose(
            np.load(
                CALIB_CAMERA_POSE_DIR / "Tb_b2c_20230918_CSE4144_front_jiacheng.npy"
            )
        )
        * pose_CV_ROS
    )
}


class CameraConfig:
    def __init__(
        self,
        uid: str,
        device_sn: str | None = None,
        pose: Pose = pose_CV_ROS,
        config: tuple[int, int, int] | dict[str, int | tuple[int, int, int]] | None = None,  # noqa: E501
        *,
        preset: str = "Default",
        align_to: Literal["Color", "Depth"] = "Color",
        color_option_kwargs={},
        depth_option_kwargs={},
        json_file: str | Path | None = None,
        parent_pose_so_name: str | None = None,
    ):  # fmt: skip
        """Camera configuration.

        :param uid: unique id of the camera
        :param device_sn: unique serial number of the camera.
                          If None, use the only RSDevice connected.
        :param pose: camera pose in world frame, following ROS frame conventions.
                     Format is forward(x), left(y) and up(z)
                     If parent_pose_so_name is not None, this is pose relative to
                     parent link.
        :param config: camera stream config, can be a tuple of (width, height, fps)
                       or a dict with format {stream_type: (param1, param2, ...)}.
                       If config is None, use default config realsense.get_default_config().
                       If config is a tuple, enables color & depth streams with config.
                       If config is a dict, enables streams given stream parameters.
                       Possible stream parameters format:
                           (width, height, fps) for video streams OR
                           fps for motion_streams
                       An example config dict:
                       {"Color": (848, 480, 30), "Depth": (848, 480, 30),
                        "Infrared 1": (848, 480, 30), "Acceleration": 250}
        :param preset: depth sensor presets
        :param align_to: align camera streams to color or depth frame.
        :param color_option_kwargs: color sensor optional keywords
        :param depth_option_kwargs: depth sensor optional keywords
        :param json_file: path to a json file containing sensor configs
        :param parent_pose_so_name: SharedObject name of camera's parent link pose
                                    in world frame. Defaults to None (camera has no
                                    mounting parent link).
        """  # noqa: E501
        self.uid = uid
        self.device_sn = device_sn
        self.pose = pose
        self.config = config

        self.preset = preset
        self.align_to = align_to
        self.color_option_kwargs = color_option_kwargs
        self.depth_option_kwargs = depth_option_kwargs
        self.json_file = json_file
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


def update_camera_cfgs_from_dict(
    camera_cfgs: dict[str, CameraConfig], cfg_dict: dict[str, Any | dict[str, Any]]
):
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

    _stream_name2obs_key = {
        "Color": "rgb",
        "Depth": "depth",
        "Infrared 1": "ir_l",
        "Infrared 2": "ir_r",
    }

    def __init__(
        self, camera_cfg: CameraConfig, *, record_bag_path: str | Path | None = None
    ):
        """
        :param record_bag_path: path to save bag recording if not None.
                                Must end with ".bag" if it's a file
        """
        self.camera_cfg = camera_cfg
        self.uid = camera_cfg.uid
        self.device_sn = camera_cfg.device_sn
        self.local_pose = camera_cfg.pose

        device = get_connected_rs_devices(
            self.device_sn
            if self.device_sn is not None
            else get_connected_rs_devices()[0]
        )
        self.product_type = device.get_info(rs.camera_info.name).split()[-1]

        self.config = camera_cfg.config
        if isinstance(self.config, tuple):
            self.config = {"Color": self.config, "Depth": self.config}
        elif self.config is None:
            self.config = get_default_stream_config(self.product_type)
        self.config = dict(sorted(self.config.items()))  # sort config by stream_name

        self.parent_pose_so_name = camera_cfg.parent_pose_so_name

        self.record_bag_path = record_bag_path

        self.device_proc = ctx.Process(
            target=RSDevice,
            name=f"RSDevice_{self.uid}",
            args=(self.device_sn, self.uid),
            kwargs={
                "config": self.config,
                "preset": camera_cfg.preset,
                "align_to": camera_cfg.align_to,
                "color_option_kwargs": camera_cfg.color_option_kwargs,
                "depth_option_kwargs": camera_cfg.depth_option_kwargs,
                "json_file": camera_cfg.json_file,
                "record_bag_path": record_bag_path,
                "run_as_process": True,
                "parent_pose_so_name": self.parent_pose_so_name,
                "local_pose": self.local_pose,
            },
        )
        start_and_wait_for_process(self.device_proc, timeout=30)

        # Create SharedObject to control RSDevice and fetch data
        self.so_joined = SharedObject(f"join_rs_{self.uid}")
        self.so_sync = SharedObject(f"sync_rs_{self.uid}")
        self.so_start = SharedObject(f"start_rs_{self.uid}")
        self.so_data_dict = {}  # {obs_key: SharedObject}
        for stream_name in self.config:
            self.so_data_dict[self._stream_name2obs_key[stream_name]] = SharedObject(
                f"rs_{self.uid}_{stream_name.lower().replace(' ', '_')}"
            )
        self.so_intr = None
        if "rgb" in self.so_data_dict or "depth" in self.so_data_dict:  # intrinsics
            self.so_intr = SharedObject(f"rs_{self.uid}_intr")
        self.so_pose = SharedObject(f"rs_{self.uid}_pose")

        self.so_start.assign(True)  # start the RSDevice
        self.intrinsic_matrix = None
        if self.so_intr is not None:
            # Wait for intrinsic matrix
            while not self.so_intr.modified:
                pass
            self.intrinsic_matrix = self.so_intr.fetch()

        self._camera_buffer = {}  # {obs_key: np.ndarray} for self.take_picture()
        self._camera_pose = self.so_pose.fetch()

    def take_picture(self):
        """Fetch images and camera pose from the camera"""
        assert self.device_proc.is_alive(), f"RSDevice for {self!r} has died"

        # Trigger camera capture in other processes
        self.so_sync.trigger()

        for obs_key, so_data in self.so_data_dict.items():
            if obs_key == "depth":
                self._camera_buffer["depth"] = (
                    so_data.fetch(lambda d: d[..., None].astype(np.float32)) / 1000.0
                    if self.product_type != "L515"
                    else 4000.0
                )
            else:
                self._camera_buffer[obs_key] = so_data.fetch()
        self._camera_pose = self.so_pose.fetch()

    def get_images(self, take_picture=False) -> dict[str, np.ndarray]:
        """Get (raw) images from the camera. Takes ~300 us for 848x480 @ 60fps
        :return rgb: color image, [H, W, 3] np.uint8 array
        :return depth: depth image, [H, W, 1] np.float32 array
        :return ir_l: left IR image, [H, W] np.uint8 array
        :return ir_r: right IR image, [H, W] np.uint8 array
        """
        if take_picture:
            self.take_picture()
        return self._camera_buffer

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
        """
        Returns a 4x4 camera model matrix in OpenCV format
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
        pose = self._camera_pose
        return dict(
            extrinsic_cv=self.get_extrinsic_matrix(pose),
            cam2world_cv=self.get_model_matrix(pose),
            intrinsic_cv=self.intrinsic_matrix,
        )

    @property
    def observation_space(self) -> spaces.Dict:
        obs_spaces = OrderedDict()
        for stream_name, params in self.config.items():
            if isinstance(params, tuple):
                W, H, _ = params
                if stream_name == "Color":
                    shape = (
                        self.config["Depth"][1::-1] + (3,)
                        if "Depth" in self.config
                        and self.camera_cfg.align_to == "Depth"
                        else (H, W, 3)
                    )
                elif stream_name == "Depth":
                    shape = (
                        self.config["Color"][1::-1] + (1,)
                        if "Color" in self.config
                        and self.camera_cfg.align_to == "Color"
                        else (H, W, 1)
                    )
                else:
                    shape = (H, W)
                dtype = np.float32 if stream_name == "Depth" else np.uint8
                obs_spaces[self._stream_name2obs_key[stream_name]] = spaces.Box(
                    low=0,
                    high=255 if dtype == np.uint8 else np.inf,
                    shape=shape,
                    dtype=dtype,
                )
            else:
                raise NotImplementedError(f"No support for {stream_name=} yet")
        return spaces.Dict(obs_spaces)

    def __del__(self):
        self.so_joined.trigger()
        self.device_proc.join()

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}: {self.uid} (S/N: {self.device_sn}) "
            f"config={self.config}>"
        )
