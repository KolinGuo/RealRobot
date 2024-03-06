"""Interface for pyrealsense2 API"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
from sapien import Pose

from .camera import pose_CV_ROS
from .logger import get_logger
from .multiprocessing import SharedObject, signal_process_ready

# https://www.intelrealsense.com/compare-depth-cameras/
SUPPORTED_RS_PRODUCTS = ("D415", "D435")

RS_DEVICES = None  # {device_sn: rs.device}


def check_rs_product_support(name: str) -> None:
    """Check if the RealSense device is supported. Print warning otherwise"""
    if name.split()[-1] not in SUPPORTED_RS_PRODUCTS:
        get_logger("realsense.py").warning(
            f'Only support {SUPPORTED_RS_PRODUCTS} currently, got "{name}"'
        )


def get_connected_rs_devices(
    device_sn: str | list[str] | None = None,
) -> list[str] | rs.device | list[rs.device]:
    """Returns list of connected RealSense devices.

    :param device_sn: list of serial numbers of devices to get.
        If not None, only return those devices in matching order.
        Else, return all connected devices' serial number
    :return devices: list of rs.device if device_sn is not None:
                     list of connected devices' serial number if device_sn is None
    """
    global RS_DEVICES

    if RS_DEVICES is None:
        RS_DEVICES = {}
        for d in rs.context().devices:
            name = d.get_info(rs.camera_info.name)
            if name.lower() != "platform camera":
                serial = d.get_info(rs.camera_info.serial_number)
                fw_version = d.get_info(rs.camera_info.firmware_version)
                usb_type = d.get_info(rs.camera_info.usb_type_descriptor)

                get_logger("realsense.py").info(
                    f"Found {name} (S/N: {serial} FW: {fw_version} on USB {usb_type})"
                )
                check_rs_product_support(name)
                RS_DEVICES[serial] = d
        get_logger("realsense.py").info(f"Found {len(RS_DEVICES)} devices")

    if device_sn is None:
        return list(RS_DEVICES.keys())
    elif isinstance(device_sn, str):
        return RS_DEVICES[device_sn]
    else:
        return [RS_DEVICES[sn] for sn in device_sn]


class RSDevice:
    """
    RealSense Device.
    Depth stream will be aligned to camera color frame and resolution
    if Color stream is enabled.

    For best depth accuracy with D435,
    set preset="High Accuracy" and use (848, 480) depth resolution

    References:
    * https://dev.intelrealsense.com/docs/d400-series-visual-presets
    * https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
    """

    # Default format for camera streams, {stream_type: stream_format}
    _default_stream_formats = {
        rs.stream.color: rs.format.rgb8,
        rs.stream.depth: rs.format.z16,
        rs.stream.infrared: rs.format.y8,
    }

    def __init__(
        self,
        device_sn: str | None = None,
        uid: str | None = None,
        config: tuple[int, int, int] | dict[str, int | tuple[int, int, int]] = (
            848,
            480,
            30,
        ),
        *,
        preset: str = "Default",
        color_option_kwargs={},
        depth_option_kwargs={},
        json_file: str | Path | None = None,
        record_bag_path: str | Path | None = None,
        run_as_process: bool = False,
        parent_pose_so_name: str | None = None,
        local_pose: Pose = pose_CV_ROS,
    ):
        """
        :param device_sn: realsense device serial number.
                          If None, use the only RSDevice connected.
        :param uid: unique camera id, e.g. "hand_camera", "front_camera"
        :param config: camera stream config, can be a tuple of (width, height, fps)
                       or a dict with format {stream_type: (param1, param2, ...)}.
                       If config is a tuple, enables color & depth streams with config.
                       If config is a dict, enables streams given stream parameters.
                       Possible stream parameters format:
                           (width, height, fps) for video streams OR
                           fps for motion_streams
                       An example config dict:
                       {"Color": (848, 480, 30), "Depth": (848, 480, 30),
                        "Infrared 1": (848, 480, 30), "Acceleration": 250}
        :param preset: depth sensor preset, available options:
                       ["Custom", "Default", "Hand", "High Accuracy",
                        "High Density", "Medium Density"].
        :param color_option_kwargs: color sensor options kwargs.
                                    Available options see self.supported_color_options
        :param depth_option_kwargs: depth sensor options kwargs.
                                    Available options see self.supported_depth_options
        :param json_file: path to a json file containing sensor configs
        :param record_bag_path: path to save bag recording if not None.
                                Must end with ".bag" if it's a file
        :param run_as_process: whether to run RSDevice as a separate process.
            If True, RSDevice needs to be created as a `mp.Process`.
            Several SharedObject are created to control RSDevice and stream data:
            * "join_rs_<device_uid>": If triggered, the RSDevice process is joined.
            * "sync_rs_<device_uid>": If triggered, all processes should fetch data.
                                      Used for synchronizing camera capture.
            * "start_rs_<device_uid>": If True, starts the RSDevice; else, stops it.
            * "rs_<device_uid>_color": rgb color image, [H, W, 3] np.uint8 np.ndarray
            * "rs_<device_uid>_depth": depth image, [H, W] np.uint16 np.ndarray
            * "rs_<device_uid>_infrared_1": Left IR image, [H, W] np.uint8 np.ndarray
            * "rs_<device_uid>_infrared_2": Right IR image, [H, W] np.uint8 np.ndarray
            * "rs_<device_uid>_intr": intrinsic matrix, [3, 3] np.float64 np.ndarray
            * "rs_<device_uid>_pose": camera pose in world frame (ROS convention)
                                      forward(x), left(y) and up(z), sapien.Pose
        :param parent_pose_so_name: If not None, it's the SharedObject name of camera's
                                    parent link pose in world frame.
        :param local_pose: camera pose in world frame (ROS convention)
                           If parent_pose_so_name is not None, this is pose
                           relative to parent link.
        """
        self.logger = get_logger("RSDevice")

        if device_sn is None:
            device_sns = get_connected_rs_devices()
            assert (
                len(device_sns) == 1
            ), f"Only 1 RSDevice should be connected, got S/Ns {device_sns}"
            device_sn = device_sns[0]
        self.device = get_connected_rs_devices(device_sn)
        self.name = self.device.get_info(rs.camera_info.name)
        self.serial_number = device_sn
        self.uid = device_sn if uid is None else uid.replace(" ", "_")
        self.usb_type = self.device.get_info(rs.camera_info.usb_type_descriptor)
        if self.usb_type != "3.2":
            self.logger.warning(
                f"Device {self!r} is connected with USB {self.usb_type}, not 3.2"
            )
        check_rs_product_support(self.name)
        self.color_sensor = self.device.first_color_sensor()
        self.depth_sensor = self.device.first_depth_sensor()
        self._read_device_info()
        # Record to a rosbag file
        self.record_bag_path = None
        if record_bag_path is not None:
            self.record_bag_path = Path(record_bag_path)
            if self.record_bag_path.suffix != ".bag":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.record_bag_path /= f"rs_{self.uid}_{timestamp}.bag"
            self.record_bag_path.parent.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.rs_config = self._create_rs_config(config)
        self.align = rs.align(rs.stream.color)

        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrices = {}  # {stream_name: intrinsics}
        self.last_frame_num = None

        self._configure_sensor_options(
            preset, color_option_kwargs, depth_option_kwargs, json_file
        )

        if run_as_process:
            self.parent_pose_so_name = parent_pose_so_name
            self.local_pose = local_pose
            self.run_as_process()

    def _read_device_info(self) -> None:
        """Reads and stores useful device information

        self.stream_name2type_idx: Mapping from stream name to stream type and index
            {"Color": (rs.stream.color, 0),
             "Infrared 1": (rs.stream.infrared, 1),  # Left IR
             "Infrared 2": (rs.stream.infrared, 2),  # Right IR
             "Depth": (rs.stream.depth, 0)}
        self.supported_configs: Mapping from stream name to supported streaming configs
            {"Color": [(848, 480, 60), ...],
             "Depth": [(848, 480, 60), ...],
             "Infrared 1": [(848, 480, 60), ...],
             "Infrared 2": [(848, 480, 60), ...]}
        self.all_intrinsics: Mapping from stream name to {stream_resolution: intrinsics}
            {"Color": {"848x480": rs.intrinsics},
             "Depth": {"848x480": rs.intrinsics},
             "Infrared 1": {"848x480": rs.intrinsics},
             "Infrared 2": {"848x480": rs.intrinsics}}
        self.all_extrinsics: Mapping from "sensor1=>sensor2" to 4x4 extrinsic np.ndarray
                "Color=>Depth" is the transformation from Color to Depth (T_depth_color)
            {"Color=>Depth": np.ndarray,
             "Depth=>Infrared 1": np.ndarray}
        """
        configs = defaultdict(list)
        all_intrinsics = defaultdict(dict)
        name2type_idx = {}
        sensor_profiles = {}  # {"Color": rs.stream_profile}

        for sensor in self.device.sensors:
            for profile in sensor.profiles:
                name = profile.stream_name()
                stream_type = profile.stream_type()
                stream_format = profile.format()
                stream_idx = profile.stream_index()

                if stream_type not in self._default_stream_formats:
                    raise NotImplementedError(f"Unknown {stream_type=}")

                # Not default format
                if stream_format != self._default_stream_formats[stream_type]:
                    continue

                if profile.is_video_stream_profile():
                    profile = profile.as_video_stream_profile()
                    width, height = profile.width(), profile.height()
                    configs[name].append((width, height, profile.fps()))
                    all_intrinsics[name][f"{width}x{height}"] = profile.intrinsics
                elif profile.is_motion_stream_profile():
                    profile = profile.as_motion_stream_profile()
                    configs[name].append(profile.fps())
                    all_intrinsics[name] = profile.get_motion_intrinsics()
                else:
                    raise TypeError(f"Unknown stream {profile=}")
                sensor_profiles[name] = profile
                name2type_idx[name] = (stream_type, stream_idx)

        all_extrinsics = {}
        for sensor1 in sensor_profiles:
            for sensor2 in sensor_profiles:
                all_extrinsics[f"{sensor1}=>{sensor2}"] = self.rs_extr2np(
                    sensor_profiles[sensor1].get_extrinsics_to(sensor_profiles[sensor2])
                )

        # {"Color": (rs.stream.color, 0), "Depth": (rs.stream.depth, 0)}
        self.stream_name2type_idx = name2type_idx
        # {"Color": [(848, 480, 60),]}
        self.supported_configs = dict(configs)
        # {"Color": {"848x480": rs.intrinsics}}
        self.all_intrinsics = dict(all_intrinsics)
        # {"Color=>Depth": np.ndarray}
        # "Color=>Depth" is the transformation from Color to Depth, i.e., T_depth_color
        self.all_extrinsics = all_extrinsics

    def _create_rs_config(
        self, config: tuple[int, int, int] | dict[str, int | tuple[int, int, int]]
    ) -> rs.config:
        """Create rs.config for rs.pipeline"""
        rs_config = rs.config()
        rs_config.enable_device(self.serial_number)

        if isinstance(config, tuple):
            assert (
                config in self.supported_configs["Color"]
            ), f"Not supported {config=} for Color stream"
            assert (
                config in self.supported_configs["Depth"]
            ), f"Not supported {config=} for Depth stream"
            width, height, fps = config
            rs_config.enable_stream(
                rs.stream.color,
                width,
                height,
                self._default_stream_formats[rs.stream.color],
                fps,
            )
            rs_config.enable_stream(
                rs.stream.depth,
                width,
                height,
                self._default_stream_formats[rs.stream.depth],
                fps,
            )
            self.config = {"Color": config, "Depth": config}
        else:
            for stream_name, params in config.items():
                assert stream_name in self.stream_name2type_idx, (
                    f"Unknown {stream_name=}. "
                    f"Available: {list(self.stream_name2type_idx.keys())}"
                )
                stream_type, stream_idx = self.stream_name2type_idx[stream_name]
                stream_format = self._default_stream_formats[stream_type]
                if isinstance(params, tuple):
                    width, height, fps = params
                    rs_config.enable_stream(
                        stream_type, stream_idx, width, height, stream_format, fps
                    )
                else:
                    fps = params
                    rs_config.enable_stream(stream_type, stream_idx, stream_format, fps)

        # warning about rs.align
        if "Depth" in self.config and "Color" not in self.config:
            self.logger.warning(
                "Color stream is not enabled. "
                "Depth stream will not be aligned to Color frame"
            )

        # Record camera streams as a rosbag file
        if self.record_bag_path is not None:
            self.logger.info(
                f'Enable recording {self!r} to file "{self.record_bag_path}"'
            )
            rs_config.enable_record_to_file(str(self.record_bag_path))
        return rs_config

    def _configure_sensor_options(
        self,
        preset="Default",
        color_option_kwargs={},
        depth_option_kwargs={},
        json_file=None,
    ) -> None:
        """Configure sensor options (depth preset, color/depth sensor options, json)"""
        # Load depth preset
        if preset not in (presets := self.supported_depth_presets):
            raise ValueError(f"Unknown {preset=}. Available presets: {presets}")
        self.depth_sensor.set_option(rs.option.visual_preset, presets.index(preset))
        self.logger.info(f'Loaded "{preset}" preset for {self!r}')

        # Set sensor options
        for key, value in color_option_kwargs.items():
            self.color_sensor.set_option(key, value)
            self.logger.info(f'Setting Color option "{key}" to {value}')
        for key, value in depth_option_kwargs.items():
            self.depth_sensor.set_option(key, value)
            self.logger.info(f'Setting Depth option "{key}" to {value}')

        # Load json config
        if json_file is not None:
            json_string = str(json.load(Path(json_file).open("r"))).replace("'", '"')
            advanced_mode = rs.rs400_advanced_mode(self.device)
            advanced_mode.load_json(json_string)
            self.logger.info(f'Loaded json config from "{json_file}"')

    def start(self) -> bool:
        """Start the streaming pipeline"""
        if self.is_running:
            self.logger.warning(
                f"Device {self!r} is already running. "
                "Please call stop() before calling start() again"
            )
            return False

        self.pipeline = rs.pipeline()

        try:
            self.pipeline_profile = self.pipeline.start(self.rs_config)
            for _ in range(20):  # wait for white balance to stabilize
                self.pipeline.wait_for_frames()
        except RuntimeError as e:
            e.args = (e.args[0] + f": {self.config=}",)
            raise e

        # Log enabled streams
        streams = self.pipeline_profile.get_streams()
        self.logger.info(f"Started device {self!r} with {len(streams)} streams")
        for i, stream in enumerate(streams):
            self.logger.info(f"Stream {i + 1}: {stream}")

        # Stores camera intrinsics
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        for frame in frames:
            profile = frame.profile
            name = profile.stream_name()
            if profile.is_video_stream_profile():
                profile = profile.as_video_stream_profile()
                self.intrinsic_matrices[name] = self.rs_intr2np(profile.intrinsics)
            elif profile.is_motion_stream_profile():
                # TODO: convert motion intrinsics to np.ndarray?
                profile = profile.as_motion_stream_profile()
                self.intrinsic_matrices[name] = profile.get_motion_intrinsics()
            else:
                raise TypeError(f"Unknown stream {profile=}")
        return True

    def wait_for_frames(self) -> dict[str, np.ndarray]:
        """
        Wait until a new set of frames becomes available.
        Each enabled stream in the pipeline is time-synchronized.

        :return ret_frames: dictionary {stream_name: np.ndarray}. Supported examples:
            * "Color": color image, [H, W, 3] np.uint8 array
            * "Depth": depth image, [H, W] np.uint16 array
            * "Infrared 1/2": infrared image, [H, W] np.uint8 array
        """
        if not self.is_running:
            self.logger.error(f"Device {self!r} is not started")
            return None

        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        self.last_frame_num = frames.get_frame_number()
        # self.logger.info(f"Received frame #{self.last_frame_num}")

        # Need to copy() so the device can release the frame from its internal memory
        ret_frames = {
            frame.profile.stream_name(): np.asarray(frame.data).copy()
            for frame in frames
        }
        return ret_frames

    def stop(self) -> bool:
        """Stop the streaming pipeline"""
        if not self.is_running:
            self.logger.warning(f"Device {self!r} is not running. Ignoring stop()")
            return False

        self.pipeline.stop()
        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrices = {}  # {stream_name: intrinsics}
        self.last_frame_num = None
        self.logger.info(f"Stopped device {self!r}")
        return True

    def run_as_process(self):
        """Run RSDevice as a separate process"""
        self.logger.info(f"Running {self!r} as a separate process")

        # RSDevice control
        device_started = False
        so_joined = SharedObject(f"join_rs_{self.uid}")
        so_sync = SharedObject(f"sync_rs_{self.uid}")
        so_start = SharedObject(f"start_rs_{self.uid}", data=False)
        # data
        so_data_dict = {}  # {stream_name: SharedObject}
        for stream_name, params in self.config.items():
            if isinstance(params, tuple):
                W, H, _ = params
                shape = (H, W, 3) if stream_name == "Color" else (H, W)
                if stream_name == "Depth" and "Color" in self.config:  # rs.align
                    shape = self.config["Color"][1::-1]  # (H, W)
                dtype = np.uint16 if stream_name == "Depth" else np.uint8
            else:
                raise NotImplementedError(f"No support for {stream_name=} yet")

            so_data_dict[stream_name] = SharedObject(
                f"rs_{self.uid}_{stream_name.lower().replace(' ', '_')}",
                data=np.zeros(shape, dtype=dtype),
            )
        if "Color" in so_data_dict or "Depth" in so_data_dict:  # intrinsics
            so_data_dict["Intrinsics"] = SharedObject(
                f"rs_{self.uid}_intr", data=np.zeros((3, 3))
            )
        so_pose = SharedObject(f"rs_{self.uid}_pose", data=self.local_pose)

        if self.parent_pose_so_name is not None:
            so_parent_pose = SharedObject(self.parent_pose_so_name)

        signal_process_ready()  # current process is ready

        while not so_joined.triggered:
            start = so_start.fetch()
            if not device_started and start:
                self.start()
                if "Intrinsics" in so_data_dict:
                    so_data_dict["Intrinsics"].assign(self.get_intrinsic_matrix())
                device_started = True
            elif device_started and not start:
                self.stop()
                device_started = False

            if device_started:
                # wait for frames
                frames = self.pipeline.wait_for_frames()
                # Fetch parent link pose and assign camera pose here for synchronization
                if self.parent_pose_so_name is not None:  # dynamic camera pose
                    so_pose.assign(so_parent_pose.fetch() * self.local_pose)
                frames = self.align.process(frames)  # align depth image to color frame

                # TODO: it's not guaranteed that the depth frame is assigned last,
                # violating the assumption in visualizers
                for frame in frames:
                    so_data_dict[frame.profile.stream_name()].assign(
                        np.asarray(frame.data)
                    )

        self.logger.info(f"Process running {self!r} is joined")
        # Unlink created SharedObject
        so_joined.unlink()
        so_sync.unlink()
        so_start.unlink()
        so_pose.unlink()
        for so_data in so_data_dict.values():
            so_data.unlink()

    def get_intrinsic_matrix(self) -> np.ndarray | dict[str, np.ndarray]:
        """Returns a 3x3 camera intrinsics matrix, available after self.start()"""
        if "Color" in self.intrinsic_matrices:
            # with rs.align, camera intrinsics is color sensor intrinsics
            return self.intrinsic_matrices["Color"]
        elif "Depth" in self.intrinsic_matrices:
            # If Color stream is not enabled,
            #   rs.align will not align depth stream to color frame
            return self.intrinsic_matrices["Depth"]
        else:
            return self.intrinsic_matrices

    @property
    def is_running(self) -> bool:
        """Returns whether the streaming pipeline is running"""
        return self.pipeline is not None

    @property
    def supported_depth_presets(self) -> list[str]:
        presets = []
        max_val = int(self.depth_sensor.get_option_range(rs.option.visual_preset).max)
        for i in range(max_val + 1):
            preset = self.depth_sensor.get_option_value_description(
                rs.option.visual_preset, i
            )
            presets.append(preset)
        return presets

    @property
    def supported_color_options(self) -> dict[int, rs.option_range]:
        options = {}
        for option in self.color_sensor.get_supported_options():
            try:
                options[option] = self.color_sensor.get_option_range(option)
            except RuntimeError:
                pass
        return options

    @property
    def supported_depth_options(self) -> dict[int, rs.option_range]:
        options = {}
        for option in self.depth_sensor.get_supported_options():
            try:
                options[option] = self.depth_sensor.get_option_range(option)
            except RuntimeError:
                pass
        return options

    @staticmethod
    def rs_intr2np(intrinsics: rs.intrinsics) -> np.ndarray:
        """Converts rs.intrinsics to 3x3 np.ndaray"""
        intr = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ])
        return intr

    @staticmethod
    def rs_extr2np(extrinsics: rs.extrinsics) -> np.ndarray:
        """Converts rs.extrinsics to 4x4 np.ndaray"""
        extr = np.eye(4)
        extr[:3, :3] = np.asarray(extrinsics.rotation).reshape(3, 3).T
        extr[:3, 3] = extrinsics.translation
        return extr

    def print_device_info(self) -> None:
        """
        Print device information (similar to running `rs-enumerate-devices -c`).
        This includes:
        supported_configs, all_intrinsics, all_extrinsics

        Only prints information with self._default_stream_formats
        """
        tab = "    "

        # ----- Device Info ----- #
        firmware_version = self.device.get_info(rs.camera_info.firmware_version)
        rec_fw_ver = self.device.get_info(rs.camera_info.recommended_firmware_version)
        physical_port = self.device.get_info(rs.camera_info.physical_port)
        debug_op_code = self.device.get_info(rs.camera_info.debug_op_code)
        advanced_mode = self.device.get_info(rs.camera_info.advanced_mode)
        product_id = self.device.get_info(rs.camera_info.product_id)
        camera_locked = self.device.get_info(rs.camera_info.camera_locked)
        product_line = self.device.get_info(rs.camera_info.product_line)
        asic_serial_number = self.device.get_info(rs.camera_info.asic_serial_number)
        firmware_update_id = self.device.get_info(rs.camera_info.firmware_update_id)
        device_info = (
            "Device info:\n"
            f"{tab}Name                          : {tab}{self.name}\n"
            f"{tab}Serial Number                 : {tab}{self.serial_number}\n"
            f"{tab}Firmware Version              : {tab}{firmware_version}\n"
            f"{tab}Recommended Firmware Version  : {tab}{rec_fw_ver}\n"
            f"{tab}Physical Port                 : {tab}{physical_port}\n"
            f"{tab}Debug Op Code                 : {tab}{debug_op_code}\n"
            f"{tab}Advanced Mode                 : {tab}{advanced_mode}\n"
            f"{tab}Product Id                    : {tab}{product_id}\n"
            f"{tab}Camera Locked                 : {tab}{camera_locked}\n"
            f"{tab}Usb Type Descriptor           : {tab}{self.usb_type}\n"
            f"{tab}Product Line                  : {tab}{product_line}\n"
            f"{tab}Asic Serial Number            : {tab}{asic_serial_number}\n"
            f"{tab}Firmware Update Id            : {tab}{firmware_update_id}\n"
        )

        # ----- Stream Profiles ----- #
        stream_profiles = ""
        for stream_name, params in self.supported_configs.items():
            n_space_after_stream = len(stream_name) + len(tab) - len("stream")
            # default format
            format = str(
                self._default_stream_formats[self.stream_name2type_idx[stream_name][0]]
            )[7:].upper()

            stream_profiles += (
                f"Stream Profiles supported by {stream_name=}\n"
                f"{tab}stream{' ' * n_space_after_stream}resolution"
                "      fps       format\n"
            )
            if isinstance(params[0], tuple):  # video streams, params: [(w, h, fps),]
                stream_profiles += "\n".join([
                    f"{tab}{stream_name}{tab} {width}x{height}"
                    f"{' ' * (len('resolution') - 2 - len(str(width)) - len(str(height)))}"
                    f"     @ {fps}Hz{' ' * (7 - len(str(fps)))}{format}"
                    for (width, height, fps) in params
                ])
            else:  # motion streams, params: [fps,]
                raise NotImplementedError(f"Not implemented for {stream_name=}")
            stream_profiles += "\n\n"

        # ----- Intrinsic Parameters ----- #
        intrinsics = "Intrinsic Parameters:\n"
        for stream_name, intr_dict in self.all_intrinsics.items():
            # default format
            format = str(
                self._default_stream_formats[self.stream_name2type_idx[stream_name][0]]
            )[7:].upper()

            for stream_cfg, intr in intr_dict.items():
                intrinsics += (
                    f' Intrinsic of "{stream_name}" / {stream_cfg} / {{{format}}}\n'
                )

                if isinstance(intr, rs.intrinsics):  # video streams, rs.intrinsics
                    dist_model_str = str(intr.model)[11:].replace("_", " ").title()
                    fovx = np.rad2deg(
                        np.arctan2(intr.ppx + 0.5, intr.fx)
                        + np.arctan2(intr.width - (intr.ppx + 0.5), intr.fx)
                    )
                    fovy = np.rad2deg(
                        np.arctan2(intr.ppy + 0.5, intr.fy)
                        + np.arctan2(intr.height - (intr.ppy + 0.5), intr.fy)
                    )
                    alpha = intr.fy / intr.fx
                    fovd = np.rad2deg(
                        2
                        * np.arctan2(
                            np.sqrt(intr.width**2 + (intr.height / alpha) ** 2) / 2,
                            intr.fx,
                        )
                    )

                    intr_mat_str = np.array2string(
                        self.rs_intr2np(intr),
                        precision=20,
                        suppress_small=True,
                        separator=", ",
                        prefix="    np.array(",
                    )
                    intrinsics += (
                        f"  Width:        {intr.width}\n"
                        f"  Height:       {intr.height}\n"
                        f"  PPX:          {intr.ppx}\n"
                        f"  PPY:          {intr.ppy}\n"
                        f"  Fx:           {intr.fx}\n"
                        f"  Fy:           {intr.fy}\n"
                        f"  Distortion:   {dist_model_str}\n"
                        f"  Coeffs:       {intr.coeffs}\n"
                        f"  FOV (deg):    {fovx:.4f} x {fovy:.4f} ({fovd:.4f})\n"
                        f"  Intrinsic mat:\n    np.array({intr_mat_str})\n\n"
                    )
                else:  # motion streams, rs.motion_device_intrinsic
                    raise NotImplementedError(f"Not implemented for {stream_name=}")

        # ----- Extrinsic Parameters ----- #
        extrinsics = "Extrinsic Parameters:\n"
        for extr_name, extr_mat in self.all_extrinsics.items():
            sensor1, sensor2 = extr_name.split("=>")
            sensor1_lower = sensor1.replace(" ", "").lower()
            sensor2_lower = sensor2.replace(" ", "").lower()
            extr_mat_str = np.array2string(
                extr_mat,
                max_line_width=200,
                precision=20,
                suppress_small=True,
                separator=", ",
                prefix="    np.array(",
            )

            extrinsics += (
                f' Extrinsic from "{sensor1}"    To     "{sensor2}" '
                f"(T_{sensor2_lower}_{sensor1_lower}):\n"
                f"    np.array({extr_mat_str})\n\n"
            )

        print(device_info)
        print(stream_profiles)
        print(intrinsics)
        print(extrinsics)

    def __del__(self):
        self.stop()

    def __repr__(self):
        if self.uid == self.serial_number:
            return (
                f"<{self.__class__.__name__}: {self.name} (S/N: {self.serial_number})>"
            )
        else:
            return (
                f"<{self.__class__.__name__}: {self.uid} "
                f"({self.name}, S/N: {self.serial_number})>"
            )


class RealSenseAPI:
    """This API should only be used for testing, use sensors.Camera instead"""

    def __init__(self, **kwargs):
        self.logger = get_logger("RealSenseAPI")

        self._connected_devices = self._load_connected_devices(**kwargs)
        self._enabled_devices = []

        self.enable_all_devices()

    def _load_connected_devices(
        self, device_sn: list[str] | None = None, **kwargs
    ) -> list[RSDevice]:
        """Return list of RSDevice
        :param device_sn: list of serial numbers of devices to load.
                          If not None, only load those devices in exact order.
        """
        if device_sn is None:
            device_sn = get_connected_rs_devices()
        elif isinstance(device_sn, str):
            device_sn = [device_sn]

        devices = [RSDevice(sn, **kwargs) for sn in device_sn]

        self.logger.info(f"Loading {len(devices)} devices")
        return devices

    def enable_all_devices(self):
        for device in self._connected_devices:
            device.start()
            self._enabled_devices.append(device)

    def capture(self) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """Capture data from all _enabled_devices.

        :return frame_dicts: list of frame_dict, {stream_name: np.ndarray}.
                             Supported examples:
                             "Color": color image, [H, W, 3] np.uint8 array
                             "Depth": depth image, [H, W] np.uint16 array
                             "Infrared 1/2": infrared image, [H, W] np.uint8 array
        """
        frame_dicts = []
        for device in self._enabled_devices:
            frame_dict = device.wait_for_frames()
            frame_dicts.append(frame_dict)

        if len(self._enabled_devices) == 1:
            frame_dicts = frame_dicts[0]
        return frame_dicts

    def disable_all_devices(self):
        for i in range(len(self._enabled_devices)):
            device = self._enabled_devices.pop()
            device.stop()

    def __del__(self):
        self.disable_all_devices()
