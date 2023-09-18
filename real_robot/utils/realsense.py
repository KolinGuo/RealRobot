from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional

import pyrealsense2 as rs
import numpy as np

from .multiprocessing import SharedObject
from .logger import get_logger
from .. import REPO_ROOT


_logger = get_logger("realsense.py")
_default_bag_path = REPO_ROOT / "rosbag_recordings"
RS_DEVICES = None  # {device_sn: rs.device}


def get_connected_rs_devices(
    device_sn: Optional[Union[str, List[str]]] = None
) -> Union[List[str], rs.device, List[rs.device]]:
    """Returns list of connected RealSense devices
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
            if name.lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                fw_version = d.get_info(rs.camera_info.firmware_version)
                usb_type = d.get_info(rs.camera_info.usb_type_descriptor)

                _logger.info(f"Found {name} (S/N: {serial} "
                             f"FW: {fw_version} on USB {usb_type})")
                assert "D435" in name, "Only support D435 currently"
                RS_DEVICES[serial] = d
        _logger.info(f"Found {len(RS_DEVICES)} devices")

    if device_sn is None:
        return list(RS_DEVICES.keys())
    elif isinstance(device_sn, str):
        return RS_DEVICES[device_sn]
    else:
        return [RS_DEVICES[sn] for sn in device_sn]


class RSDevice:
    """RealSense Device, only support D435 for now

    For best depth accuracy with D435,
        set preset="High Accuracy" and use (848, 480) depth resolution

    References:
    * https://dev.intelrealsense.com/docs/d400-series-visual-presets
    * https://dev.intelrealsense.com/docs/tuning-depth-cameras-for-best-performance
    """

    def __init__(self, device_sn: str,
                 color_config=(848, 480, 30), depth_config=(848, 480, 30), *,
                 preset="Default", color_option_kwargs={}, depth_option_kwargs={},
                 record_bag=False, bag_path=_default_bag_path, run_as_process=False):
        """
        :param device_sn: realsense device serial number
        :param color_config: color sensor config, (width, height, fps)
        :param depth_config: depth sensor config, (width, height, fps)
        :param preset: depth sensor preset, available options:
                       ["Custom", "Default", "Hand", "High Accuracy",
                        "High Density", "Medium Density", "Remove Ir Pattern"].
        :param color_option_kwargs: color sensor options kwargs.
                                    Available options see self.supported_color_options
        :param depth_option_kwargs: depth sensor options kwargs.
                                    Available options see self.supported_depth_options
        :param record_bag: whether to record camera streams as a rosbag file.
        :param bag_path: path to save bag recording. Must end with ".bag" if it's a file
        :param run_as_process: whether to run RSDevice as a separate process.
            If True, RSDevice needs to be created as a `mp.Process`.
            Several SharedObject are created to control RSDevice and fetch data:
            * "rs_<device_sn>_start": If True, starts the RSDevice; else, stops it.
            * "rs_<device_sn>_joined": If True, the RSDevice process is joined.
            * "rs_<device_sn>_color": color image, [H, W, 3] np.uint8 np.ndarray
            * "rs_<device_sn>_depth": depth image, [H, W] np.uint16 np.ndarray
            * "rs_<device_sn>_intr": intrinsic matrix, [3, 3] np.float64 np.ndarray
        """
        self.logger = get_logger("RSDevice")

        self.device = get_connected_rs_devices(device_sn)
        self.name = self.device.get_info(rs.camera_info.name)
        self.serial_number = device_sn
        assert "D435" in self.name, f"Only support D435 currently, get {self!r}"
        self.color_sensor = self.device.first_color_sensor()
        self.depth_sensor = self.device.first_depth_sensor()
        # Record to a rosbag file
        self.record_bag = record_bag
        self.bag_path = Path(bag_path)
        if self.bag_path.suffix != ".bag":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.bag_path = self.bag_path / f"rs_{device_sn}_{timestamp}.bag"
        self.bag_path.parent.mkdir(parents=True, exist_ok=True)

        self.config = self._create_rs_config(color_config, depth_config)
        self.align = rs.align(rs.stream.color)
        self.width, self.height = color_config[0], color_config[1]

        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrix = None
        self.last_frame_num = None

        self._load_depth_preset(preset)
        self._set_sensor_options(color_option_kwargs, depth_option_kwargs)

        if run_as_process:
            self.run_as_process()

    def _create_rs_config(self, color_config: tuple, depth_config: tuple) -> rs.config:
        config = rs.config()
        if color_config is not None:
            assert color_config in self.supported_color_configs, \
                f"Not supported {color_config = }"
            width, height, fps = color_config
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if depth_config is not None:
            assert depth_config in self.supported_depth_configs, \
                f"Not supported {depth_config = }"
            width, height, fps = depth_config
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Record camera streams as a rosbag file
        if self.record_bag:
            self.logger.info(f'Enable recording {self!r} to file "{self.bag_path}"')
            config.enable_record_to_file(str(self.bag_path))
        return config

    def _load_depth_preset(self, preset="Default"):
        if preset not in (presets := self.supported_depth_presets):
            raise ValueError(f"No preset named {preset}. "
                             f"Available presets {presets}")

        self.depth_sensor.set_option(rs.option.visual_preset,
                                     presets.index(preset))
        self.logger.info(f'Loaded "{preset}" preset for {self!r}')

    def _set_sensor_options(self, color_option_kwargs, depth_option_kwargs):
        for key, value in color_option_kwargs.items():
            self.color_sensor.set_option(key, value)
            self.logger.info(f'Setting Color "{key}" to {value}')

        for key, value in depth_option_kwargs.items():
            self.depth_sensor.set_option(key, value)
            self.logger.info(f'Setting Depth "{key}" to {value}')

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Returns a 3x3 camera intrinsics matrix, available after self.start()"""
        return self.intrinsic_matrix

    def start(self) -> bool:
        """Start the streaming pipeline"""
        if self.is_running:
            self.logger.warning(f"Device {self!r} is already running. "
                                "Please call stop() before calling start() again")
            return False

        self.pipeline = rs.pipeline()

        self.config.enable_device(self.serial_number)
        self.pipeline_profile = self.pipeline.start(self.config)

        for _ in range(20):  # wait for white balance to stabilize
            self.pipeline.wait_for_frames()

        streams = self.pipeline_profile.get_streams()
        self.logger.info(f"Started device {self!r} with {len(streams)} streams")
        for i, stream in enumerate(streams):
            self.logger.info(f"Stream {i+1}: {stream}")

        # with rs.align, camera intrinsics is color sensor intrinsics
        stream_profile = self.pipeline_profile.get_stream(rs.stream.color)
        intrinsics = stream_profile.as_video_stream_profile().intrinsics
        self.intrinsic_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                          [0, intrinsics.fy, intrinsics.ppy],
                                          [0, 0, 1]])
        return True

    def wait_for_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wait until a new set of frames becomes available.
        Each enabled stream in the pipeline is time-synchronized.
        :return color_image: color image, [H, W, 3] np.uint8 array
        :return depth_image: depth image, [H, W] np.uint16 array
        """
        assert self.pipeline is not None, "Device is not started"

        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        self.last_frame_num = frames.get_frame_number()
        # self.logger.info(f"Received frame #{self.last_frame_num}")

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Need to copy() so the device can release the frame from its internal memory
        depth_image = np.asarray(depth_frame.data).copy()
        color_image = np.asarray(color_frame.data).copy()

        return color_image, depth_image

    def stop(self) -> bool:
        """Stop the streaming pipeline"""
        if not self.is_running:
            self.logger.warning(f"Device {self!r} is not running. Ignoring stop()")
            return False

        self.pipeline.stop()
        self.pipeline = None
        self.pipeline_profile = None
        self.intrinsic_matrix = None
        self.last_frame_num = None
        self.logger.info(f"Stopped device {self!r}")
        return True

    def run_as_process(self):
        """Run RSDevice as a separate process"""
        self.logger.info(f"Running {self!r} as a separate process")

        # RSDevice control
        device_started = False
        so_start = SharedObject(f"rs_{self.serial_number}_start", data=False)
        so_joined = SharedObject(f"rs_{self.serial_number}_joined", data=False)
        # data
        so_color = SharedObject(
            f"rs_{self.serial_number}_color",
            data=np.zeros((self.height, self.width, 3), dtype=np.uint8)
        )
        so_depth = SharedObject(
            f"rs_{self.serial_number}_depth",
            data=np.zeros((self.height, self.width), dtype=np.uint16)
        )
        so_intr = SharedObject(f"rs_{self.serial_number}_intr", data=np.zeros((3, 3)))

        while not so_joined.fetch():
            start = so_start.fetch()
            if not device_started and start:
                self.start()
                so_intr.assign(self.intrinsic_matrix)
                device_started = True
            elif device_started and not start:
                self.stop()
                device_started = False

            if device_started:
                # wait for frames
                frames = self.pipeline.wait_for_frames()
                frames = self.align.process(frames)
                so_color.assign(np.asarray(frames.get_color_frame().data))
                so_depth.assign(np.asarray(frames.get_depth_frame().data))

        self.logger.info(f"Process running {self!r} is joined")
        # Unlink SharedObject
        so_start.unlink()
        so_joined.unlink()
        so_color.unlink()
        so_depth.unlink()
        so_intr.unlink()

    @property
    def is_running(self) -> bool:
        """Returns whether the streaming pipeline is running"""
        return self.pipeline is not None

    @property
    def supported_color_configs(self) -> List[Tuple[int]]:
        """Return supported color configs as (width, height, fps)"""
        configs = []
        format = rs.format.rgb8
        for profile in self.color_sensor.get_stream_profiles():
            profile = profile.as_video_stream_profile()
            if profile.format() == format:
                configs.append((profile.width(), profile.height(), profile.fps()))
        return configs

    @property
    def supported_depth_configs(self) -> List[Tuple[int]]:
        """Return supported depth configs as (width, height, fps)"""
        configs = []
        format = rs.format.z16
        for profile in self.depth_sensor.get_stream_profiles():
            profile = profile.as_video_stream_profile()
            if profile.format() == format:
                configs.append((profile.width(), profile.height(), profile.fps()))
        return configs

    @property
    def supported_depth_presets(self) -> List[str]:
        presets = []
        for i in range(10):
            preset = self.depth_sensor.get_option_value_description(
                rs.option.visual_preset, i
            )
            if preset == "UNKNOWN":
                break
            presets.append(preset)
        return presets

    @property
    def supported_color_options(self) -> Dict[int, rs.option_range]:
        options = {}
        for option in self.color_sensor.get_supported_options():
            try:
                options[option] = self.color_sensor.get_option_range(option)
            except RuntimeError:
                pass
        return options

    @property
    def supported_depth_options(self) -> Dict[int, rs.option_range]:
        options = {}
        for option in self.depth_sensor.get_supported_options():
            try:
                options[option] = self.depth_sensor.get_option_range(option)
            except RuntimeError:
                pass
        return options

    def __del__(self):
        self.stop()

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name} (S/N: {self.serial_number})>"


class RealSenseAPI:
    def __init__(self, **kwargs):
        self.logger = get_logger("RealSenseAPI")

        self._connected_devices = self._load_connected_devices(**kwargs)
        self._enabled_devices = []

        self.enable_all_devices()

    def _load_connected_devices(self, device_sn: Optional[List[str]] = None,
                                **kwargs) -> List[RSDevice]:
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

    def capture(self):
        """Capture data from all _enabled_devices.
        If n_cam == 1, first dimension is squeezed.
        :return color_image: color image, [n_cam, H, W, 3] np.uint8 array
        :return depth_image: depth image, [n_cam, H, W] np.uint16 array
        """
        color_images = []
        depth_images = []
        for device in self._enabled_devices:
            color_image, depth_image = device.wait_for_frames()
            color_images.append(color_image)
            depth_images.append(depth_image)

        if len(self._enabled_devices) == 1:
            color_images = color_images[0]
            depth_images = depth_images[0]
        else:
            color_images = np.stack(color_images)
            depth_images = np.stack(depth_images)

        return color_images, depth_images

    def disable_all_devices(self):
        for i in range(len(self._enabled_devices)):
            device = self._enabled_devices.pop()
            device.stop()

    def __del__(self):
        self.disable_all_devices()
