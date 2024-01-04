"""Unittests for real_robot.sensors.camera"""

from time import perf_counter_ns

import numpy as np
import pyrealsense2 as rs
from sapien import Pose

from real_robot.sensors.camera import Camera, CameraConfig
from real_robot.utils.realsense import get_connected_rs_devices


class TestCameraFPS:
    """Test camera streaming speed"""

    _configs = [
        (1280, 720, 30),
        (1280, 720, 15),
        (848, 480, 60),
        (848, 480, 30),
        (848, 480, 15),
        (640, 480, 60),
        (640, 480, 30),
        (640, 480, 15),
        (640, 360, 60),
        (640, 360, 30),
        (640, 360, 15),
        (424, 240, 60),
        (424, 240, 30),
        (424, 240, 15),
    ]

    def test_fps(self):
        device_sn = get_connected_rs_devices()[0]

        for width, height, fps in self._configs:
            camera_cfg = CameraConfig(
                "front_camera",
                device_sn,
                Pose(),
                (width, height, fps),
                preset="High Accuracy",
                depth_option_kwargs={rs.option.exposure: 1500},
            )
            camera = Camera(camera_cfg)

            n_iters = fps * 5
            start_time_ns = perf_counter_ns()
            for _ in range(n_iters):
                while not camera.so_data_dict["depth"].modified:
                    pass
                images = camera.get_images()
            elapsed_ns = perf_counter_ns() - start_time_ns

            rgb, depth = images["rgb"], images["depth"]
            assert rgb.shape == (height, width, 3), rgb.shape
            assert rgb.dtype == np.uint8, rgb.dtype
            assert depth.shape == (height, width, 1), depth.shape
            assert depth.dtype == np.float32, depth.dtype

            elapsed = elapsed_ns / n_iters / 1e9
            iter_max_time = 1.0 / fps + 300e-6
            assert (
                elapsed <= iter_max_time
            ), f"{elapsed:.4g} > {iter_max_time} for {(width, height, fps)}"
            print(f"{elapsed = :.6g} {iter_max_time = :.6g} {(width, height, fps)}")
            del camera

    def test_record_bag_fps(self):
        device_sn = get_connected_rs_devices()[0]

        for width, height, fps in self._configs:
            camera_cfg = CameraConfig(
                "front_camera",
                device_sn,
                Pose(),
                (width, height, fps),
                preset="High Accuracy",
                depth_option_kwargs={rs.option.exposure: 1500},
            )
            camera = Camera(camera_cfg, record_bag=True, bag_path="/tmp")

            n_iters = fps * 5
            start_time_ns = perf_counter_ns()
            for _ in range(n_iters):
                while not camera.so_data_dict["depth"].modified:
                    pass
                images = camera.get_images()
            elapsed_ns = perf_counter_ns() - start_time_ns

            rgb, depth = images["rgb"], images["depth"]
            assert rgb.shape == (height, width, 3), rgb.shape
            assert rgb.dtype == np.uint8, rgb.dtype
            assert depth.shape == (height, width, 1), depth.shape
            assert depth.dtype == np.float32, depth.dtype

            elapsed = elapsed_ns / n_iters / 1e9
            iter_max_time = 1.0 / fps + 300e-6
            assert (
                elapsed <= iter_max_time
            ), f"{elapsed:.4g} > {iter_max_time} for {(width, height, fps)}"
            print(f"{elapsed = :.6g} {iter_max_time = :.6g} {(width, height, fps)}")
            del camera


if __name__ == "__main__":
    t = TestCameraFPS()
    t.test_record_bag_fps()
