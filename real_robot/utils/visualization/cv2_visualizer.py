import os
import math
import time
from typing import List, Tuple

import numpy as np
import cv2

from .utils import draw_mask, colorize_mask
from ..multiprocessing import SharedObject
from ..logger import get_logger


class CV2Visualizer:
    """OpenCV visualizer for RGB and depth images, fps is updated in window title"""

    def __init__(self, window_name="Images", run_as_process=False, stream_camera=False):
        """
        :param window_name: window name
        :param run_as_process: whether to run CV2Visualizer as a separate process.
            If True, CV2Visualizer needs to be created as a `mp.Process`.
            Several SharedObject are mounted to control CV2Visualizer and feed data:
              * "join_viscv2" (created): If True, the CV2Visualizer process is joined.
              * "draw_vis": If True, redraw the images.
              * "sync_rs_<device_uid>": If True, capture from RSDevice.
              Data unique to CV2Visualizer:
              * "viscv2_<image_uid>_color": rgb color image
              * "viscv2_<image_uid>_depth": depth image
              * "viscv2_<image_uid>_mask": object mask
              Data shared with O3DGUIVisualizer
              * "vis_<image_uid>_color": rgb color image
              * "vis_<image_uid>_depth": depth image
              * "vis_<image_uid>_mask": object mask
              RSDevice camera feeds
              * "rs_<device_uid>_color": rgb color image, [H, W, 3] np.uint8 np.ndarray
              * "rs_<device_uid>_depth": depth image, [H, W] np.uint16 np.ndarray
              Corresponding object mask
              * "rs_<device_uid>_mask": object mask, [H, W] bool/np.uint8 np.ndarray
            Acceptable visualization data format:
            * RGB color images: [H, W, 3] np.uint8 np.ndarray
            * Depth images: [H, W] or [H, W, 1] np.uint16/np.floating np.ndarray
            * Object mask images: [H, W] bool/np.uint8 np.ndarray
        :param stream_camera: whether to redraw camera stream when a new frame arrives
        """
        self.logger = get_logger("CV2Visualizer")

        self.window_name = window_name
        self.stream_camera = stream_camera
        self.last_timestamp_ns = time.time_ns()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if run_as_process:
            self.run_as_process()

    @staticmethod
    def preprocess_image(image: np.ndarray, depth_scale=1000.0) -> np.ndarray:
        """Preprocess image for plotting with cv2 (RGB2BGR, applyColorMap for depth)
        :param image: depth or RGB color image
        :param depth_scale: used to apply color map on np.floating depth_image
        :return image: color image in BGR format, [H, W, 3] np.uint8 np.ndarray
        """
        ndim = image.ndim
        channels = image.shape[-1]
        dtype = image.dtype
        if ndim == 2 or (ndim == 3 and channels == 1):  # depth
            # Depth image colormap is taken from
            # https://github.com/IntelRealSense/librealsense/blob/8ffb17b027e100c2a14fa21f01f97a1921ec1e1b/wrappers/python/examples/opencv_viewer_example.py#L56
            if np.issubdtype(dtype, np.floating):
                alpha = 0.03 * depth_scale
            elif dtype == np.uint16:
                alpha = 0.03
            else:
                raise TypeError(f"Unknown depth image dtype: {dtype}")
            return cv2.applyColorMap(
                cv2.convertScaleAbs(image, alpha=alpha), cv2.COLORMAP_JET
            )
        elif ndim == 3 and channels == 3 and dtype == np.uint8:  # rgb
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError(f"Unknown image type: {image.shape=} {dtype=}")

    @staticmethod
    def get_image_layout(n_image: int) -> Tuple[int, int]:
        """Get layout of Images (n_rows, n_cols) where n_rows >= n_cols"""
        s = math.isqrt(n_image)
        s_squared = s*s
        if s_squared == n_image:
            return s, s
        elif n_image <= s_squared+s:
            return s+1, s
        else:
            return s+1, s+1

    def show_images(self, images: List[np.ndarray]):
        """Show the list of images, support non-equal size (cv2.rerize to max size)
        :param images: List of np.ndarray images. Supports depth or RGB color images.
                       If depth image, dtype can be np.uint16 or np.floating
                       If RGB image, dtype must be np.uint8
        """
        if (n_image := len(images)) == 0:
            return
        images = [self.preprocess_image(image) for image in images]

        # Resize non-equal sized image to max_shape
        max_shape = images[np.argmax([image.size for image in images])].shape
        max_H, max_W, _ = max_shape
        images = [image if image.shape == max_shape
                  else cv2.resize(image, (max_W, max_H),
                                  interpolation=cv2.INTER_NEAREST_EXACT)
                  for image in images]

        if n_image == 1:
            vis_image = images[0]
        elif n_image < 4:
            vis_image = np.vstack(images)
        else:
            n_rows, n_cols = self.get_image_layout(n_image)
            vis_image = np.zeros((max_H * n_rows, max_W * n_cols, 3), dtype=np.uint8)
            for r in range(n_rows):
                for c in range(n_cols):
                    if (idx := n_cols*r + c) >= n_image:
                        break
                    vis_image[max_H*r:max_H*(r+1),
                              max_W*c:max_W*(c+1)] = images[idx]

        # Add fps to window title (overlay with cv2.putText is slower)
        cur_timestamp_ns = time.time_ns()
        fps = 1e9 / (cur_timestamp_ns - self.last_timestamp_ns)
        self.last_timestamp_ns = cur_timestamp_ns
        cv2.setWindowTitle(self.window_name,
                           f"{self.window_name} {max_W}x{max_H} @ {fps:6.2f}fps")

        cv2.imshow(self.window_name, vis_image)
        cv2.pollKey()

    def run_as_process(self):
        """Run CV2Visualizer as a separate process"""
        self.logger.info(f"Running {self!r} as a separate process")

        # CV2Visualizer control
        so_joined = SharedObject("join_viscv2", data=False)
        so_draw = SharedObject("draw_vis")

        so_vis_data = {}

        while not so_joined.fetch():
            # Sort names so they are ordered as color, depth, mask
            exist_so_data_names = sorted([
                p for p in os.listdir("/dev/shm")
                if p.startswith(("rs_", "vis_", "viscv2_"))
                and p.endswith(("_color", "_depth", "_mask"))
            ])

            if so_draw.fetch():  # triggers redraw
                images = []
                for so_name in exist_so_data_names:
                    if so_name in so_vis_data:
                        so_data = so_vis_data[so_name]
                    else:
                        so_data = so_vis_data[so_name] = SharedObject(so_name)

                    image = so_data.fetch()
                    if so_name.endswith("_color"):
                        color_image = image
                    if so_name.endswith("_mask"):
                        images.append(draw_mask(color_image, image))
                        images.append(colorize_mask(image))
                    else:
                        images.append(image)
                self.show_images(images)

            self.render()

        self.logger.info(f"Process running {self!r} is joined")
        # Unlink created SharedObject
        so_joined.unlink()

    def clear_image(self):
        """Show a black image"""
        cv2.imshow(self.window_name, np.zeros((128, 128, 3), dtype=np.uint8))
        cv2.pollKey()

    def render(self):
        """Update renderer to show image and respond to mouse and keyboard events"""
        cv2.pollKey()

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.window_name})>"
