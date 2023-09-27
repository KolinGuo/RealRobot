import os
import math
import time
import functools
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import cv2

from .utils import draw_mask, colorize_mask
from ..multiprocessing import SharedObject, SharedObjectDefaultDict
from ..logger import get_logger


class CV2Visualizer:
    """OpenCV visualizer for RGB and depth images, fps is updated in window title"""

    def __init__(self, window_name="Images", run_as_process=False, stream_camera=False):
        """
        :param window_name: window name
        :param run_as_process: whether to run CV2Visualizer as a separate process.
          If True, CV2Visualizer needs to be created as a `mp.Process`.
          Several SharedObject are mounted to control CV2Visualizer and fetch data:
              Only "join_viscv2" is created by this process.
            * "join_viscv2": If triggered, the CV2Visualizer process is joined.
            * "draw_vis": If triggered, redraw the images.
            * "sync_rs_<device_uid>": If triggered, capture from RSDevice.
            Corresponding data have the same prefix (implemented as sorting)
            * Data unique to CV2Visualizer have prefix "viscv2_<image_uid>_"
            * Data shared with O3DGUIVisualizer have prefix "vis_<image_uid>_"
            * RSDevice camera feeds have prefix "rs_<device_uid>_"
              * "rs_<device_uid>_color": rgb color image, [H, W, 3] np.uint8 np.ndarray
              * "rs_<device_uid>_depth": depth image, [H, W] np.uint16 np.ndarray
              * "rs_<device_uid>_mask": object mask, [H, W] bool/np.uint8 np.ndarray
          Acceptable visualization SharedObject data formats and suffixes:
          * "_color": RGB color images, [H, W, 3] np.uint8 np.ndarray
          * "_depth": Depth images, [H, W] or [H, W, 1] np.uint16/np.floating np.ndarray
          * "_mask": Object mask images, [H, W] bool/np.uint8 np.ndarray
        :param stream_camera: whether to redraw camera stream when a new frame arrives
        """
        self.logger = get_logger("CV2Visualizer")

        self.window_name = window_name
        self.last_timestamp_ns = time.time_ns()
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        if run_as_process:
            self.stream_camera = stream_camera
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
        if ndim == 3 and channels == 3 and dtype == np.uint8:  # rgb
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif ndim == 2 or (ndim == 3 and channels == 1):  # depth
            # Depth image colormap is taken from
            # https://github.com/IntelRealSense/librealsense/blob/8ffb17b027e100c2a14fa21f01f97a1921ec1e1b/wrappers/python/examples/opencv_viewer_example.py#L56
            if dtype == np.uint16:
                alpha = 0.03
            elif np.issubdtype(dtype, np.floating):
                alpha = 0.03 * depth_scale
            else:
                raise TypeError(f"Unknown depth image dtype: {dtype}")
            return cv2.applyColorMap(
                cv2.convertScaleAbs(image, alpha=alpha), cv2.COLORMAP_JET
            )
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

    @staticmethod
    def resize_image(image: np.ndarray, max_H: int, max_W: int,
                     interpolation=cv2.INTER_NEAREST_EXACT) -> np.ndarray:
        """Resize image to the requested size while preserving its aspect ratio"""
        # background is white
        new_image = np.full((max_H, max_W, 3), 255, np.uint8)

        h, w, _ = image.shape
        if (h_ratio := (max_H / h)) < (w_ratio := (max_W / w)):
            new_w = math.floor(w * h_ratio)
            start_i = (max_W - new_w) // 2
            new_image[:, start_i:start_i+new_w] = cv2.resize(
                image, (new_w, max_H), interpolation=interpolation
            )
            return new_image
        else:
            new_h = math.floor(h * w_ratio)
            start_i = (max_H - new_h) // 2
            new_image[start_i:start_i+new_h, :] = cv2.resize(
                image, (max_W, new_h), interpolation=interpolation
            )
            return new_image

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
                  else self.resize_image(image, max_H, max_W) for image in images]

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
        so_joined = SharedObject("join_viscv2")
        so_draw = SharedObject("draw_vis")
        so_dict = SharedObjectDefaultDict()  # {so_name: SharedObject}

        # {"rs_<device_uid>_color": image}
        vis_data = defaultdict(functools.partial(np.full, shape=(480, 848, 3),
                                                 fill_value=255, dtype=np.uint8))

        while not so_joined.triggered:
            # Sort names so they are ordered as color, depth, mask
            all_so_names = sorted(os.listdir("/dev/shm"))

            so_data_names = [
                p for p in all_so_names
                if p.startswith(("rs_", "vis_", "viscv2_"))
                and p.endswith(("_color", "_depth", "_mask"))
            ]

            # ----- Capture from RSDevice stream -----
            if self.stream_camera:  # capture whenever a new frame comes in
                updated = False
                for so_data_name in [p for p in all_so_names if p.startswith("rs_")
                                     and p.endswith(("_color", "_depth"))]:
                    if (so_data := so_dict[so_data_name]).modified:
                        vis_data[so_data_name] = so_data.fetch()
                        updated = True
                if updated:  # camera stream is updated, redraw images
                    images = []
                    for so_data_name in so_data_names:
                        if so_data_name.endswith("_mask"):
                            images += [vis_data[f"{so_data_name}_overlay"],
                                       vis_data[f"{so_data_name}_colorized"]]
                        else:
                            images.append(vis_data[so_data_name])
                    self.show_images(images)
            else:  # synchronized capturing with env (no redraw here)
                # for each camera sync, check if capture is triggered
                for so_name in [p for p in all_so_names if p.startswith("sync_rs_")]:
                    if so_dict[so_name].triggered:
                        if (so_data_name := f"{so_name[5:]}_color") in all_so_names:
                            vis_data[so_data_name] = so_dict[so_data_name].fetch()
                        if (so_data_name := f"{so_name[5:]}_depth") in all_so_names:
                            vis_data[so_data_name] = so_dict[so_data_name].fetch()

            # ----- Fetch data and draw -----
            if so_draw.triggered:  # triggers redraw
                images = []
                for so_data_name in so_data_names:
                    # Fetch data or use captured RSDevice data
                    if so_data_name.endswith("_color"):
                        if so_data_name.startswith("rs_"):
                            color_image = image = vis_data[so_data_name]
                        else:
                            vis_data[so_data_name] = color_image = image \
                                = so_dict[so_data_name].fetch()
                    elif (so_data_name.endswith("_depth") and
                          so_data_name.startswith("rs_")):
                        image = vis_data[so_data_name]
                    else:
                        vis_data[so_data_name] = image = so_dict[so_data_name].fetch()

                    # Visualize mask by overlaying to color_image and colorizing mask
                    if so_data_name.endswith("_mask"):
                        vis_data[f"{so_data_name}_overlay"] = mask_overlay \
                            = draw_mask(color_image, image)
                        vis_data[f"{so_data_name}_colorized"] = mask_colorized \
                            = colorize_mask(image)
                        images += [mask_overlay, mask_colorized]
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
