import math
from typing import List, Tuple

import numpy as np
import cv2


class CV2Visualizer:
    def __init__(self, window_name="Images"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

        cv2.imshow(self.window_name, vis_image)
        cv2.pollKey()

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
