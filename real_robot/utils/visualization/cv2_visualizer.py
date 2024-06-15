from __future__ import annotations

import functools
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np

from ..logger import get_logger
from ..multiprocessing import (
    SharedObject,
    SharedObjectDefaultDict,
    signal_process_ready,
)
from .utils import colorize_mask, draw_mask


class CV2Visualizer:
    """OpenCV visualizer for RGB and depth images, fps is updated in window title"""

    images: list[np.ndarray] = []  # images being visualized
    # images boundaries [[[x_min, x_max+1], [y_min, y_max+1]]]
    image_boundaries: np.ndarray = np.empty((0, 2, 2), dtype=np.floating)

    class DrawingMode(Enum):
        """Drawing mode when draw_with_mouse()"""

        Box = 0
        Point = 1

    # Attributes for drawing points / bbox
    _done_drawing: bool = True  # whether drawing with mouse is done
    _in_drawing: bool = False  # whether currently in drawing
    selected_image_idx: int | None = None  # previously selected image during drawing
    _image: np.ndarray | None = None  # the image currently in drawing
    _extra_ret_data: Any = None  # extra data returned from _update_drawing_fn()
    _mouse_pos: tuple[int, int] = (-1, -1)  # mouse position when selecting image
    _CTRLKEY: int = 0  # whether CTRL key is pressed
    _drawing_mode: DrawingMode = DrawingMode.Box  # Drawing mode
    points: np.ndarray = np.empty((0, 2), dtype=int)  # Drawn points: [x, y]
    point_labels: np.ndarray = np.empty(0, dtype=int)  # Drawn point labels: (0, 1)
    boxes: np.ndarray = np.empty((0, 4), dtype=int)  # Drawn boxes XYXY coordinates
    box_labels: np.ndarray = np.empty(0, dtype=int)  # Drawn box labels: (0, 1)

    def __init__(
        self,
        window_name="Images",
        *,
        update_drawing_fn: Optional[
            Callable[[np.ndarray, dict[str, np.ndarray]], tuple[np.ndarray, Any]]
        ] = None,
        run_as_process=False,
        stream_camera=False,
    ):
        """
        :param window_name: window name
        :param update_drawing_fn: callback function with drawn points/boxes
            during drawing.
            Inputs to this function are an RGB image and a dictionary of
            {"points": np.ndarray, "point_labels": np.ndarray,
             "boxes": np.ndarray, "box_labels": np.ndarray}
            Outputs are the modified image and any additional data.
        :param run_as_process: whether to run CV2Visualizer as a separate process.
          If True, CV2Visualizer needs to be created as a `mp.Process`.
          Several SharedObject are mounted to control CV2Visualizer and fetch data:
              Only "join_viscv2" is created by this process.
            * "join_viscv2": If triggered, the CV2Visualizer process is joined.
            * "draw_vis": If triggered, redraw the images.
            * "reset_vis": If triggered, call self.clear_image().
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
        cv2.setMouseCallback(window_name, self.on_mouse)
        cv2.displayOverlay(self.window_name, "Press 'd' to go into drawing mode", 5000)
        if update_drawing_fn is not None:
            self._update_drawing_fn = update_drawing_fn

        if run_as_process:
            self.stream_camera = stream_camera
            self.run_as_process()

    @staticmethod
    def preprocess_image(image: np.ndarray, depth_scale=1000.0) -> np.ndarray:
        """Preprocess image for plotting with cv2 (RGB2BGR, applyColorMap for depth)
        :param image: depth, grayscale or RGB color image
        :param depth_scale: used to apply color map on np.floating depth_image
        :return image: color image in BGR format, [H, W, 3] np.uint8 np.ndarray
        """
        ndim = image.ndim
        channels = image.shape[-1]
        dtype = image.dtype
        if ndim == 3 and channels == 3 and dtype == np.uint8:  # rgb
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif ndim == 2 or (ndim == 3 and channels == 1):  # depth or grayscale
            # Depth image colormap is taken from
            # https://github.com/IntelRealSense/librealsense/blob/8ffb17b027e100c2a14fa21f01f97a1921ec1e1b/wrappers/python/examples/opencv_viewer_example.py#L56
            if dtype == np.uint16:
                alpha = 0.03
            elif np.issubdtype(dtype, np.floating):
                alpha = 0.03 * depth_scale
            elif dtype == np.uint8:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                raise TypeError(f"Unknown depth image dtype: {dtype}")
            return cv2.applyColorMap(
                cv2.convertScaleAbs(image, alpha=alpha), cv2.COLORMAP_JET
            )
        else:
            raise NotImplementedError(f"Unknown image type: {image.shape=} {dtype=}")

    @staticmethod
    def get_image_layout(n_image: int) -> tuple[int, int]:
        """Get layout of Images (n_rows, n_cols) where n_rows >= n_cols"""
        s = math.isqrt(n_image)
        s_squared = s * s
        if s_squared == n_image:
            return s, s
        elif n_image <= s_squared + s:
            return s + 1, s
        else:
            return s + 1, s + 1

    @staticmethod
    def resize_image(
        image: np.ndarray, max_H: int, max_W: int, interpolation=cv2.INTER_NEAREST_EXACT
    ) -> np.ndarray:
        """Resize image to the requested size while preserving its aspect ratio"""
        # background is white
        new_image = np.full((max_H, max_W, 3), 255, np.uint8)

        h, w, _ = image.shape
        if (h_ratio := (max_H / h)) < (w_ratio := (max_W / w)):
            new_w = math.floor(w * h_ratio)
            start_i = (max_W - new_w) // 2
            new_image[:, start_i : start_i + new_w] = cv2.resize(
                image, (new_w, max_H), interpolation=interpolation
            )
            return new_image
        else:
            new_h = math.floor(h * w_ratio)
            start_i = (max_H - new_h) // 2
            new_image[start_i : start_i + new_h, :] = cv2.resize(
                image, (max_W, new_h), interpolation=interpolation
            )
            return new_image

    def show_images(self, images: list[np.ndarray]):
        """Show the list of images, support non-equal size (cv2.rerize to max size)

        :param images: List of np.ndarray images. Supports depth or RGB color images.
            If depth image, dtype can be np.uint16 or np.floating
            If RGB image, dtype must be np.uint8
        """
        if (n_image := len(images)) == 0:
            return
        self.images = images  # save input images
        images = [self.preprocess_image(image) for image in images]

        # Resize non-equal sized image to max_shape
        max_shape = images[np.argmax([image.size for image in images])].shape
        max_H, max_W, _ = max_shape
        images = [
            (
                image
                if image.shape == max_shape
                else self.resize_image(image, max_H, max_W)
            )
            for image in images
        ]

        if n_image == 1:
            vis_image = images[0]
            self.image_boundaries = np.asarray([[[0, max_W], [0, max_H]]])
        elif n_image < 4:
            vis_image = np.vstack(images)
            self.image_boundaries = np.asarray([
                [[0, max_W], [max_H * i, max_H * (i + 1)]] for i in range(n_image)
            ])
        else:
            n_rows, n_cols = self.get_image_layout(n_image)
            vis_image = np.zeros((max_H * n_rows, max_W * n_cols, 3), dtype=np.uint8)
            self.image_boundaries = np.empty((0, 2, 2), dtype=int)
            for r in range(n_rows):
                for c in range(n_cols):
                    if (idx := n_cols * r + c) >= n_image:
                        break
                    vis_image[
                        max_H * r : max_H * (r + 1), max_W * c : max_W * (c + 1)
                    ] = images[idx]
                    self.image_boundaries = np.concatenate([
                        self.image_boundaries,
                        np.asarray([
                            [[max_W * c, max_W * (c + 1)], [max_H * r, max_H * (r + 1)]]
                        ]),
                    ])

        # Add fps to window title (overlay with cv2.putText is slower)
        cur_timestamp_ns = time.time_ns()
        fps = 1e9 / (cur_timestamp_ns - self.last_timestamp_ns)
        self.last_timestamp_ns = cur_timestamp_ns
        cv2.setWindowTitle(
            self.window_name, f"{self.window_name} {max_W}x{max_H} @ {fps:6.2f}fps"
        )

        cv2.imshow(self.window_name, vis_image)
        self.render()

    def run_as_process(self):
        """Run CV2Visualizer as a separate process"""
        self.logger.info(f"Running {self!r} as a separate process")

        # CV2Visualizer control
        so_joined = SharedObject("join_viscv2")
        so_draw = SharedObject("draw_vis")
        so_reset = SharedObject("reset_vis")
        so_dict = SharedObjectDefaultDict()  # {so_name: SharedObject}

        # {"rs_<device_uid>_color": image}
        vis_data = defaultdict(
            functools.partial(
                np.full, shape=(480, 848, 3), fill_value=255, dtype=np.uint8
            )
        )

        def get_so_data_names(all_so_names: list[str]) -> list[str]:
            """Get so_data_names acceptable by CV2Visualizer"""
            valid_names = [
                p
                for p in all_so_names
                if p.startswith(("rs_", "vis_", "viscv2_"))
                and p.endswith((
                    "_color",
                    "_depth",
                    "_infrared_1",
                    "_infrared_2",
                    "_mask",
                ))
            ]
            if len(valid_names) == 0:
                self.logger.warning(
                    "No valid shm data names found under /dev/shm. The shm file names "
                    "must have prefix in ['rs_', 'vis_', 'viscv2_'] and "
                    "suffix in ['_color', '_depth', '_mask']"
                )
            return valid_names

        rs_so_name_suffix = ("_color", "_depth", "_infrared_1", "_infrared_2")

        signal_process_ready()  # current process is ready

        while not so_joined.triggered:
            # Sort names so they are ordered as color, depth, mask
            all_so_names = sorted(os.listdir("/dev/shm"))

            # ----- Reset ----- #
            if so_reset.triggered:  # triggers reset
                self.clear_image()
                so_dict = SharedObjectDefaultDict()  # {so_name: SharedObject}
                # {"rs_<device_uid>_color": image}
                vis_data = defaultdict(
                    functools.partial(
                        np.full, shape=(480, 848, 3), fill_value=255, dtype=np.uint8
                    )
                )

            # ----- Capture and update from RSDevice stream ----- #
            if self.stream_camera:  # capture whenever a new frame comes in
                updated = False
                for so_data_name in [
                    p
                    for p in all_so_names
                    if p.startswith("rs_") and p.endswith(rs_so_name_suffix)
                ]:
                    if (so_data := so_dict[so_data_name]).modified:
                        vis_data[so_data_name] = so_data.fetch()
                        updated = True
                if updated:  # camera stream is updated, redraw images
                    images = []
                    for so_data_name in get_so_data_names(all_so_names):
                        if so_data_name.endswith("_mask"):
                            images += [
                                vis_data[f"{so_data_name}_overlay"],
                                vis_data[f"{so_data_name}_colorized"],
                            ]
                        else:
                            images.append(vis_data[so_data_name])
                    self.show_images(images)
            else:  # synchronized capturing with env (no redraw here)
                # for each camera sync, check if capture is triggered
                for so_name in [p for p in all_so_names if p.startswith("sync_rs_")]:
                    if so_dict[so_name].triggered:
                        for so_data_name in [
                            f"{so_name[5:]}{suffix}" for suffix in rs_so_name_suffix
                        ]:
                            if so_data_name in all_so_names:
                                vis_data[so_data_name] = so_dict[so_data_name].fetch()

            # ----- Fetch data and draw ----- #
            if so_draw.triggered:  # triggers redraw
                # Sort names so they are ordered as color, depth, mask
                all_so_names = sorted(os.listdir("/dev/shm"))  # get most updated list
                images = []
                for so_data_name in get_so_data_names(all_so_names):
                    # Fetch data or use captured RSDevice data
                    if so_data_name.endswith("_color"):
                        if so_data_name.startswith("rs_"):
                            color_image = image = vis_data[so_data_name]
                        else:
                            vis_data[so_data_name] = color_image = image = so_dict[
                                so_data_name
                            ].fetch()
                    elif so_data_name.startswith("rs_") and not so_data_name.endswith(
                        "_mask"
                    ):
                        image = vis_data[so_data_name]
                    else:
                        vis_data[so_data_name] = image = so_dict[so_data_name].fetch()

                    # Visualize mask by overlaying to color_image and colorizing mask
                    if so_data_name.endswith("_mask"):
                        vis_data[f"{so_data_name}_overlay"] = mask_overlay = draw_mask(
                            color_image, image
                        )
                        vis_data[f"{so_data_name}_colorized"] = mask_colorized = (
                            colorize_mask(image)
                        )
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
        if cv2.pollKey() == ord("d"):
            self.draw_with_mouse()

    def draw_with_mouse(
        self,
        *,
        update_drawing_fn: Optional[
            Callable[[np.ndarray, dict[str, np.ndarray]], tuple[np.ndarray, Any]]
        ] = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], Any] | None:
        """Enter drawing mode with mouse

        :param update_drawing_fn: function to be called with the image and drawn
            points/boxes.
            Inputs to this function are an RGB image and a dictionary of
            {"points": np.ndarray, "point_labels": np.ndarray,
            "boxes": np.ndarray, "box_labels": np.ndarray}
            Outputs are the modified image and any additional data.
        :return: Original image before drawing, drawn labels dict, _extra_ret_data
            returned from self._update_drawing_fn()
        """
        self.selected_image_idx = None
        self._image = None  # the image currently in drawing
        if update_drawing_fn is not None:
            self._update_drawing_fn = update_drawing_fn
        self._extra_ret_data = None
        self._mouse_pos = (-1, -1)
        self._CTRLKEY = 0
        self._drawing_mode = self.DrawingMode.Box
        self.points = np.empty((0, 2), dtype=int)  # Drawn points: [x, y]
        self.point_labels = np.empty(0, dtype=int)  # Drawn point labels: (0, 1)
        self.boxes = np.empty((0, 4), dtype=int)  # Drawn boxes XYXY coordinates
        self.box_labels = np.empty(0, dtype=int)  # Drawn box labels: (0, 1)
        self._done_drawing = False

        init_overlay_text = (
            "Drawing mode! 's/x' to select/delete image, 'Esc/Enter/Space' to quit"
        )
        cv2.displayOverlay(self.window_name, init_overlay_text)

        def get_drawing_overlay_text():
            return (
                f"{str(self._drawing_mode).split('.')[-1]} drawing mode! "
                "'Enter/Space' to apply, 'Esc' to quit, 'm' to change mode, "
                "'z' to remove last drawing, 'r' to remove all drawings, "
                "hold 'Ctrl' to draw with negative label"
            )

        def draw_labels(image: np.ndarray, with_cursor=False) -> np.ndarray:
            """Draw points/boxes labels on image

            :param image: RGB image to drawn onto
            :return image: drawn image in BGR format
            """
            image = self.preprocess_image(image)
            point_radius = int(np.sqrt(np.prod(image.shape[:2])) * 0.005)
            line_thickness = int(np.sqrt(np.prod(image.shape[:2])) * 0.004)
            # Current mouse position
            if with_cursor:
                image = cv2.circle(  # type: ignore
                    image,
                    self._mouse_pos,
                    radius=point_radius,
                    color=(0, 0, 255) if self._CTRLKEY else (0, 255, 0),
                    thickness=-1,
                )
            # Drawn points
            for point, label in zip(self.points, self.point_labels):
                image = cv2.circle(
                    image,
                    point,
                    radius=point_radius,
                    color=(0, 255, 0) if label else (0, 0, 255),
                    thickness=-1,
                )
            # Drawn boxes
            for (x0, y0, x1, y1), label in zip(self.boxes, self.box_labels):
                image = cv2.rectangle(  # type: ignore
                    image,
                    (x0, y0),
                    (x1, y1),
                    color=(0, 255, 0) if label else (0, 0, 255),
                    thickness=line_thickness,
                )
            return image

        def get_image_idx() -> int | None:
            """Get image index that the mouse is pointing at"""
            image_idx_mask = np.all(
                [
                    (self.image_boundaries[:, :, 0] <= self._mouse_pos).all(1),
                    (self._mouse_pos < self.image_boundaries[:, :, 1]).all(1),
                ],
                axis=0,
            )
            if image_idx_mask.any():
                return int(np.flatnonzero(image_idx_mask))
            else:
                self.logger.warning("No selection, please move mouse to select")
                return None

        if len(self.images) == 1:  # when there is only one image, skip selection
            self.selected_image_idx = 0
            self._image = self.images[self.selected_image_idx].copy()
            cv2.displayOverlay(self.window_name, get_drawing_overlay_text())

        while True:
            if self._in_drawing:
                cv2.imshow(self.window_name, draw_labels(self._image, with_cursor=True))  # type: ignore
                cv2.pollKey()
                continue

            if (key := cv2.pollKey()) in [13, 32]:  # ENTER or Space
                self._done_drawing = True
                if (
                    self.selected_image_idx is not None
                    and self._image is not None
                    and (len(self.points) > 0 or len(self.boxes) > 0)
                ):
                    self.images.insert(
                        self.selected_image_idx + 1,
                        cv2.cvtColor(draw_labels(self._image), cv2.COLOR_BGR2RGB),
                    )
                self.show_images(self.images)
                break
            elif key == 27:  # ESC, stop current drawing and restart
                if self.selected_image_idx is None:
                    self._done_drawing = True
                    break
                else:
                    self.selected_image_idx = None
                    self._image = None
                    self._extra_ret_data = None
                    self._mouse_pos = (-1, -1)
                    self.points = np.empty((0, 2), dtype=int)
                    self.point_labels = np.empty(0, dtype=int)
                    self.boxes = np.empty((0, 4), dtype=int)
                    self.box_labels = np.empty(0, dtype=int)
                    self.show_images(self.images)
                    cv2.displayOverlay(self.window_name, init_overlay_text)
            elif (
                key == ord("x") and (image_idx := get_image_idx()) is not None
            ):  # delete image
                self.images.pop(image_idx)
                self.show_images(self.images)
            elif (
                key == ord("s")  # select image
                and self.selected_image_idx is None
                and (image_idx := get_image_idx()) is not None
            ):
                self.selected_image_idx = image_idx
                self._image = self.images[self.selected_image_idx].copy()
                self.points = np.empty((0, 2), dtype=int)
                self.point_labels = np.empty(0, dtype=int)
                self.boxes = np.empty((0, 4), dtype=int)
                self.box_labels = np.empty(0, dtype=int)
                cv2.displayOverlay(self.window_name, get_drawing_overlay_text())
            if self._image is None:
                continue

            # Draw image
            cv2.imshow(self.window_name, draw_labels(self._image, with_cursor=True))

            if key == ord("m"):  # change drawing mode
                self._drawing_mode = (
                    self.DrawingMode.Box
                    if self._drawing_mode == self.DrawingMode.Point
                    else self.DrawingMode.Point
                )
                cv2.displayOverlay(self.window_name, get_drawing_overlay_text())
            elif key == ord("z"):  # remove previous drawing
                if (
                    self._drawing_mode == self.DrawingMode.Point
                    and len(self.points) > 0
                ):
                    self.points = self.points[:-1]
                    self.point_labels = self.point_labels[:-1]
                    self.call_update_drawing_fn()
                elif self._drawing_mode == self.DrawingMode.Box and len(self.boxes) > 0:
                    self.boxes = self.boxes[:-1]
                    self.box_labels = self.box_labels[:-1]
                    self.call_update_drawing_fn()
            elif key == ord("r"):  # remove all drawings
                self.points = np.empty((0, 2), dtype=int)
                self.point_labels = np.empty(0, dtype=int)
                self.boxes = np.empty((0, 4), dtype=int)
                self.box_labels = np.empty(0, dtype=int)
                self.call_update_drawing_fn()

        cv2.displayOverlay(
            self.window_name, "Drawing ended. Press 'd' to go into drawing mode", 5000
        )

        if self.selected_image_idx is not None:
            return (
                self.images[self.selected_image_idx].copy(),
                self.drawn_labels,
                self._extra_ret_data,
            )
        else:
            return None

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: Any = None):
        """
        Callback function for mouse events.

        :param event: one of the cv2.MouseEventTypes:
            [EVENT_MOUSEMOVE,
            EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN, EVENT_MBUTTONDOWN,
            EVENT_LBUTTONUP, EVENT_RBUTTONUP, EVENT_MBUTTONUP,
            EVENT_LBUTTONDBLCLK, EVENT_RBUTTONDBLCLK, EVENT_MBUTTONDBLCLK,
            EVENT_MOUSEWHEEL, EVENT_MOUSEHWHEEL]
            https://docs.opencv.org/4.9.0/d0/d90/group__highgui__window__flags.html#ga927593befdddc7e7013602bca9b079b0
        :param x: The x-coordinate of the mouse event.
        :param y: The y-coordinate of the mouse event.
        :param flags: one of the cv2.MouseEventFlags:
            [EVENT_FLAG_LBUTTON, EVENT_FLAG_RBUTTON, EVENT_FLAG_MBUTTON,
            EVENT_FLAG_CTRLKEY, EVENT_FLAG_SHIFTKEY, EVENT_FLAG_ALTKEY]
            https://docs.opencv.org/4.9.0/d0/d90/group__highgui__window__flags.html#gaab4dc057947f70058c80626c9f1c25ce
        :param param: The optional user data passed by cv2.setMouseCallback().
        """
        if self._done_drawing:
            return

        self._mouse_pos = (x, y)
        self._CTRLKEY = flags & cv2.EVENT_FLAG_CTRLKEY

        if self._image is None:  # No selected image
            return

        image_bounds = np.stack([[0, 0], self._image.shape[1::-1]]).T
        if self._drawing_mode == self.DrawingMode.Box:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._in_drawing = True
                self.boxes = np.vstack([self.boxes, [x, y, x, y]])
                self.box_labels = np.hstack([
                    self.box_labels,
                    0 if self._CTRLKEY else 1,
                ])
            elif event == cv2.EVENT_LBUTTONUP:
                # Clip mouse position to be within image bounds
                self._mouse_pos = (x1, y1) = tuple(  # type: ignore
                    np.clip(self._mouse_pos, image_bounds[:, 0], image_bounds[:, 1] - 1)
                )
                self._in_drawing = False
                # Enforce boxes to be XYXY coordinates
                x0, y0 = self.boxes[-1, :2]
                self.boxes[-1] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
                self.call_update_drawing_fn()
            elif event == cv2.EVENT_MOUSEMOVE and self._in_drawing:  # for visualization
                self.boxes[-1, 2:] = self._mouse_pos
        elif self._drawing_mode == self.DrawingMode.Point:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._in_drawing = True
            elif event == cv2.EVENT_LBUTTONUP:
                self._in_drawing = False
                if not (
                    (image_bounds[:, 0] <= self._mouse_pos).all()
                    and (self._mouse_pos < image_bounds[:, 1]).all()
                ):
                    self.logger.warning("Drawn points outside image bounds, ignoring")
                    return
                self.points = np.vstack([self.points, self._mouse_pos])
                self.point_labels = np.hstack([
                    self.point_labels,
                    0 if self._CTRLKEY else 1,
                ])
                self.call_update_drawing_fn()

    @staticmethod
    def _update_drawing_fn(
        image: np.ndarray, drawn_labels: dict[str, np.ndarray], /
    ) -> tuple[np.ndarray, Any]:
        """Callback function with drawn points/boxes during drawing"""
        return image, None

    def call_update_drawing_fn(self) -> None:
        """Call self._update_drawing_fn with current drawings
        This happens when the user adds/removes any drawing.
        """
        if self.selected_image_idx is None:
            self.logger.error("No image selected, please select image first")
            return

        cv2.setWindowTitle(self.window_name, "Busy updating drawings...")
        self._image, self._extra_ret_data = self._update_drawing_fn(
            self.images[self.selected_image_idx].copy(), self.drawn_labels
        )
        cv2.setWindowTitle(self.window_name, self.window_name)

    @property
    def drawn_labels(self) -> dict[str, np.ndarray]:
        return {
            "points": self.points,
            "point_labels": self.point_labels,
            "boxes": self.boxes,
            "box_labels": self.box_labels,
        }

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.window_name}>"
