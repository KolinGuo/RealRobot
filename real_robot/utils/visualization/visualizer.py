from typing import Dict, List, Union, Any, Optional
import time

import numpy as np
import open3d as o3d

from .cv2_visualizer import CV2Visualizer
from .o3d_gui_visualizer import O3DGUIVisualizer
from .utils import draw_mask, colorize_mask
from ..multiprocessing import ctx, SharedObject, start_and_wait_for_process
from ..lib3d import np2pcd
from ..logger import get_logger
try:
    from pynput.keyboard import Key, KeyCode, Listener
except ImportError as e:
    get_logger("visualizer.py").warning(f"ImportError: {e}")


pause_render = False


def _on_key_press(key):
    if key == KeyCode.from_char('p'):
        global pause_render
        pause_render = True


class Visualizer:
    def __init__(self, *,
                 run_as_process=False, stream_camera=False, stream_robot=False):
        """Visualizer managing CV2Visualizer and O3DGUIVisualizer
        :param run_as_process: whether to run CV2Visualizer and O3DGUIVisualizer
                               as separate processes.
        :param stream_camera: whether to redraw camera stream when a new frame arrives
        :param stream_robot: whether to update robot mesh when a new robot state arrives
        """
        if run_as_process:
            self.cv2vis_proc = ctx.Process(
                target=CV2Visualizer, name="CV2Visualizer", args=(),
                kwargs=dict(
                    run_as_process=True,
                    stream_camera=stream_camera,
                )
            )
            start_and_wait_for_process(self.cv2vis_proc, timeout=30)

            self.o3dvis_proc = ctx.Process(
                target=O3DGUIVisualizer, name="O3DGUIVisualizer", args=(),
                kwargs=dict(
                    run_as_process=True,
                    stream_camera=stream_camera,
                    stream_robot=stream_robot,
                )
            )
            start_and_wait_for_process(self.o3dvis_proc, timeout=30)

            # Create SharedObject to control visualizer and feed data
            self.so_cv2vis_joined = SharedObject("join_viscv2")
            self.so_o3dvis_joined = SharedObject("join_viso3d")
            self.so_draw = SharedObject("draw_vis")
            self.so_reset = SharedObject("reset_vis")
            self.so_data_dict = {}  # {so_data_name: SharedObject(so_data_name)}
        else:
            self.cv2vis = CV2Visualizer()
            self.o3dvis = O3DGUIVisualizer()

            self.key_listener = Listener(on_press=_on_key_press)
            self.key_listener.start()

        self.run_as_process = run_as_process

    def reset(self, obs_dict={}):
        if self.run_as_process:
            self.so_reset.trigger()  # triggers reset
            time.sleep(1e-3)  # sleep a while to wait for visualizer to finish reset
            # Unlink created SharedObject
            for so_data in self.so_data_dict.values():
                so_data.unlink()
            self.so_data_dict = {}
        else:
            self.cv2vis.clear_image()
            self.o3dvis.clear_geometries()

        if len(obs_dict) > 0:
            self.show_obs(obs_dict)
            self.render()

    def _show_obs_async(self, obs_dict: Dict[str, Union[SharedObject._object_types]]):
        """Render observations
        :param obs_dict: dict, {so_data_name: obs_data}
                         See CV2Visualizer.__init__.__doc__ and
                             O3DGUIVisualizer.__init__.__doc__
                         for acceptable so_data_name
        """
        for so_data_name, data in obs_dict.items():
            if so_data_name not in self.so_data_dict:
                self.so_data_dict[so_data_name] = SharedObject(so_data_name, data=data)
            else:
                self.so_data_dict[so_data_name].assign(data)

    def _show_obs_sync(self, *, camera_names: Optional[List[str]] = None,
                       **obs_dict: Dict[str, Union[np.ndarray,
                                                   List[np.ndarray],
                                                   o3d.geometry.Geometry]]):
        """Render observations
        :param camera_names: camera names if obs_data are from multiple cameras
                             The order should match with order of obs_data
        :param obs_dict: dict, {obs_name: obs_data}
                         obs_name must contain one of
                         ['color_image', 'depth_image', 'mask',
                          'xyz_image', 'points', 'pts', 'bbox', 'mesh']
                         obs_data can be np.ndarray from one camera or
                         a list of np.ndarray from multiple cameras
        """
        images = {}  # {name_with_group: image}
        o3d_geometries = {}  # {name_with_group: geometry}

        color_images = []
        for obs_name, obs_data in obs_dict.items():
            if not isinstance(obs_data, list):
                obs_name, obs_data = [obs_name], [obs_data]
            else:
                # Prepend camera_name to obs_name
                obs_name = [f"{cam_name}/{obs_name}" for cam_name in camera_names]

            for i, (name, obs) in enumerate(zip(obs_name, obs_data)):
                if "color_image" in name:  # color image
                    images[name] = obs
                    color_images.append(obs)
                elif "depth_image" in name:  # depth image
                    images[name] = obs
                elif "mask" in name:  # mask images
                    if len(color_images) > 0:
                        images[name+"_overlay"] = draw_mask(color_images[i], obs)
                    images[name] = colorize_mask(obs)
                elif "xyz_image" in name:  # xyz_image
                    colors = None
                    if len(color_images) > 0:
                        colors = color_images[i].reshape(-1, 3) / 255.0
                    o3d_geometries[name] = np2pcd(obs.reshape(-1, 3), colors)
                elif "points" in name or "pts" in name:  # point clouds
                    o3d_geometries[name] = np2pcd(obs.reshape(-1, 3))
                elif "bbox" in name:  # bounding boxes
                    assert isinstance(obs,
                                      (o3d.geometry.AxisAlignedBoundingBox,
                                       o3d.geometry.OrientedBoundingBox)), \
                        f"Not a bbox: {type(obs) = }"
                    o3d_geometries[name] = obs
                elif "mesh" in name:  # TriangleMesh
                    assert isinstance(obs, o3d.geometry.TriangleMesh), \
                        f"Not a mesh: {type(obs) = }"
                    o3d_geometries[name] = obs
                else:
                    raise NotImplementedError(f"Unknown object {name = }")

        # Sort images based on key
        self.cv2vis.show_images([img for _, img in sorted(images.items())])
        self.o3dvis.add_geometries(o3d_geometries)

    def show_obs(self, obs_dict: Dict[str, Any]) -> None:
        """Render observations

        :param obs_dict: dict, {so_data_name: obs_data}
                         See CV2Visualizer.__init__.__doc__ and
                             O3DGUIVisualizer.__init__.__doc__
                         for acceptable so_data_name
        """
        if self.run_as_process:
            self._show_obs_async(obs_dict)
        else:
            self._show_obs_sync(**obs_dict)

    def render(self):
        # TODO: What does pause_render do for run_as_process?
        if self.run_as_process:
            self.so_draw.trigger()
        else:
            global pause_render
            if pause_render:
                self.o3dvis.toggle_pause(True)
                pause_render = False

            # Render visualizer
            # self.o3d_vis.render() returns only when not paused or single_step
            self.cv2vis.render()
            self.o3dvis.render(render_step_fn=self.cv2vis.render)

    def close(self):
        """Close visualizers"""
        if self.run_as_process:
            self.so_cv2vis_joined.trigger()
            self.cv2vis_proc.join()
            self.so_o3dvis_joined.trigger()
            self.o3dvis_proc.join()

            # Unlink created SharedObject
            for so_data in self.so_data_dict.values():
                so_data.unlink()
            self.so_draw.unlink()
            self.so_reset.unlink()
        else:
            self.cv2vis.close()
            self.o3dvis.close()

    def __del__(self):
        if not self.run_as_process:
            self.key_listener.stop()
        self.close()
