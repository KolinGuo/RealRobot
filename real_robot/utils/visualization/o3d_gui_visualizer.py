import os
import glob
import time
import platform
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import List, Union, Dict, Optional

import numpy as np
from urchin import URDF
import open3d as o3d
from open3d.utility import Vector3dVector
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from .gripper_utils import XArmGripper
from ..camera import T_GL_CV, T_CV_GL, T_ROS_GL, T_ROS_CV, depth2xyz
from ..multiprocessing import (
    SharedObject, SharedObjectDefaultDict, signal_process_ready
)
from ..logger import get_logger

isMacOS = (platform.system() == "Darwin")

_o3d_geometry_type = Union[o3d.geometry.Geometry3D,
                           o3d.t.geometry.Geometry,
                           rendering.TriangleMeshModel]


class O3DGeometryDefaultDict(dict):
    """This defaultdict helps to store open3d geometries by name (only known at runtime)
    so we don't need to frequently create them.
    Used in O3DGUIVisualizer.run_as_process()
    """

    def __missing__(self, name: str) -> _o3d_geometry_type:
        if name.endswith("_pcd"):
            geometry = self[name] = o3d.geometry.PointCloud()
        elif name.endswith("_frame"):
            geometry = self[name] = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5
            )
        elif name.endswith("_bbox"):
            geometry = self[name] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=[-1e-3]*3, max_bound=[1e-3]*3
            )
        else:
            raise ValueError(f"Unknown {name=}")
        return geometry


class Settings:
    UNLIT = "defaultUnlit"
    UNLIT_LINE = "unlitLine"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    GROUND_PLANE = {
        "XY": rendering.Scene.GroundPlane.XY,
        "XZ": rendering.Scene.GroundPlane.XZ,
        "YZ": rendering.Scene.GroundPlane.YZ,
    }

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.show_camera = True
        self.show_ground = False
        self.ground_plane = self.GROUND_PLANE["XY"]
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.UNLIT_LINE: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.UNLIT_LINE].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT_LINE].shader = Settings.UNLIT_LINE
        self._materials[Settings.UNLIT_LINE].line_width = 3.0
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns
        # a reference, not a copy, so if we change the property of a material,
        # then switch to another one, then come back,
        # the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

    def apply_material_prefab(self, name: str):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name: str):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)


@dataclass
class GeometryNode:
    """A geometry node in gui.TreeView() for easy parent-node lookup
    NOTE: geometry group node with no children is not allowed
    :ivar name: full geometry name containing all nested group names starting
                from root, separated by '/'.
                That is, name is unique based on its path from root node.
    :ivar id: item id in gui.TreeView
    :ivar parent: parent GeometryNode instance.
    :ivar children: list of children GeometryNode instances.
    :ivar cell: gui.Widget for changing checkbox
    :ivar mat_shader_index: _shader item index
    :ivar mat_prefab_text: _material_prefab item text
    :ivar mat_color: material color
    :ivar mat_point_size: material point size
    """
    name: str
    id: int = -1
    parent: 'GeometryNode' = None
    children: List['GeometryNode'] = field(default_factory=list)
    cell: gui.Widget = None
    # Material settings values
    mat_changed: bool = False
    mat_shader_index: int = 1  # Settings.UNLIT
    mat_prefab_text: str = Settings.DEFAULT_MATERIAL_NAME
    mat_color: gui.Color = gui.Color(0.9, 0.9, 0.9, 1.0)
    mat_point_size: float = 3.0

    @property
    def display_name(self) -> str:
        """Returns the display name in geometries_panel"""
        # self.parent can be root node
        if self.parent is not None and self.parent.parent is not None:
            # NOTE: removeprefix is not in python3.8
            # return self.name.removeprefix(self.parent.name + '/')
            return self.name[len(prefix):] if self.name.startswith(
                (prefix := self.parent.name + '/')) else self.name
        return self.name

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__}: '{self.name}' (id={self.id}), "
                f"parent='{self.parent.name}', "
                f"children={[f'{c.name}' for c in self.children]}>")


class O3DGUIVisualizer:
    """Open3D GUI Application based on
    http://www.open3d.org/docs/latest/python_api/open3d.visualization.gui.html
    References:
    https://github.com/isl-org/Open3D/blob/master/cpp/open3d/visualization/visualizer/O3DVisualizer.cpp
    https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/all_widgets.py
    https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/add_geometry.py
    https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/vis_gui.py
    https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/mouse_and_point_coord.py
    https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/
    """

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    MATERIAL_NAMES = ["Standard (Lit)", "Unlit", "Unlit Line", "Normal Map", "Depth"]
    MATERIAL_SHADERS = [
        Settings.LIT, Settings.UNLIT, Settings.UNLIT_LINE,
        Settings.NORMALS, Settings.DEPTH
    ]

    def __init__(self, window_name="Point Clouds", window_size=(1920, 1080), *,
                 run_as_process=False, stream_camera=False, stream_robot=False):
        """
        :param window_name: window name
        :param window_size: (width, height)
        :param run_as_process: whether to run O3DGUIVisualizer as a separate process.
          If True, O3DGUIVisualizer needs to be created as a `mp.Process`.
          Several SharedObject are mounted to control O3DGUIVisualizer and fetch data:
              Only "join_viso3d" is created by this process.
            * "join_viso3d": If triggered, the O3DGUIVisualizer process is joined.
            * "draw_vis": If triggered, redraw the images.
            * "reset_vis": If triggered, call self.clear_geometries().
            * "sync_rs_<device_uid>": If triggered, capture from RSDevice.
            * "sync_xarm7_<robot_uid>": If triggered, fetch joint states from robot.
            Corresponding data have the same prefix (implemented as sorting)
            * Data unique to O3DGUIVisualizer have prefix "viso3d_<data_uid>_"
            * Data shared with CV2Visualizer have prefix "vis_<data_uid>_"
            * RSDevice camera feeds have prefix "rs_<device_uid>_"
              * "rs_<device_uid>_color": rgb color image, [H, W, 3] np.uint8 np.ndarray
              * "rs_<device_uid>_depth": depth image, [H, W] np.uint16 np.ndarray
              * "rs_<device_uid>_intr": intrinsic matrix, [3, 3] np.float64 np.ndarray
              * "rs_<device_uid>_pose": camera pose in world frame (ROS convention)
                                        forward(x), left(y) and up(z), sapien.Pose
            * xArm7 state feeds have prefix "xarm7_<robot_uid>_"
              * "xarm7_<robot_uid>_urdf_path": xArm7 URDF path, str
              * "xarm7_<robot_uid>_qpos": xArm7 joint angles, [8,] np.float32 np.ndarray
            Grouping can be specified with '|' in <data_uid> (e.g., "front_camera|cube")
              <device_uid>, <robot_uid>, and <data_uid> must not be the same
          Acceptable <data_uid> suffixes with its acceptable data member suffixes:
            (data members in brackets are optional)
          * "_camera": PointCloud capture: ("_depth", "_intr", ["_color", "_pose"])
                       For rs camera stream, "_pose" is in ROS convention
                       For camera capture, "_pose" is in OpenCV convention
          * "_pcd": PointCloud: ("_pts", ["_color", "_pose"]),
                                ("_xyzimg", ["_color", "_pose"])
          * "*": Robot mesh: ("_urdf_path", "_qpos")
          * "_frame": Coordinate frame: ("_pose",)
          * "_bbox": bounding box pts: ("_bounds", ["_pose"])
          * "*": Robot gripper mesh / lineset: ("_gposes", "_gscores", "_gqvals")
                 E.g., "viso3d_CGN_grasps|obj1_gposes"
                 Also mounts "robot_gripper_urdf_path" to load gripper URDF.

          Acceptable visualization SharedObject data formats:
          * "_color": RGB color images, [H, W, 3] np.uint8 np.ndarray
                      or pts color, [N, 3] np.uint8 np.ndarray
          * "_depth": Depth images, [H, W] or [H, W, 1] np.uint16/np.floating np.ndarray
          * "_intr": camera intrinsic matrix, [3, 3] np.floating np.ndarray
          * "_pose": object / camera pose, sapien.Pose
          * "_xyzimg": xyz image, [H, W, 3] np.floating np.ndarray
          * "_pts": points, [N, 3] np.floating np.ndarray
          * "_urdf_path": robot URDF path, str
          * "_qpos": robot qpos, [ndof,] np.float32 np.ndarray
          * "_bounds": AxisAlignedBoundingBox bounds, (xyz_min, xyz_max),
                       [2, 3] np.floating np.ndarray
          * "_gposes": Gripper poses in world frame, [N, 4, 4] np.floating np.ndarray
          * "_gscores": Gripper pose confidence scores, [N,] np.floating np.ndarray
          * "_gqvals": Gripper joint values, [N,] np.floating np.ndarray
        :param stream_camera: whether to redraw camera stream when a new frame arrives
        :param stream_robot: whether to update robot mesh when a new robot state arrives
        """
        self.logger = get_logger("O3DGUIVisualizer")

        # We need to initialize the application, which finds the necessary shaders
        # for rendering and prepares the cross-platform window abstraction.
        gui.Application.instance.initialize()

        self.window_name = window_name
        self.window = gui.Application.instance.create_window(
            window_name, *window_size
        )
        self.paused = False
        self.single_step = False
        self.not_closed = True

        self.camera_poses = {"default": np.eye(4, dtype=np.float32)}
        # Internally, geometry_name and geometry_group_name contains all nested
        # group names starting from root, separated by '/'.
        # That is, names are unique based on its path from root node.
        self.geometries = {}  # {geometry_name: GeometryNode}
        self.geometry_groups = {}  # {geometry_group_name: GeometryNode}
        self.id_to_geometry_nodes = {}  # {id: GeometryNode}

        self.construct_gui()

        # Picked points
        self.picked_pts = []  # [(x, y, z)]
        self.picked_pts_pcd_name = "__picked_points__"
        self.picked_pts_pcd = o3d.geometry.PointCloud()
        self.picked_pts_pcd_mat = rendering.MaterialRecord()
        self.picked_pts_pcd_mat.base_color = [0.9, 0.9, 0.9, 1.0]
        self.picked_pts_pcd_mat.shader = Settings.UNLIT
        self.picked_pts_pcd_mat.point_size = int(3 * 2)
        self._scene.scene.add_geometry(self.picked_pts_pcd_name,
                                       self.picked_pts_pcd,
                                       self.picked_pts_pcd_mat)

        # For computing update fps
        self.last_timestamp_ns = time.time_ns()

        if run_as_process:
            self.stream_camera = stream_camera
            self.stream_robot = stream_robot
            self.run_as_process()

    def construct_gui(self):
        """Construct the GUI visualizer"""
        self.settings = Settings()
        resource_path = gui.Application.instance.resource_path
        self.settings.new_ibl_name = resource_path + "/" + self.DEFAULT_IBL
        w = self.window  # for more concise code

        # 3D scene
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.set_on_sun_direction_changed(self._on_sun_dir)

        # ---- Settings panel ----
        # Rather than specifying sizes in pixels, which may vary in size based
        # on the monitor, especially on macOS which has 220 dpi monitors, use
        # the em-size. This way sizings will be proportional to the font size,
        # which will create a more visually consistent size across platforms.
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        # Widgets are laid out in layouts: gui.Horiz, gui.Vert, gui.VGrid, and
        # gui.CollapsableVert. By nesting the layouts we can achieve complex
        # designs. Usually we use a vertical layout as the topmost widget,
        # since widgets tend to be organized from top to bottom.
        # Within that, we usually have a series of horizontal layouts for each
        # row. All layouts take a spacing parameter, which is the spacing
        # between items in the widget, and a margins parameter, which specifies
        # the spacing of the left, top, right, bottom margins. (This acts like
        # the 'padding' property in CSS.)
        self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em,
                                                       0.25 * em, 0.25 * em))

        # -- Render controls widget --
        self._paused_checkbox = gui.Checkbox("Pause")
        self._paused_checkbox.set_on_checked(self.toggle_pause)
        self._single_step_button = gui.Button("Single Step")
        self._single_step_button.set_on_clicked(self._on_single_step)
        self._single_step_button.horizontal_padding_em = 0.5
        self._single_step_button.vertical_padding_em = 0
        self._render_info = gui.Label("")
        h = gui.Horiz(0.25 * em, gui.Margins(0.25 * em, 0.25 * em,
                                             0.25 * em, 0.25 * em))
        h.add_child(self._paused_checkbox)
        h.add_child(self._single_step_button)
        h.add_child(self._render_info)
        h.add_stretch()
        self._settings_panel.add_child(h)
        self._fps_label = gui.Label("FPS: NaN      ")
        h = gui.Horiz(0.25 * em, gui.Margins(0.25 * em, 0.25 * em,
                                             0.25 * em, 0.25 * em))
        h.add_child(self._fps_label)
        h.add_stretch()
        self._settings_panel.add_child(h)

        # -- View controls widget --
        # Create a collapsible vertical widget, which takes up enough vertical
        # space for all its children when open, but only enough for text when
        # closed. This is useful for property pages, so the user can hide sets
        # of properties they rarely use.
        view_ctrls = gui.CollapsableVert("View Controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        # Mouse controls
        def _create_mouse_control_button(
            name: str, mode: gui.SceneWidget.Controls
        ) -> gui.Button:
            button = gui.Button(name)
            button.horizontal_padding_em = 0.5
            button.vertical_padding_em = 0
            button.toggleable = True
            button.set_on_clicked(partial(self._set_mouse_mode, mode))
            self._mouse_control_buttons[mode] = button
            return button

        self._mouse_control_buttons = {}
        view_ctrls.add_child(gui.Label("Mouse Controls"))
        # We want two rows of buttons, so make two horizontal layouts. We also
        # want the buttons centered, which we can do be putting a stretch item
        # as the first and last item. Stretch items take up as much space as
        # possible, and since there are two, they will each take half the extra
        # space, thus centering the buttons.
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(_create_mouse_control_button(
            "Arcball", gui.SceneWidget.Controls.ROTATE_CAMERA
        ))
        h.add_child(_create_mouse_control_button(
            "Fly", gui.SceneWidget.Controls.FLY
        ))
        h.add_child(_create_mouse_control_button(
            "Model", gui.SceneWidget.Controls.ROTATE_MODEL
        ))
        h.add_stretch()
        view_ctrls.add_child(h)
        h = gui.Horiz(0.25 * em)  # row 2
        h.add_stretch()
        h.add_child(_create_mouse_control_button(
            "Sun", gui.SceneWidget.Controls.ROTATE_SUN
        ))
        h.add_child(_create_mouse_control_button(
            "Environment", gui.SceneWidget.Controls.ROTATE_IBL
        ))
        h.add_stretch()
        view_ctrls.add_child(h)
        self._set_mouse_mode(gui.SceneWidget.Controls.ROTATE_CAMERA)

        # Render camera pose
        self._camera_list = gui.Combobox()
        self._camera_list.add_item("default")
        self._camera_list.set_on_selection_changed(self._on_camera_list)
        self._camera_list.tooltip = "Set the rendering camera to stored camera poses"
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Cameras"))
        grid.add_child(self._camera_list)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(view_ctrls)

        # -- Scene controls widget --
        scene_ctrls = gui.CollapsableVert("Scene", 0.25 * em,
                                          gui.Margins(em, 0, 0, 0))
        # Coordinate axes and skybox
        self._show_axes = gui.Checkbox("Show axis")
        self._show_axes.set_on_checked(self._on_show_axes)
        self._show_camera = gui.Checkbox("Show camera")
        self._show_camera.set_on_checked(self._on_show_camera)
        h = gui.Horiz(em)
        h.add_child(self._show_axes)
        h.add_child(self._show_camera)
        scene_ctrls.add_child(h)
        # Ground plane
        self._show_ground = gui.Checkbox("Show ground")
        self._show_ground.set_on_checked(self._on_show_ground)
        self._ground_plane = gui.Combobox()
        self._ground_plane.add_item("XY")
        self._ground_plane.add_item("XZ")
        self._ground_plane.add_item("YZ")
        self._ground_plane.set_on_selection_changed(self._on_ground_plane)
        self._show_skybox = gui.Checkbox("Show skybox")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        h = gui.Horiz(em)
        h.add_child(self._show_ground)
        h.add_child(self._ground_plane)
        h.add_child(self._show_skybox)
        scene_ctrls.add_child(h)

        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        self._profiles = gui.Combobox()
        for name in sorted(Settings.LIGHTING_PROFILES.keys()):
            self._profiles.add_item(name)
        self._profiles.add_item(Settings.CUSTOM_PROFILE_NAME)
        self._profiles.set_on_selection_changed(self._on_lighting_profile)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("BG Color"))
        grid.add_child(self._bg_color)
        grid.add_child(gui.Label("Lighting"))
        grid.add_child(self._profiles)
        scene_ctrls.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(scene_ctrls)

        # -- Advanced lighting controls widget --
        advanced = gui.CollapsableVert("Advanced Lighting", 0,
                                       gui.Margins(em, 0, 0, 0))
        advanced.set_is_open(False)

        self._use_ibl = gui.Checkbox("HDR map")
        self._use_ibl.set_on_checked(self._on_use_ibl)
        self._use_sun = gui.Checkbox("Sun")
        self._use_sun.set_on_checked(self._on_use_sun)
        advanced.add_child(gui.Label("Light sources"))
        h = gui.Horiz(em)
        h.add_child(self._use_ibl)
        h.add_child(self._use_sun)
        advanced.add_child(h)

        self._ibl_map = gui.Combobox()
        for ibl in glob.glob(gui.Application.instance.resource_path + "/*_ibl.ktx"):
            self._ibl_map.add_item(os.path.basename(ibl[:-8]))
        self._ibl_map.selected_text = self.DEFAULT_IBL
        self._ibl_map.set_on_selection_changed(self._on_new_ibl)
        self._ibl_intensity = gui.Slider(gui.Slider.INT)
        self._ibl_intensity.set_limits(0, 200000)
        self._ibl_intensity.set_on_value_changed(self._on_ibl_intensity)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("HDR map"))
        grid.add_child(self._ibl_map)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._ibl_intensity)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Environment"))
        advanced.add_child(grid)

        self._sun_intensity = gui.Slider(gui.Slider.INT)
        self._sun_intensity.set_limits(0, 200000)
        self._sun_intensity.set_on_value_changed(self._on_sun_intensity)
        self._sun_dir = gui.VectorEdit()
        self._sun_dir.set_on_value_changed(self._on_sun_dir)
        self._sun_color = gui.ColorEdit()
        self._sun_color.set_on_value_changed(self._on_sun_color)
        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Intensity"))
        grid.add_child(self._sun_intensity)
        grid.add_child(gui.Label("Direction"))
        grid.add_child(self._sun_dir)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._sun_color)
        advanced.add_fixed(separation_height)
        advanced.add_child(gui.Label("Sun (Directional light)"))
        advanced.add_child(grid)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(advanced)

        # -- Material controls widget --
        material_settings = gui.CollapsableVert("Material Settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        material_settings.set_is_open(False)

        self._shader = gui.Combobox()
        self._shader.add_item(self.MATERIAL_NAMES[0])
        self._shader.add_item(self.MATERIAL_NAMES[1])
        self._shader.add_item(self.MATERIAL_NAMES[2])
        self._shader.add_item(self.MATERIAL_NAMES[3])
        self._shader.add_item(self.MATERIAL_NAMES[4])
        self._shader.set_on_selection_changed(self._on_shader)
        self._shader.selected_index = GeometryNode.mat_shader_index
        self._material_prefab = gui.Combobox()
        for prefab_name in sorted(Settings.PREFAB.keys()):
            self._material_prefab.add_item(prefab_name)
        self._material_prefab.selected_text = Settings.DEFAULT_MATERIAL_NAME
        self._material_prefab.set_on_selection_changed(self._on_material_prefab)
        self._material_color = gui.ColorEdit()
        self._material_color.set_on_value_changed(self._on_material_color)
        self._material_color.color_value = gui.Color(*self.settings.material.base_color)
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 30)
        self._point_size.set_on_value_changed(self._on_point_size)
        self._point_size.double_value = self.settings.material.point_size

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Type"))
        grid.add_child(self._shader)
        grid.add_child(gui.Label("Material"))
        grid.add_child(self._material_prefab)
        grid.add_child(gui.Label("Color"))
        grid.add_child(self._material_color)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        material_settings.add_child(grid)

        h = gui.Horiz(0.25 * em)
        self._material_reset_button = gui.Button("Reset")
        self._material_reset_button.set_on_clicked(self._on_material_reset)
        self._material_reset_button.horizontal_padding_em = 0.5
        self._material_reset_button.vertical_padding_em = 0
        h.add_child(self._material_reset_button)
        h.add_stretch()
        material_settings.add_child(h)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)

        # -- Geometries widget --
        geometries_panel = gui.CollapsableVert("Geometries", 0.25 * em,
                                               gui.Margins(em, 0, 0, 0))
        self._geometry_tree = gui.TreeView()
        self._geometry_tree.can_select_items_with_children = True
        self._geometry_tree.set_on_selection_changed(self._on_geometry_tree)
        # Add root node for concise code and operations on all geometries
        self.root_geometry_node = self._create_geometry_node("World")
        self._geometry_tree.selected_item = self.root_geometry_node.id
        h = gui.Horiz(0.25 * em)
        self._geometry_remove_button = gui.Button("Remove")
        self._geometry_remove_button.set_on_clicked(self._on_geometry_remove)
        self._geometry_remove_button.horizontal_padding_em = 0.5
        self._geometry_remove_button.vertical_padding_em = 0
        h.add_child(self._geometry_remove_button)
        h.add_stretch()

        geometries_panel.add_child(self._geometry_tree)
        geometries_panel.add_child(h)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(geometries_panel)
        # ----

        # Normally our user interface can be children of all one layout
        # (usually a vertical layout), which is then the only child of the
        # window. In our case we want the scene to take up all the space and
        # the settings panel to go above it. We can do this custom layout by
        # providing an on_layout callback. The on_layout callback should set
        # the frame (position + size) of every child correctly.
        # After the callback is done the window will layout the grandchildren.
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", self.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", self.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", self.MENU_OPEN)
            file_menu.add_item("Export Current Image...", self.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", self.MENU_QUIT)
            panels_menu = gui.Menu()
            panels_menu.add_item("Settings", self.MENU_SHOW_SETTINGS)
            panels_menu.set_checked(self.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", self.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running
                # application (in our case, probably "Python"), regardless of
                # what we call it. This is the application menu, and it is
                # where the About..., Preferences..., and Quit menu items
                # typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Panels", panels_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Panels", panels_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(self.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(self.MENU_EXPORT,
                                     self._on_menu_export)
        w.set_on_menu_item_activated(self.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(self.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(self.MENU_ABOUT, self._on_menu_about)
        # ----

        # ---- Floating info widget for point coordinate ----
        self._point_coord_info = gui.Label("")
        self._point_coord_info.visible = False

        w.add_child(self._point_coord_info)

        # ---- MouseEvent ----
        self._scene.set_on_mouse(self._on_scene_mouse_event)

        self._apply_settings()

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)
        for camera_name in self.camera_poses:
            if camera_name != "default":
                self._scene.scene.show_geometry(f"{camera_name}_lineset",
                                                self.settings.show_camera)
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._scene.scene.show_ground_plane(self.settings.show_ground,
                                            self.settings.ground_plane)
        if self.settings.new_ibl_name is not None:
            self._scene.scene.scene.set_indirect_light(
                self.settings.new_ibl_name)
            # Clear new_ibl_name, so we don't keep reloading this image every
            # time the settings are applied.
            self.settings.new_ibl_name = None
        self._scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self._scene.scene.scene.set_indirect_light_intensity(
            self.settings.ibl_intensity)
        sun_color = [
            self.settings.sun_color.red, self.settings.sun_color.green,
            self.settings.sun_color.blue
        ]
        self._scene.scene.scene.set_sun_light(self.settings.sun_dir, sun_color,
                                              self.settings.sun_intensity)
        self._scene.scene.scene.enable_sun_light(self.settings.use_sun)

        if self.settings.apply_material:
            self._update_selected_geometry_material()  # update only selected
            self.settings.apply_material = False

        self._bg_color.color_value = self.settings.bg_color
        self._show_skybox.checked = self.settings.show_skybox
        self._show_axes.checked = self.settings.show_axes
        self._show_camera.checked = self.settings.show_camera
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c
        self._point_size.double_value = self.settings.material.point_size

    def _update_selected_geometry_material(self, reset=False):
        """Update selected geometries using self.settings.material
        Update material settings values in GeometryNode
        """
        # Selected node can be root_geometry_node
        unvisited_nodes = [
            self.id_to_geometry_nodes[self._geometry_tree.selected_item]
        ]
        while len(unvisited_nodes) > 0:
            node = unvisited_nodes.pop()

            # Update material settings values in GeometryNode
            node.mat_changed = not reset
            node.mat_shader_index = self._shader.selected_index
            node.mat_prefab_text = self._material_prefab.selected_text
            color = self._material_color.color_value
            node.mat_color = gui.Color(color.red, color.green, color.blue,
                                       color.alpha)  # create a copy
            node.mat_point_size = self._point_size.double_value

            # No geometry corresponds to group nodes, so no update is needed
            if len(node.children) > 0 or node is self.root_geometry_node:
                unvisited_nodes.extend(node.children)
            else:  # geometry node
                # Update geometry material using self.settings.material
                self._scene.scene.modify_geometry_material(
                    node.name, self.settings.material
                )

    def _on_layout(self, layout_context: gui.LayoutContext):
        """Sets a floating settings panel"""
        # The on_layout callback should set the frame (position + size)
        # of every child correctly. After the callback is done the window
        # will layout the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        # ---- Settings panel ----
        width = 20 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y,
                                              width, height)
        # ---- Floating info widget for point coordinate ----
        pref = self._point_coord_info.calc_preferred_size(
            layout_context, gui.Widget.Constraints()
        )
        self._point_coord_info.frame = gui.Rect(
            r.x, r.get_bottom() - pref.height, pref.width, pref.height
        )

    # ---------------------------------------------------------------------- #
    # Render controls callbacks and methods
    # ---------------------------------------------------------------------- #
    def toggle_pause(self, paused: bool):
        # Update _paused_checkbox if called as a method
        self._paused_checkbox.checked = self.paused = paused

    def _on_single_step(self):
        """Callback function when _single_step_button is clicked"""
        self.single_step = True

    def update_render_info(self, text: str, color=[1.0, 1.0, 1.0]):
        self._render_info.text = text
        self._render_info.text_color = gui.Color(*color)
        self.window.post_redraw()  # force redraw

    # ---------------------------------------------------------------------- #
    # View controls callbacks
    # ---------------------------------------------------------------------- #
    def _set_mouse_mode(self, mode: gui.SceneWidget.Controls):
        for button in self._mouse_control_buttons.values():
            button.is_on = False
        self._mouse_control_buttons[mode].is_on = True
        self._scene.set_view_controls(mode)

    def _on_camera_list(self, name: str, index: int):
        def transform_to_lookat(T: np.ndarray):
            """Convert a 4x4 transformation matrix to look_at parameters:
            Camera frame is right(x), up(y), backwards(z)
            :return center: the point the camera is looking at
            :return eye: camera position
            :return up: Y axis of the camera frame
            """
            up = T[:3, 1]
            eye = T[:3, -1]
            center = eye - T[:3, 2]
            return center, eye, up

        self._scene.look_at(*transform_to_lookat(self.camera_poses[name]))

    # ---------------------------------------------------------------------- #
    # Scene controls callbacks
    # ---------------------------------------------------------------------- #
    def _on_show_axes(self, show: bool):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_show_camera(self, show: bool):
        self.settings.show_camera = show
        self._apply_settings()

    def _on_show_skybox(self, show: bool):
        self.settings.show_skybox = show
        self._apply_settings()

    def _on_show_ground(self, show: bool):
        self.settings.show_ground = show
        self._apply_settings()

    def _on_ground_plane(self, name: str, index: int):
        self.settings.ground_plane = Settings.GROUND_PLANE[name]
        self._apply_settings()

    def _on_bg_color(self, new_color: gui.Color):
        self.settings.bg_color = new_color
        self._apply_settings()

    # ---------------------------------------------------------------------- #
    # Advanced lighting controls callbacks
    # ---------------------------------------------------------------------- #
    def _on_use_ibl(self, use: bool):
        self.settings.use_ibl = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_use_sun(self, use: bool):
        self.settings.use_sun = use
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_lighting_profile(self, name: str, index: int):
        if name != Settings.CUSTOM_PROFILE_NAME:
            self.settings.apply_lighting_profile(name)
            self._apply_settings()

    def _on_new_ibl(self, name: str, index: int):
        self.settings.new_ibl_name = (gui.Application.instance.resource_path
                                      + "/" + name)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_ibl_intensity(self, intensity: float):
        self.settings.ibl_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_intensity(self, intensity: float):
        self.settings.sun_intensity = int(intensity)
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_dir(self, sun_dir: np.ndarray):
        self.settings.sun_dir = sun_dir
        self._profiles.selected_text = Settings.CUSTOM_PROFILE_NAME
        self._apply_settings()

    def _on_sun_color(self, color: gui.Color):
        self.settings.sun_color = color
        self._apply_settings()

    # ---------------------------------------------------------------------- #
    # Material controls callbacks
    # ---------------------------------------------------------------------- #
    def _on_shader(self, name: str, index: int):
        self.settings.material = self.settings._materials[
            self.MATERIAL_SHADERS[index]
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_prefab(self, name: str, index: int):
        self.settings.apply_material_prefab(name)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_color(self, color: gui.Color):
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.apply_material = True
        self._apply_settings()

    def _on_point_size(self, size: float):
        self.settings.material.point_size = int(size)
        self.settings.material.line_width = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_material_reset(self):
        """Callback function when _material_reset_button is clicked"""
        # Reset to default material settings
        self._update_material_from_node(GeometryNode)
        # Update selected geometries using self.settings.material
        self._update_selected_geometry_material(reset=True)

    def _update_material_from_node(self, node: GeometryNode):
        """Update material widget and settings.material from node
        :param node: GeometryNode
        """
        # Update material control widget
        self._shader.selected_index = shader_index = node.mat_shader_index
        self._material_prefab.selected_text = prefab = node.mat_prefab_text
        self._material_prefab.enabled = prefab_enabled = shader_index == 0
        self._material_color.color_value = color = node.mat_color
        self._point_size.double_value = point_size = node.mat_point_size

        # Update material (same as material controls callbacks)
        self.settings.material = self.settings._materials[
            self.MATERIAL_SHADERS[shader_index]
        ]
        if prefab_enabled:  # Settings.LIT
            self.settings.apply_material_prefab(prefab)
        self.settings.material.base_color = [
            color.red, color.green, color.blue, color.alpha
        ]
        self.settings.material.point_size = int(point_size)

    # ---------------------------------------------------------------------- #
    # Geometries callbacks
    # ---------------------------------------------------------------------- #
    def _on_geometry_toggle(self, show: bool, node: GeometryNode):
        """Callback function when tree cell is toggled
        If a GeometryNode is toggled on,
            toggle on all its children nodes and show their geometries, and
            toggle on all its parent nodes
        If a GeometryNode is toggled off,
            toggle off all its children nodes and hide their geometries
        :param node: toggled GeometryNode
        """
        # Toggle on/off all its children nodes and show/hide their geometries
        unvisited_nodes = [node]
        while len(unvisited_nodes) > 0:
            n = unvisited_nodes.pop()
            n.cell.checkbox.checked = show

            # group node or root node
            if len(n.children) > 0 or n is self.root_geometry_node:
                unvisited_nodes.extend(n.children)
            else:  # geometry node
                self._scene.scene.show_geometry(n.name, show)

        # If a GeometryNode is toggled on,
        #   toggle on all its parent nodes
        if show:
            while (node := node.parent) is not None:
                node.cell.checkbox.checked = True

    def _on_geometry_tree(self, new_item_id: int):
        """Callback function when a geometry in _geometry_tree is selected
        Update material control widget GUI and update self.settings.material
        """
        node = self.id_to_geometry_nodes[new_item_id]
        self._update_material_from_node(node)

    def _on_geometry_remove(self):
        """Callback function when _geometry_remove_button is clicked"""
        self._remove_geometry_node(
            self.id_to_geometry_nodes[self._geometry_tree.selected_item]
        )

    # ---------------------------------------------------------------------- #
    # Menu callbacks
    # ---------------------------------------------------------------------- #
    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename: str):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename: str):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            self.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget,
        # you can treat it similar to a Window for layout and put all the
        # widgets in a layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    # ---------------------------------------------------------------------- #
    # MouseEvent callbacks
    # ---------------------------------------------------------------------- #
    def _on_scene_mouse_event(
        self, event: gui.MouseEvent
    ) -> gui.SceneWidget.EventCallbackResult:
        """Callback function for mouse event on SceneWidget self._scene
        :param event: gui.MouseEvent
        :return res: gui.SceneWidget.EventCallbackResult, one of the following
            IGNORED : Event handler ignored the event, widget will handle event
                      normally
            HANDLED : Event handler handled the event, but widget will still
                      handle the event normally. This is useful when you are
                      augmenting base functionality
            CONSUMED : Event handler consumed the event, event handling stops,
                       widget will not handle the event. This is useful when
                       you are replacing functionality
        """
        if (event.is_button_down(gui.MouseButton.LEFT)
                and event.is_modifier_down(gui.KeyModifier.CTRL)
                and event.type == event.Type.BUTTON_DOWN):  # CTRL + LEFT Down

            def depth_callback(depth_image: o3d.geometry.Image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS)
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                    self.picked_pts = []
                else:
                    world_xyz = self._scene.scene.camera.unproject(
                        x, y, depth,
                        self._scene.frame.width, self._scene.frame.height
                    ).flatten()
                    text = "({:.3f}, {:.3f}, {:.3f})".format(*world_xyz)
                    self.picked_pts = [world_xyz]

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_label():
                    self._point_coord_info.text = text
                    self._point_coord_info.visible = (text != "")
                    # We are sizing the info label to be exactly
                    # the right size, so since the text likely changed width,
                    # we need to re-layout to set the new frame.
                    self.window.set_needs_layout()

                    # ---- Update picked_points scene geometry ----
                    self._scene.scene.remove_geometry(self.picked_pts_pcd_name)
                    # Update points and color
                    self.picked_pts_pcd.points = Vector3dVector(self.picked_pts)
                    self.picked_pts_pcd.paint_uniform_color([1.0, 0.0, 1.0])
                    # Update material point_size
                    self.picked_pts_pcd_mat.point_size = int(
                        self.settings.material.point_size * 2
                    )
                    self._scene.scene.add_geometry(self.picked_pts_pcd_name,
                                                   self.picked_pts_pcd,
                                                   self.picked_pts_pcd_mat)

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label
                )

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.SceneWidget.EventCallbackResult.HANDLED
        return gui.SceneWidget.EventCallbackResult.IGNORED

    # ---------------------------------------------------------------------- #
    # Methods
    # ---------------------------------------------------------------------- #
    def load(self, path: str, *, name: str = None):
        """Load and add a geometry from file"""
        geometry_name = Path(path).stem if name is None else name

        geometry = None
        geometry_type = o3d.io.read_file_geometry_type(path)

        mesh = None
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            mesh = o3d.io.read_triangle_model(path)
        if mesh is None:
            self.logger.debug(f"{path} appears to be a point cloud")
            cloud = None
            try:
                cloud = o3d.io.read_point_cloud(path)
            except Exception:
                pass
            if cloud is not None:
                if not cloud.has_normals():
                    cloud.estimate_normals()
                cloud.normalize_normals()
                geometry = cloud
            else:
                self.logger.error(f"Failed to read points from {path}")

        if geometry is not None or mesh is not None:
            self.add_geometry(geometry_name,
                              geometry if mesh is None else mesh)

    def add_geometry(self, name: str, geometry: _o3d_geometry_type,
                     show: bool = None) -> bool:
        """Add a geometry to scene and update the _geometries_tree
        :param name: geometry name separated by '/', str.
                     Group names are nested starting from root.
                     Geometry and geometry group with same names can coexist.
        :param geometry: Open3D geometry
        :param show: whether to show geometry after loading
        :return success: whether geometry is successfully added
        """
        name = name.split('/')
        # group_names can be [], ["g1"], ["g1", "g1/g2"]
        group_names = ['/'.join(name[:i]) for i in range(1, len(name))]
        name = '/'.join(name)
        # Get leaf node as parent_node
        parent_node = self.root_geometry_node
        new_group_idx = 0
        for i, group_name in enumerate(group_names):
            if group_name not in self.geometry_groups:
                break
            parent_node = self.geometry_groups[group_name]
            new_group_idx = i + 1
        # New groups needed to be added
        group_names = group_names[new_group_idx:]

        # Add geometry to scene
        # Remove geometry with the same name
        if name in self.geometries:
            T_world_obj = self._scene.scene.get_geometry_transform(name)
            self._scene.scene.remove_geometry(name)
        try:
            unlit_line_geometry = False
            if isinstance(geometry, rendering.TriangleMeshModel):
                self._scene.scene.add_model(name, geometry)
            elif isinstance(geometry, (o3d.geometry.LineSet,
                                       o3d.geometry.AxisAlignedBoundingBox,
                                       o3d.geometry.OrientedBoundingBox)):
                unlit_line_geometry = True
                self._scene.scene.add_geometry(
                    name, geometry, self.settings._materials[Settings.UNLIT_LINE]
                )
            elif isinstance(geometry, (o3d.geometry.Geometry3D,
                                       o3d.t.geometry.Geometry)):
                self._scene.scene.add_geometry(
                    name, geometry, self.settings.material
                )
        except Exception as e:
            self.logger.error(e)

        # NOTE: sometimes scene.add_geometry will fail with no warning/error
        # E.g., when adding an empty pointcloud: o3d.geometry.PointCloud()
        if not self._scene.scene.has_geometry(name):
            self.logger.warning(f"Failed to add geometry {name}: {geometry}")
            # Remove geometry node if exists
            # because geometry is already removed from scene
            if name in self.geometries:
                self.logger.warning(f"Removing geometry {name} from scene")
                self.remove_geometry(name)
            return False

        if name not in self.geometries:  # adding new geometry
            # Update camera pose
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
            # Store the new camera pose as default
            self.update_camera_pose(
                "default", self._scene.scene.camera.get_model_matrix()
            )

            # Update GUI
            # Add geometry group to _geometries_tree
            for group_name in group_names:
                parent_node = self._create_geometry_node(group_name,
                                                         parent_node)
                self.geometry_groups[group_name] = parent_node
            # Add geometry to _geometries_tree
            node = self._create_geometry_node(name, parent_node)
            if unlit_line_geometry:
                node.mat_shader_index = 2  # Settings.UNLIT_LINE
            self.geometries[name] = node
        else:  # changing existing geometry
            if (node := self.geometries[name]).mat_changed:  # update geometry material
                current_selected_item = self._geometry_tree.selected_item
                self._on_geometry_tree(node.id)
                # Update geometry material using self.settings.material
                self._scene.scene.modify_geometry_material(
                    node.name, self.settings.material
                )
                self._on_geometry_tree(current_selected_item)
            # Update geometry pose to previous geometry pose
            self._scene.scene.set_geometry_transform(name, T_world_obj)

        # Toggle geometry checkbox and show/hide
        self._on_geometry_toggle(
            show if show is not None else node.cell.checkbox.checked,
            self.geometries[name]
        )
        return True

    def add_geometries(self, geometry_dict: Dict[str, _o3d_geometry_type],
                       show: bool = None):
        """Add multiple geometries (allow for computing update fps)
        :param geometry_dict: dictionary with format {name: Open3D geometry}
        :param show: whether to show geometry after loading
        """
        for name, geometry in geometry_dict.items():
            self.add_geometry(name, geometry, show)

        # Compute fps
        cur_timestamp_ns = time.time_ns()
        fps = 1e9 / (cur_timestamp_ns - self.last_timestamp_ns)
        self.last_timestamp_ns = cur_timestamp_ns
        self._fps_label.text = f"FPS: {fps:6.2f}"

    def _create_geometry_node(
        self, name: str, parent_node: Optional[GeometryNode] = None
    ) -> GeometryNode:
        """Create a GeometryNode and update GUI"""
        child_node = GeometryNode(name, parent=parent_node)
        parent_id = self._geometry_tree.get_root_item()
        if parent_node is not None:
            parent_node.children.append(child_node)
            parent_id = parent_node.id

        child_node.cell = cell = gui.CheckableTextTreeCell(
            child_node.display_name, True,  # always show initially
            partial(self._on_geometry_toggle, node=child_node)
        )
        child_node.id = child_id = self._geometry_tree.add_item(
            parent_id, cell
        )
        self.id_to_geometry_nodes[child_id] = child_node
        return child_node

    def clear_geometries(self):
        """Remove all geometries"""
        self._remove_geometry_node(self.root_geometry_node)

    def remove_geometry(self, name: str):
        """Remove a geometry from scene and update the _geometries_tree
        :param name: geometry or geometry group name separated by '/', str.
                     Group names are nested starting from root.
                     Geometry and geometry group with same names can coexist.
        """
        if name in self.geometries:
            self._remove_geometry_node(self.geometries[name])
        elif name in self.geometry_groups:
            self._remove_geometry_node(self.geometry_groups[name])
        else:
            self.logger.error(f"No geometry or geometry group with {name = }")

    def _remove_geometry_node(self, node: GeometryNode):
        """Remove a GeometryNode and its children, update GUI and scene.
        NOTE: geometry group node with no children is not allowed
        :param node: geometry node or geometry group node or root_geometry_node
        """
        # Do not remove root_geometry_node
        if node is self.root_geometry_node:
            unvisited_nodes = node.children.copy()
        else:
            unvisited_nodes = [node]

        while len(unvisited_nodes) > 0:  # for all children nodes
            n = unvisited_nodes.pop()
            n.parent.children.remove(n)
            self.id_to_geometry_nodes.pop(n.id)
            # Remove an item and all its children from _geometry_tree
            self._geometry_tree.remove_item(n.id)

            if len(n.children) > 0:  # group node
                unvisited_nodes.extend(n.children)
                self.geometry_groups.pop(n.name)
            else:  # geometry node
                self._scene.scene.remove_geometry(n.name)
                self.geometries.pop(n.name)

        # If a parent group node has no children, remove it
        while (node is not self.root_geometry_node
               and (node := node.parent) is not self.root_geometry_node
               and len(node.children) == 0):
            node.parent.children.remove(node)
            self.geometry_groups.pop(node.name)
            self.id_to_geometry_nodes.pop(node.id)
            # Remove an item and all its children from _geometry_tree
            self._geometry_tree.remove_item(node.id)

        # Update _geometry_tree.selected_item
        if self._geometry_tree.selected_item not in self.id_to_geometry_nodes:
            self._geometry_tree.selected_item = item_id = list(
                self.id_to_geometry_nodes.keys()
            )[-1]
            # Update material widget
            self._on_geometry_tree(item_id)

    def export_image(self, path: str, width: int, height: int):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def set_focused_camera(self, name: str):
        """Set the camera view to look at
        :param name: existing camera name in self._camera_list
        """
        assert name in self.camera_poses, f"Camera {name=} does not exist"
        self._camera_list.selected_text = name
        self._on_camera_list(name, list(self.camera_poses.keys()).index(name))

    @staticmethod
    def get_camera_lineset(width: int, height: int,
                           K: np.ndarray, far=1.0) -> o3d.geometry.LineSet:
        """Create a camera lineset with annotated up-direction
        :param width: camera image width
        :param height: camera image height
        :param K: camera intrinsic matrix, [3, 3] np.floating np.ndarray
        :param far: camera far clipping plane to draw the lineset, unit in meters
        """
        lineset = o3d.geometry.LineSet.create_camera_visualization(
            width, height, K, np.eye(4), scale=far
        )

        points = np.asarray(lineset.points)
        p_00, p_10, p_01 = points[1], points[2], points[4]
        up_axis = p_00 - p_01
        up_gap = up_axis * 0.1
        up_axis /= np.linalg.norm(up_axis)

        lineset.points.append(p_00 + up_gap)
        lineset.points.append(p_10 + up_gap)
        lineset.points.append((p_00 + p_10) / 2 + up_gap
                              + up_axis * np.linalg.norm(p_00 - p_10) / 2 / np.sqrt(3))
        lineset.lines.append([5, 6])
        lineset.lines.append([5, 7])
        lineset.lines.append([6, 7])
        lineset.paint_uniform_color([0, 0, 1])
        return lineset

    def add_camera(self, camera_name: str, width: int, height: int, K: np.ndarray,
                   T: np.ndarray = None, fmt: str = "GL"):
        """Add a camera to view from (OpenGL convention)
        Camera frame is right(x), up(y), backwards(z)
        :param camera_name: camera unique name
        :param width: camera image width
        :param height: camera image height
        :param K: camera intrinsic matrix, [3, 3] np.floating np.ndarray
        :param T: camera pose in world frame, [4, 4] np.floating np.ndarray
        :param fmt: camera frame conventions, available: ["GL", "ROS", "CV"]
        """
        # New camera
        if camera_name not in self.camera_poses:
            self._camera_list.add_item(camera_name)
            # convert lineset to GL frame convention
            lineset = self.get_camera_lineset(width, height, K).transform(T_GL_CV)
            lineset_name = f"{camera_name}_lineset"
            self._scene.scene.add_geometry(
                lineset_name, lineset, self.settings._materials[Settings.UNLIT_LINE]
            )
            self._scene.scene.show_geometry(lineset_name, self.settings.show_camera)
            # Initial pose is np.eye(4) (OpenCV convention)
            self.camera_poses[camera_name] = T_CV_GL
            self._scene.scene.set_geometry_transform(lineset_name, T_CV_GL)

        if T is not None:
            self.update_camera_pose(camera_name, T, fmt)

    def update_camera_pose(self, camera_name, T: np.ndarray, fmt: str = "GL"):
        """Update viewing camera pose (OpenGL convention)
        Camera frame is right(x), up(y), backwards(z)
        :param camera_name: camera unique name
        :param T: camera pose in world frame, [4, 4] np.floating np.ndarray
        :param fmt: camera frame conventions, available: ["GL", "ROS", "CV"]
        """
        if fmt == "GL":
            self.camera_poses[camera_name] = T_GL = T
        elif fmt == "ROS":
            self.camera_poses[camera_name] = T_GL = T @ T_ROS_GL
        elif fmt == "CV":
            self.camera_poses[camera_name] = T_GL = T @ T_CV_GL
        else:
            raise ValueError(f"Unknown camera pose format {fmt=}")
        # Update camera lineset pose
        if camera_name != "default":
            self._scene.scene.set_geometry_transform(f"{camera_name}_lineset", T_GL)

    def run_as_process(self):
        """Run O3DGUIVisualizer as a separate process"""
        # TODO: size-variable pointcloud support is not implemented yet,
        #       Need support for size-variable np.ndarray in SharedObject
        self.logger.info(f"Running {self!r} as a separate process")

        # O3DGUIVisualizer control
        so_joined = SharedObject("join_viso3d")
        so_draw = SharedObject("draw_vis")
        so_reset = SharedObject("reset_vis")
        so_dict = SharedObjectDefaultDict()  # {so_name: SharedObject}

        data_dict = O3DGeometryDefaultDict()  # {geometry name: o3d geometry}
        urdf_data_dict = {}  # {xarm7_<robot_uid>: (URDF, [geometry name])}
        gripper = XArmGripper()

        def fetch_rs_camera_stream_and_update_pcd(camera_name: str, pcd, all_so_names):
            """Fetch intr, color, depth, pose streams and update pcd attributes"""
            # Took ~1.4 ms to fetch 848x480 image streams
            K = so_dict[f"rs_{camera_name}_intr"].fetch()
            pts_color = None
            if (so_data_name := f"rs_{camera_name}_color") in all_so_names:
                pts_color = so_dict[so_data_name].fetch(lambda x: x/255.).reshape(-1, 3)
            depth_image = so_dict[f"rs_{camera_name}_depth"].fetch()
            T_world_camROS = so_dict[f"rs_{camera_name}_pose"].fetch()\
                .to_transformation_matrix()

            if pts_color is not None:
                pcd.colors = Vector3dVector(pts_color)
            pcd.points = Vector3dVector(depth2xyz(depth_image, K).reshape(-1, 3))
            pcd.transform(T_world_camROS @ T_ROS_CV)
            self.add_camera(camera_name, *depth_image.shape[1::-1], K,
                            T_world_camROS, fmt="ROS")

        def init_urdf_geometries(robot_name: str, *, urdf_so_name: str = None):
            """Initialize robot geometries by reading from URDF
            and adding all meshes to visualizer
            :param urdf_so_name: name of SharedObject containing URDF path
            """
            if robot_name not in urdf_data_dict:
                urdf_path = so_dict[f"{robot_name}_urdf_path" if urdf_so_name is None
                                    else urdf_so_name].fetch()
                geometry_dir = urdf_path.rsplit('/', 1)[0]
                robot = URDF.load(urdf_path, lazy_load_meshes=True)
                urdf_geo_names = []

                for link in robot.link_fk():
                    n_visuals = len(link.visuals)
                    for i, visual in enumerate(link.visuals):
                        geo_name = (f"{robot_name}/{link.name}" if n_visuals == 1
                                    else f"{robot_name}/{link.name}_{i}")
                        self.load(f"{geometry_dir}/{visual.geometry.mesh.filename}",
                                  name=geo_name)
                        urdf_geo_names.append(geo_name)
                urdf_data_dict[robot_name] = (robot, urdf_geo_names)

                update_urdf_geometries(robot_name)

        def update_urdf_geometries(robot_name: str, *,
                                   qpos: np.ndarray = None, pose: np.ndarray = None):
            """Update robot geometries using qpos and FK
            :param pose: T_world_urdf pose, [4, 4] np.floating np.ndarray
            """
            robot, urdf_geo_names = urdf_data_dict[robot_name]
            if qpos is None:
                qpos = np.zeros(len(robot.actuated_joints))
            for geo_name, T in zip(urdf_geo_names,
                                   robot.visual_geometry_fk(qpos).values()):
                # TODO: currently assumes robot base is world frame
                self._scene.scene.set_geometry_transform(
                    geo_name, T if pose is None else pose @ T
                )

        signal_process_ready()  # current process is ready

        while not so_joined.triggered:
            # Sort names so they are ordered as color, depth, mask
            all_so_names = sorted(os.listdir("/dev/shm"))

            # ----- Reset ----- #
            if so_reset.triggered:  # triggers reset
                self.clear_geometries()
                so_dict = SharedObjectDefaultDict()  # {so_name: SharedObject}
                data_dict = O3DGeometryDefaultDict()  # {geometry name: o3d geometry}
                urdf_data_dict = {}  # {xarm7_<robot_uid>: (URDF, [geometry name])}

            # ----- Capture and update from RSDevice stream ----- #
            if self.stream_camera:  # capture whenever a new frame comes in
                redraw_geometry_uids = []
                for so_data_name in [p for p in all_so_names
                                     if p.startswith("rs_") and p.endswith("_depth")]:
                    if so_dict[so_data_name].modified:
                        camera_name = so_data_name[3:-6]
                        data_uid = f"{camera_name}/captured_pcd"

                        fetch_rs_camera_stream_and_update_pcd(
                            camera_name, data_dict[data_uid], all_so_names
                        )

                        redraw_geometry_uids.append(data_uid)

                if len(redraw_geometry_uids) > 0:  # redraw scene for camera stream
                    self.add_geometries({data_uid: data_dict[data_uid]
                                         for data_uid in redraw_geometry_uids})
            else:  # synchronized capturing with env (no redraw here)
                # Capture color, depth, and pose stream
                # for each camera sync, check if capture is triggered
                for so_name in [p for p in all_so_names if p.startswith("sync_rs_")]:
                    if so_dict[so_name].triggered:
                        camera_name = so_name[8:]
                        data_uid = f"{camera_name}/captured_pcd"

                        fetch_rs_camera_stream_and_update_pcd(
                            camera_name, data_dict[data_uid], all_so_names
                        )

            # ----- Capture and update from robot state stream ----- #
            if self.stream_robot:  # update whenever a new robot state comes in
                for so_data_name in [p for p in all_so_names
                                     if p.startswith("xarm7_") and p.endswith("_qpos")]:
                    if (so_data := so_dict[so_data_name]).modified:
                        robot_name = so_data_name[:-5]  # xarm7_<robot_uid>
                        init_urdf_geometries(robot_name)
                        update_urdf_geometries(robot_name, qpos=so_data.fetch())
            else:
                for so_name in [p for p in all_so_names if p.startswith("sync_xarm7_")]:
                    robot_name = so_name[5:]  # xarm7_<robot_uid>
                    init_urdf_geometries(robot_name)
                    if so_dict[so_name].triggered:
                        update_urdf_geometries(
                            robot_name, qpos=so_dict[f"{robot_name}_qpos"].fetch()
                        )

            # ----- Fetch data and draw ----- #
            if so_draw.triggered:  # triggers redraw
                redraw_geometry_uids = set()

                so_data_names = [
                    p for p in all_so_names
                    if p.startswith(("rs_", "vis_", "viso3d_"))
                    and p.endswith(("_color", "_depth", "_pose", "_xyzimg", "_pts",
                                    "_qpos", "_bounds", "_gposes"))
                ]
                for so_data_name in so_data_names:
                    data_source, data_uid = so_data_name.split('_', 1)
                    data_uid, data_fmt = data_uid.replace('|', '/').rsplit('_', 1)
                    if data_source == "rs":  # all capturing / updating is already done
                        redraw_geometry_uids.add(f"{data_uid}/captured_pcd")
                        continue

                    # Fetch data
                    if data_fmt == "color":  # PointCloud.colors
                        if data_uid.endswith("_camera"):  # camera capture
                            data_uid = f"{data_uid}/captured_pcd"
                        data_dict[data_uid].colors = Vector3dVector(
                            so_dict[so_data_name].fetch(lambda x: x/255.).reshape(-1, 3)
                        )
                        redraw_geometry_uids.add(data_uid)  # redraw
                    elif data_fmt == "depth":  # camera capture
                        camera_name = data_uid
                        data_prefix = f"{data_source}_{camera_name}".replace('/', '|')
                        data_uid = f"{camera_name}/captured_pcd"

                        K = so_dict[f"{data_prefix}_intr"].fetch()
                        depth_image = so_dict[so_data_name].fetch()

                        data_dict[data_uid].points = Vector3dVector(depth2xyz(
                            depth_image, K,
                            1000.0 if depth_image.dtype == np.uint16 else 1.0
                        ).reshape(-1, 3))
                        self.add_camera(camera_name, *depth_image.shape[1::-1], K)
                        redraw_geometry_uids.add(data_uid)  # redraw
                    elif data_fmt == "pose":  # object / camera pose
                        T = so_dict[so_data_name].fetch().to_transformation_matrix()
                        if data_uid.endswith("_camera"):  # camera capture
                            self.update_camera_pose(data_uid, T, fmt="CV")
                            data_uid = f"{data_uid}/captured_pcd"
                        if data_uid not in self.geometries:  # add for the first time
                            self.add_geometry(data_uid, data_dict[data_uid])
                        if not self._scene.scene.has_geometry(data_uid):
                            self.logger.error(f"Geometry {data_uid=} is not in scene")
                        # NOTE: it's also possible to rescale coord frames
                        # with set_geometry_transform (maybe add another slider?)
                        self._scene.scene.set_geometry_transform(data_uid, T)
                    elif data_fmt == "pts":  # PointCloud.points
                        data_dict[data_uid].points = Vector3dVector(
                            so_dict[so_data_name].fetch()
                        )
                        redraw_geometry_uids.add(data_uid)  # redraw
                    elif data_fmt == "xyzimg":  # xyz_image after depth2xyz
                        data_dict[data_uid].points = Vector3dVector(
                            so_dict[so_data_name].fetch().reshape(-1, 3)
                        )
                        redraw_geometry_uids.add(data_uid)  # redraw
                    elif data_fmt == "qpos":  # robot joint states
                        robot_name = so_data_name[:-5]  # xarm7_<robot_uid>
                        init_urdf_geometries(robot_name)
                        update_urdf_geometries(robot_name,
                                               qpos=so_dict[so_data_name].fetch())
                    elif data_fmt == "bounds":  # bbox
                        bounds = so_dict[so_data_name].fetch()  # [xyz_min, xyz_max]
                        data_dict[data_uid].min_bound = bounds[0]
                        data_dict[data_uid].max_bound = bounds[1]
                        redraw_geometry_uids.add(data_uid)  # redraw
                    elif data_fmt == "gposes":  # gripper grasp poses
                        # CGN_grasps/obj1/best_grasp_mesh
                        grasp_mesh_name = f"{data_uid}/best_grasp_mesh"
                        init_urdf_geometries(grasp_mesh_name,
                                             urdf_so_name="robot_gripper_urdf_path")
                        grasp_poses_world = so_dict[so_data_name].fetch()
                        grasp_scores = so_dict[f"{so_data_name[:-7]}_gscores"].fetch()
                        grasp_qvals = so_dict[f"{so_data_name[:-7]}_gqvals"].fetch()

                        # update gripper mesh
                        max_score_idx = grasp_scores.argmax()
                        update_urdf_geometries(
                            grasp_mesh_name, qpos=[0]*6+[grasp_qvals[max_score_idx]]*2,
                            pose=grasp_poses_world[max_score_idx]
                        )
                        # add gripper lineset
                        lineset = gripper.get_control_points_lineset(
                            gripper.get_control_points(grasp_qvals, grasp_poses_world)
                        )
                        data_dict[f"{data_uid}/grasp_lineset"] = lineset
                        redraw_geometry_uids.add(f"{data_uid}/grasp_lineset")  # redraw
                    else:
                        raise ValueError(f"Unknown {so_data_name = }")

                self.add_geometries({data_uid: data_dict[data_uid]
                                     for data_uid in redraw_geometry_uids})

            self.render()

        self.logger.info(f"Process running {self!r} is joined")
        # Unlink created SharedObject
        so_joined.unlink()

    def render(self, render_step_fn=None):
        """Update GUI and respond to mouse and keyboard events for one tick
        :param render_step_fn: additional render step function to call.
        """
        # Update info: start rendering
        self.update_render_info("GUI running", color=[0.0, 1.0, 0.0])

        while self.not_closed:
            self.not_closed = gui.Application.instance.run_one_tick()
            if render_step_fn is not None:
                render_step_fn()

            if not self.paused or (self.paused and self.single_step):
                self.single_step = False
                break

        # Update info: pause rendering (need run_one_tick to update)
        self.update_render_info("GUI paused", color=[1.0, 0.0, 0.0])
        self.not_closed = gui.Application.instance.run_one_tick()

    def close(self):
        self.window.close()
        self.not_closed = gui.Application.instance.run_one_tick()
        self.window = None

    def __del__(self):
        """Can segfault if not closed before delete"""
        self.close()

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.window_name}>"


if __name__ == "__main__":
    visualizer = O3DGUIVisualizer()

    while visualizer.not_closed:
        visualizer.render()

    # Run the event loop. This will not return until the last window is closed.
    # gui.Application.instance.run()
