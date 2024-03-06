from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, TypeVar, Union

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from urchin import URDF

from real_robot.utils.logger import get_logger

_N = TypeVar("_N", bound=int)


def np2pcd(
    points: np.ndarray[tuple[_N, Literal[3]], np.dtype[np.floating]]
    | o3d.geometry.PointCloud,
    colors: Optional[np.ndarray[tuple[_N, Literal[3]], np.dtype[np.floating]]] = None,
    normals: Optional[np.ndarray[tuple[_N, Literal[3]], np.dtype[np.floating]]] = None,
) -> o3d.geometry.PointCloud:
    """Convert numpy array to open3d PointCloud."""
    if isinstance(points, o3d.geometry.PointCloud):
        return points
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.copy())
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


O3D_GEOMETRIES = (
    o3d.geometry.Geometry3D,
    o3d.t.geometry.Geometry,
    rendering.TriangleMeshModel,
)
ANY_O3D_GEOMETRY = Union[O3D_GEOMETRIES]


def transform_geometry(geometry: ANY_O3D_GEOMETRY, T: np.ndarray) -> ANY_O3D_GEOMETRY:
    """Apply transformation to o3d geometry, always returns a copy

    :param T: transformation matrix, [4, 4] np.floating np.ndarray
    """
    if isinstance(geometry, rendering.TriangleMeshModel):
        out_geometry = rendering.TriangleMeshModel()
        out_geometry.meshes = [
            rendering.TriangleMeshModel.MeshInfo(
                deepcopy(mesh_info.mesh).transform(T),
                mesh_info.mesh_name,
                mesh_info.material_idx,
            )
            for mesh_info in geometry.meshes
        ]
        out_geometry.materials = geometry.materials
    elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
        out_geometry = deepcopy(geometry).transform(T)
    else:
        raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return out_geometry


O3D_GEOMETRY_LIST = Union[tuple(list[t] for t in O3D_GEOMETRIES)]


def merge_geometries(geometries: O3D_GEOMETRY_LIST) -> ANY_O3D_GEOMETRY:
    """Merge a list of o3d geometries, must be of same type"""
    geometry_types = set([type(geometry) for geometry in geometries])
    assert len(geometry_types) == 1, f"Not the same geometry type: {geometry_types = }"

    merged_geometry = next(iter(geometry_types))()
    for i, geometry in enumerate(geometries):
        if isinstance(geometry, rendering.TriangleMeshModel):
            num_materials = len(merged_geometry.materials)
            merged_geometry.meshes += [
                rendering.TriangleMeshModel.MeshInfo(
                    deepcopy(mesh_info.mesh),
                    f"mesh_{i}_{mesh_info.mesh_name}".strip("_"),
                    mesh_info.material_idx + num_materials,
                )
                for mesh_info in geometry.meshes
            ]
            merged_geometry.materials += geometry.materials
        elif isinstance(geometry, (o3d.geometry.Geometry3D, o3d.t.geometry.Geometry)):
            merged_geometry += geometry
        else:
            raise TypeError(f"Unknown o3d geometry type: {type(geometry)}")
    return merged_geometry


def sample_pcd_from_mesh(
    mesh: o3d.geometry.TriangleMesh | rendering.TriangleMeshModel,
    number_of_points: int = 100,
    method: str = "uniform",
    use_triangle_normal: bool = False,
    seed: Optional[int] = None,
    **kwargs,
) -> o3d.geometry.PointCloud:
    """Sample points from an Open3D mesh

    :param mesh: an Open3D TriangleMesh or rendering.TriangleMeshModel.
    :param number_of_points: number of points to sample.
    :param method: "uniform" or "poisson_disk".
        * uniform: sample uniformly (faster).
        * poisson_disk: sample such that each point has approximately the same
        distance to the neighbouring points (slower).
    :param use_triangle_normal: If True, assigns the triangle normals instead of
                                the interpolated vertex normals to the returned points.
                                The triangle normals will be computed and added to
                                the mesh if necessary.
    :param seed: If not None, set Open3D's random seed.
    :param kwargs: additional kwargs for sample_points_poisson_disk():
                   init_factor=5, pcl=None.
    """
    if isinstance(mesh, rendering.TriangleMeshModel):
        mesh = merge_geometries([mesh_info.mesh for mesh_info in mesh.meshes])

    if seed is not None:
        o3d.utility.random.seed(seed)

    if method == "uniform":
        return mesh.sample_points_uniformly(
            number_of_points, use_triangle_normal=use_triangle_normal
        )
    elif method == "poisson_disk":
        return mesh.sample_points_poisson_disk(
            number_of_points, use_triangle_normal=use_triangle_normal, **kwargs
        )
    else:
        raise ValueError(f"Unknown {method=}, choices: ['uniform', 'poisson_disk']")


def convert_mesh_format(mesh_path: str | Path, export_suffix=".glb") -> str:
    """Convert mesh format to glb for open3d.io.read_triangle_model()"""
    try:
        import trimesh
    except ImportError as e:
        get_logger("real_robot").critical("Failed to import trimesh: %s", e)

    mesh_format = Path(mesh_path).suffix[1:].lower()
    assert (
        mesh_format in trimesh.exchange.load.mesh_formats()  # type: ignore
    ), f"mesh format {mesh_path} not supported"

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), f"mesh type {type(mesh)} not supported"

    with tempfile.NamedTemporaryFile(suffix=export_suffix, delete=False) as f:
        file_path = f.name
        mesh.export(file_path)

    return file_path


O3D_PCD_FORMATS = ("xyz", "xyzn", "xyzrgb", "pts", "ply", "pcd")
O3D_MESH_FORMATS = ("ply", "stl", "obj", "off", "gltf", "glb")


def load_geometry(
    path: str | Path, *, logger=get_logger("real_robot")
) -> rendering.TriangleMeshModel | o3d.geometry.PointCloud | None:
    """Load a geometry from file as an Open3D geometry

    :param path: path to a geometry file supported by open3d.
                 https://www.open3d.org/docs/release/tutorial/geometry/file_io.html
    """
    if (mesh_format := Path(path).suffix[1:].lower()) in O3D_PCD_FORMATS:
        pass
    elif mesh_format not in O3D_MESH_FORMATS:
        path = convert_mesh_format(path, export_suffix=".glb")

    geometry_type = o3d.io.read_file_geometry_type(str(path))

    if geometry_type & o3d.io.CONTAINS_TRIANGLES:
        return o3d.io.read_triangle_model(str(path))

    logger.debug("%s appears to be a point cloud", path)
    cloud = None
    try:
        cloud = o3d.io.read_point_cloud(str(path))
    except Exception:
        pass
    if cloud is not None:
        if not cloud.has_normals():
            cloud.estimate_normals()
        cloud.normalize_normals()
        return cloud

    logger.error("Failed to read points from %s", path)
    return None


def load_urdf_geometries(
    urdf_path: str | Path,
    *,
    skip_links: Optional[list[str]] = None,
    qpos: Optional[np.ndarray] = None,
    base_pose: np.ndarray = np.eye(4),
    return_pose: bool = False,
    merge: bool = False,
    logger=get_logger("real_robot"),
) -> tuple[
    URDF,
    dict[
        str,
        tuple[rendering.TriangleMeshModel, np.ndarray] | rendering.TriangleMeshModel,
    ]
    | rendering.TriangleMeshModel,
]:
    """Load a robot geometries from a URDF file

    :param urdf_path: path to a URDF file.
    :param skip_links: names of links to skip loading.
    :param qpos: robot joint positions. If None, all joints are at zero positions.
    :param base_pose: T_world_urdfbase pose, [4, 4] np.floating np.ndarray
    :param return_pose: Whether to return the geometries pose in world frame
                        instead of applying the pose transformation to the geometries.
    :param merge: Whether to merge the geometries into a single
                  rendering.TriangleMeshModel.
    """
    urdf_path = Path(urdf_path).resolve()
    geometry_dir = urdf_path.parent

    robot: URDF = URDF.load(urdf_path, lazy_load_meshes=True)

    # Filter based on skip_links
    link_names = [link.name for link in robot.links]
    if skip_links is not None:
        link_names = [name for name in link_names if name not in skip_links]

    # Load URDF geometries
    urdf_geometries = {}  # {geometry_name: rendering.TriangleMeshModel}
    for link in robot.link_fk(links=link_names):
        n_visuals = len(link.visuals)
        for i, visual in enumerate(link.visuals):
            geo_name = link.name if n_visuals == 1 else f"{link.name}_{i}"
            urdf_geometries[geo_name] = load_geometry(
                f"{geometry_dir}/{visual.geometry.mesh.filename}",
                logger=logger,
            )

    if qpos is None:
        qpos = np.zeros(len(robot.actuated_joints))

    # Apply the pose transformation when merging the geometries
    if merge:
        return_pose = False
    for (geometry_name, geometry), T_base_geom in zip(
        urdf_geometries.items(),
        robot.visual_geometry_fk(qpos, links=link_names).values(),
    ):
        urdf_geometries[geometry_name] = (
            (geometry, base_pose @ T_base_geom)
            if return_pose
            else transform_geometry(geometry, base_pose @ T_base_geom)
        )

    if merge:
        return robot, merge_geometries(urdf_geometries.values())
    else:
        return robot, urdf_geometries
