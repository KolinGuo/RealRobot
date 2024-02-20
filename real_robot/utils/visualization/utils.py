import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from real_robot.utils.logger import get_logger

_rng = np.random.RandomState(0)
_palette = ((_rng.random((3 * 255)) * 0.7 + 0.3) * 255).astype(np.uint8).tolist()
_palette = [0, 0, 0] + _palette


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Colorize a predicted mask
    :param pred_mask: [H, W] bool/np.uint8 np.ndarray
    :return mask: colorized mask, [H, W, 3] np.uint8 np.ndarray
    """
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode="RGB")
    return np.asarray(save_mask)


def draw_mask(rgb_img, mask, alpha=0.5, id_countour=False) -> np.ndarray:
    """Overlay predicted mask on rgb image
    :param rgb_img: RGB image, [H, W, 3] np.uint8 np.ndarray
    :param mask: [H, W] bool/np.uint8 np.ndarray
    :param alpha: overlay transparency
    :return img_mask: mask-overlayed image, [H, W, 3] np.uint8 np.ndarray
    """
    img_mask = rgb_img.copy()
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on binary mask
            if id <= 255:
                color = _palette[id * 3 : id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = rgb_img * (1 - alpha) + np.ones_like(
                rgb_img
            ) * alpha * np.asarray(color)
            binary_mask = mask == id

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = mask != 0
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = rgb_img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0
    return img_mask


def convert_mesh_format(mesh_path: str | Path, export_suffix=".glb") -> str:
    """Convert mesh format to glb for open3d.io.read_triangle_model()"""
    try:
        import trimesh
    except ImportError as e:
        get_logger("real_robot").critical("Failed to import trimesh: %s", e)

    mesh_format = Path(mesh_path).suffix[1:]
    assert (
        mesh_format in trimesh.exchange.load.mesh_formats()  # type: ignore
    ), f"mesh format {mesh_path} not supported"

    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh), f"mesh type {type(mesh)} not supported"

    with tempfile.NamedTemporaryFile(suffix=export_suffix, delete=False) as f:
        file_path = f.name
        mesh.export(file_path)

    return file_path
