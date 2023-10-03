from pathlib import Path
from importlib.metadata import version
__version__ = version("real_robot")

from .utils.logger import logger

ASSET_DIR = Path(__file__).resolve().parent / "assets"
