from importlib.metadata import version
from pathlib import Path

__version__ = version("real_robot")

from real_robot.utils.logger import Logger

LOGGER = Logger()

ASSET_DIR = Path(__file__).resolve().parent / "assets"
