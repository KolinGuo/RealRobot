import os
from pathlib import Path
from importlib.metadata import version
__version__ = version("real_robot")

from .utils.logger import logger
from ._root_dir import REPO_ROOT

# FIXME: include hec results in real_robot
REPO_ROOT = Path(os.getenv("REAL_ROBOT_ROOT", REPO_ROOT))
if not REPO_ROOT.is_dir():
    raise FileNotFoundError("RealRobot repo root does not exist. Please specify it "
                            'with environment variable "REAL_ROBOT_ROOT"')

ASSET_DIR = Path(__file__).resolve().parent / "assets"
