from importlib.metadata import version
from pathlib import Path

__version__ = version("real_robot")

ASSET_DIR = Path(__file__).resolve().parent / "assets"
