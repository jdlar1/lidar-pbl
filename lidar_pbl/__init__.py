import toml
from pathlib import Path


import matplotlib

from .core.lidar_dataset import LidarDataset
from .utils import in_wsl


def get_version() -> str:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    return pyproject["tool"]["poetry"]["version"]


__version__ = get_version()

if in_wsl():
    matplotlib.use("GTK3Agg")
