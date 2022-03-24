__version__ = "0.1.0"

import matplotlib

from .core.lidar_dataset import LidarDataset
from .utils import in_wsl

if in_wsl():
    matplotlib.use("GTK3Agg")
