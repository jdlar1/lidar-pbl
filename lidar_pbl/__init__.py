__version__ = "0.1.0"

from .utils import in_wsl
import matplotlib

if in_wsl():
    matplotlib.use("GTK3Agg")
