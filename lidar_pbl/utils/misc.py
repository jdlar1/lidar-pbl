from pathlib import Path
from platform import uname

import toml
import numpy as np


def in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def moving_average(x: np.ndarray, w: int = 3) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w
