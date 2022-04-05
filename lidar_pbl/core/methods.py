import warnings

from matplotlib import pyplot as plt
import numpy as np


def gradient_pbl(
    lidar_profile: np.ndarray,
    min_grad: float = -2,
    max_grad: float = 0.5,
    min_height: float = 0,
    max_height=3000,
    bin_res: float = 3.75,
) -> np.ndarray:
    """Gives the pblh heights given profiles

    Args:
        lidar_profile (np.ndarray): 2D array of lidar profile
        max_grad (float, optional): A max thereshold for the gradient avoiding clouds or other interferences. Defaults to None.
        max_height (int, optional): Max height to seek PBL (in meters). Defaults to 3000.

    Returns:
        np.ndarray: 1D array of pbl heights
    """

    safe_profile = lidar_profile.copy()
    safe_profile[safe_profile <= 0] = 1e-10
    dimension = safe_profile.ndim

    if dimension == 1:
        gradient = np.gradient(np.log10(safe_profile))
    else:
        gradient = np.gradient(np.log10(safe_profile))[1]

    gradient[gradient > 0] = 0

    if max_grad is not None:
        gradient[gradient < min_grad] = 0

    # plt.plot(gradient[0])
    # plt.show()

    heights = np.arange(0, lidar_profile.shape[0]) * bin_res
    index_top = np.searchsorted(heights, max_height, side="right") - 1
    index_bottom = np.searchsorted(heights, min_height, side="right") - 1
    print("index_top", index_top)
    print("index_bottom", index_bottom)

    if max_grad is not None:
        gradient[gradient > max_grad] = 0

    min_axis = 0 if dimension == 1 else 1
    mins = np.argmin(gradient, axis=min_axis)

    return mins
