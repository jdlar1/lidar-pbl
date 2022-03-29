import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
import matplotlib as mpl


def plot_profile(
    bin: np.ndarray, bin_res: float = 3.75, max_height: None | float = None
) -> None:
    """
    Plots a raw lidar scan.
    Args:
        bin (np.ndarray): The raw lidar scan.
        bin_res (float, optional): The resolution of the lidar scan. Defaults to 3.75.
    """

    bin_number = np.arange(0, bin.shape[0])
    heights = bin_number * bin_res

    if max_height is None:
        plt.plot(bin, heights)
    else:
        index = np.searchsorted(heights, max_height)
        plt.plot(bin[:index], heights[:index])

    plt.xlabel("Lidar Signal [a.u.]")
    plt.ylabel("Height [m]")

    plt.show()


def quicklook(
    bin2d: np.ndarray,
    bin_res: float = 3.75,
    max_height: None | float = None,
    bin_zero: int = 0,
):
    """
    Quicklook of the data.
    Args:
        bin2d (np.ndarray): The raw lidar scan.
        bin_res (float, optional): The resolution of the lidar scan. Defaults to 3.75.
        max_height (int | float, optional): The maximum height to plot. Defaults to None.
        bin_zero (int, optional): The number of bins to be removed from the top. Defaults to 0.
    """

    bin_number = np.arange(0, bin2d.shape[0])
    heights = bin_number * bin_res

    if max_height is None:
        data, h = bin2d[:, bin_zero:].T, heights[bin_zero:],
    else:
        index = np.searchsorted(heights, max_height)
        data, h= bin2d[:, bin_zero: index].T, heights[bin_zero:index]

    print(f'min: {np.min(data)}, max: {np.max(data)}')
    params = {
        "aspect": "auto",
        "cmap": "jet",
        "interpolation": "bilinear",
        "origin": "lower",
        "norm": colors.Normalize(vmax=data.max() * 0.008)
    }

    plt.figure(figsize=(10, 4))
    plt.imshow(data, **params)

    plt.xlabel("Lidar Signal [a.u.]")
    plt.ylabel("Height [m]")
    plt.colorbar()

    plt.show()
