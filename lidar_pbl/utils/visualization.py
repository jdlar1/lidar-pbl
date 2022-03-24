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
    Plots a 2D lidar scan.
    Args:
        bin2d (np.ndarray): The 2D lidar scan.
        bin_res (float, optional): The resolution of the lidar scan. Defaults to 3.75.
    """
    params = {
        "origin": "lower",
        "aspect": "auto",
        "interpolation": "bilinear",
        "cmap": "jet",
        # "norm": colors.LogNorm()
        "norm": colors.LogNorm(clip=True),
    }

    heights = np.arange(0, bin2d.shape[0]) * bin_res
    bin_number = np.arange(0, bin2d.shape[1])

    heights = bin_number * bin_res
    plt.figure(figsize=(12, 7))

    if max_height is None:
        plt.imshow(bin2d.T[:, bin_zero:], **params)
        plt.gca().yaxis.set_major_formatter(lambda x, _: heights[x])
        # plt.yticks(heights[bin_zero:])
    else:
        index = np.searchsorted(heights, max_height)
        plt.imshow(bin2d[:, bin_zero:index].T, **params)
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: heights[int(x)])
        )
        # plt.yticks(heights[bin_zero:index])

    plt.show()
