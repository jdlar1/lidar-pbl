import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.ticker import FuncFormatter
import pendulum


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


def quicklook(
    bin2d: np.ndarray,
    dates: list[pendulum.DateTime],
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

    if len(dates) != bin2d.shape[0]:
        raise ValueError("The length of dates does not match the number of rows.")

    bin_number = np.arange(0, bin2d.shape[1])
    heights = bin_number * bin_res

    if max_height is None:
        data, h = (
            bin2d[:, bin_zero:].T,
            heights[bin_zero:],
        )
    else:
        index = np.searchsorted(heights, max_height)
        data, h = bin2d[:, bin_zero:index].T, heights[bin_zero:index]

    params = {
        "aspect": "auto",
        "cmap": "jet",
        "interpolation": "antialiased",
        "origin": "lower",
        "norm": colors.LogNorm(vmax=data.max() * 0.004, clip=True),
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.imshow(data, **params)

    ax.set(xlabel="Time", ylabel="Height [m]", title="Lidar Scan")

    @FuncFormatter
    def format_heights(x, pos=None):
        if x >= h.shape[0]:
            return ""
        return f"{h[int(x)]}"

    @FuncFormatter
    def format_dates(x, pos=None):
        if x >= dates.shape[0]:
            return ""
        return dates[int(x)].format("HH:mm:ss")

    ax.yaxis.set_major_formatter(format_heights)
    ax.xaxis.set_major_formatter(format_dates)
    fig.autofmt_xdate(rotation=45)

    fig.colorbar(ax.images[0], ax=ax, label="Lidar Signal [a.u.]")
