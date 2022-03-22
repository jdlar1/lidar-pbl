import numpy as np
import matplotlib.pyplot as plt

def plot_profile(bin: np.ndarray ,bin_res: float = 3.75, max_height: None | float = None) -> None:
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

