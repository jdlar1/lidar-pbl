import numpy as np


def rcs(
    bins: np.ndarray, bin_res: float = 3.75, background_height: float = 8000
) -> np.ndarray:
    """Calculates the range-corrected signal.
    Args:
        bins (np.ndarray): The raw lidar scan.
        bin_res (float, optional): The resolution of the lidar scan. Defaults to 3.75.
    """
    print(bins.shape)

    bin_number = np.arange(0, bins.shape[1])
    heights = bin_number * bin_res
    heights = np.tile(heights, (bins.shape[0], 1))
    index = np.searchsorted(heights[0], background_height)

    background_value = np.mean(bins[:, :index])
    bins_cleaned = bins - background_value

    return bins_cleaned * heights * heights
