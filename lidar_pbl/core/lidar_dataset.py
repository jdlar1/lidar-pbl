import pathlib

import numpy as np
from matplotlib import pyplot as plt

from lidar_pbl.core.types import InputType
from lidar_pbl.core.methods import gradient_pbl, variance_pbl, wavelet_pbl
from lidar_pbl.utils import (
    read_npz,
    plot_profile,
    rcs,
    quicklook,
    read_txts,
)


class LidarDataset:
    """
    Class to handle Lidar data.
    """

    def __init__(
        self,
        data_path: pathlib.Path | str,
        dark_current: pathlib.Path | np.ndarray,
        data_type: InputType = InputType.NPZ,
        bin_res: float = 3.75,
        bin_zero: int = 12,
    ):
        """Initialize LidarDataset for data processing

        Args:
            data_path (pathlib.Path | str): The path of the directory if data_type is NPZ, or the path of the txt file if data_type is TXT.
            data_type (InputType, optional): The type of the data. Defaults to InputType.NPZ.
        """
        self._data: np.ndarray = np.empty(0)
        self.bin_res: float = bin_res
        self.bin_zero: int = bin_zero

        if isinstance(dark_current, np.ndarray):
            if dark_current.shape != self.data[0].shape:
                raise ValueError(
                    "The dark current shape does not match the data shape."
                )
            dark_mean = np.mean(dark_current, axis=0)
            self.dark_current = dark_mean
        elif isinstance(dark_current, pathlib.Path | str):
            dark_matrix = np.load(dark_current)
            self.dark_current = np.mean(dark_matrix, axis=0)
        else:
            raise ValueError("Invalid dark_current type.")

        match data_type:
            case InputType.TXT:
                temp_data, self.dates = read_txts(data_path)
            case InputType.NPZ:
                temp_data, self.dates = read_npz(data_path)
            case _:
                raise ValueError(f"Invalid data_type: {data_type}")

        temp_data: np.ndarray = temp_data - self.dark_current
        temp_data[temp_data < 0] = 0
        self.data = temp_data
        self.heights = np.arange(temp_data.shape[1]) * self.bin_res

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data):
        self.rcs = rcs(data)
        self._data = data

    def plot_profile(
        self,
        bin_number: int = 0,
        max_height: float = None,
    ):
        """Plot the profile of the data.

        Args:
            bin_number (int, optional): The number of bins. Defaults to 0.
            max_height (int | float, optional): The maximum height to plot. Defaults to None.
        """
        plot_profile(self.rcs[bin_number], max_height=max_height)

    def quicklook(self, max_height: float = None):
        """Quicklook of the data.

        Args:
            bin_number (int, optional): The number of bins. Defaults to 0.
            max_height (int | float, optional): The maximum height to plot. Defaults to None.
        """

        quicklook(
            self.rcs,
            self.dates,
            max_height=max_height,
            bin_zero=self.bin_zero,
            bin_res=self.bin_res,
        )

    def gradient_pbl(
        self, min_height: float = 0, min_grad: float = -0.08, max_height: float = 3000
    ):
        """Gradient PBL height criterias

        Args:
            min_grad (float, optional): _description_. Defaults to 0.08.
            max_height (int, optional): _description_. Defaults to 3000.
        """
        height_index = np.searchsorted(self.heights, max_height)
        min_height_index = np.searchsorted(self.heights, min_height)
        points = gradient_pbl(
            self.rcs[:, min_height_index:height_index], min_grad=min_grad
        )

        plt.scatter(
            np.arange(points.size),
            points - self.bin_zero + min_height_index,
            marker="^",
            label=f"Gradient method",
            alpha=0.5,
            s=15,
        )

    def variance_pbl(self, max_height=3000, min_height=0, window_size=10):
        height_index = np.searchsorted(self.heights, max_height)
        min_height_index = np.searchsorted(self.heights, min_height)
        element, variance = variance_pbl(
            self.rcs[:, min_height_index:height_index], window_size=window_size
        )

        plt.scatter(
            element + window_size // 2,
            variance - self.bin_zero + min_height_index,
            marker="o",
            label=f"Variance method",
            alpha=0.5,
            s=15,
            c="g",
        )

    def show(self, legend=True) -> None:
        """Show the data.

        Args:
            bins (int, optional): The number of bins. Defaults to 0.
        """
        if legend:
            plt.legend()
        plt.show()

    def wavelet_pbl(self, max_height=3000, min_height=0, a_meters: float = 20):
        height_index = np.searchsorted(self.heights, max_height)
        min_height_index = np.searchsorted(self.heights, min_height)
        a = a_meters // self.bin_res

        points = wavelet_pbl(self.rcs[:, min_height_index:height_index], a=a)

        plt.scatter(
            np.arange(points.size),
            points - self.bin_zero + min_height_index,
            marker="^",
            label=f"Wavelet method",
            alpha=0.5,
            s=15,
            c="k",
        )

