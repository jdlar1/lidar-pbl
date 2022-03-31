import pathlib

import numpy as np

from lidar_pbl.core.types import InputType
from lidar_pbl.utils import (
    read_npz,
    plot_profile,
    rcs,
    quicklook,
    txt_to_npz,
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
    ):
        """Initialize LidarDataset for data processing

        Args:
            data_path (pathlib.Path | str): The path of the directory if data_type is NPZ, or the path of the txt file if data_type is TXT.
            data_type (InputType, optional): The type of the data. Defaults to InputType.NPZ.
        """
        self._data: np.ndarray | None = None
        self.bin_res: float = bin_res

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

        temp_data = temp_data - self.dark_current
        temp_data[temp_data < 0] = 0
        self.data = temp_data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data):
        self.rcs = rcs(data)
        self._data = data

    def rcs_diff(self, bins: int = 0, degree: int = 1):
        """Calculate the RCS difference between two lidar scans."""
        return np.diff(self.rcs[bins])

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
        quicklook(self.rcs, self.dates, max_height=max_height, bin_res=self.bin_res)