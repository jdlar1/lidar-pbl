import re
import pathlib

import pendulum
import numpy as np


def read_txts(
    dirpath: pathlib.Path | str,
) -> tuple[np.ndarray, list[pendulum.DateTime]]:
    """Opens all txt files in a directory and return a 2 numpy array of them sorted and a list of dates.

    Args:
        dirpath (pathlib.Path | str): The dir where those txt files are.

    Returns:
        tuple[np.ndarray, list[Any]]: 2D values for time and space dependence and a list of dates.
    """
    txts = list(map(str, pathlib.Path(dirpath).glob("*.txt")))
    txts.sort()
    res = [
        re.match(
            r"^.+(RS|DC)(\d{2})(\d{1})(\d{2})(\d{2})\.(\d{2})(\d{2})\d+\.txt$", txt
        )
        for txt in txts
    ]
    dates = [
        pendulum.parse(
            f"20{res.group(2)}-0{res.group(3)}-{res.group(4)}T{res.group(5)}:{res.group(6)}:{res.group(7)}"
        )
        for res in res
        if res is not None
    ]
    data = [np.loadtxt(txt, skiprows=5) for txt in txts]

    cache_file = pathlib.Path(dirpath) / ".cache.npz"
    np.savez(cache_file, data=data, dates=dates)

    return data, dates


def read_npz(
    filepath: pathlib.Path | str,
) -> tuple[np.ndarray, list[pendulum.DateTime]]:
    """Opens a npz file and return a 2 numpy array of them sorted and a list of dates.

    Args:
        filepath (pathlib.Path | str): The filepath of the npz file.

    Returns:
        tuple[np.ndarray, list[Any]]: 2D values for time and space dependence and a list of dates.
    """
    npz = np.load(filepath, allow_pickle=True)

    return npz["data"], npz["dates"]


def txt_to_npz(
    dirpath: pathlib.Path | str, filepath_output: pathlib.Path | str
) -> None:
    """Converts a lot of txt files to a single npz file.
    Which can be opened with:
    lidar = np.load('filepath_output.npz')
    lidar['data'] is a list of numpy arrays.
    lidar['dates'] is a list of pendulum dates.
    Args:
        dirpath (pathlib.Path | str): _description_
        filepath_output (pathlib.Path | str): _description_

    Returns:
        _type_: None
    """
    txts = list(map(str, pathlib.Path(dirpath).glob("*.txt")))
    txts.sort()
    res = [
        re.match(r"^.+RS/RS(\d{2})(\d)(\d{2})(\d{2})\.(\d{2})(\d{2})\d+\.txt$", txt)
        for txt in txts
    ]
    dates = [
        pendulum.parse(
            f"20{res.group(1)}-0{res.group(2)}-{res.group(3)}T{res.group(4)}:{res.group(5)}:{res.group(6)}"
        )
        for res in res
    ]
    data = [np.loadtxt(txt, skiprows=5) for txt in txts]
    filepath_output = (
        filepath_output
        if filepath_output.endswith(".npz")
        else filepath_output + ".npz"
    )
    np.savez(filepath_output, data=data, dates=dates)
