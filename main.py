from lidar_pbl import LidarDataset
from lidar_pbl.utils.visualization import quicklook

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    lidar_dataset = LidarDataset(
        data_path="./data/2021/08/12/RS/data.npz",
        dark_current="./data/dark_current/dark_current.npy",
        data_type="NPZ",
    )
    print('max:', lidar_dataset.data.max())
    print('min:', lidar_dataset.data.min())
    print('min_rcs:', lidar_dataset.rcs.min())

    lidar_dataset.quicklook(max_height=2000)
    # print(lidar_dataset.rcs)


if __name__ == "__main__":
    main()
