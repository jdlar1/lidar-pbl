import matplotlib.pyplot as plt
from lidar_pbl import LidarDataset
from lidar_pbl.utils.visualization import quicklook

import numpy as np

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

    print('argmin: ', np.argmin(lidar_dataset.data, axis = 1))
        

    lidar_dataset.quicklook(max_height=1500)
    # print(lidar_dataset.rcs)


if __name__ == "__main__":
    main()
