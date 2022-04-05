import matplotlib.pyplot as plt
import numpy as np

from lidar_pbl import LidarDataset
from lidar_pbl.core.methods import gradient_pbl
from lidar_pbl.utils.misc import moving_average


def main():
    lidar_dataset = LidarDataset(
        data_path="./data/2021/08/12/RS/data.npz",
        dark_current="./data/dark_current/dark_current.npy",
        data_type="NPZ",
    )

    lidar_dataset.quicklook(max_height=2000)
    lidar_dataset.gradient_pbl(max_height=1500, min_grad=-0.05)
    # lidar_dataset

    lidar_dataset.show()


if __name__ == "__main__":
    main()
