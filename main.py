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

    # print(lidar_dataset.rcs[:1, :300].shape)

    # lidar_dataset.
    # plt.plot(lidar_dataset.rcs[0][:600], label = "rcs")
    gradient_heights = gradient_pbl(lidar_dataset.rcs[:, :350], min_grad=-0.08)
    lidar_dataset.quicklook(max_height=2000)
    print(gradient_heights.shape)
    plt.scatter(np.arange(gradient_heights.shape[0]), gradient_heights, label="gradient")
    # print(lidar_dataset.rcs[:2,:600].shape)

    plt.show()


if __name__ == "__main__":
    main()
