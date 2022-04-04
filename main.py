import matplotlib.pyplot as plt

from lidar_pbl import LidarDataset
from lidar_pbl.core.methods import gradient
from lidar_pbl.utils.misc import moving_average


def main():
    lidar_dataset = LidarDataset(
        data_path="./data/2021/08/12/RS/data.npz",
        dark_current="./data/dark_current/dark_current.npy",
        data_type="NPZ",
    )

    # lidar_dataset.
    # lidar_dataset.quicklook(max_height=2000)
    # plt.plot(lidar_dataset.rcs[0][:600], label = "rcs")
    plt.plot(gradient(lidar_dataset.rcs[0][:600]), label="gradient")
    plt.plot(
        moving_average(gradient(lidar_dataset.rcs[0][:310]), 3), label="moving average"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
