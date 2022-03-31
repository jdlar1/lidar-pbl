import matplotlib.pyplot as plt

from lidar_pbl import LidarDataset
from lidar_pbl.core.methods import gradient


def main():
    lidar_dataset = LidarDataset(
        data_path="./data/2021/08/12/RS/data.npz",
        dark_current="./data/dark_current/dark_current.npy",
        data_type="NPZ",
    )

    # lidar_dataset.quicklook(max_height=2000)
    # plt.plot(lidar_dataset.rcs[0][:600], label = "rcs")
    plt.plot(gradient(lidar_dataset.rcs[0][:600]), label = "gradient")

    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
