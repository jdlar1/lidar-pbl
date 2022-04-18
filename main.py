from lidar_pbl import LidarDataset
from lidar_pbl.core.methods import wavelet_pbl


def main():
    lidar_dataset = LidarDataset(
        data_path="./data/2021/08/12/RS/data.npz",
        dark_current="./data/dark_current/dark_current.npy",
        data_type="NPZ",
    )

    # lidar_dataset.quicklook(max_height=2000)
    # lidar_dataset.gradient_pbl(min_height=400, max_height=1250, min_grad=-0.05)
    # lidar_dataset.variance_pbl(min_height=400, max_height=1250)
    lidar_dataset.wavelet_pbl(min_height=400, max_height=1250, a_meters= 4)

    # lidar_dataset.show()


if __name__ == "__main__":
    main()
