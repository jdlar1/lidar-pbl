import numpy as np


def gradient(lidar_profile: np.ndarray) -> np.ndarray:
    """Calculates the gradient of the lidar profile.
    Args:
        lidar_profile (np.ndarray): The lidar profile.
    """
    return np.gradient(np.log(lidar_profile))
