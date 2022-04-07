import pywt
import numpy as np


def gradient_pbl(
    lidar_profile: np.ndarray,
    min_grad: float = -2,
    max_grad: float = 0.5,
    min_height: float = 0,
    max_height=3000,
    bin_res: float = 3.75,
) -> np.ndarray:
    """Gives the pblh heights given profiles

    Args:
        lidar_profile (np.ndarray): 2D array of lidar profile
        max_grad (float, optional): A max thereshold for the gradient avoiding clouds or other interferences. Defaults to None.
        max_height (int, optional): Max height to seek PBL (in meters). Defaults to 3000.

    Returns:
        np.ndarray: 1D array of pbl heights
    """

    safe_profile = lidar_profile.copy()
    safe_profile[safe_profile <= 0] = 1e-10
    dimension = safe_profile.ndim

    if dimension == 1:
        gradient = np.gradient(np.log10(safe_profile))
    else:
        gradient = np.gradient(np.log10(safe_profile))[1]

    # gradient2 = np.gradient(gradient)[1]

    # num = 0
    # final = 300
    # plt.plot(gradient[num][10:final])
    # plt.plot(np.gradient(gradient[num][10:final]))
    # plt.show()

    if max_grad is not None:
        gradient[gradient < min_grad] = 0
    gradient[gradient > 0] = 0
    # gradient2[gradient2 > 0] = 0

    # plt.plot(gradient[num][10:final])
    # plt.plot(gradient2[num][10:final])
    # plt.show()

    heights = np.arange(0, lidar_profile.shape[0]) * bin_res
    index_top = np.searchsorted(heights, max_height, side="right") - 1
    index_bottom = np.searchsorted(heights, min_height, side="right") - 1

    if max_grad is not None:
        gradient[gradient > max_grad] = 0

    min_axis = 0 if dimension == 1 else 1
    mins = np.argmin(gradient, axis=min_axis)

    return mins


def variance_pbl(
    lidar_profile: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:

    window_number = lidar_profile.shape[0] // window_size
    window_element = np.arange(window_number) * window_size

    variance = np.zeros([window_number, lidar_profile.shape[1]])

    for i, window in enumerate(window_element):
        start = window
        end = start + window_size

        temp_var = np.var(lidar_profile[start:end, :], axis=0)

        variance[i, :] = temp_var

        # var_window = np.var(lidar_profile[start:end], axis=0)
        # variance = np.hstack([variance, var_window])

    variance_vote = np.argmax(variance, axis=1)

    return window_element, variance_vote


def haar(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.piecewise(
        x, [b - a / 2 <= x and x <= b, b <= x and x <= b + a / 2], [1, -1, 0]
    )


def wavelet_pbl(
    lidar_profile: np.ndarray,
) -> np.ndarray:
    wavelet = pywt.dwt2(lidar_profile, "bior1.3", axes=(0, 1))
    return wavelet
