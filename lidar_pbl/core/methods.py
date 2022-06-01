import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# Apply the default theme
sns.set_theme()

sns.set(
    font_scale=1.4,
)


def gradient_pbl(
    lidar_profile: np.ndarray,
    min_grad: float = -2,
    max_grad: float = 0.5,
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
    # gradient2[gradient2 > 0] = 0

    # plt.plot(gradient[num][10:final])
    # plt.plot(gradient2[num][10:final])
    # plt.show()

    if max_grad is not None:
        gradient[gradient > max_grad] = 0

    min_axis = 0 if dimension == 1 else 1
    mins = np.argmin(gradient, axis=min_axis)

    # current_index = 80

    # try:
    #     heights = np.arange(lidar_profile.shape[1]) * 3.75 + 400
    # except IndexError:
    #     heights = np.arange(lidar_profile.shape[0]) * 3.75 + 400

    # palette = sns.color_palette()

    # fig, ax = plt.subplots(figsize=(7, 13))
    # # ax.plot(np.log10(safe_profile)[current_index], heights, label="RCS profile", color=palette[0], linewidth=1.5)
    # ax2 = ax.twiny()
    # ax2.plot(gradient[current_index], heights, label="Gradient method", color=palette[1], linewidth=1.5)

    # ax.legend(loc="upper left", fontsize="medium")
    # ax2.legend(loc="upper right", fontsize="medium")

    # ax.set(xlabel="RCS log10 [a.u]", ylabel="Height [m]")
    # ax2.set(xlabel="Gradient [a.u]")

    # ax2.grid(False)

    # fig.savefig("gradient.png", dpi=600)
    # plt.show()


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

    palette = sns.color_palette()

    current_index = 80
    heights = np.arange(lidar_profile.shape[1]) * 3.75 + 400

    fig, ax = plt.subplots(figsize=(7, 13))

    window = lidar_profile[current_index: current_index + window_size, :]

    maxes = np.nanmax(window, axis=0)
    mins = np.nanmin(window, axis=0)

    # ax.plot(lidar_profile[current_index], heights, label="RCS profile (10 values)", color=palette[0], linewidth=1)
    ax.fill_betweenx(heights, mins, maxes, alpha = 0.5, color = palette[0], label="RCS window (10 values)")
    ax2 = ax.twiny()

    ax2.plot(variance[8],heights, label="Variance method", color=palette[1])

    ax.set(xlabel="RCS [a.u]", ylabel="Height [m]")
    ax2.set(xlabel="Variance [a.u]")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # ax.legend(loc="upper right")
    # ax2.legend(loc="upper left")
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax2.grid(False)

    fig.savefig("variance.png", dpi=600)
    # plt.show()

    return window_element, variance_vote


def haar(array_shape: int, a: float = 1, b: float = 1 / 2) -> np.ndarray:
    x = np.arange(0, array_shape)
    return np.piecewise(
        x,
        [
            np.logical_and(b - a / 2 <= x, x <= b),
            np.logical_and(b <= x, x <= b + a / 2),
        ],
        [1, -1, 0],
    )


def wavelet_pbl(
    lidar_profile: np.ndarray,
    a: int = 4,
) -> np.ndarray:
    def single_row_wavelet(row: np.ndarray) -> np.ndarray:
        res = np.zeros(row.shape)

        for [idx], _ in np.ndenumerate(row):
            _fn = row * haar(row.shape[0], a, (2 * idx + 1) / 2)
            _int = np.sum(_fn)
            res[idx] += _int

        res[0:a] = np.nan
        res[-a:] = np.nan

        return res / np.sqrt(a)

    wavelets = np.apply_along_axis(single_row_wavelet, 1, lidar_profile)
    wavelet_vote = np.nanargmax(wavelets, axis=1)

    palette = sns.color_palette()

    current_index = 80

    heights = np.arange(lidar_profile.shape[1]) * 3.75 + 400

    fig, ax = plt.subplots(1, 2,figsize=(12, 13), sharey=True)
    row = lidar_profile[current_index]

    ax[0].plot(row, heights, color=palette[0], linewidth=2, label="RCS profile")
    # ax[0]fill_betweenx(heights, mins, maxes, alpha = 0.5, color = palette[0], label="RCS window (10 values)")
    ax2 = ax[1]

    # ax2.plot(variance[8],heights, label="Variance method", color=palette[1], linewidth = 5)

    ax[0].set(xlabel="RCS [a.u]", ylabel="Height [m]")
    ax2.set(xlabel="Wavelet [a.u]")

    plt.locator_params(axis='x', nbins=5)

    ranging = np.arange(1, 40, 11)
    for c, i in enumerate(ranging):
        a = i
        ax2.plot(single_row_wavelet(row), heights, color=palette[c + 1], linewidth=0.8, label=f"a = {a * 3.75} m")


    ax[0].legend(loc="upper right")
    ax2.legend(loc="upper right")
    # ax2.legend(loc="upper left")
    # ax2.grid(False)

    fig.savefig("wavelet.png", dpi=600)

    return wavelet_vote
