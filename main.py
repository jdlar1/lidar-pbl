from lidar_pbl.utils import read_npz, plot_profile, rcs

data, dates = read_npz("./data/2021/08/12/RS/data.npz")
rcs_d = rcs(data[:2])

plot_profile(rcs_d[0], max_height=1_500)
plot_profile(rcs_d[1], max_height=1_500)
