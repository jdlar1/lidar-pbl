from lidar_pbl.utils import read_npz, plot_profile, rcs, quicklook, txt_to_npz

data, dates = read_npz("./data/2021/08/12/RS/data.npz")
data_rcs = rcs(data)

quicklook(data[:-40, :], bin_res=3.75, max_height=1_500, bin_zero=0)
