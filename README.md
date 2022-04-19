# lidar-pbl: CLI for PBL calculation from lidar data

[![PyPI Latest Release](https://img.shields.io/pypi/v/lidar-pbl.svg)](https://pypi.org/project/lidar-pbl/)

## What is it?

    **lidar-pbl** is a command line tool for to handle Licel txt output files. It aims to provide a simple and easy to use interface to calculate the PBL with classical methods (gradient, variance, wavelet).

## Usage
**CLI**
```bash
$ lidar-pbl quicklook 
  \\ data/2021/08/12/RS 
  \\ data/dark_current/
  \\ --methods
```
For help type `lidar-pbl --help`	too.
**From python file**
This library can also be easily imported as a python module

```python	
lidar_dataset = LidarDataset(
        data_dir="data/2021/08/12/RS",
        dark_current_dir="data/dark_current/",
    )

    lidar_dataset.quicklook(max_height=2000)
    lidar_dataset.gradient_pbl(min_height=400, max_height=1250, min_grad=-0.05)
    lidar_dataset.wavelet_pbl(min_height=400, max_height=1250, a_meters=4)
    lidar_dataset.variance_pbl(min_height=400, max_height=1250)
```	



