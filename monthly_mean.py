import os
import time
import numpy as np
import random
import sys
import netCDF4 as ncdf

import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter , LatitudeFormatter

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from tqdm.autonotebook import tqdm

from sklearn.metrics import mean_squared_error, r2_score

def get_data(data, target_field):
    if target_field.endswith("850"):
        target_field = target_field[:-3]
        return data[target_field][0,23]
    if target_field.endswith("200"):
        target_field = target_field[:-3]
        return data[target_field][0,12]
    if target_field.endswith("500"):
        target_field = target_field[:-3]
        return data[target_field][0,18]
    if target_field.endswith("600"):
        target_field = target_field[:-3]
        return data[target_field][0,19]
    if target_field == "PRECC":
        return data["PRECC"][0]*24*3600*1000
    return data[target_field][0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target_field", help="The target field to be analyzed")
    args = parser.parse_args()
    expname = ["SPCAM", "NNCAM", "NNCAM(PhyC)", "CAM5", "rb1", "rb2"]#, "new50_rh1d"]
    # caserealname = ["spcam.baseline", "2021_11_15", "baseline_nn_rh", "2022_11_10"]
    caserealname = ["spcam.baseline", "2021_11_15", "crash1_rh_rerun0612", "2022_11_10", "conv_mem_spinup5", "conv_mem_spinup5"]

    spcam_path    = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
    nncam_path    = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"
    nncamrh_path  = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"
    # nncamrh_path  = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist/"
    cam5_path     = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"
    rb1_path = "/share3/chenj209/CESM_logs/replay_buffer_spinup5_seed1117_full/nc_files/"
    rb2_path = "/share3/chenj209/CESM_logs/replay_buffer_spin5_seed1_full/nc_files/"


    filepath = [spcam_path, nncam_path, nncamrh_path, cam5_path, rb1_path, rb2_path]

    print(filepath)
    months = 12*6

    # yearbuff  = ['0002', '0003', '0004', '0005', '0006']#, 2002]#,2003]
    # yearbuff  = [1998, 1999, 2000, 2001, 2002, 2003]
    yearbuff  = [1998,1999, 2000, 2001, 2002, 2003]
    monthbuff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    target_field = args.target_field
    monthly_mean = np.zeros((len(expname),months))

    # 读取数据
    ii=0

    for iyear in tqdm(yearbuff, desc="year"):
        for imonth in tqdm(monthbuff, desc="month"):
            for icase, casename in enumerate(caserealname):
                filename = "{}.cam.h0.{:04d}-{:02d}.nc".format(casename,iyear, imonth)
                data     = ncdf.Dataset(os.path.join(filepath[icase], filename),'r')
                target_data = get_data(data, target_field)
                monthly_mean[icase,ii] = np.mean(target_data)
            ii += 1
    np.save(args.target_field + "_monthly_mean.npy", monthly_mean)
