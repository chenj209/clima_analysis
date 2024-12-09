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

from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
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
    lenth = 12*6

    # yearbuff  = ['0002', '0003', '0004', '0005', '0006']#, 2002]#,2003]
    yearbuff  = [1998, 1999, 2000, 2001, 2002, 2003]
    #yearbuff  = [1998]
    monthbuff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    precip    = np.zeros((len(expname),lenth,96,144))

    # 读取数据
    ii=0

    winter_months = []
    summer_months = []

    for iyear in yearbuff:
        for imonth in monthbuff:
            if imonth in [12, 1, 2]:
                w_i = (iyear - 1998) * 12 + (imonth - 1)
                print(w_i)
                winter_months.append(w_i)

            if imonth in [6, 7, 8]:
                s_i = (iyear - 1998) * 12 + (imonth - 1)
                print(s_i)
                summer_months.append(s_i)
            
            for icase, casename in enumerate(caserealname):
                    
                filename = "{}.cam.h0.{:04d}-{:02d}.nc".format(casename,iyear, imonth)
                data     = ncdf.Dataset(os.path.join(filepath[icase], filename),'r')
                precip[icase, ii,:,:] = data['PRECC'][0,:,:]*24*3600*1000

            ii+=1

            print("{:04d}-{:02d}".format(iyear, imonth))

    #for i in [0, 1, 71]:
    #    winter_months.remove(i)

    #for i in [5, 6, 7]:
    #    summer_months.remove(i)
    DJF_precip = np.mean(precip[:, winter_months, :, :], axis=1)
    JJA_precip = np.mean(precip[:, summer_months, :, :], axis=1)
    ANN_precip = precip.mean(axis=1)
    np.save("all_precip", precip)
    np.save("DJF_precip", DJF_precip)
    np.save("JJA_precip", JJA_precip)
    np.save("ANN_precip", ANN_precip)
