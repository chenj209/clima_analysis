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
from config import *

if __name__ == "__main__":
    # caserealname = ["spcam.baseline", "2021_11_15", "baseline_nn_rh", "2022_11_10"]
    caserealname = CASE_NAMES

    #spcam_path    = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
    #nncam_path    = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"
    #nncamrh_path  = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"
    # nncamrh_path  = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist/"
    #cam5_path     = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"
    #rb1_path = "/share3/chenj209/CESM_logs/replay_buffer_spinup5_seed1117_full/nc_files/"
    #rb2_path = "/share3/chenj209/CESM_logs/replay_buffer_spin5_seed1_full/nc_files/"


    #filepath = [spcam_path, nncam_path, nncamrh_path, cam5_path, rb1_path, rb2_path]
    filepath = DATA_PATHS

    print(filepath)
    months = 12*6

    # yearbuff  = ['0002', '0003', '0004', '0005', '0006']#, 2002]#,2003]
    # yearbuff  = [1998, 1999, 2000, 2001, 2002, 2003]
    #yearbuff  = [1998,1999, 2000, 2001, 2002, 2003]
    yearbuff  = [y for y in range(START_YEAR, END_YEAR+1)]
    monthbuff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    target_field = args.target_field
    monthly_rmse = np.zeros((NUM_CASES,months))

    # yearbuff  = ['0002', '0003', '0004', '0005', '0006']#, 2002]#,2003]
    precip    = np.zeros((NUM_CASES,lenth,96,144))

    # 读取数据
    ii=0

    for iyear in tqdm(yearbuff, desc="year"):
        for imonth in tqdm(monthbuff, desc="month"):
            filename = "{}.cam.h0.{:04d}-{:02d}.nc".format(caserealname[0],iyear, imonth)
            spcam_nc = ncdf.Dataset(os.path.join(filepath[0], filename),'r')
            spcam_data = get_data(spcam_nc, target_field)
            ps = spcam_nc.variables["PS"][0]
            hyai = spcam_nc.variables["hyai"]
            hybi = spcam_nc.variables["hybi"]
            thickness = get_thickness_from_ps_2d(ps, hybi, hyai)
            horizontal_mass_factors, mass_factor = get_grid_mass(thickness)
            if np.all(spcam_data == 0):
                raise Exception(f"{filename} all zero {args.target_field}")
            for icase, casename in enumerate(caserealname[1:]):
                filename = "{}.cam.h0.{:04d}-{:02d}.nc".format(casename,iyear, imonth)
                file_path = os.path.join(filepath[icase+1], filename)
                if not os.path.isfile(file_path):
                    monthly_rmse[icase,ii] = np.nan
                else:
                    data     = ncdf.Dataset(os.path.join(filepath[icase+1], filename),'r')
                    target_data = get_data(data, target_field)
                    if len(spcam_data.shape) == 3:
                        rmse_weighted = np.sqrt(np.sum((spcam_data - target_data)**2*mass_factor))
                    elif len(spcam_data.shape) == 2:
                        rmse_weighted = np.sqrt(np.sum((spcam_data - target_data)**2*horizontal_mass_factors))
                    monthly_rmse[icase,ii] = rmse_weighted
            ii += 1

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
