import numpy as np
import os
import netCDF4 as nc
NUM_YEARS = 5
NUM_CASES = 2
lenth = 5 * 12
#expname = ["SPCAM", "NNCAM", "NNCAM(PhyC)", "CAM5"]#, "new50_rh1d"]
caserealname = ["spcam.baseline", "2021_11_15", "baseline_nn_rh", "2022_11_10"]
# caserealname = ["spcam.baseline", "2021_11_15", "crash1_rh_rerun0612", "2022_11_10"]

spcam_path    = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
nncam_path    = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"
# nncamrh_path  = "/temp_share/stabilities.analysis/hist.plot/hist.nc-data/case1_6years/"
nncamrh_path  = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist/"
cam5_path     = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"

#filepath = [spcam_path, nncam_path, nncamrh_path, cam5_path]
filepath = [spcam_path]

print(filepath)

# yearbuff  = ['0002', '0003', '0004', '0005', '0006']#, 2002]#,2003]
yearbuff  = [1998, 1999, 2000, 2001, 2002]
monthbuff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
precip    = np.zeros((2,lenth,52,144))

# 读取数据
ii=0

winter_months = []
summer_months = []

trmm_nc = nc.Dataset("/cust_users/chenj209/trmm_data/monthly_mean_1998_onwards.nc")
trmm_precp = trmm_nc["PCP"]
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
        
        precip[0, ii] = trmm_precp[ii]
        for icase, casename in enumerate(caserealname):
                
            filename = "{}.cam.h0.{:04d}-{:02d}.nc".format(casename,iyear, imonth)
            data     = nc.Dataset(os.path.join(filepath[icase], filename),'r')
            precip[icase+1, ii,:,:] = data['PRECC'][0,22:74,:]*24*3600*1000

        ii+=1

        print("{:04d}-{:02d}".format(iyear, imonth))

DJF_precip = np.mean(precip[:, winter_months, :, :], axis=1)
JJA_precip = np.mean(precip[:, summer_months, :, :], axis=1)
ANN_precip = precip.mean(axis=1)

# save all three means
np.save("DJF_precip.npy", DJF_precip)
np.save("JJA_precip.npy", JJA_precip)
np.save("ANN_precip.npy", ANN_precip)