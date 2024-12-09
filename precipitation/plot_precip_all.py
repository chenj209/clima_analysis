import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import glob

# Parameters
season = ["ANN", "JJA", "DJF"]
firstyr, lastyr = 1980, 1982
nyears = 2
startfile, endfile = 1, 12
ntime = 12

expname = ["SPCAM", "NNCAM", "NNCAM(PhyC)", "CAM5"]
ncases = len(expname) * len(season)

listyears = [1999,2000,2001,2002,2003]
listmonths = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

field_name = "Precip"

cam5_name = "2022_11_10"
cam5_path = "/temp_share/nncam-cases/nncam-diag_cam5/2022_11_10/atm/hist/"

nncam_name = "2021_11_15"
nncam_path = "/temp_share/nncam-cases/nncam-couple/2021_11_15/atm/hist/"

nncamrh_name = "baseline_nn_rh"
nncamrh_path = "/temp_share/nncam-cases/neuroGCM/baseline_nn_rh/atm/hist_old/"

spcam_name = "spcam.baseline"
spcam_path = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"

# Load data
def load_data(path, name, var_name):
    print("loading: ", path, name, var_name)
    files = []
    for year in listyears:
        fs = glob.glob(f"/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/spcam.baseline.cam.h0.{year}-*.nc")
        files.extend(fs)
    #print("opening: ", files)
    ds = xr.open_mfdataset(files)
    return ds[var_name] * 24 * 3600 * 1000

spcam_data = load_data(spcam_path, spcam_name, 'PRECC')
nncam_data = load_data(nncam_path, nncam_name, 'PRECC')
nncamrh_data = load_data(nncamrh_path, nncamrh_name, 'PRECC')
cam5_data = load_data(cam5_path, cam5_name, 'PRECT')

# Calculate seasonal means
def seasonal_mean(data):
    annual = data.mean('time')
    jja = data.sel(time=data['time.month'].isin([6, 7, 8])).mean('time')
    djf = data.sel(time=data['time.month'].isin([12, 1, 2])).mean('time')
    return [annual, jja, djf]

field = np.array([
    seasonal_mean(spcam_data),
    seasonal_mean(nncam_data),
    seasonal_mean(nncamrh_data),
    seasonal_mean(cam5_data)
]).reshape(ncases, spcam_data.shape[1], spcam_data.shape[2])

# Calculate differences
field_diff = np.array([
    field[1:4] - field[0],
    field[4:7] - field[0],
    field[7:10] - field[0]
])

# Calculate area-weighted averages and RMSEs
def area_weighted_average(data, lat):
    weights = np.cos(np.deg2rad(lat))
    return np.average(data, weights=weights)

lat = spcam_data.lat
precip_areaavg = np.array([area_weighted_average(f, lat) for f in field])
precip_diff_areaavg = np.array([area_weighted_average(f, lat) for f in field_diff])

rmse = np.sqrt(np.array([area_weighted_average((field[i+3] - field[i])**2, lat) for i in range(9)]))

# Plotting function
def plot_map(ax, data, title, cmap, levels, add_colorbar=False):
    data, lon = add_cyclic_point(data, coord=spcam_data.lon)
    cs = ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels, extend='both')
    ax.coastlines()
    ax.set_global()
    ax.set_title(title)
    if add_colorbar:
        plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05)

# Create plots
fig, axs = plt.subplots(4, 3, figsize=(15, 20), subplot_kw={'projection': ccrs.Robinson()})

levels = np.array([0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17])
for i, ax in enumerate(axs.flat):
    title = f"{chr(97+i)}) {expname[i//3]} {season[i%3]} mean:{precip_areaavg[i]:.3f}"
    plot_map(ax, field[i], title, 'Blues', levels)

plt.tight_layout()
plt.savefig('precip_comparison0829.png', dpi=300, bbox_inches='tight')

# Difference plots
fig, axs = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': ccrs.Robinson()})

diff_levels = np.arange(-10, 11, 1)
for i, ax in enumerate(axs.flat):
    title = f"{chr(97+i)}) Diff {expname[(i//3)+1]} {season[i%3]} rmse:{rmse[i]:.3f}"
    plot_map(ax, field_diff[i//3, i%3], title, 'RdBu_r', diff_levels)

plt.tight_layout()
plt.savefig('precip_difference0829.png', dpi=300, bbox_inches='tight')
