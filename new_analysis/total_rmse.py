import os
import netCDF4 as nc
import numpy as np

def get_t2m(file_path):
    with nc.Dataset(file_path) as fid:
        t2m = fid.variables['T2M'][:]
    return t2m

def area_RMSE(model, obs, lat):
    '''
    calculate weighted area RMSE
    input data is numpy array
    model and obs should have shape (lat, lon)
    '''
    # Calculate weights based on latitude
    weights = np.cos(np.deg2rad(lat))
    
    # Expand weights to match data dimensions
    weights = weights[:, np.newaxis]
    
    # Calculate squared difference
    bias_2 = (model - obs) ** 2
    
    # Apply weights and take mean
    bias_2_weighted = bias_2 * weights
    bias_2_mean = np.sum(bias_2_weighted) / np.sum(weights)
    
    return np.sqrt(bias_2_mean)

def compute_annual_rmse(spcam_path, model_path):
    """
    Compute monthly RMSE between SPCAM and a model for the year 1998
    
    Parameters:
    -----------
    spcam_path : str
        Path to SPCAM data directory
    model_path : str 
        Path to model data directory
        
    Returns:
    --------
    rmse_array : list
        List of 12 monthly RMSE values
    """
    sample_data = nc.Dataset(os.path.join(spcam_path, f"spcam.baseline.cam.h0.1998-01.nc"))
    rmse_array = []
    for i in range(1,13):
        spcam_t2m = get_t2m(os.path.join(spcam_path, f"spcam.baseline.cam.h0.1998-{i:02d}.nc"))[:]
        model_t2m = get_t2m(os.path.join(model_path, f"conv_mem_spinup5.cam.h0.1998-{i:02d}.nc"))[:]
        rmse = area_RMSE(spcam_t2m, model_t2m, sample_data.variables["lat"])
        rmse_array.append(rmse)
    return rmse_array

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    print(sys.argv[2])
    print(compute_annual_rmse(sys.argv[1], sys.argv[2]))
