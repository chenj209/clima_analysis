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

def area_RMSE_3d(data1, data2, lat):
    """Calculate area-weighted RMSE for 3D data (time, lat, lon)"""
    # Calculate weights based on latitude
    weights = np.cos(np.deg2rad(lat))
    weights = weights[:, np.newaxis]  # Shape (lat, 1)
    
    # Calculate squared difference across all times
    bias_2 = (data1 - data2) ** 2  # Shape (time, lat, lon)
    
    # Apply weights to each timestep
    bias_2_weighted = bias_2 * weights  # Broadcasting to (time, lat, lon)
    
    # Take mean over lat, lon for each timestep, then mean over time
    bias_2_mean = np.sum(bias_2_weighted, axis=(1,2)) / np.sum(weights)
    final_mean = np.mean(bias_2_mean)
    
    return np.sqrt(final_mean)

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
    all_spcam_t2m = []
    all_model_t2m = []
    
    # First collect monthly RMSEs and concatenate data
    for i in range(1,13):
        spcam_t2m = get_t2m(os.path.join(spcam_path, f"spcam.baseline.cam.h0.1998-{i:02d}.nc"))[:]
        model_t2m = get_t2m(os.path.join(model_path, f"conv_mem_spinup5.cam.h0.1998-{i:02d}.nc"))[:]
        rmse = area_RMSE(spcam_t2m, model_t2m, sample_data.variables["lat"])
        rmse_array.append(rmse)
        all_spcam_t2m.append(spcam_t2m)
        all_model_t2m.append(model_t2m)
    
    # Calculate overall RMSE using concatenated data
    spcam_concat = np.concatenate(all_spcam_t2m, axis=0)
    model_concat = np.concatenate(all_model_t2m, axis=0)
        
    total_rmse = area_RMSE_3d(spcam_concat, model_concat, sample_data.variables["lat"])
    return rmse_array, total_rmse

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    print(sys.argv[2])
    rmse_array, total_rmse = compute_annual_rmse(sys.argv[1], sys.argv[2])
    print("total_rmse: ", total_rmse)
    print("rmse_array: ", rmse_array)
