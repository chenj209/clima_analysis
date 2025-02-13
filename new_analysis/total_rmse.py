import os
import netCDF4 as nc
import numpy as np

def get_t2m(file_path):
    with nc.Dataset(file_path) as fid:
        t2m = fid.variables['TREFHT'][:]
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
    Compute seasonal and annual RMSE and correlation between SPCAM and a model for the year 1998
    
    Parameters:
    -----------
    spcam_path : str
        Path to SPCAM data directory
    model_path : str 
        Path to model data directory
        
    Returns:
    --------
    metrics : dict
        Dictionary containing RMSE and correlation for each season and annual mean
    """
    sample_data = nc.Dataset(os.path.join(spcam_path, f"spcam.baseline.cam.h0.1998-01.nc"))
    lat = sample_data.variables["lat"][:]
    
    # Initialize monthly data arrays
    monthly_spcam = []
    monthly_model = []
    
    # Load all monthly data
    for i in range(1,13):
        spcam_t2m = get_t2m(os.path.join(spcam_path, f"spcam.baseline.cam.h0.1998-{i:02d}.nc"))
        model_t2m = get_t2m(os.path.join(model_path, f"conv_mem_share3.cam.h0.1998-{i:02d}.nc"))
        monthly_spcam.append(spcam_t2m)
        monthly_model.append(model_t2m)
    
    # Convert to numpy arrays
    monthly_spcam = np.array(monthly_spcam)
    monthly_model = np.array(monthly_model)
    
    # Get model name from path
    model_name = os.path.basename(model_path).split('/')[-1].split('.')[0]
    
    # Calculate seasonal and annual means
    annual_spcam = np.mean(monthly_spcam, axis=0)
    annual_model = np.mean(monthly_model, axis=0)
    annual_bias = annual_model - annual_spcam
    np.save(f'{model_name}_annual_bias.npy', annual_bias)
    
    djf_spcam = np.mean(monthly_spcam[[11,0,1],:,:], axis=0)  # Dec,Jan,Feb
    djf_model = np.mean(monthly_model[[11,0,1],:,:], axis=0)
    djf_bias = djf_model - djf_spcam
    np.save(f'{model_name}_djf_bias.npy', djf_bias)
    
    mam_spcam = np.mean(monthly_spcam[2:5,:,:], axis=0)  # Mar,Apr,May
    mam_model = np.mean(monthly_model[2:5,:,:], axis=0)
    mam_bias = mam_model - mam_spcam
    np.save(f'{model_name}_mam_bias.npy', mam_bias)
    
    jja_spcam = np.mean(monthly_spcam[5:8,:,:], axis=0)  # Jun,Jul,Aug
    jja_model = np.mean(monthly_model[5:8,:,:], axis=0)
    jja_bias = jja_model - jja_spcam
    np.save(f'{model_name}_jja_bias.npy', jja_bias)
    
    son_spcam = np.mean(monthly_spcam[8:11,:,:], axis=0)  # Sep,Oct,Nov
    son_model = np.mean(monthly_model[8:11,:,:], axis=0)
    son_bias = son_model - son_spcam
    np.save(f'{model_name}_son_bias.npy', son_bias)
    
    # Calculate metrics for each period
    metrics = {}
    
    # Helper function for correlation
    def spatial_correlation(x, y, weights):
        x_w = x * weights
        y_w = y * weights
        x_mean = np.sum(x_w) / np.sum(weights)
        y_mean = np.sum(y_w) / np.sum(weights)
        cov = np.sum(weights * (x - x_mean) * (y - y_mean)) / np.sum(weights)
        var_x = np.sum(weights * (x - x_mean)**2) / np.sum(weights)
        var_y = np.sum(weights * (y - y_mean)**2) / np.sum(weights)
        return cov / np.sqrt(var_x * var_y)
    
    weights = np.cos(np.deg2rad(lat))[:,np.newaxis]
    
    for period, (spcam, model) in {
        'annual': (annual_spcam, annual_model),
        'djf': (djf_spcam, djf_model),
        'mam': (mam_spcam, mam_model),
        'jja': (jja_spcam, jja_model),
        'son': (son_spcam, son_model)
    }.items():
        metrics[period] = {
            'rmse': area_RMSE(model, spcam, lat),
            'correlation': spatial_correlation(spcam, model, weights)
        }
        
    return metrics

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    print(sys.argv[2])
    metrics = compute_annual_rmse(sys.argv[1], sys.argv[2])
    print("\nMetrics Summary:")
    print("-" * 50)
    for period in metrics:
        print(f"\n{period.upper()}:")
        print(f"  RMSE: {metrics[period]['rmse']:.4f}")
        print(f"  Correlation: {metrics[period]['correlation']:.4f}")
    print("-" * 50)