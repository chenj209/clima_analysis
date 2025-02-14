import os
import netCDF4 as nc
import numpy as np
import json

def get_t2m(file_path):
    with nc.Dataset(file_path) as fid:
        t2m = fid.variables['TREFHT'][:]
    return t2m

def get_precip(file_path):
    with nc.Dataset(file_path) as fid:
        precip = fid.variables['PRECC'][:]*24*3600*1000
    return precip

def get_Q850(file_path):
    with nc.Dataset(file_path) as fid:
        Q850 = fid.variables['Q'][0,23]
        if np.all(Q850 == Q850.flatten()[0]):
            print(f"Constant Q850 in {file_path}")
    return Q850

def get_Q500(file_path):
    with nc.Dataset(file_path) as fid:
        Q500 = fid.variables['Q'][0,18]
        if np.all(Q500 == Q500.flatten()[0]):
            print(f"Constant Q500 in {file_path}")
    return Q500

def get_T850(file_path):
    with nc.Dataset(file_path) as fid:
        T850 = fid.variables['T'][0,23]
    return T850

def get_T500(file_path):
    with nc.Dataset(file_path) as fid:
        T500 = fid.variables['T'][0,18]
    return T500

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

def spatial_correlation(x, y, lat):
    # Normalize by weights
    weights = np.cos(np.deg2rad(lat))[:,np.newaxis]
    x_weighted = x * weights 
    y_weighted = y * weights
    x_weighted = x_weighted / np.sum(weights*144)
    y_weighted = y_weighted / np.sum(weights*144)
        
    # Use np.corrcoef with weighted data
    correlation_matrix = np.corrcoef(x_weighted.flatten(), y_weighted.flatten())
    return correlation_matrix[0,1]
    
def compute_annual_rmse(spcam_path, model_path, year=1998):
    """
    Compute seasonal and annual RMSE and correlation between SPCAM and a model for the specified year
    for multiple variables: T2m, precipitation, Q850, Q500, T850, T500
    
    Parameters:
    -----------
    spcam_path : str
        Path to SPCAM data directory
    model_path : str 
        Path to model data directory
    year : int, optional
        Year to analyze (default: 1998)
        
    Returns:
    --------
    metrics : dict
        Dictionary containing RMSE and correlation for each season and variable
    """
    sample_data = nc.Dataset(os.path.join(spcam_path, f"spcam.baseline.cam.h0.{year}-01.nc"))
    lat = sample_data.variables["lat"][:]
    
    # Define variables and their getter functions
    variables = {
        't2m': get_t2m,
        'precip': get_precip,
        'q850': get_Q850,
        'q500': get_Q500,
        't850': get_T850,
        't500': get_T500
    }
    
    # Initialize data storage
    monthly_data = {
        var: {'spcam': [], 'model': []} for var in variables
    }
    

    # Load all monthly data for each variable
    for i in range(1,13):
        spcam_file = os.path.join(spcam_path, f"spcam.baseline.cam.h0.{year}-{i:02d}.nc")
        
        # Try conv_mem_share3 first, fallback to conv_mem_spinup5 if not found
        # Find first file matching cam.h0 pattern
        model_files = [f for f in os.listdir(model_path) if 'cam.h0' in f]
        if model_files:
            model_prefix = model_files[0].split('.cam.h0')[0]
            model_file = os.path.join(model_path, f"{model_prefix}.cam.h0.{year}-{i:02d}.nc")
        else:
            raise FileNotFoundError(f"No cam.h0 files found in {model_path}")
        
        for var, getter_func in variables.items():
            monthly_data[var]['spcam'].append(getter_func(spcam_file))
            monthly_data[var]['model'].append(getter_func(model_file))
    
    # Convert lists to numpy arrays
    for var in variables:
        monthly_data[var]['spcam'] = np.array(monthly_data[var]['spcam'])
        monthly_data[var]['model'] = np.array(monthly_data[var]['model'])
    
    # Get model name from path
    model_name = model_path.split('/')[-2]
    
    # Define seasons
    seasons = {
        'annual': slice(None),
        'djf': [11,0,1],
        'mam': [2,3,4],
        'jja': [5,6,7],
        'son': [8,9,10]
    }
    
    metrics = {}
    
    # Calculate metrics for each variable and season
    for var in variables:
        metrics[var] = {}
        
        for season, idx in seasons.items():
            spcam_mean = np.mean(monthly_data[var]['spcam'][idx,:,:], axis=0)
            model_mean = np.mean(monthly_data[var]['model'][idx,:,:], axis=0)
            
            # Calculate and save bias
            bias = model_mean - spcam_mean
            np.save(f'{model_name}_{var}_{season}_bias.npy', bias)
            
            # Calculate metrics
            metrics[var][season] = {
                'correlation': float(spatial_correlation(spcam_mean, model_mean, lat)),  # Convert to float for JSON serialization
                'rmse': float(area_RMSE(model_mean, spcam_mean, lat))  # Convert to float for JSON serialization
            }
            
            print(f"{var} {season}:")
            print(f"  RMSE: {metrics[var][season]['rmse']:.4f}")
            print(f"  Correlation: {metrics[var][season]['correlation']:.4f}")
    
    # Save metrics to JSON file
    metrics_filename = f'{model_name}_metrics_{year}.json'
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_filename}")
    
    return metrics

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    print(sys.argv[2])
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 1998
    metrics = compute_annual_rmse(sys.argv[1], sys.argv[2], year)
    print("\nMetrics Summary:")
    print("-" * 50)
    for var in metrics:
        print(f"\n{var.upper()}:")
        for season in metrics[var]:
            print(f"\n  {season.upper()}:")
            print(f"    RMSE: {metrics[var][season]['rmse']:.4f}")
            print(f"    Correlation: {metrics[var][season]['correlation']:.4f}")
    print("-" * 50)