import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from total_rmse import get_precip

def get_Q_all_levels(file_path):
    """Get specific humidity for all vertical levels"""
    with nc.Dataset(file_path) as fid:
        Q = fid.variables['Q'][0,:]  # Get all levels
        lev = fid.variables['lev'][:]  # Get pressure levels
    return Q, lev

def get_T_all_levels(file_path):
    """Get temperature for all vertical levels"""
    with nc.Dataset(file_path) as fid:
        T = fid.variables['T'][0,:]  # Get all levels
        lev = fid.variables['lev'][:]  # Get pressure levels
    return T, lev

def compute_zonal_mean(data, lat):
    """Compute zonal mean of data"""
    return np.mean(data, axis=1)

def compute_zonal_bias(model_data, spcam_data, lat):
    """Compute zonal mean bias between model and SPCAM"""
    model_zonal = compute_zonal_mean(model_data, lat)
    spcam_zonal = compute_zonal_mean(spcam_data, lat)
    return model_zonal - spcam_zonal

def plot_zonal_mean_and_bias(lat, lev, spcam_data, model_data, variable_name, output_dir):
    """Plot zonal mean and bias for a given variable with vertical levels"""
    # Compute zonal means and bias
    spcam_zonal = compute_zonal_mean(spcam_data, lat)
    model_zonal = compute_zonal_mean(model_data, lat)
    bias = model_zonal - spcam_zonal
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot zonal means
    im1 = ax1.contourf(lat, lev, spcam_zonal.T, cmap='RdBu_r')
    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Pressure (hPa)')
    ax1.set_title(f'SPCAM {variable_name}')
    plt.colorbar(im1, ax=ax1, label=variable_name)
    ax1.invert_yaxis()  # Invert y-axis to show pressure decreasing upward
    
    # Plot model zonal mean
    im2 = ax2.contourf(lat, lev, model_zonal.T, cmap='RdBu_r')
    ax2.set_xlabel('Latitude')
    ax2.set_ylabel('Pressure (hPa)')
    ax2.set_title(f'Model {variable_name}')
    plt.colorbar(im2, ax=ax2, label=variable_name)
    ax2.invert_yaxis()
    
    # Plot bias
    im3 = ax3.contourf(lat, lev, bias.T, cmap='RdBu_r')
    ax3.set_xlabel('Latitude')
    ax3.set_ylabel('Pressure (hPa)')
    ax3.set_title(f'{variable_name} Bias')
    plt.colorbar(im3, ax=ax3, label='Bias')
    ax3.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'zonal_mean_{variable_name}.png'))
    plt.close()

def plot_spatial_mean(data, lat, lon, variable_name, output_dir):
    """Plot spatial mean of data"""
    plt.figure(figsize=(10, 6))
    plt.contourf(lon, lat, data, cmap='RdBu_r')
    plt.colorbar(label=variable_name)
    plt.title(f'Annual Mean {variable_name}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, f'spatial_mean_{variable_name}.png'))
    plt.close()

def analyze_zonal_means(spcam_path, model_path, year=1998, output_dir='zonal_analysis_output'):
    """Main function to compute and plot zonal means and biases"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample data for coordinates
    sample_data = nc.Dataset(os.path.join(spcam_path, f"spcam.baseline.cam.h0.{year}-01.nc"))
    lat = sample_data.variables["lat"][:]
    lon = sample_data.variables["lon"][:]
    lev = sample_data.variables["lev"][:]
    
    # Initialize data storage
    variables = {
        'Q': get_Q_all_levels,
        'T': get_T_all_levels,
        'precip': get_precip
    }
    
    monthly_data = {var: {'spcam': [], 'model': []} for var in variables}
    
    # Load all monthly data
    for i in range(1, 13):
        spcam_file = os.path.join(spcam_path, f"spcam.baseline.cam.h0.{year}-{i:02d}.nc")
        model_files = [f for f in os.listdir(model_path) if 'cam.h0' in f]
        if model_files:
            model_prefix = model_files[0].split('.cam.h0')[0]
            model_file = os.path.join(model_path, f"{model_prefix}.cam.h0.{year}-{i:02d}.nc")
        else:
            raise FileNotFoundError(f"No cam.h0 files found in {model_path}")
        
        for var, getter_func in variables.items():
            if var in ['Q', 'T']:
                spcam_data, _ = getter_func(spcam_file)
                model_data, _ = getter_func(model_file)
            else:
                spcam_data = getter_func(spcam_file)
                model_data = getter_func(model_file)
            
            monthly_data[var]['spcam'].append(spcam_data)
            monthly_data[var]['model'].append(model_data)
    
    # Convert lists to numpy arrays and compute annual means
    annual_means = {}
    for var in variables:
        spcam_data = np.array(monthly_data[var]['spcam'])
        model_data = np.array(monthly_data[var]['model'])
        
        # Compute annual means
        spcam_annual = np.mean(spcam_data, axis=0)
        model_annual = np.mean(model_data, axis=0)
        annual_means[var] = {'spcam': spcam_annual, 'model': model_annual}
        
        if var in ['Q', 'T']:
            # Plot zonal means and biases with vertical levels
            plot_zonal_mean_and_bias(lat, lev, spcam_annual, model_annual, var, output_dir)
            
            # Plot spatial means at specific levels (e.g., 850hPa and 500hPa)
            for level_idx, level in enumerate(lev):
                if level in [850, 500]:
                    plot_spatial_mean(spcam_annual[level_idx], lat, lon, f'SPCAM {var} {level}hPa', output_dir)
                    plot_spatial_mean(model_annual[level_idx], lat, lon, f'Model {var} {level}hPa', output_dir)
                    bias = model_annual[level_idx] - spcam_annual[level_idx]
                    plot_spatial_mean(bias, lat, lon, f'{var} {level}hPa Bias', output_dir)
        else:
            # For precipitation, use the original plotting functions
            plot_zonal_mean_and_bias(lat, None, spcam_annual, model_annual, var, output_dir)
            plot_spatial_mean(spcam_annual, lat, lon, f'SPCAM {var}', output_dir)
            plot_spatial_mean(model_annual, lat, lon, f'Model {var}', output_dir)
            bias = model_annual - spcam_annual
            plot_spatial_mean(bias, lat, lon, f'{var} Bias', output_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python zonal_mean_analysis.py <spcam_path> <model_path> [year]")
        sys.exit(1)
    
    spcam_path = sys.argv[1]
    model_path = sys.argv[2]
    year = int(sys.argv[3]) if len(sys.argv) > 3 else 1998
    
    analyze_zonal_means(spcam_path, model_path, year) 