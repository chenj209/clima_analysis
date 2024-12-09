import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Configuration flags
USE_HISTOGRAM = True  # Set to False to use KDE
USE_OCEAN_MASK = True  # Set to False to use all data (land + ocean)

# Set parameters
start_date = '1999-01-01'
end_date = '2003-12-31'
lat_range = (-30, 30)  # Updated latitude range
var_name = 'precip'

# File paths
file_paths = {
    'TRMM': './TRMM.day.1999-2003.nc',
    'CAM5': './2021_10_21.cam.h1.day.1999-2003.nc',
    'NNCAM(PhyC)': './baseline-nn_rh-daily_cp/baseline_nn_rh.cam.h1.day.1999-2003.nc',
    'NNCAM': './2021_11_15.cam.h1.day.1999-2003.nc',
    'SPCAM': './spcam.cam.h1.day.1999-2003.nc'
}

# Plot configuration
PLOT_CONFIG = {
    'main': {
        'xlabel': 'mm/day',
        'ylabel': 'Probability Density (%)',
        'title': 'Precipitation 1999-2003 Bin size: 1mm/day',
        'ylim': (1e-5, 100),
        'xlim': (0, 160),
        'xticks': range(0, 161, 20),
        'yticks': {
            'values': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            'labels': ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1', '10', '100']
        }
    },
    'zoomed': {
        'xlabel': 'mm/day',
        'ylabel': 'Probability Density (%)',
        'title': 'Precipitation 1999-2003 (Zoomed)',
        'ylim': (1e-5, 100),
        'xlim': (2e-1, 40)
    },
    'common': {
        'figsize': (10, 8),
        'legend_loc': 'upper right',
        'legend_fontsize': 12,
        'label_fontsize': 12,
        'title_fontsize': 14,
        'tick_fontsize': 10,
        'colors': ['black', 'orange', 'red', 'green', 'blue']
    }
}

def load_landfrac(spcam_file):
    with xr.open_dataset(spcam_file) as ds:
        landfrac = ds['LANDFRAC'].isel(time=0)
    return landfrac

def load_data(file_path, var_name, start_date, end_date, lat_range, landfrac):
    ds = xr.open_dataset(file_path)
    data = ds[var_name].sel(lat=slice(*lat_range), time=slice(start_date, end_date))
    # if data is empty, try loading without time
    if data.isnull().all():
        data = ds[var_name].sel(lat=slice(*lat_range))
    
    # Ensure landfrac has the same lat/lon coordinates as the data
    landfrac = landfrac.sel(lat=slice(*lat_range))
    landfrac = landfrac.interp_like(data.isel(time=0))
    
    return data, landfrac

def create_ocean_mask(landfrac, threshold=0):
    return landfrac <= threshold

def apply_ocean_mask(data, ocean_mask):
    return data.where(ocean_mask)

def calculate_pdf(data, config):
    bin_min, bin_max = config['xlim']
    num_bins = bin_max + 1  # Use bin_max + 1 as the number of bins
    
    # Convert xarray DataArray to flattened numpy array
    flat_data = data.values.flatten()
    
    if USE_HISTOGRAM:
        hist, bin_edges = np.histogram(flat_data, bins=np.linspace(bin_min, bin_max, num_bins), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist * 100  # Multiply by 100 to convert to percentage
    else:
        kde = stats.gaussian_kde(flat_data)
        x = np.linspace(bin_min, bin_max, num_bins)
        y = kde(x) * 100  # Multiply by 100 to convert to percentage
        return x, y

def main():
    spcam_name = "spcam.baseline"
    spcam_path = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
    spcam_file = spcam_path + spcam_name + ".cam.h0.1998-01.nc"
    landfrac = load_landfrac(spcam_file)

    # Create and save ocean mask
    ocean_mask = create_ocean_mask(landfrac)
    np.save('ocean_mask.npy', ocean_mask.values)

    # Load and process data
    print("Starting data loading and processing")
    results = {}
    for name, file_path in file_paths.items():
        print(f"Loading data for {name} from {file_path}")
        data, landfrac_for_data = load_data(file_path, var_name, start_date, end_date, lat_range, landfrac)
        if name != 'TRMM':
            print(f"Converting {name} data to mm/day")
            data *= 24 * 3600 * 1000  # Convert to mm/day
        
        if USE_OCEAN_MASK:
            # Create and apply ocean mask
            ocean_mask = create_ocean_mask(landfrac_for_data)
            data = apply_ocean_mask(data, ocean_mask)
        
        print(f"Calculating {'histogram' if USE_HISTOGRAM else 'KDE'} for {name}")
        x, y = calculate_pdf(data, PLOT_CONFIG['main'])
        results[name] = (x, y)
    print("Data loading and processing completed")

    # Create figures directory if it doesn't exist
    figures_dir = './figures'
    os.makedirs(figures_dir, exist_ok=True)

    # Update plot titles and filenames
    lat_range_str = f"{abs(lat_range[0])}S-{lat_range[1]}N"
    mask_str = "ocean_only" if USE_OCEAN_MASK else "land_and_ocean"
    title_suffix = f" ({lat_range_str}, {'Ocean Only' if USE_OCEAN_MASK else 'Land and Ocean'})"
    filename_suffix = f"_{lat_range_str}_{mask_str}"

    # Plotting
    print("\nStarting main plot")
    plt.figure(figsize=PLOT_CONFIG['common']['figsize'])
    
    for (name, (x, y)), color in zip(results.items(), PLOT_CONFIG['common']['colors']):
        print(f"Plotting {name}")
        plt.semilogy(x, y, label=name, color=color, linewidth=2)

    print("Setting plot labels and title")
    plt.xlabel(PLOT_CONFIG['main']['xlabel'], fontweight='bold', fontsize=PLOT_CONFIG['common']['label_fontsize'])
    plt.ylabel(PLOT_CONFIG['main']['ylabel'], fontweight='bold', fontsize=PLOT_CONFIG['common']['label_fontsize'])
    plt.title(f"{PLOT_CONFIG['main']['title']}{title_suffix} ({'Histogram' if USE_HISTOGRAM else 'KDE'})", fontweight='bold', fontsize=PLOT_CONFIG['common']['title_fontsize'])
    plt.legend(loc=PLOT_CONFIG['common']['legend_loc'], fontsize=PLOT_CONFIG['common']['legend_fontsize'])
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    print("Setting plot limits and ticks")
    plt.ylim(*PLOT_CONFIG['main']['ylim'])
    plt.xlim(*PLOT_CONFIG['main']['xlim'])
    
    plt.xticks(PLOT_CONFIG['main']['xticks'], fontsize=PLOT_CONFIG['common']['tick_fontsize'])
    plt.yticks(PLOT_CONFIG['main']['yticks']['values'],
               PLOT_CONFIG['main']['yticks']['labels'],
               fontsize=PLOT_CONFIG['common']['tick_fontsize'])

    print("Saving main plot")
    plt.savefig(os.path.join(figures_dir, f'Precip_PDF_NNCAM0903_python{filename_suffix}_{"histogram" if USE_HISTOGRAM else "kde"}.pdf'))
    plt.close()

    # Zoomed plot
    print("\nStarting zoomed plot")
    plt.figure(figsize=PLOT_CONFIG['common']['figsize'])
    for (name, (x, y)), color in zip(results.items(), PLOT_CONFIG['common']['colors']):
        print(f"Plotting {name} (zoomed)")
        plt.loglog(x, y, label=name, color=color, linewidth=2)

    print("Setting zoomed plot labels and title")
    plt.xlabel(PLOT_CONFIG['zoomed']['xlabel'], fontweight='bold', fontsize=PLOT_CONFIG['common']['label_fontsize'])
    plt.ylabel(PLOT_CONFIG['zoomed']['ylabel'], fontweight='bold', fontsize=PLOT_CONFIG['common']['label_fontsize'])
    plt.title(f"{PLOT_CONFIG['zoomed']['title']}{title_suffix} ({'Histogram' if USE_HISTOGRAM else 'KDE'})", fontweight='bold', fontsize=PLOT_CONFIG['common']['title_fontsize'])
    plt.legend(loc=PLOT_CONFIG['common']['legend_loc'], fontsize=PLOT_CONFIG['common']['legend_fontsize'])
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    print("Setting zoomed plot limits")
    plt.ylim(*PLOT_CONFIG['zoomed']['ylim'])
    plt.xlim(*PLOT_CONFIG['zoomed']['xlim'])

    plt.xticks(fontsize=PLOT_CONFIG['common']['tick_fontsize'])
    plt.yticks(fontsize=PLOT_CONFIG['common']['tick_fontsize'])

    print("Saving zoomed plot")
    plt.savefig(os.path.join(figures_dir, f'Precip_zoom_PDF_NNCAM0903_python{filename_suffix}_{"histogram" if USE_HISTOGRAM else "kde"}.pdf'))
    plt.close()

    print("Plotting process completed")

if __name__ == "__main__":
    main()