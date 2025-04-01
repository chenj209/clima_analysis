import numpy as np
import pandas as pd
import netCDF4 as nc
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import re
import argparse
import xarray as xr
import calendar

def parse_args():
    parser = argparse.ArgumentParser(description='Find regions with strong precipitation events')
    parser.add_argument('--threshold', type=float, default=200,
                      help='Precipitation threshold in mm/day (default: 200)')
    parser.add_argument('--consecutive-hours', type=int, default=3,
                      help='Number of consecutive hours required above threshold (default: 3)')
    return parser.parse_args()

def load_precipitation_data():
    """Load precipitation data from all files."""
    nc_dir_path = "/Users/jianda/Projects/experience_replay_data/nncam/"
    files = os.listdir(nc_dir_path)
    
    all_data = []
    all_times = []
    time_data_pairs = []  # Store (time, data) pairs for sorting
    
    for file in files:
        if "h1" in file and file.endswith(".nc"):
            print(f"\nLoading {file}...")
            try:
                nc_file = nc.Dataset(nc_dir_path + file)
                precip = nc_file["cp"][:]
                precip = precip * 24 * 60 * 60 * 1000  # Convert to mm/day
                times = nc_file["time"]
                time_values = nc.num2date(times[:], times.units)
                
                print(f"Shape of precipitation data: {precip.shape}")
                print(f"Time range: {time_values[0]} to {time_values[-1]}")
                print(f"Number of time steps: {len(time_values)}")
                
                # Reshape to (time, lon*lat)
                n_time = precip.shape[0]
                n_grid = precip.shape[1] * precip.shape[2]
                precip_reshaped = precip.reshape(n_time, n_grid)
                
                # Store time and data together
                time_data_pairs.extend([(t, d) for t, d in zip(time_values, precip_reshaped)])
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
    
    if not time_data_pairs:
        raise ValueError("No data files could be loaded")
    
    # Sort by time
    time_data_pairs.sort(key=lambda x: x[0])
    
    # Separate time and data after sorting
    all_times = [pair[0] for pair in time_data_pairs]
    all_data = [pair[1] for pair in time_data_pairs]
    
    # Stack the data
    precip_data = np.stack(all_data, axis=0)
    print(f"\nFinal data shape: {precip_data.shape}")
    print(f"Total number of time steps: {len(all_times)}")
    
    return precip_data, all_times

def load_precip_data(nc_dir_path, exclude_pattern=None, include_pattern=None):
    """
    Load precipitation data from h1.nc files in the given directory.
    Args:
        nc_dir_path (str): The directory path containing the h1.nc files.
        include_pattern (str): A regex pattern to include only specific files.
        exclude_pattern (str): A regex pattern to exclude specific files. include pattern is proritized over exclude pattern.
    Returns:
        pd.DataFrame: A concatenated DataFrame of precipitation data with time index.
    """
    dfs = []
    files = os.listdir(nc_dir_path)
    if exclude_pattern is not None:
        files = [file for file in files if not re.search(exclude_pattern, file)]
        excluded_files = [file for file in files if re.search(exclude_pattern, file)]
    if include_pattern is not None:
        included_files = [file for file in excluded_files if re.search(include_pattern, file)]
        files.extend(included_files)
    lat = None
    lon = None
    for file in files:
        if "h1" in file and file.endswith(".nc"):
            print(file)
            nc_file = nc.Dataset(nc_dir_path + file)
            if lat is None:
                lat = nc_file["lat"][:]
            if lon is None:
                lon = nc_file["lon"][:]
            nc_data = nc_file["cp"][:]
            nc_data = nc_data * 24 * 60 * 60 * 1000
            print(nc_data.mean())
            time = nc_file["time"]
            # Convert time to datetime
            time_values = nc.num2date(time[:], time.units)
            # Reshape precipitation arrays to 2D (time, lat*lon)
            nc_data_flat = nc_data.reshape(nc_data.shape[0], -1)
            # Convert cftime DatetimeGregorian to pandas datetime
            time_values_pd = pd.to_datetime([str(t) for t in time_values])
            # Create DataFrames with time index
            nc_df = pd.DataFrame(nc_data_flat, index=time_values_pd)
            dfs.append(nc_df)
            nc_file.close()
    concat_df = pd.concat(dfs)
    concat_df = concat_df.sort_index()
    time_values = concat_df.index
    return concat_df, time_values, lat, lon


def find_strong_precip_regions(precip_data, threshold, consecutive_hours=1, time_values=None):
    """Find regions with strong precipitation events.
    
    Args:
        precip_data: Array of precipitation data
        threshold: Precipitation threshold in mm/day
        consecutive_hours: Number of consecutive hours required
        time_values: List of datetime objects for each timestep
    """
    # Find indices where precipitation exceeds threshold
    strong_precip_indices = np.where(precip_data > threshold)
    time_indices = strong_precip_indices[0]
    grid_indices = strong_precip_indices[1]
    
    print(f"\nAnalysis for threshold {threshold} mm/day:")
    print(f"Total number of strong precipitation events: {len(time_indices)}")
    
    # Calculate frequencies for all grid points
    n_lon = 144  # Number of longitude points
    n_lat = 96   # Number of latitude points
    total_timesteps = precip_data.shape[0]
    
    # Create a 2D array to store frequencies
    frequencies = np.zeros((n_lat, n_lon))
    
    # Count occurrences at each grid point
    for t, g in zip(time_indices, grid_indices):
        lon_idx = g % n_lon
        lat_idx = g // n_lon
        frequencies[lat_idx, lon_idx] += 1
    
    # Convert to percentages
    frequencies = (frequencies / total_timesteps) * 100
    
    # Find the location with maximum frequency
    max_lat_idx, max_lon_idx = np.unravel_index(np.argmax(frequencies), frequencies.shape)
    max_freq = frequencies[max_lat_idx, max_lon_idx]
    max_freq_grid_idx = max_lat_idx * n_lon + max_lon_idx
    
    # Get average precipitation at this location
    location_precip = precip_data[:, max_freq_grid_idx]
    avg_precip = np.mean(location_precip)
    
    # Get monthly distribution for all events
    all_event_times = [time_values[i] for i in time_indices]
    all_months = [t.month for t in all_event_times]
    monthly_counts = np.zeros(12)
    for month in all_months:
        monthly_counts[month-1] += 1
    
    # Get events at most frequent location for detailed analysis
    strong_precip_times = time_indices[grid_indices == max_freq_grid_idx]
    print(f"Number of strong events at most frequent location: {len(strong_precip_times)}")
    
    # Get the actual times for these events
    event_times = [time_values[i] for i in strong_precip_times] if time_values is not None else []
    print("\nStrong precipitation event times at most frequent location:")
    for t in event_times:
        print(f"  {t}")
    
    # Find extended time windows
    extended_windows = []
    if consecutive_hours > 1:
        # Sort time indices
        sorted_indices = np.sort(strong_precip_times)
        current_window = [sorted_indices[0]]
        
        for idx in sorted_indices[1:]:
            if idx - current_window[-1] <= consecutive_hours:
                current_window.append(idx)
            else:
                if len(current_window) >= consecutive_hours:
                    extended_windows.append(current_window)
                current_window = [idx]
        
        if len(current_window) >= consecutive_hours:
            extended_windows.append(current_window)
    
    # For plotting, we want to show all individual events, not just the start of consecutive windows
    return {
        'lat_idx': max_lat_idx,
        'lon_idx': max_lon_idx,
        'frequency': max_freq,
        'frequencies': frequencies,  # 2D array of frequencies
        'avg_precip': avg_precip,
        'monthly_counts': monthly_counts,  # Now contains all events
        'extended_windows': extended_windows,
        'event_times': event_times,
        'all_time_indices': time_indices,  # Add all time indices
        'all_grid_indices': grid_indices    # Add all grid indices
    }

def analyze_time_windows(strong_precip_mask, time_values, grid_idx, window_size=30):
    """Analyze time windows of strong precipitation events"""
    # Get the time series for the most frequent region
    region_events = strong_precip_mask[:, grid_idx]
    
    # Find all time indices where strong precipitation occurs
    event_times = time_values[region_events]
    
    # Group events by month to find seasonal patterns
    event_months = pd.Series(event_times).dt.month
    monthly_counts = event_months.value_counts().sort_index()
    
    # Find consecutive events
    consecutive_events = []
    current_window = []
    
    for i in range(len(region_events)):
        if region_events[i]:
            current_window.append(time_values[i])
        elif current_window:
            if len(current_window) >= window_size:
                consecutive_events.append((current_window[0], current_window[-1]))
            current_window = []
    
    # Check the last window if it exists
    if current_window and len(current_window) >= window_size:
        consecutive_events.append((current_window[0], current_window[-1]))
    
    return monthly_counts, consecutive_events

def plot_strong_precip_regions(frequencies, lat, lon, monthly_counts, threshold, all_time_indices=None, all_grid_indices=None, time_values=None, precip_data=None, max_freq_grid_idx=None, lat_idx=None, lon_idx=None):
    """Plot the frequency of strong precipitation events and monthly distribution."""
    plt.figure(figsize=(15, 15))
    
    # Plot frequency map
    ax1 = plt.subplot(221, projection=ccrs.PlateCarree(central_longitude=180))
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Create a mesh grid for plotting
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    
    # Plot the frequency data
    c = ax1.contourf(lon_mesh, lat_mesh, frequencies,
                     transform=ccrs.PlateCarree(),
                     cmap='YlOrRd')
    plt.colorbar(c, ax=ax1, label='Frequency (%)')
    
    # Set title
    ax1.set_title(f'Frequency of Precipitation > {threshold} mm/day')
    
    # Add gridlines
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Add box around most frequent location
    box_width = 5  # degrees
    box_height = 5  # degrees
    lon_center = lon[lon_idx]
    lat_center = lat[lat_idx]
    box = plt.Rectangle((lon_center - box_width/2, lat_center - box_height/2), 
                       box_width, box_height,
                       fill=False, color='red', linewidth=2, 
                       transform=ccrs.PlateCarree())
    ax1.add_patch(box)
    
    # Plot monthly distribution for all events
    ax2 = plt.subplot(222)
    months = range(1, 13)
    ax2.bar(months, monthly_counts)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Number of Events')
    ax2.set_title('Monthly Distribution of All Strong Precipitation Events')
    ax2.set_xticks(months)
    ax2.grid(True, alpha=0.3)
    
    # Plot time series of all events
    if all_time_indices is not None and all_grid_indices is not None:
        ax3 = plt.subplot(223)
        # Convert time indices to actual dates
        event_dates = [time_values[i] for i in all_time_indices]
        event_dates = [pd.to_datetime(str(t)) for t in event_dates]
        
        # Count overlapping points at the same time
        time_counts = {}
        for date in event_dates:
            time_counts[date] = time_counts.get(date, 0) + 1
        
        # Plot points with size based on count
        max_count = max(time_counts.values())
        for date, grid_idx in zip(event_dates, all_grid_indices):
            count = time_counts[date]
            size = 10 + (count / max_count) * 90  # Scale size from 10 to 100
            ax3.scatter(date, grid_idx, s=size, color='blue', alpha=0.6)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Grid Point')
        ax3.set_title(f'All Strong Precipitation Events (Total: {len(all_time_indices)})')
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis to show months nicely
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Set y-axis limits to show all grid points
        ax3.set_ylim(0, 13824)  # Total number of grid points (96 * 144)
    
    # Plot time evolution at most frequent location
    if time_values is not None and precip_data is not None and max_freq_grid_idx is not None:
        ax4 = plt.subplot(224)
        location_precip = precip_data[:, max_freq_grid_idx]
        
        # Convert cftime to matplotlib datetime and ensure they're sorted
        time_values_mpl = [pd.to_datetime(str(t)) for t in time_values]
        
        # Format x-axis to show months
        ax4.plot(time_values_mpl, location_precip, 'b-', alpha=0.5, label='Precipitation')
        # Highlight strong precipitation events
        strong_events = location_precip > threshold
        ax4.scatter(np.array(time_values_mpl)[strong_events], 
                   location_precip[strong_events], 
                   color='red', s=20, label=f'Events > {threshold} mm/day')
        ax4.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Precipitation (mm/day)')
        ax4.set_title(f'Time Evolution at ({lon[lon_idx]:.2f}°E, {lat[lat_idx]:.2f}°N) [Grid idx: {max_freq_grid_idx}]')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Format x-axis to show months nicely
        ax4.xaxis.set_major_locator(mdates.MonthLocator())
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'strong_precip_regions_{int(threshold)}mm.png')
    plt.close()

def main():
    """Main function to run the analysis."""
    args = parse_args()
    
    # Load precipitation data
    precip_array, time_values = load_precipitation_data()
    
    # Get lat/lon information from one of the files
    nc_dir_path = "/Users/jianda/Projects/experience_replay_data/nncam/"
    first_file = next(f for f in os.listdir(nc_dir_path) if "h1" in f and f.endswith(".nc"))
    ds = nc.Dataset(os.path.join(nc_dir_path, first_file))
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    ds.close()
    
    # Find strong precipitation regions
    result = find_strong_precip_regions(
        precip_array, 
        threshold=args.threshold,
        consecutive_hours=args.consecutive_hours,
        time_values=time_values
    )
    
    # Get the region with highest frequency
    max_lat = lat[result['lat_idx']]
    max_lon = lon[result['lon_idx']]
    max_freq_grid_idx = result['lat_idx'] * 144 + result['lon_idx']
    
    # Plot results with threshold
    plot_strong_precip_regions(
        result['frequencies'],
        lat, 
        lon, 
        result['monthly_counts'], 
        args.threshold,
        result['all_time_indices'],
        result['all_grid_indices'],
        time_values,
        precip_array,
        max_freq_grid_idx,
        result['lat_idx'],
        result['lon_idx']
    )
    
    print(f"\nStrong Precipitation Analysis Results:")
    print(f"Threshold: {args.threshold} mm/day")
    print(f"Most frequent region: {max_lat:.2f}°N, {max_lon:.2f}°E")
    print(f"Frequency: {result['frequency']:.2f}% of time")
    print(f"Average precipitation in this region: {result['avg_precip']:.2f} mm/day")
    
    print("\nMonthly distribution of all strong precipitation events:")
    for month, count in enumerate(result['monthly_counts'], 1):
        print(f"{calendar.month_name[month]}: {int(count)} events")
    
    if result['extended_windows']:
        print(f"\nFound {len(result['extended_windows'])} extended time windows of strong precipitation")
        for window in result['extended_windows']:
            start_time = time_values[window[0]]
            end_time = time_values[window[-1]]
            print(f"Window: {start_time} to {end_time}")
    else:
        print("\nNo extended time windows of strong precipitation found")
    
    print(f"\nOutput saved as: strong_precip_regions_{int(args.threshold)}mm.png")

if __name__ == "__main__":
    main() 