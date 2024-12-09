import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def load_landfrac(spcam_file):
    with xr.open_dataset(spcam_file) as ds:
        landfrac = ds['LANDFRAC'].isel(time=0)
    return landfrac

def create_ocean_mask(landfrac):
    return landfrac <= 0

def main():
    # Load landfrac to get lat and lon coordinates
    spcam_name = "spcam.baseline"
    spcam_path = "/temp_share/nncam-cases/nncam-diag_spcam/spcam.baseline/atm/hist/"
    spcam_file = spcam_path + spcam_name + ".cam.h0.1998-01.nc"
    landfrac = load_landfrac(spcam_file)

    # Create ocean mask
    ocean_mask = create_ocean_mask(landfrac)

    # Create a figure with a global projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot the ocean mask
    im = ax.imshow(ocean_mask, transform=ccrs.PlateCarree(), cmap='coolwarm', 
                   extent=[landfrac.lon.min(), landfrac.lon.max(), 
                           landfrac.lat.min(), landfrac.lat.max()],
                   vmin=0, vmax=1)

    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Ocean Mask (0: Land, 1: Ocean)')

    # Set title
    plt.title('Ocean Mask (landfrac <= 0)')

    # Save the figure
    plt.savefig('ocean_mask_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save the ocean mask as a .npy file
    np.save('ocean_mask.npy', ocean_mask.values)

if __name__ == "__main__":
    main()