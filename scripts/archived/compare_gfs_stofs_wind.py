#!/usr/bin/env python3
"""
Compare GFS f000 wind with STOFS wind forcing to verify temporal alignment.

This script:
1. Reads GFS GRIB2 f000 file for a given date
2. Reads STOFS uvgrd10m file (first timestep)
3. Creates side-by-side comparison plots over US East Coast
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Try to import required libraries
try:
    import xarray as xr
except ImportError:
    os.system("pip install xarray netCDF4")
    import xarray as xr

try:
    import pygrib
    HAS_PYGRIB = True
except ImportError:
    HAS_PYGRIB = False
    print("pygrib not available, trying cfgrib...")
    try:
        import cfgrib
        HAS_CFGRIB = True
    except ImportError:
        HAS_CFGRIB = False


def read_gfs_wind(gfs_path: Path):
    """Read U10 and V10 from GFS GRIB2 file."""
    print(f"Reading GFS: {gfs_path}")

    if HAS_PYGRIB:
        return read_gfs_wind_pygrib(gfs_path)
    elif HAS_CFGRIB:
        return read_gfs_wind_cfgrib(gfs_path)
    else:
        raise ImportError("Neither pygrib nor cfgrib available!")


def read_gfs_wind_pygrib(gfs_path: Path):
    """Read GFS wind using pygrib."""
    grbs = pygrib.open(str(gfs_path))

    # Find 10m U and V wind components
    u10_grb = None
    v10_grb = None

    for grb in grbs:
        if grb.parameterName == '10 metre U wind component' or \
           (grb.shortName == '10u' and grb.level == 10):
            u10_grb = grb
        elif grb.parameterName == '10 metre V wind component' or \
             (grb.shortName == '10v' and grb.level == 10):
            v10_grb = grb

        if u10_grb and v10_grb:
            break

    if u10_grb is None or v10_grb is None:
        # Try alternative search
        grbs.rewind()
        for grb in grbs:
            name = grb.name.lower() if hasattr(grb, 'name') else ''
            if '10 metre u' in name or 'u-component of wind' in name:
                if grb.level == 10 or 'surface' in str(grb.typeOfLevel).lower():
                    u10_grb = grb
            elif '10 metre v' in name or 'v-component of wind' in name:
                if grb.level == 10 or 'surface' in str(grb.typeOfLevel).lower():
                    v10_grb = grb

    if u10_grb is None or v10_grb is None:
        raise ValueError("Could not find 10m wind components in GRIB file")

    u10 = u10_grb.values
    v10 = v10_grb.values
    lat, lon = u10_grb.latlons()
    valid_time = u10_grb.validDate

    grbs.close()

    # Convert longitude from 0-360 to -180-180
    lon = np.where(lon > 180, lon - 360, lon)

    return u10, v10, lat, lon, valid_time


def read_gfs_wind_cfgrib(gfs_path: Path):
    """Read GFS wind using cfgrib/xarray."""
    # Read U component
    ds_u = xr.open_dataset(gfs_path, engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround',
                                                              'level': 10,
                                                              'shortName': '10u'}})

    # Read V component
    ds_v = xr.open_dataset(gfs_path, engine='cfgrib',
                          backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround',
                                                              'level': 10,
                                                              'shortName': '10v'}})

    u10 = ds_u['u10'].values
    v10 = ds_v['v10'].values
    lat = ds_u['latitude'].values
    lon = ds_u['longitude'].values
    valid_time = ds_u['valid_time'].values

    # Convert longitude from 0-360 to -180-180
    lon = np.where(lon > 180, lon - 360, lon)

    return u10, v10, lat, lon, valid_time


def read_stofs_wind_remote(url: str, timestep: int = 0):
    """Read STOFS wind from remote URL (first timestep only)."""
    print(f"Reading STOFS from: {url}")
    print(f"  Timestep: {timestep}")

    # Try to open with xarray (will download minimal data)
    try:
        # Open the dataset
        ds = xr.open_dataset(url, engine='netcdf4')

        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Dimensions: {dict(ds.dims)}")

        # Get wind components at first timestep
        if 'u10' in ds:
            u10 = ds['u10'].isel(time=timestep).values
            v10 = ds['v10'].isel(time=timestep).values
        elif 'uwind' in ds:
            u10 = ds['uwind'].isel(time=timestep).values
            v10 = ds['vwind'].isel(time=timestep).values
        else:
            print(f"  Available variables: {list(ds.data_vars)}")
            raise ValueError("Cannot find wind variables")

        # Get coordinates
        if 'x' in ds and 'y' in ds:
            lon = ds['x'].values
            lat = ds['y'].values
        elif 'longitude' in ds and 'latitude' in ds:
            lon = ds['longitude'].values
            lat = ds['latitude'].values
        elif 'lon' in ds and 'lat' in ds:
            lon = ds['lon'].values
            lat = ds['lat'].values
        else:
            print(f"  Available coords: {list(ds.coords)}")
            raise ValueError("Cannot find coordinates")

        time = ds['time'].isel(time=timestep).values

        ds.close()
        return u10, v10, lat, lon, time

    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None, None, None


def read_stofs_wind_local(filepath: Path, timestep: int = 0):
    """Read STOFS wind from local file."""
    print(f"Reading STOFS from: {filepath}")

    ds = xr.open_dataset(filepath)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")

    # Detect variable names
    u_var = None
    v_var = None
    for var in ds.data_vars:
        if 'uwnd' in var.lower() or 'u10' in var.lower() or 'uwind' in var.lower():
            u_var = var
        if 'vwnd' in var.lower() or 'v10' in var.lower() or 'vwind' in var.lower():
            v_var = var

    if u_var is None or v_var is None:
        print(f"  Cannot find wind variables in: {list(ds.data_vars)}")
        return None, None, None, None, None

    print(f"  Using: {u_var}, {v_var}")

    u10 = ds[u_var].isel(time=timestep).values
    v10 = ds[v_var].isel(time=timestep).values

    # Get coordinates - STOFS uses unstructured mesh
    if 'x' in ds.coords:
        lon = ds['x'].values
        lat = ds['y'].values
    elif 'lon' in ds:
        lon = ds['lon'].values
        lat = ds['lat'].values

    time = ds['time'].isel(time=timestep).values

    ds.close()
    return u10, v10, lat, lon, time


def compute_wind_speed(u, v):
    """Compute wind speed magnitude."""
    return np.sqrt(u**2 + v**2)


def plot_comparison(gfs_data, stofs_data, output_path: Path, region='east_coast'):
    """Create side-by-side comparison plots."""

    u_gfs, v_gfs, lat_gfs, lon_gfs, time_gfs = gfs_data
    u_stofs, v_stofs, lat_stofs, lon_stofs, time_stofs = stofs_data

    # Compute wind speeds
    ws_gfs = compute_wind_speed(u_gfs, v_gfs)
    ws_stofs = compute_wind_speed(u_stofs, v_stofs) if u_stofs is not None else None

    # Region bounds for US East Coast
    if region == 'east_coast':
        lon_min, lon_max = -82, -65
        lat_min, lat_max = 24, 46
    elif region == 'gulf':
        lon_min, lon_max = -100, -80
        lat_min, lat_max = 18, 31
    else:
        lon_min, lon_max = -180, 180
        lat_min, lat_max = -90, 90

    # Create figure
    if ws_stofs is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]

    # Common colormap settings
    vmin, vmax = 0, 20
    cmap = 'viridis'

    # Plot GFS
    ax = axes[0]

    # For GFS, we need to handle the 2D lat/lon grid
    if len(lon_gfs.shape) == 1:
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_gfs, lat_gfs)
    else:
        lon_grid, lat_grid = lon_gfs, lat_gfs

    # Mask to region
    mask_lon = (lon_grid >= lon_min) & (lon_grid <= lon_max)
    mask_lat = (lat_grid >= lat_min) & (lat_grid <= lat_max)
    mask = mask_lon & mask_lat

    # Create masked data for plotting
    ws_gfs_masked = np.where(mask, ws_gfs, np.nan)

    im = ax.pcolormesh(lon_grid, lat_grid, ws_gfs_masked,
                       vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'GFS f000 Wind Speed (m/s)\n{time_gfs}')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Wind Speed (m/s)')

    # Add wind vectors (subsampled)
    skip = 8  # Plot every 8th vector
    ax.quiver(lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
              u_gfs[::skip, ::skip], v_gfs[::skip, ::skip],
              scale=200, alpha=0.5, width=0.002)

    # Plot STOFS if available
    if ws_stofs is not None and len(axes) > 1:
        ax = axes[1]

        # STOFS is on unstructured mesh - scatter plot
        # Mask to region
        mask = ((lon_stofs >= lon_min) & (lon_stofs <= lon_max) &
                (lat_stofs >= lat_min) & (lat_stofs <= lat_max))

        sc = ax.scatter(lon_stofs[mask], lat_stofs[mask],
                       c=ws_stofs[mask], s=1,
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'STOFS Wind Forcing (m/s)\n{time_stofs}')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label='Wind Speed (m/s)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare GFS and STOFS wind forcing')
    parser.add_argument('--gfs', type=str, required=True, help='Path to GFS GRIB2 file')
    parser.add_argument('--stofs', type=str, help='Path or URL to STOFS uvgrd10m file')
    parser.add_argument('--stofs-timestep', type=int, default=0, help='STOFS timestep (0=first)')
    parser.add_argument('--output', type=str, default='wind_comparison.png')
    parser.add_argument('--region', type=str, default='east_coast',
                       choices=['east_coast', 'gulf', 'global'])
    args = parser.parse_args()

    # Read GFS
    gfs_data = read_gfs_wind(Path(args.gfs))
    print(f"GFS valid time: {gfs_data[4]}")
    print(f"GFS shape: {gfs_data[0].shape}")
    print(f"GFS lat range: {gfs_data[2].min():.1f} to {gfs_data[2].max():.1f}")
    print(f"GFS lon range: {gfs_data[3].min():.1f} to {gfs_data[3].max():.1f}")

    # Read STOFS if provided
    if args.stofs:
        if args.stofs.startswith('http'):
            stofs_data = read_stofs_wind_remote(args.stofs, args.stofs_timestep)
        else:
            stofs_data = read_stofs_wind_local(Path(args.stofs), args.stofs_timestep)

        if stofs_data[0] is not None:
            print(f"STOFS time: {stofs_data[4]}")
            print(f"STOFS nodes: {len(stofs_data[0])}")
    else:
        stofs_data = (None, None, None, None, None)

    # Create comparison plot
    plot_comparison(gfs_data, stofs_data, Path(args.output), args.region)


if __name__ == '__main__':
    main()
