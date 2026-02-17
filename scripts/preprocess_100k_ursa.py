#!/usr/bin/env python3
"""
Preprocess STOFS data for 100k mesh on URSA.
Extracts water elevation and interpolates GFS forcing to mesh nodes.
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Domain bounds
LON_MIN, LON_MAX = -77.0, -72.0
LAT_MIN, LAT_MAX = 37.0, 42.0


def process_single_date(args):
    """Process a single date - designed for parallel execution."""
    date_str, mesh_file, stofs_dir, gfs_dir, output_dir = args

    output_file = output_dir / f'processed_{date_str}.npz'

    # Skip if already processed
    if output_file.exists():
        return f"{date_str}: Already processed, skipping"

    try:
        # Load mesh
        mesh = np.load(mesh_file)
        lon = mesh['lon']
        lat = mesh['lat']
        original_indices = mesh['original_stofs_indices']
        num_nodes = len(lon)

        # Find STOFS file
        stofs_file = stofs_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'
        if not stofs_file.exists():
            return f"{date_str}: STOFS file not found"

        # Extract elevation at mesh nodes
        with nc.Dataset(stofs_file) as ds:
            # Time dimension
            n_times = len(ds.dimensions['time'])

            # Extract elevation for our nodes
            elevation = np.zeros((n_times, num_nodes), dtype=np.float32)
            zeta_full = ds.variables['zeta'][:]  # (time, nodes)

            for t in range(n_times):
                elevation[t, :] = zeta_full[t, original_indices]

        # Process GFS forcing
        gfs_file = gfs_dir / f'gfs_{date_str}.npz'
        if gfs_file.exists():
            gfs_data = np.load(gfs_file)

            # Interpolate GFS to mesh nodes
            forcing = interpolate_gfs_to_mesh(gfs_data, lon, lat, n_times)
        else:
            # Create dummy forcing if GFS not available
            logger.warning(f"{date_str}: GFS file not found, using zeros")
            forcing = {
                'u10': np.zeros((n_times, num_nodes), dtype=np.float32),
                'v10': np.zeros((n_times, num_nodes), dtype=np.float32),
                'pressure': np.zeros((n_times, num_nodes), dtype=np.float32),
            }

        # Compute derived forcing variables
        u10 = forcing['u10']
        v10 = forcing['v10']
        pressure = forcing['pressure']

        wind_speed = np.sqrt(u10**2 + v10**2)
        wind_speed_sq = wind_speed**2
        wind_dir = np.arctan2(v10, u10)

        # Pressure gradients (simple finite difference in space)
        dP_dx = np.zeros_like(pressure)
        dP_dy = np.zeros_like(pressure)

        # Save processed data
        np.savez_compressed(
            output_file,
            elevation=elevation.astype(np.float32),
            u10=u10.astype(np.float32),
            v10=v10.astype(np.float32),
            pressure=pressure.astype(np.float32),
            wind_speed=wind_speed.astype(np.float32),
            wind_speed_sq=wind_speed_sq.astype(np.float32),
            wind_dir=wind_dir.astype(np.float32),
            dP_dx=dP_dx.astype(np.float32),
            dP_dy=dP_dy.astype(np.float32),
        )

        return f"{date_str}: Success ({n_times} timesteps)"

    except Exception as e:
        return f"{date_str}: Error - {str(e)}"


def interpolate_gfs_to_mesh(gfs_data, mesh_lon, mesh_lat, n_times):
    """Interpolate GFS gridded data to unstructured mesh nodes."""

    gfs_lon = gfs_data['lon']
    gfs_lat = gfs_data['lat']

    # GFS variables
    u10_grid = gfs_data['u10']  # (time, lat, lon)
    v10_grid = gfs_data['v10']
    pressure_grid = gfs_data['pressure'] if 'pressure' in gfs_data else gfs_data.get('mslp', np.zeros_like(u10_grid))

    n_mesh = len(mesh_lon)
    gfs_times = min(u10_grid.shape[0], n_times)

    u10_mesh = np.zeros((n_times, n_mesh), dtype=np.float32)
    v10_mesh = np.zeros((n_times, n_mesh), dtype=np.float32)
    pressure_mesh = np.zeros((n_times, n_mesh), dtype=np.float32)

    # Interpolation points
    points = np.column_stack([mesh_lat, mesh_lon])

    for t in range(gfs_times):
        # U10
        interp = RegularGridInterpolator(
            (gfs_lat, gfs_lon), u10_grid[t],
            method='linear', bounds_error=False, fill_value=0
        )
        u10_mesh[t] = interp(points)

        # V10
        interp = RegularGridInterpolator(
            (gfs_lat, gfs_lon), v10_grid[t],
            method='linear', bounds_error=False, fill_value=0
        )
        v10_mesh[t] = interp(points)

        # Pressure
        interp = RegularGridInterpolator(
            (gfs_lat, gfs_lon), pressure_grid[t],
            method='linear', bounds_error=False, fill_value=101325
        )
        pressure_mesh[t] = interp(points)

    return {
        'u10': u10_mesh,
        'v10': v10_mesh,
        'pressure': pressure_mesh,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS data for 100k mesh')
    parser.add_argument('--mesh-file', type=str, required=True,
                        help='Path to mesh.npz')
    parser.add_argument('--stofs-dir', type=str, required=True,
                        help='Directory containing raw STOFS data (with date subdirs)')
    parser.add_argument('--gfs-dir', type=str, required=True,
                        help='Directory containing GFS forcing files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for processed files')
    parser.add_argument('--start-date', type=str, default='20230108',
                        help='Start date (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default='20260124',
                        help='End date (YYYYMMDD)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')

    args = parser.parse_args()

    mesh_file = Path(args.mesh_file)
    stofs_dir = Path(args.stofs_dir)
    gfs_dir = Path(args.gfs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate date list
    start = datetime.strptime(args.start_date, '%Y%m%d')
    end = datetime.strptime(args.end_date, '%Y%m%d')

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)

    logger.info(f"Processing {len(dates)} dates")
    logger.info(f"  Mesh: {mesh_file}")
    logger.info(f"  STOFS: {stofs_dir}")
    logger.info(f"  GFS: {gfs_dir}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Workers: {args.workers}")

    # Prepare arguments for parallel processing
    task_args = [
        (date, mesh_file, stofs_dir, gfs_dir, output_dir)
        for date in dates
    ]

    # Process in parallel
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_date, arg): arg[0] for arg in task_args}

        for future in as_completed(futures):
            date = futures[future]
            result = future.result()
            completed += 1

            if 'Error' in result:
                errors += 1
                logger.error(result)
            elif 'Success' in result:
                logger.info(f"[{completed}/{len(dates)}] {result}")
            else:
                logger.info(f"[{completed}/{len(dates)}] {result}")

    logger.info(f"\nComplete: {completed - errors} succeeded, {errors} errors")


if __name__ == '__main__':
    main()
