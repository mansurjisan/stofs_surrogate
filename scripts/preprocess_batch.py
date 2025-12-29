#!/usr/bin/env python3
"""Batch preprocess multiple dates for training."""

import os
import gc
import time
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.ndimage import map_coordinates
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path('/mnt/e/Drive2/Good/STOFS_TRAINING_DATA')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_25k')
MESH_FILE = OUTPUT_DIR / 'mesh_25k.npz'

BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0
NOWCAST_HOURS = 5

# Complete dates (have CWL + wind + pressure)
COMPLETE_DATES = [
    '20251101',
    '20251106', '20251107', '20251108', '20251109', '20251110',
    '20251111', '20251112', '20251113', '20251114', '20251115',
    '20251116', '20251117', '20251118', '20251119', '20251120',
    '20251121', '20251122', '20251123', '20251124', '20251125',
    '20251126', '20251127', '20251128', '20251129', '20251130',
]


def fast_interpolate_to_nodes(data_3d, grid_lat, grid_lon, node_lat, node_lon):
    """Vectorized interpolation using scipy.ndimage.map_coordinates."""
    num_times = data_3d.shape[0]
    num_nodes = len(node_lon)

    lat_sort = np.argsort(grid_lat)
    lon_sort = np.argsort(grid_lon)
    grid_lat_s = grid_lat[lat_sort]
    grid_lon_s = grid_lon[lon_sort]

    lat_frac = np.interp(node_lat, grid_lat_s, np.arange(len(grid_lat_s)))
    lon_frac = np.interp(node_lon, grid_lon_s, np.arange(len(grid_lon_s)))

    coords = np.array([lat_frac, lon_frac])
    result = np.zeros((num_times, num_nodes), dtype=np.float32)

    for t in range(num_times):
        data = data_3d[t][lat_sort][:, lon_sort].astype(np.float32)
        result[t] = map_coordinates(data, coords, order=1, mode='nearest')

    return result


def process_date(date, mesh):
    """Process a single date."""
    global_indices = mesh['global_indices']
    node_lon = mesh['lon']
    node_lat = mesh['lat']

    date_dir = DATA_DIR / date
    cwl_file = date_dir / 'stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = date_dir / 'stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = date_dir / 'stofs_2d_glo.t00z.pressfc.nc'

    # Check files exist
    for f, name in [(cwl_file, 'CWL'), (wind_file, 'Wind'), (pres_file, 'Pressure')]:
        if not f.exists():
            logger.error(f"  {name} NOT FOUND: {f}")
            return False

    start_time = time.time()

    # 1. Load CWL
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(str(cwl_file), 'r')
    zeta = nc_cwl.variables['zeta']

    full_times = zeta.shape[0]
    time_indices = list(range(NOWCAST_HOURS, full_times))
    num_times = len(time_indices)

    elevation = np.zeros((num_times, len(global_indices)), dtype=np.float32)
    for i, t in enumerate(time_indices):
        elev_t = np.array(zeta[t, global_indices], dtype=np.float32)
        elev_t = np.where(elev_t < -9000, np.nan, elev_t)
        elevation[i] = elev_t

    nc_cwl.close()

    # 2. Load wind
    logger.info("  Loading wind...")
    nc_wind = NCDataset(str(wind_file), 'r')

    grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    margin = 2.0
    lon_mask = (grid_lon >= BBOX['lon_min'] - margin) & (grid_lon <= BBOX['lon_max'] + margin)
    lat_mask = (grid_lat >= BBOX['lat_min'] - margin) & (grid_lat <= BBOX['lat_max'] + margin)
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    u_all = np.array(nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    v_all = np.array(nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_wind.close()

    # 3. Load pressure
    logger.info("  Loading pressure...")
    nc_pres = NCDataset(str(pres_file), 'r')
    p_all = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_pres.close()

    # Match timesteps
    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]

    met_times = u_all.shape[0]
    common_times = min(num_times, met_times)
    elevation = elevation[:common_times]
    u_all = u_all[:common_times]
    v_all = v_all[:common_times]
    p_all = p_all[:common_times]

    # 4. Interpolate forcing
    logger.info("  Interpolating forcing...")
    u10 = fast_interpolate_to_nodes(u_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    v10 = fast_interpolate_to_nodes(v_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    pressure = fast_interpolate_to_nodes(p_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)

    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    del u_all, v_all, p_all
    gc.collect()

    elapsed = time.time() - start_time

    # Save
    out_file = OUTPUT_DIR / f'processed_{date}.npz'
    np.savez_compressed(
        str(out_file),
        elevation=elevation,
        u10=u10,
        v10=v10,
        pressure=pressure,
    )
    
    logger.info(f"  Saved: {out_file.name} ({common_times} timesteps, {elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch preprocess STOFS data')
    parser.add_argument('--dates', nargs='+', help='Specific dates to process (YYYYMMDD)')
    parser.add_argument('--skip-existing', action='store_true', default=True, help='Skip already processed dates')
    parser.add_argument('--force', action='store_true', help='Reprocess even if exists')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("BATCH PREPROCESSING FOR TRAINING")
    logger.info("=" * 70)

    # Load mesh
    logger.info(f"Loading mesh from {MESH_FILE}")
    mesh = dict(np.load(str(MESH_FILE)))
    logger.info(f"Mesh: {len(mesh['lon'])} nodes")

    # Determine which dates to process
    if args.dates:
        dates_to_process = args.dates
    else:
        dates_to_process = COMPLETE_DATES

    # Check which dates already processed
    existing = set()
    for f in OUTPUT_DIR.glob('processed_*.npz'):
        date = f.stem.replace('processed_', '')
        existing.add(date)
    
    logger.info(f"Already processed: {len(existing)} dates")

    # Process each date
    success = 0
    failed = 0
    skipped = 0

    for i, date in enumerate(dates_to_process):
        logger.info(f"\n[{i+1}/{len(dates_to_process)}] Processing {date}...")
        
        if date in existing and not args.force:
            logger.info(f"  Skipping (already exists)")
            skipped += 1
            continue

        try:
            if process_date(date, mesh):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            failed += 1

        gc.collect()

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total dates: {len(dates_to_process)}")
    logger.info(f"Processed: {success}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
