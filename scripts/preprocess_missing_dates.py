#!/usr/bin/env python3
"""
Preprocess missing dates (20251116, 20251127) for 25k node mesh.
Uses the existing mesh_25k.npz file.
"""

import os
import gc
import time
from pathlib import Path
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.ndimage import map_coordinates
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = '/mnt/e/Drive2/Good/STOFS_TRAINING_DATA'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / 'data/processed_25k')
MESH_FILE = f'{OUTPUT_DIR}/mesh_25k.npz'

# Dates to reprocess
MISSING_DATES = ['20251116', '20251127']

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

# Normalization constants
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0
NOWCAST_HOURS = 5


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


def preprocess_date(date_str: str, mesh_data: dict):
    """Preprocess a single date."""

    date_dir = f'{DATA_DIR}/{date_str}'
    cwl_file = f'{date_dir}/stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = f'{date_dir}/stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = f'{date_dir}/stofs_2d_glo.t00z.pressfc.nc'

    logger.info(f"\nProcessing {date_str}...")
    start_time = time.time()

    # Check files exist
    for f, name in [(cwl_file, 'CWL'), (wind_file, 'Wind'), (pres_file, 'Pressure')]:
        if not os.path.exists(f):
            logger.error(f"  {name} file not found: {f}")
            return None

    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']

    # 1. Load CWL
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(cwl_file, 'r')
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
    logger.info(f"    CWL shape: {elevation.shape}")

    # 2. Load wind forcing
    logger.info("  Loading wind...")
    nc_wind = NCDataset(wind_file, 'r')

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

    # 3. Load pressure forcing
    logger.info("  Loading pressure...")
    nc_pres = NCDataset(pres_file, 'r')
    p_all = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_pres.close()

    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]

    # Ensure same number of timesteps
    met_times = u_all.shape[0]
    common_times = min(num_times, met_times)
    elevation = elevation[:common_times]
    u_all = u_all[:common_times]
    v_all = v_all[:common_times]
    p_all = p_all[:common_times]

    # 4. Interpolate forcing to mesh nodes
    logger.info("  Interpolating forcing to mesh nodes...")

    u10 = fast_interpolate_to_nodes(u_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    v10 = fast_interpolate_to_nodes(v_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    pressure = fast_interpolate_to_nodes(p_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)

    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    del u_all, v_all, p_all
    gc.collect()

    elapsed = time.time() - start_time
    logger.info(f"  Done: {common_times} timesteps in {elapsed:.1f}s")

    return {
        'date': date_str,
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
    }


def main():
    logger.info("=" * 70)
    logger.info("PREPROCESSING MISSING DATES FOR 25K MESH")
    logger.info("=" * 70)
    logger.info(f"Dates to process: {MISSING_DATES}")

    # Load existing mesh
    if not os.path.exists(MESH_FILE):
        logger.error(f"Mesh file not found: {MESH_FILE}")
        return

    logger.info(f"Loading mesh from {MESH_FILE}")
    mesh = np.load(MESH_FILE)
    mesh_data = {
        'global_indices': mesh['global_indices'],
        'lon': mesh['lon'],
        'lat': mesh['lat'],
    }
    logger.info(f"  Nodes: {len(mesh_data['lon']):,}")

    # Process each date
    for date_str in MISSING_DATES:
        try:
            data = preprocess_date(date_str, mesh_data)

            if data is None:
                continue

            out_file = f'{OUTPUT_DIR}/processed_{date_str}.npz'
            np.savez_compressed(
                out_file,
                elevation=data['elevation'],
                u10=data['u10'],
                v10=data['v10'],
                pressure=data['pressure'],
            )
            logger.info(f"  Saved to {out_file}")

            del data
            gc.collect()

        except Exception as e:
            logger.error(f"  Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
