#!/usr/bin/env python3
"""
Preprocess STOFS CWL data with GFS forcing from NPZ files. (PARALLEL VERSION)

This version combines the optimized I/O for NetCDF reads with multiprocessing
to parallelize processing across multiple dates.

Reads:
- CWL: netCDF files from stofs_data/{date}/stofs_2d_glo.t00z.fields.cwl.nc
- GFS: NPZ files from gfs_forcing/{date}/gfs_{date}_regional.npz

Outputs:
- processed_{date}.npz with aligned elevation and forcing data

Usage:
    python scripts/preprocess_npz_parallel.py --cwl-dir /path/to/stofs_data --gfs-dir /path/to/gfs_forcing --output-dir /path/to/output
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
import multiprocessing
import functools
from multiprocessing import cpu_count
from tqdm import tqdm

# Configure logging
# In multiprocessing, basicConfig needs to be handled carefully.
# This setup is fine for the main process. Workers will inherit the logger.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    logger.error("netCDF4 required: pip install netCDF4")
    raise

# ============================================================ 
# CONFIGURATION
# ============================================================ 

# Time settings - VERIFIED: GFS f000 aligns with STOFS Hour 7
NOWCAST_HOURS = 7  # Skip first 7 hours of CWL (nowcast period)
CWL_TOTAL_HOURS = 186  # Total CWL timesteps

# Spatial settings - Mid-Atlantic + New England for winter storms
BBOX = {
    'lon_min': -77.0,
    'lon_max': -66.0,
    'lat_min': 37.0,
    'lat_max': 45.0
}
MAX_NODES = 40000
MIN_DEPTH = 0.1

# Normalization
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


def load_gfs_npz(gfs_file: Path, bbox: dict = None):
    """
    Load GFS data from regional NPZ file.
    """
    data = np.load(gfs_file)
    result = {
        'u10': data['u10'], 'v10': data['v10'], 'sp': data['sp'],
        'fhr': data['fhr'], 'lat': data['lat'], 'lon': data['lon'],
    }
    if bbox:
        lat_1d = result['lat'][:, 0] if result['lat'].ndim == 2 else result['lat']
        lon_1d = result['lon'][0, :] if result['lon'].ndim == 2 else result['lon']
        lat_mask = (lat_1d >= bbox['lat_min']) & (lat_1d <= bbox['lat_max'])
        lon_mask = (lon_1d >= bbox['lon_min']) & (lon_1d <= bbox['lon_max'])
        lat_idx, lon_idx = np.where(lat_mask)[0], np.where(lon_mask)[0]
        if len(lat_idx) > 0 and len(lon_idx) > 0:
            i0, i1 = lat_idx[0], lat_idx[-1] + 1
            j0, j1 = lon_idx[0], lon_idx[-1] + 1
            result['u10'] = result['u10'][:, i0:i1, j0:j1]
            result['v10'] = result['v10'][:, i0:i1, j0:j1]
            result['sp'] = result['sp'][:, i0:i1, j0:j1]
            result['lat'] = result['lat'][i0:i1, j0:j1]
            result['lon'] = result['lon'][i0:i1, j0:j1]
    return result


def interpolate_gfs_temporal(gfs_data: dict, target_hours: list) -> dict:
    """
    Interpolate GFS data temporally to target hourly timesteps.
    """
    available_hours = gfs_data['fhr']
    u10_interp = interp1d(available_hours, gfs_data['u10'], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    v10_interp = interp1d(available_hours, gfs_data['v10'], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    sp_interp = interp1d(available_hours, gfs_data['sp'], axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
    return {
        'u10': u10_interp(target_hours).astype(np.float32),
        'v10': v10_interp(target_hours).astype(np.float32),
        'sp': sp_interp(target_hours).astype(np.float32),
        'lat': gfs_data['lat'], 'lon': gfs_data['lon'],
    }


def interpolate_to_mesh(data_3d: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray,
                        node_lat: np.ndarray, node_lon: np.ndarray) -> np.ndarray:
    """
    Interpolate gridded data to unstructured mesh nodes using bilinear interpolation.
    """
    num_times, num_nodes = data_3d.shape[0], len(node_lon)
    lat_1d = grid_lat[:, 0] if grid_lat.ndim == 2 else grid_lat
    lon_1d = grid_lon[0, :] if grid_lon.ndim == 2 else grid_lon
    lat_sort, lon_sort = np.argsort(lat_1d), np.argsort(lon_1d)
    lat_1d_s, lon_1d_s = lat_1d[lat_sort], lon_1d[lon_sort]
    lat_frac = np.interp(node_lat, lat_1d_s, np.arange(len(lat_1d_s)))
    lon_frac = np.interp(node_lon, lon_1d_s, np.arange(len(lon_1d_s)))
    coords = np.array([lat_frac, lon_frac])
    result = np.zeros((num_times, num_nodes), dtype=np.float32)
    for t in range(num_times):
        data = data_3d[t][lat_sort][:, lon_sort].astype(np.float32)
        result[t] = map_coordinates(data, coords, order=1, mode='nearest')
    return result


def select_wet_nodes(cwl_file: str, bbox: dict, max_nodes: int, min_depth: float):
    """Select wet nodes from CWL file within bounding box."""
    logger.info(f"Selecting wet nodes from {cwl_file}")
    with NCDataset(cwl_file, 'r') as nc:
        x, y = np.array(nc.variables['x'][:]), np.array(nc.variables['y'][:])
        depth = np.array(nc.variables['depth'][:])
        element = np.array(nc.variables['element'][:]) - 1
        bbox_mask = (x >= bbox['lon_min']) & (x <= bbox['lon_max']) & (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
        depth_mask = depth >= min_depth
        zeta = nc.variables['zeta']
        timesteps_to_check = list(range(NOWCAST_HOURS, min(zeta.shape[0], NOWCAST_HOURS + 10)))
        nan_count = np.zeros(len(x), dtype=np.int32)
        for t in timesteps_to_check:
            zeta_t = np.array(zeta[t, :], dtype=np.float32)
            nan_count += ((zeta_t < -9000) | np.isnan(zeta_t)).astype(np.int32)
        data_mask = (1.0 - (nan_count / len(timesteps_to_check))) >= 0.8
        combined_mask = bbox_mask & depth_mask & data_mask
        wet_indices = np.where(combined_mask)[0]
    logger.info(f"  Wet nodes in bbox: {len(wet_indices):,}")
    if len(wet_indices) > max_nodes:
        logger.info(f"  Subsampling to {max_nodes:,} nodes")
        np.random.seed(42)
        selected_idx = np.random.choice(len(wet_indices), max_nodes, replace=False)
        global_indices = wet_indices[np.sort(selected_idx)]
    else:
        global_indices = wet_indices
    logger.info(f"  Selected {len(global_indices):,} nodes")
    return {'global_indices': global_indices, 'lon': x[global_indices], 'lat': y[global_indices],
            'depth': depth[global_indices], 'element': element}


def build_edges(element, global_indices):
    """Build edge connectivity from triangular elements."""
    global_to_local = {g: l for l, g in enumerate(global_indices)}
    selected_set = set(global_indices)
    edges = set()
    for tri in element:
        local_nodes = [global_to_local[n] for n in tri if n in selected_set]
        if len(local_nodes) >= 2:
            for i in range(len(local_nodes)):
                for j in range(i + 1, len(local_nodes)):
                    edges.add(tuple(sorted((local_nodes[i], local_nodes[j]))))
    edge_list = np.array(list(edges), dtype=np.int64).T
    return np.hstack([edge_list, edge_list[[1, 0]]])

def preprocess_date(date_str: str, mesh_data: dict, cwl_dir: Path, gfs_dir: Path):
    """Preprocess a single date, returning data or an error message."""
    cwl_file = cwl_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'
    gfs_file = gfs_dir / date_str / f'gfs_{date_str}_regional.npz'
    if not cwl_file.exists(): return date_str, False, "CWL file not found"
    if not gfs_file.exists(): return date_str, False, "GFS file not found"

    global_indices = mesh_data['global_indices']
    with NCDataset(str(cwl_file), 'r') as nc_cwl:
        zeta = nc_cwl.variables['zeta']
        t_start, t_end = NOWCAST_HOURS, min(zeta.shape[0], CWL_TOTAL_HOURS)
        num_times = t_end - t_start
        idx_min, idx_max = global_indices.min(), global_indices.max() + 1
        zeta_block = np.array(zeta[t_start:t_end, idx_min:idx_max], dtype=np.float32)
        local_indices = global_indices - idx_min
        elevation = zeta_block[:, local_indices]
        del zeta_block
        elevation = np.where(elevation < -9000, np.nan, elevation)

    gfs_data = load_gfs_npz(gfs_file, BBOX)
    target_gfs_hours = list(range(num_times))
    gfs_hourly = interpolate_gfs_temporal(gfs_data, target_gfs_hours)
    
    node_lat, node_lon = mesh_data['lat'], mesh_data['lon']
    u10 = interpolate_to_mesh(gfs_hourly['u10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    v10 = interpolate_to_mesh(gfs_hourly['v10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    pressure = interpolate_to_mesh(gfs_hourly['sp'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    return date_str, True, {'elevation': elevation, 'u10': u10, 'v10': v10, 'pressure': pressure}

def process_and_save_wrapper(date_str, mesh_data, cwl_dir, gfs_dir, output_dir):
    """Wrapper to run processing and save the output for a single date."""
    try:
        date_str, success, result = preprocess_date(date_str, mesh_data, cwl_dir, gfs_dir)
        if success:
            out_file = output_dir / f'processed_{date_str}.npz'
            np.savez_compressed(str(out_file), **result)
            return date_str, True, out_file.name
        else:
            return date_str, False, result
    except Exception as e:
        # Log exception from the worker process
        worker_logger = logging.getLogger(f"worker-{date_str}")
        worker_logger.error(f"Error processing {date_str}: {e}", exc_info=True)
        return date_str, False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS with GFS (NPZ format) - PARALLEL')
    parser.add_argument('--cwl-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/stofs_data', help='CWL data directory')
    parser.add_argument('--gfs-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/gfs_forcing', help='GFS data directory')
    parser.add_argument('--output-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/processed', help='Output directory')
    parser.add_argument('--dates', nargs='+', help='Specific dates to process')
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--skip-existing', action='store_true', default=True, help="Skip dates if output file already exists")
    parser.add_argument('--num-workers', type=int, default=cpu_count(), help=f'Number of worker processes (default: {cpu_count()})')
    args = parser.parse_args()

    # Setup directories
    cwl_dir, gfs_dir, output_dir = Path(args.cwl_dir), Path(args.gfs_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70 + f"\nPREPROCESSING STOFS WITH GFS (PARALLEL) - Using {args.num_workers} workers\n" + "="*70)
    logger.info(f"CWL dir: {cwl_dir}\nGFS dir: {gfs_dir}\nOutput: {output_dir}")
    logger.info(f"Domain: {BBOX}\nMax nodes: {args.max_nodes:,}")

    # Discover and filter dates
    if args.dates:
        dates_to_process = args.dates
    else:
        cwl_dates = {d.name for d in cwl_dir.iterdir() if d.is_dir() and d.name.isdigit()}
        gfs_dates = {d.name for d in gfs_dir.iterdir() if d.is_dir() and d.name.isdigit()}
        dates_to_process = sorted(list(cwl_dates & gfs_dates))

    if args.skip_existing:
        existing_dates = {f.name.split('_')[1].split('.')[0] for f in output_dir.glob('processed_*.npz')}
        dates_to_process = [d for d in dates_to_process if d not in existing_dates]
        logger.info(f"Found {len(existing_dates)} existing files, skipping them.")

    if not dates_to_process:
        logger.info("No new dates to process. Exiting.")
        return

    logger.info(f"Found {len(dates_to_process)} new dates to process.")

    # Build and save mesh from the first available date
    first_cwl = cwl_dir / dates_to_process[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(str(first_cwl), BBOX, args.max_nodes, MIN_DEPTH)
    edge_index = build_edges(mesh_data['element'], mesh_data['global_indices'])
    del mesh_data['element']  # Don't need full element array anymore
    mesh_data['edge_index'] = edge_index
    mesh_file = output_dir / 'mesh.npz'
    np.savez_compressed(str(mesh_file), **mesh_data)
    logger.info(f"Mesh saved: {len(mesh_data['lon']):,} nodes, {edge_index.shape[1]//2:,} edges")

    # Create a partial function for the worker
    worker_func = functools.partial(process_and_save_wrapper,
                                    mesh_data=mesh_data, cwl_dir=cwl_dir,
                                    gfs_dir=gfs_dir, output_dir=output_dir)

    # Process dates in parallel
    success_count, fail_count = 0, 0
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(dates_to_process), desc="Processing dates") as pbar:
            for date_str, success, message in pool.imap_unordered(worker_func, dates_to_process):
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    logger.warning(f"Failed processing {date_str}: {message}")
                pbar.update(1)

    logger.info(f"\nCompleted processing. Successful: {success_count}, Failed: {fail_count}")

if __name__ == '__main__':
    main()
