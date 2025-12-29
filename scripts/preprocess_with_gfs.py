#!/usr/bin/env python3
"""
Preprocess STOFS CWL data with GFS atmospheric forcing.

This script:
1. Downloads GFS forcing data (wind, pressure) from AWS/NOMADS
2. Interpolates GFS regular grid data to STOFS mesh nodes
3. Aligns temporal resolution between CWL and GFS
4. Filters out dry/land nodes

GFS Data Sources:
- AWS: s3://noaa-gfs-bdp-pds/ (operational, ~2 weeks retention)
- NOMADS: https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/
- Historical: https://data.rda.ucar.edu/ (requires registration)

Usage:
    python scripts/preprocess_with_gfs.py --data-dir /path/to/stofs --dates 20230108 20230109
"""

import os
import gc
import time
import argparse
import subprocess
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
import logging
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_DIR = '/mnt/f/STOFS_TRAINING_DATA/stofs_data'
DEFAULT_OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_gfs'
GFS_CACHE_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/gfs_cache'

MAX_NODES = 25000

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

MIN_DEPTH_THRESHOLD = 0.1

# Normalization
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Time settings
NOWCAST_HOURS = 6
CWL_DT_HOURS = 1  # CWL output interval
GFS_DT_HOURS = 3  # GFS output interval (typical)

# GFS data sources
GFS_AWS_BUCKET = 'noaa-gfs-bdp-pds'
GFS_NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod'


def check_gfs_tools():
    """Check if required tools for GFS processing are available."""
    tools = {
        'wgrib2': False,
        'wget': False,
        'aws': False,
    }

    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, timeout=5)
            tools[tool] = result.returncode == 0
        except:
            pass

    return tools


def download_gfs_from_aws(date_str: str, output_dir: Path, bbox: dict) -> dict:
    """
    Download GFS data from AWS for a specific date.

    Args:
        date_str: Date in YYYYMMDD format
        output_dir: Directory to save downloaded files
        bbox: Bounding box for subsetting

    Returns:
        dict with paths to downloaded files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # GFS file pattern on AWS
    # s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/00/atmos/gfs.t00z.pgrb2.0p25.fHHH

    base_url = f's3://{GFS_AWS_BUCKET}/gfs.{date_str}/00/atmos'

    files_needed = []
    # Download forecast hours 0-186 at 3-hour intervals to match CWL
    for fhr in range(0, 187, 3):
        files_needed.append(f'gfs.t00z.pgrb2.0p25.f{fhr:03d}')

    logger.info(f"Downloading {len(files_needed)} GFS files for {date_str}...")

    downloaded = []
    for fname in files_needed:
        local_path = output_dir / fname
        if local_path.exists():
            downloaded.append(local_path)
            continue

        s3_path = f'{base_url}/{fname}'
        try:
            cmd = ['aws', 's3', 'cp', s3_path, str(local_path), '--no-sign-request']
            subprocess.run(cmd, check=True, capture_output=True)
            downloaded.append(local_path)
        except Exception as e:
            logger.warning(f"Failed to download {fname}: {e}")

    return {'files': downloaded, 'date': date_str}


def download_gfs_from_nomads(date_str: str, output_dir: Path, bbox: dict) -> dict:
    """
    Download GFS data from NOMADS (alternative source).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f'{GFS_NOMADS_BASE}/gfs.{date_str}/00/atmos'

    files_needed = []
    for fhr in range(0, 187, 3):
        files_needed.append(f'gfs.t00z.pgrb2.0p25.f{fhr:03d}')

    downloaded = []
    for fname in files_needed:
        local_path = output_dir / fname
        if local_path.exists():
            downloaded.append(local_path)
            continue

        url = f'{base_url}/{fname}'
        try:
            cmd = ['wget', '-q', '-O', str(local_path), url]
            subprocess.run(cmd, check=True, timeout=300)
            downloaded.append(local_path)
        except Exception as e:
            logger.warning(f"Failed to download {fname}: {e}")

    return {'files': downloaded, 'date': date_str}


def extract_gfs_variables(grib_files: list, bbox: dict, output_nc: Path):
    """
    Extract wind and pressure from GFS GRIB2 files to NetCDF.

    Uses wgrib2 to extract:
    - UGRD:10 m above ground (u-component of wind)
    - VGRD:10 m above ground (v-component of wind)
    - PRES:surface (surface pressure)
    """
    logger.info(f"Extracting GFS variables from {len(grib_files)} files...")

    # This requires wgrib2 to be installed
    # Alternative: use pygrib or cfgrib

    try:
        import cfgrib
        import xarray as xr

        datasets = []
        for grib_file in sorted(grib_files):
            try:
                # Open with cfgrib, filter to needed variables
                ds = xr.open_dataset(
                    grib_file,
                    engine='cfgrib',
                    filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 10}
                )
                datasets.append(ds)
            except Exception as e:
                logger.warning(f"Could not read {grib_file}: {e}")

        if datasets:
            combined = xr.concat(datasets, dim='time')
            # Subset to bbox
            lon_min, lon_max = bbox['lon_min'], bbox['lon_max']
            lat_min, lat_max = bbox['lat_min'], bbox['lat_max']

            # Handle longitude convention (0-360 vs -180-180)
            if combined.longitude.min() >= 0:
                lon_min = lon_min % 360
                lon_max = lon_max % 360

            subset = combined.sel(
                longitude=slice(lon_min - 2, lon_max + 2),
                latitude=slice(lat_max + 2, lat_min - 2)
            )

            subset.to_netcdf(output_nc)
            return True

    except ImportError:
        logger.warning("cfgrib not available, trying wgrib2...")

    # Fallback to wgrib2
    try:
        temp_csv = output_nc.with_suffix('.csv')

        for grib_file in grib_files:
            # Extract u10, v10, surface pressure
            cmd = [
                'wgrib2', str(grib_file),
                '-match', ':(UGRD|VGRD):10 m above ground:|:PRES:surface:',
                '-csv', str(temp_csv)
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        # Parse CSV and create NetCDF... (complex, skip for now)
        logger.warning("wgrib2 extraction requires additional parsing")
        return False

    except Exception as e:
        logger.error(f"Failed to extract GFS variables: {e}")
        return False


def interpolate_gfs_to_mesh(gfs_data: dict, node_lon: np.ndarray, node_lat: np.ndarray) -> dict:
    """
    Interpolate GFS regular grid data to unstructured mesh nodes.

    Args:
        gfs_data: dict with 'u10', 'v10', 'pressure', 'lat', 'lon', 'time'
        node_lon: mesh node longitudes
        node_lat: mesh node latitudes

    Returns:
        dict with interpolated u10, v10, pressure at mesh nodes
    """
    logger.info("Interpolating GFS data to mesh nodes...")

    grid_lat = gfs_data['lat']
    grid_lon = gfs_data['lon']
    num_times = gfs_data['u10'].shape[0]
    num_nodes = len(node_lon)

    # Handle longitude convention
    if grid_lon.max() > 180:
        grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Sort grid for interpolation
    lat_sort = np.argsort(grid_lat)
    lon_sort = np.argsort(grid_lon)
    grid_lat_s = grid_lat[lat_sort]
    grid_lon_s = grid_lon[lon_sort]

    # Compute fractional indices for bilinear interpolation
    lat_frac = np.interp(node_lat, grid_lat_s, np.arange(len(grid_lat_s)))
    lon_frac = np.interp(node_lon, grid_lon_s, np.arange(len(grid_lon_s)))
    coords = np.array([lat_frac, lon_frac])

    u10_interp = np.zeros((num_times, num_nodes), dtype=np.float32)
    v10_interp = np.zeros((num_times, num_nodes), dtype=np.float32)
    pres_interp = np.zeros((num_times, num_nodes), dtype=np.float32)

    for t in range(num_times):
        u_grid = gfs_data['u10'][t][lat_sort][:, lon_sort].astype(np.float32)
        v_grid = gfs_data['v10'][t][lat_sort][:, lon_sort].astype(np.float32)
        p_grid = gfs_data['pressure'][t][lat_sort][:, lon_sort].astype(np.float32)

        u10_interp[t] = map_coordinates(u_grid, coords, order=1, mode='nearest')
        v10_interp[t] = map_coordinates(v_grid, coords, order=1, mode='nearest')
        pres_interp[t] = map_coordinates(p_grid, coords, order=1, mode='nearest')

    return {
        'u10': u10_interp,
        'v10': v10_interp,
        'pressure': pres_interp,
    }


def interpolate_forcing_temporal(forcing: dict, cwl_times: np.ndarray, gfs_times: np.ndarray) -> dict:
    """
    Interpolate forcing data temporally to match CWL timesteps.

    GFS is typically 3-hourly, CWL is hourly.
    """
    logger.info("Interpolating forcing to CWL timesteps...")

    num_cwl_times = len(cwl_times)
    num_nodes = forcing['u10'].shape[1]

    u10_interp = np.zeros((num_cwl_times, num_nodes), dtype=np.float32)
    v10_interp = np.zeros((num_cwl_times, num_nodes), dtype=np.float32)
    pres_interp = np.zeros((num_cwl_times, num_nodes), dtype=np.float32)

    for i, cwl_t in enumerate(cwl_times):
        # Find surrounding GFS times
        idx = np.searchsorted(gfs_times, cwl_t)

        if idx == 0:
            u10_interp[i] = forcing['u10'][0]
            v10_interp[i] = forcing['v10'][0]
            pres_interp[i] = forcing['pressure'][0]
        elif idx >= len(gfs_times):
            u10_interp[i] = forcing['u10'][-1]
            v10_interp[i] = forcing['v10'][-1]
            pres_interp[i] = forcing['pressure'][-1]
        else:
            # Linear interpolation
            t0, t1 = gfs_times[idx-1], gfs_times[idx]
            w = (cwl_t - t0) / (t1 - t0)

            u10_interp[i] = (1-w) * forcing['u10'][idx-1] + w * forcing['u10'][idx]
            v10_interp[i] = (1-w) * forcing['v10'][idx-1] + w * forcing['v10'][idx]
            pres_interp[i] = (1-w) * forcing['pressure'][idx-1] + w * forcing['pressure'][idx]

    return {
        'u10': u10_interp,
        'v10': v10_interp,
        'pressure': pres_interp,
    }


def fast_interpolate_to_nodes(data_3d, grid_lat, grid_lon, node_lat, node_lon):
    """Vectorized interpolation from regular grid to mesh nodes."""
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


def select_wet_nodes(cwl_file: str, bbox: dict, max_nodes: int, min_depth: float):
    """Select wet nodes within bounding box."""
    logger.info(f"Selecting wet nodes from {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])
    element = np.array(nc.variables['element'][:]) - 1

    logger.info(f"Full mesh: {len(x):,} nodes")

    # Filter by bbox and depth
    bbox_mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    depth_mask = depth >= min_depth

    # Validate with CWL data
    zeta = nc.variables['zeta']
    timesteps_to_check = list(range(NOWCAST_HOURS, min(zeta.shape[0], NOWCAST_HOURS + 10)))

    nan_count = np.zeros(len(x), dtype=np.int32)
    for t in timesteps_to_check:
        zeta_t = np.array(zeta[t, :], dtype=np.float32)
        is_invalid = (zeta_t < -9000) | np.isnan(zeta_t)
        nan_count += is_invalid.astype(np.int32)

    valid_ratio = 1.0 - (nan_count / len(timesteps_to_check))
    data_mask = valid_ratio >= 0.8

    nc.close()

    combined_mask = bbox_mask & depth_mask & data_mask
    wet_indices = np.where(combined_mask)[0]
    logger.info(f"Wet nodes in bbox: {len(wet_indices):,}")

    if len(wet_indices) == 0:
        raise ValueError("No wet nodes found!")

    # Subsample if needed
    if len(wet_indices) > max_nodes:
        logger.info(f"Subsampling to {max_nodes:,} nodes")
        wet_coords = np.column_stack([x[wet_indices], y[wet_indices]])

        selected_local = [0]
        remaining = set(range(len(wet_indices)))
        remaining.remove(0)
        np.random.seed(42)

        while len(selected_local) < min(1000, max_nodes) and remaining:
            selected_coords = wet_coords[selected_local]
            tree = cKDTree(selected_coords)
            remaining_sample = list(remaining)[:5000]
            remaining_coords = wet_coords[remaining_sample]
            dists, _ = tree.query(remaining_coords)
            farthest_local = remaining_sample[np.argmax(dists)]
            selected_local.append(farthest_local)
            remaining.remove(farthest_local)

        if len(selected_local) < max_nodes:
            remaining_list = list(remaining)
            np.random.shuffle(remaining_list)
            selected_local.extend(remaining_list[:max_nodes - len(selected_local)])

        selected_local = np.array(selected_local[:max_nodes])
        global_indices = wet_indices[selected_local]
    else:
        global_indices = wet_indices

    logger.info(f"Selected {len(global_indices):,} nodes")

    return {
        'global_indices': global_indices,
        'lon': x[global_indices],
        'lat': y[global_indices],
        'depth': depth[global_indices],
        'element': element,
    }


def build_edges_for_selected_nodes(element, global_indices):
    """Build edge connectivity."""
    logger.info("Building edges...")

    global_to_local = {g: l for l, g in enumerate(global_indices)}
    selected_set = set(global_indices)

    edges = set()
    for tri in element:
        local_nodes = []
        for node in tri:
            if node in selected_set:
                local_nodes.append(global_to_local[node])
        if len(local_nodes) >= 2:
            for i in range(len(local_nodes)):
                for j in range(i + 1, len(local_nodes)):
                    n1, n2 = local_nodes[i], local_nodes[j]
                    edges.add((min(n1, n2), max(n1, n2)))

    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    edge_index = np.array([src, dst], dtype=np.int64)
    logger.info(f"Created {edge_index.shape[1]:,} edges")

    return edge_index


def preprocess_date_with_gfs(date_str: str, mesh_data: dict, data_dir: Path,
                              gfs_dir: Path = None, use_stofs_forcing: bool = False):
    """
    Preprocess a single date with GFS forcing.

    Args:
        date_str: Date in YYYYMMDD format
        mesh_data: Mesh data dict
        data_dir: STOFS CWL data directory
        gfs_dir: GFS data directory (if using raw GFS)
        use_stofs_forcing: If True, use STOFS uvgrd/pressfc files instead of GFS
    """
    cwl_file = data_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'

    if not cwl_file.exists():
        logger.warning(f"CWL not found: {cwl_file}")
        return None

    logger.info(f"Processing {date_str}...")
    start_time = time.time()

    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']

    # 1. Load CWL
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(str(cwl_file), 'r')
    zeta = nc_cwl.variables['zeta']
    cwl_time = np.array(nc_cwl.variables['time'][:])

    full_times = zeta.shape[0]
    time_indices = list(range(NOWCAST_HOURS, full_times))
    num_times = len(time_indices)

    elevation = np.zeros((num_times, len(global_indices)), dtype=np.float32)
    for i, t in enumerate(time_indices):
        elev_t = np.array(zeta[t, global_indices], dtype=np.float32)
        elev_t = np.where(elev_t < -9000, np.nan, elev_t)
        elevation[i] = elev_t

    nc_cwl.close()

    nan_pct = 100 * np.isnan(elevation).sum() / elevation.size
    logger.info(f"    CWL: {elevation.shape}, NaN: {nan_pct:.1f}%")

    # 2. Load forcing
    u10 = None
    v10 = None
    pressure = None

    if use_stofs_forcing:
        # Try STOFS forcing files
        wind_file = data_dir / date_str / 'stofs_2d_glo.t00z.uvgrd10m.nc'
        pres_file = data_dir / date_str / 'stofs_2d_glo.t00z.pressfc.nc'

        if wind_file.exists() and pres_file.exists():
            logger.info("  Loading STOFS forcing...")

            nc_wind = NCDataset(str(wind_file), 'r')
            grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
            grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
            grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

            margin = 2.0
            lon_mask = (grid_lon >= BBOX['lon_min'] - margin) & (grid_lon <= BBOX['lon_max'] + margin)
            lat_mask = (grid_lat >= BBOX['lat_min'] - margin) & (grid_lat <= BBOX['lat_max'] + margin)
            lon_idx = np.where(lon_mask)[0]
            lat_idx = np.where(lat_mask)[0]

            u_all = np.array(nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
            v_all = np.array(nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
            nc_wind.close()

            nc_pres = NCDataset(str(pres_file), 'r')
            p_all = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
            nc_pres.close()

            grid_lon_sub = grid_lon[lon_idx]
            grid_lat_sub = grid_lat[lat_idx]

            # Match timesteps
            common_times = min(num_times, u_all.shape[0])
            elevation = elevation[:common_times]

            logger.info("  Interpolating forcing...")
            u10 = fast_interpolate_to_nodes(u_all[:common_times], grid_lat_sub, grid_lon_sub, node_lat, node_lon)
            v10 = fast_interpolate_to_nodes(v_all[:common_times], grid_lat_sub, grid_lon_sub, node_lat, node_lon)
            pressure = fast_interpolate_to_nodes(p_all[:common_times], grid_lat_sub, grid_lon_sub, node_lat, node_lon)
            pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    # If no forcing loaded, set to zeros (CWL-only mode)
    if u10 is None:
        logger.info("  No forcing data - using zeros")
        u10 = np.zeros_like(elevation)
        v10 = np.zeros_like(elevation)
        pressure = np.zeros_like(elevation)

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    return {
        'date': date_str,
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
    }


def get_available_dates(data_dir: Path):
    """Get dates with CWL data."""
    dates = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            cwl = d / 'stofs_2d_glo.t00z.fields.cwl.nc'
            if cwl.exists():
                dates.append(d.name)
    return dates


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS with GFS forcing')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--min-depth', type=float, default=MIN_DEPTH_THRESHOLD)
    parser.add_argument('--dates', nargs='+', default=None)
    parser.add_argument('--use-stofs-forcing', action='store_true',
                        help='Use STOFS forcing files if available')
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("PREPROCESSING STOFS WITH GFS FORCING")
    logger.info("=" * 70)

    # Check tools
    tools = check_gfs_tools()
    logger.info(f"Available tools: {tools}")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    dates = args.dates if args.dates else get_available_dates(data_dir)
    if not dates:
        logger.error("No dates found!")
        return

    logger.info(f"Found {len(dates)} dates")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mesh
    first_cwl = data_dir / dates[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(str(first_cwl), BBOX, args.max_nodes, args.min_depth)
    edge_index = build_edges_for_selected_nodes(mesh_data['element'], mesh_data['global_indices'])

    mesh_file = output_dir / 'mesh.npz'
    np.savez_compressed(str(mesh_file),
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
    )
    logger.info(f"Mesh saved: {len(mesh_data['lon']):,} nodes")

    # Check existing
    existing = set()
    if args.skip_existing and not args.force:
        for f in output_dir.glob('processed_*.npz'):
            existing.add(f.stem.replace('processed_', ''))

    # Process dates
    success = 0
    for i, date_str in enumerate(dates):
        if date_str in existing:
            continue

        try:
            data = preprocess_date_with_gfs(
                date_str, mesh_data, data_dir,
                use_stofs_forcing=args.use_stofs_forcing
            )
            if data:
                out_file = output_dir / f'processed_{date_str}.npz'
                np.savez_compressed(str(out_file),
                    elevation=data['elevation'],
                    u10=data['u10'],
                    v10=data['v10'],
                    pressure=data['pressure'],
                )
                success += 1
                del data
                gc.collect()
        except Exception as e:
            logger.error(f"Error: {e}")

    logger.info(f"\nCompleted: {success} dates processed")


if __name__ == '__main__':
    main()
