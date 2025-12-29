#!/usr/bin/env python3
"""
Preprocess STOFS CWL data with GFS forcing - with proper temporal alignment.

Handles the temporal mismatch between:
- CWL: Hourly output (6hr nowcast + 180hr forecast = 186 hours, use hours 6-185)
- GFS: Hourly f000-f120, 3-hourly f123-f384

This script:
1. Loads CWL hourly data (skips 6hr nowcast)
2. Loads GFS at available hours
3. Interpolates GFS to match CWL hourly timesteps
4. Extracts U10, V10, surface pressure
5. Interpolates spatially from GFS grid to STOFS mesh nodes

Usage:
    python scripts/preprocess_with_gfs_aligned.py \
        --cwl-dir /path/to/stofs_data \
        --gfs-dir /path/to/gfs_forcing \
        --output-dir /path/to/output
"""

import os
import gc
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    logger.error("netCDF4 required: pip install netCDF4")
    raise

try:
    import pygrib
    HAS_PYGRIB = True
except ImportError:
    HAS_PYGRIB = False
    logger.warning("pygrib not available, will try cfgrib")

# ============================================================
# CONFIGURATION
# ============================================================

# Time settings
# VERIFIED: GFS f000 aligns with STOFS Hour 7 (not Hour 6)
# Testing showed offset=+1 gives best match (avg diff 0.04 m/s vs 0.13 m/s)
NOWCAST_HOURS = 7  # Skip first 7 hours of CWL (nowcast period)
CWL_TOTAL_HOURS = 186  # Total CWL timesteps
CWL_FORECAST_HOURS = CWL_TOTAL_HOURS - NOWCAST_HOURS  # 179 hours we use

# GFS forecast hours to use
# CWL hour 7 (first forecast hour) = GFS f000
# CWL hour 8 = GFS f001, etc.
# So we need GFS f000-f178 for CWL hours 7-185
#
# GFS availability:
#   f000-f120: hourly (121 files)
#   f123-f180: 3-hourly (20 files)
GFS_HOURS_HOURLY = list(range(0, 121))  # f000 to f120
GFS_HOURS_3HOURLY = list(range(123, 181, 3))  # f123, f126, ..., f180
GFS_HOURS_ALL = GFS_HOURS_HOURLY + GFS_HOURS_3HOURLY

# Target GFS hours for full 179-hour forecast (0-178)
TARGET_GFS_HOURS = list(range(0, 179))  # f000 to f178

# Spatial settings - Mid-Atlantic + New England for winter storms
# Norfolk to Portland ME (focused domain for better resolution)
BBOX = {
    'lon_min': -77.0,
    'lon_max': -66.0,
    'lat_min': 37.0,
    'lat_max': 45.0
}
MAX_NODES = 40000  # Balanced for resolution vs coverage
MIN_DEPTH = 0.1

# Normalization
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


def read_gfs_grib(grib_file: Path, bbox: dict = None):
    """
    Read U10, V10, and surface pressure from GFS GRIB2 file.

    Args:
        grib_file: Path to GRIB2 file
        bbox: Optional bounding box for subsetting

    Returns:
        dict with 'u10', 'v10', 'pressure', 'lat', 'lon'
    """
    if not HAS_PYGRIB:
        raise ImportError("pygrib required for reading GRIB files")

    grbs = pygrib.open(str(grib_file))

    data = {}

    for grb in grbs:
        name = grb.name.lower()

        if '10 metre u' in name and grb.level == 10:
            data['u10'] = grb.values
            lats, lons = grb.latlons()
            data['lat'] = lats[:, 0]
            data['lon'] = lons[0, :]

        elif '10 metre v' in name and grb.level == 10:
            data['v10'] = grb.values

        elif 'surface pressure' in name and grb.typeOfLevel == 'surface':
            data['pressure'] = grb.values

    grbs.close()

    # Convert lon from 0-360 to -180-180 if needed
    if data.get('lon') is not None and data['lon'].max() > 180:
        data['lon'] = np.where(data['lon'] > 180, data['lon'] - 360, data['lon'])

    # Subset to bbox if provided
    if bbox and 'lat' in data:
        lat_mask = (data['lat'] >= bbox['lat_min'] - 2) & (data['lat'] <= bbox['lat_max'] + 2)
        lon_mask = (data['lon'] >= bbox['lon_min'] - 2) & (data['lon'] <= bbox['lon_max'] + 2)

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

        if len(lat_idx) > 0 and len(lon_idx) > 0:
            data['lat'] = data['lat'][lat_idx]
            data['lon'] = data['lon'][lon_idx]
            data['u10'] = data['u10'][lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            data['v10'] = data['v10'][lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
            data['pressure'] = data['pressure'][lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    return data


def interpolate_gfs_temporal(gfs_data_by_hour: dict, target_hours: list) -> dict:
    """
    Interpolate GFS data temporally to match target hourly timesteps.

    Args:
        gfs_data_by_hour: dict mapping forecast hour -> {'u10': array, 'v10': array, 'pressure': array}
        target_hours: list of target hours (e.g., [6, 7, 8, ..., 185])

    Returns:
        dict with 'u10', 'v10', 'pressure' arrays of shape (len(target_hours), lat, lon)
    """
    available_hours = sorted(gfs_data_by_hour.keys())

    if not available_hours:
        raise ValueError("No GFS data available")

    # Get array shape from first available hour
    first_data = gfs_data_by_hour[available_hours[0]]
    shape = first_data['u10'].shape

    logger.info(f"Interpolating GFS from {len(available_hours)} hours to {len(target_hours)} hours")

    # Stack data for interpolation
    u10_stack = np.array([gfs_data_by_hour[h]['u10'] for h in available_hours])
    v10_stack = np.array([gfs_data_by_hour[h]['v10'] for h in available_hours])
    pres_stack = np.array([gfs_data_by_hour[h]['pressure'] for h in available_hours])

    # Create interpolation functions
    u10_interp = interp1d(available_hours, u10_stack, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    v10_interp = interp1d(available_hours, v10_stack, axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    pres_interp = interp1d(available_hours, pres_stack, axis=0, kind='linear',
                           bounds_error=False, fill_value='extrapolate')

    # Interpolate to target hours
    u10_hourly = u10_interp(target_hours).astype(np.float32)
    v10_hourly = v10_interp(target_hours).astype(np.float32)
    pres_hourly = pres_interp(target_hours).astype(np.float32)

    return {
        'u10': u10_hourly,
        'v10': v10_hourly,
        'pressure': pres_hourly,
        'lat': first_data.get('lat'),
        'lon': first_data.get('lon'),
    }


def interpolate_to_mesh(data_3d: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray,
                        node_lat: np.ndarray, node_lon: np.ndarray) -> np.ndarray:
    """
    Interpolate gridded data to unstructured mesh nodes.

    Args:
        data_3d: Array of shape (time, lat, lon)
        grid_lat: 1D array of grid latitudes
        grid_lon: 1D array of grid longitudes
        node_lat: 1D array of mesh node latitudes
        node_lon: 1D array of mesh node longitudes

    Returns:
        Array of shape (time, num_nodes)
    """
    num_times = data_3d.shape[0]
    num_nodes = len(node_lon)

    # Sort grid coordinates for interpolation
    lat_sort = np.argsort(grid_lat)
    lon_sort = np.argsort(grid_lon)
    grid_lat_s = grid_lat[lat_sort]
    grid_lon_s = grid_lon[lon_sort]

    # Compute fractional indices
    lat_frac = np.interp(node_lat, grid_lat_s, np.arange(len(grid_lat_s)))
    lon_frac = np.interp(node_lon, grid_lon_s, np.arange(len(grid_lon_s)))
    coords = np.array([lat_frac, lon_frac])

    result = np.zeros((num_times, num_nodes), dtype=np.float32)

    for t in range(num_times):
        # Reorder data to match sorted coordinates
        data = data_3d[t][lat_sort][:, lon_sort].astype(np.float32)
        result[t] = map_coordinates(data, coords, order=1, mode='nearest')

    return result


def select_wet_nodes(cwl_file: str, bbox: dict, max_nodes: int, min_depth: float):
    """Select wet nodes from CWL file."""
    logger.info(f"Selecting wet nodes from {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])
    element = np.array(nc.variables['element'][:]) - 1

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

    return {
        'global_indices': global_indices,
        'lon': x[global_indices],
        'lat': y[global_indices],
        'depth': depth[global_indices],
        'element': element,
    }


def build_edges(element, global_indices):
    """Build edge connectivity."""
    global_to_local = {g: l for l, g in enumerate(global_indices)}
    selected_set = set(global_indices)

    edges = set()
    for tri in element:
        local_nodes = [global_to_local[n] for n in tri if n in selected_set]
        if len(local_nodes) >= 2:
            for i in range(len(local_nodes)):
                for j in range(i + 1, len(local_nodes)):
                    n1, n2 = local_nodes[i], local_nodes[j]
                    edges.add((min(n1, n2), max(n1, n2)))

    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    return np.array([src, dst], dtype=np.int64)


def preprocess_date(date_str: str, mesh_data: dict, cwl_dir: Path, gfs_dir: Path):
    """
    Preprocess a single date with aligned CWL and GFS data.

    Temporal alignment:
    - CWL hour 6 (first forecast hour after nowcast) = GFS f000
    - CWL hour 7 = GFS f001
    - CWL hour N = GFS f(N-6)
    - CWL hours 6-185 (180 timesteps) = GFS f000-f179
    """
    cwl_file = cwl_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'
    gfs_date_dir = gfs_dir / date_str

    if not cwl_file.exists():
        logger.warning(f"CWL not found: {cwl_file}")
        return None

    if not gfs_date_dir.exists():
        logger.warning(f"GFS dir not found: {gfs_date_dir}")
        return None

    logger.info(f"Processing {date_str}...")

    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']

    # 1. Load CWL (skip first 6 hours = nowcast)
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(str(cwl_file), 'r')
    zeta = nc_cwl.variables['zeta']

    # CWL hours 6 to min(total, 186) - these are forecast hours
    cwl_hour_indices = list(range(NOWCAST_HOURS, min(zeta.shape[0], CWL_TOTAL_HOURS)))
    num_times = len(cwl_hour_indices)

    elevation = np.zeros((num_times, len(global_indices)), dtype=np.float32)
    for i, t in enumerate(cwl_hour_indices):
        elev_t = np.array(zeta[t, global_indices], dtype=np.float32)
        elev_t = np.where(elev_t < -9000, np.nan, elev_t)
        elevation[i] = elev_t

    nc_cwl.close()
    logger.info(f"    CWL shape: {elevation.shape} (hours {cwl_hour_indices[0]}-{cwl_hour_indices[-1]})")

    # 2. Load GFS - need f000 to f(num_times-1) to match CWL
    # CWL hour 6 -> GFS f000, CWL hour 7 -> GFS f001, etc.
    logger.info("  Loading GFS...")
    gfs_data_by_hour = {}

    # GFS hours we need: 0 to (num_times - 1)
    max_gfs_hour = num_times - 1
    needed_gfs_hours = [h for h in GFS_HOURS_ALL if h <= max_gfs_hour]

    for fhr in needed_gfs_hours:
        # Try different filename patterns
        patterns = [
            f'gfs.{date_str}.f{fhr:03d}.grib2',
            f'gfs.0p25.{date_str}00.f{fhr:03d}.grib2',
        ]

        gfs_file = None
        for pattern in patterns:
            candidate = gfs_date_dir / pattern
            if candidate.exists():
                gfs_file = candidate
                break

        if gfs_file:
            try:
                data = read_gfs_grib(gfs_file, BBOX)
                gfs_data_by_hour[fhr] = data
            except Exception as e:
                logger.warning(f"    Could not read f{fhr:03d}: {e}")

    if not gfs_data_by_hour:
        logger.warning("  No GFS data loaded!")
        return None

    logger.info(f"    Loaded {len(gfs_data_by_hour)} GFS hours (f{min(gfs_data_by_hour.keys()):03d}-f{max(gfs_data_by_hour.keys()):03d})")

    # 3. Interpolate GFS temporally to hourly (f000, f001, ..., f{num_times-1})
    # Target hours: 0, 1, 2, ..., num_times-1
    target_gfs_hours = list(range(num_times))

    logger.info("  Interpolating GFS temporally...")
    gfs_hourly = interpolate_gfs_temporal(gfs_data_by_hour, target_gfs_hours)

    # 4. Interpolate GFS spatially to mesh nodes
    logger.info("  Interpolating GFS to mesh...")
    u10 = interpolate_to_mesh(gfs_hourly['u10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    v10 = interpolate_to_mesh(gfs_hourly['v10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    pressure = interpolate_to_mesh(gfs_hourly['pressure'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)

    # Normalize pressure
    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    logger.info(f"    Final shapes - elevation: {elevation.shape}, u10: {u10.shape}")
    logger.info(f"    Alignment: CWL hours {cwl_hour_indices[0]}-{cwl_hour_indices[-1]} <-> GFS f000-f{num_times-1:03d}")

    return {
        'date': date_str,
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS with GFS (aligned)')
    parser.add_argument('--cwl-dir', type=str, required=True, help='CWL data directory')
    parser.add_argument('--gfs-dir', type=str, required=True, help='GFS data directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--dates', nargs='+', help='Specific dates to process')
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    args = parser.parse_args()

    cwl_dir = Path(args.cwl_dir)
    gfs_dir = Path(args.gfs_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("PREPROCESSING STOFS WITH GFS (TEMPORALLY ALIGNED)")
    logger.info("=" * 70)
    logger.info(f"CWL dir: {cwl_dir}")
    logger.info(f"GFS dir: {gfs_dir}")
    logger.info(f"Output: {output_dir}")

    # Get dates
    if args.dates:
        dates = args.dates
    else:
        dates = sorted([d.name for d in cwl_dir.iterdir()
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8
                       and (d / 'stofs_2d_glo.t00z.fields.cwl.nc').exists()])

    if not dates:
        logger.error("No dates found!")
        return

    logger.info(f"Found {len(dates)} dates")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mesh from first date
    first_cwl = cwl_dir / dates[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(str(first_cwl), BBOX, args.max_nodes, MIN_DEPTH)
    edge_index = build_edges(mesh_data['element'], mesh_data['global_indices'])

    # Save mesh
    mesh_file = output_dir / 'mesh.npz'
    np.savez_compressed(str(mesh_file),
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
    )
    logger.info(f"Mesh saved: {len(mesh_data['lon']):,} nodes")

    # Process dates
    success = 0
    for i, date_str in enumerate(dates):
        out_file = output_dir / f'processed_{date_str}.npz'

        if args.skip_existing and out_file.exists():
            logger.info(f"[{i+1}/{len(dates)}] Skipping {date_str} (exists)")
            continue

        try:
            data = preprocess_date(date_str, mesh_data, cwl_dir, gfs_dir)

            if data:
                np.savez_compressed(str(out_file),
                    elevation=data['elevation'],
                    u10=data['u10'],
                    v10=data['v10'],
                    pressure=data['pressure'],
                )
                success += 1
                logger.info(f"  Saved: {out_file.name}")

            del data
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nCompleted: {success}/{len(dates)} dates")


if __name__ == '__main__':
    main()
