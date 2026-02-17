#!/usr/bin/env python3
"""
Improved preprocessing for STOFS data - filters out dry/land nodes.

Key improvements over v1:
1. Filters out land nodes (negative depth) before mesh creation
2. Uses depth threshold to select only consistently wet nodes
3. Validates data quality by checking NaN patterns
4. Option to use CWL data from first date to identify wet nodes

Usage:
    python scripts/preprocess_25k_v2.py --data-dir /path/to/stofs_data
"""

import os
import gc
import time
import argparse
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# Default paths (can be overridden via arguments)
DEFAULT_DATA_DIR = '/mnt/f/STOFS_TRAINING_DATA/stofs_data'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / 'data/processed_25k_v2')

# Node selection
MAX_NODES = 25000

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

# Depth threshold for wet nodes (positive depth = underwater)
# Nodes with depth < this value are considered land/dry
MIN_DEPTH_THRESHOLD = 0.1  # meters (slightly above 0 to handle coastal nodes)

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Time settings
# STOFS t00z cycle: 6 hours nowcast + 180 hours forecast
# CWL and forcing both follow this structure
# Skip the nowcast period (hours 0-5) to use only forecast data
NOWCAST_HOURS = 6  # Skip first 6 hours (nowcast period)


def select_wet_nodes(cwl_file: str, bbox: dict, max_nodes: int,
                     min_depth: float = MIN_DEPTH_THRESHOLD,
                     validate_with_data: bool = True):
    """
    Select wet nodes within bounding box, filtering out land/dry nodes.

    Args:
        cwl_file: Path to CWL NetCDF file
        bbox: Bounding box dict with lon_min, lon_max, lat_min, lat_max
        max_nodes: Maximum number of nodes to select
        min_depth: Minimum depth threshold (nodes with depth < this are filtered out)
        validate_with_data: If True, also check CWL data for NaN patterns

    Returns:
        dict with mesh data for wet nodes only
    """
    logger.info(f"Selecting wet nodes from {cwl_file}")
    logger.info(f"  Min depth threshold: {min_depth}m")

    nc = NCDataset(cwl_file, 'r')

    # Get coordinates
    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])

    # Get element connectivity for edge building
    element = np.array(nc.variables['element'][:]) - 1  # 0-indexed

    logger.info(f"Full mesh: {len(x):,} nodes")

    # Step 1: Filter by bounding box
    bbox_mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    logger.info(f"Nodes in bbox: {bbox_mask.sum():,}")

    # Step 2: Filter by depth (positive depth = underwater)
    # In ADCIRC convention: positive depth = below sea level (wet)
    #                       negative depth = above sea level (dry/land)
    depth_mask = depth >= min_depth
    logger.info(f"Nodes with depth >= {min_depth}m: {depth_mask.sum():,}")

    # Step 3: Optionally validate with actual CWL data
    if validate_with_data:
        logger.info("Validating with CWL data to identify consistently wet nodes...")
        zeta = nc.variables['zeta']

        # Check a few timesteps to identify consistently wet nodes
        timesteps_to_check = list(range(NOWCAST_HOURS, min(zeta.shape[0], NOWCAST_HOURS + 10)))

        # Count non-NaN values per node
        nan_count = np.zeros(len(x), dtype=np.int32)
        for t in timesteps_to_check:
            zeta_t = np.array(zeta[t, :], dtype=np.float32)
            # Fill values are typically < -9000
            is_invalid = (zeta_t < -9000) | np.isnan(zeta_t)
            nan_count += is_invalid.astype(np.int32)

        # Nodes that are valid (not NaN) for at least 80% of checked timesteps
        valid_ratio = 1.0 - (nan_count / len(timesteps_to_check))
        data_mask = valid_ratio >= 0.8
        logger.info(f"Nodes with >= 80% valid data: {data_mask.sum():,}")
    else:
        data_mask = np.ones(len(x), dtype=bool)

    nc.close()

    # Combine all masks
    combined_mask = bbox_mask & depth_mask & data_mask
    wet_indices = np.where(combined_mask)[0]
    logger.info(f"Wet nodes in bbox: {len(wet_indices):,}")

    if len(wet_indices) == 0:
        raise ValueError("No wet nodes found! Check bounding box and depth threshold.")

    # Subsample if too many nodes
    if len(wet_indices) > max_nodes:
        logger.info(f"Subsampling from {len(wet_indices):,} to {max_nodes:,} nodes")

        wet_coords = np.column_stack([x[wet_indices], y[wet_indices]])

        # Use farthest point sampling for good spatial coverage
        selected_local = [0]
        remaining = set(range(len(wet_indices)))
        remaining.remove(0)

        np.random.seed(42)

        # First get ~1000 well-spread points using farthest point sampling
        logger.info("  Phase 1: Farthest point sampling for initial spread...")
        while len(selected_local) < min(1000, max_nodes) and remaining:
            selected_coords = wet_coords[selected_local]
            tree = cKDTree(selected_coords)

            remaining_sample = list(remaining)[:5000]
            remaining_coords = wet_coords[remaining_sample]

            dists, _ = tree.query(remaining_coords)
            farthest_local = remaining_sample[np.argmax(dists)]

            selected_local.append(farthest_local)
            remaining.remove(farthest_local)

        # Random sample the rest for speed
        logger.info(f"  Phase 2: Random sampling for remaining {max_nodes - len(selected_local):,} nodes...")
        if len(selected_local) < max_nodes:
            remaining_list = list(remaining)
            np.random.shuffle(remaining_list)
            selected_local.extend(remaining_list[:max_nodes - len(selected_local)])

        selected_local = np.array(selected_local[:max_nodes])
        global_indices = wet_indices[selected_local]
    else:
        global_indices = wet_indices

    logger.info(f"Selected {len(global_indices):,} wet nodes")

    return {
        'global_indices': global_indices,
        'lon': x[global_indices],
        'lat': y[global_indices],
        'depth': depth[global_indices],
        'element': element,
    }


def build_edges_for_selected_nodes(element, global_indices):
    """Build edge connectivity for selected nodes."""
    logger.info("Building edge connectivity...")

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

    logger.info(f"Created {edge_index.shape[1]:,} directed edges")

    return edge_index


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


def preprocess_date(date_str: str, mesh_data: dict, data_dir: Path):
    """Preprocess a single date."""
    date_dir = data_dir / date_str
    cwl_file = date_dir / 'stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = date_dir / 'stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = date_dir / 'stofs_2d_glo.t00z.pressfc.nc'

    # Check files exist
    for f, name in [(cwl_file, 'CWL'), (wind_file, 'Wind'), (pres_file, 'Pressure')]:
        if not f.exists():
            logger.warning(f"  {name} file not found: {f}")
            return None

    logger.info(f"\nProcessing {date_str}...")
    start_time = time.time()

    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']

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
        # Convert fill values to NaN
        elev_t = np.where(elev_t < -9000, np.nan, elev_t)
        elevation[i] = elev_t

    nc_cwl.close()

    # Check data quality
    nan_pct = 100 * np.isnan(elevation).sum() / elevation.size
    logger.info(f"    CWL shape: {elevation.shape}, NaN: {nan_pct:.1f}%")

    if nan_pct > 20:
        logger.warning(f"    WARNING: High NaN percentage ({nan_pct:.1f}%)")

    # 2. Load wind
    logger.info("  Loading wind...")
    nc_wind = NCDataset(str(wind_file), 'r')

    # Try different variable names for coordinates
    if 'grid_xt' in nc_wind.variables:
        grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
        grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
    else:
        grid_lon = np.array(nc_wind.variables['lon'][:], dtype=np.float32)
        grid_lat = np.array(nc_wind.variables['lat'][:], dtype=np.float32)

    # Convert to -180 to 180 if needed
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Subset forcing data to region
    margin = 2.0
    lon_mask = (grid_lon >= BBOX['lon_min'] - margin) & (grid_lon <= BBOX['lon_max'] + margin)
    lat_mask = (grid_lat >= BBOX['lat_min'] - margin) & (grid_lat <= BBOX['lat_max'] + margin)
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    # Try different variable names for wind
    if 'ugrd10m' in nc_wind.variables:
        u_all = np.array(nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
        v_all = np.array(nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    else:
        u_all = np.array(nc_wind.variables['u10'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
        v_all = np.array(nc_wind.variables['v10'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)

    nc_wind.close()

    # 3. Load pressure
    logger.info("  Loading pressure...")
    nc_pres = NCDataset(str(pres_file), 'r')
    pres_raw = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_pres.close()

    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]

    # Match timesteps
    met_times = u_all.shape[0]
    common_times = min(num_times, met_times)

    if common_times < num_times:
        logger.info(f"  Truncating to {common_times} common timesteps (CWL={num_times}, Met={met_times})")
        elevation = elevation[:common_times]

    u_all = u_all[:common_times]
    v_all = v_all[:common_times]
    pres_raw = pres_raw[:common_times]

    # 4. Interpolate forcing
    logger.info("  Interpolating forcing to mesh nodes...")
    u10 = fast_interpolate_to_nodes(u_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    v10 = fast_interpolate_to_nodes(v_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    pressure = fast_interpolate_to_nodes(pres_raw, grid_lat_sub, grid_lon_sub, node_lat, node_lon)

    # Normalize pressure
    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    del u_all, v_all, pres_raw
    gc.collect()

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
    """Get list of dates with complete data."""
    dates = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            cwl = d / 'stofs_2d_glo.t00z.fields.cwl.nc'
            wind = d / 'stofs_2d_glo.t00z.uvgrd10m.nc'
            pres = d / 'stofs_2d_glo.t00z.pressfc.nc'
            if cwl.exists() and wind.exists() and pres.exists():
                dates.append(d.name)
    return dates


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS data (v2 - with dry node filtering)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Path to STOFS data directory')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for processed files')
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES,
                        help='Maximum number of nodes')
    parser.add_argument('--min-depth', type=float, default=MIN_DEPTH_THRESHOLD,
                        help='Minimum depth threshold (m) for wet nodes')
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Specific dates to process (YYYYMMDD)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip already processed dates')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing all dates')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip CWL data validation for wet nodes')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("PREPROCESSING STOFS DATA (v2 - WET NODES ONLY)")
    logger.info("=" * 70)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max nodes: {args.max_nodes}")
    logger.info(f"Min depth: {args.min_depth}m")
    logger.info(f"BBOX: {BBOX}")

    # Check data directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Get available dates
    if args.dates:
        dates = args.dates
    else:
        dates = get_available_dates(data_dir)

    if not dates:
        logger.error("No dates with complete data found!")
        return

    logger.info(f"Found {len(dates)} dates with complete data")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build mesh from first date
    first_cwl = data_dir / dates[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(
        str(first_cwl),
        BBOX,
        args.max_nodes,
        min_depth=args.min_depth,
        validate_with_data=not args.no_validate
    )

    # Build edge connectivity
    edge_index = build_edges_for_selected_nodes(
        mesh_data['element'],
        mesh_data['global_indices']
    )

    # Save mesh
    mesh_file = output_dir / 'mesh_25k.npz'
    np.savez_compressed(
        str(mesh_file),
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
    )
    logger.info(f"\nMesh saved to {mesh_file}")
    logger.info(f"  Nodes: {len(mesh_data['lon']):,}")
    logger.info(f"  Edges: {edge_index.shape[1]:,}")
    logger.info(f"  Depth range: [{mesh_data['depth'].min():.2f}, {mesh_data['depth'].max():.2f}]m")

    # Check which dates already processed
    existing = set()
    if args.skip_existing and not args.force:
        for f in output_dir.glob('processed_*.npz'):
            date = f.stem.replace('processed_', '')
            existing.add(date)
        logger.info(f"Already processed: {len(existing)} dates")

    # Step 2: Process each date
    success = 0
    failed = 0
    skipped = 0

    for i, date_str in enumerate(dates):
        if date_str in existing and not args.force:
            logger.info(f"\n[{i+1}/{len(dates)}] Skipping {date_str} (already exists)")
            skipped += 1
            continue

        try:
            logger.info(f"\n[{i+1}/{len(dates)}] Processing {date_str}...")
            data = preprocess_date(date_str, mesh_data, data_dir)

            if data is None:
                failed += 1
                continue

            # Save
            out_file = output_dir / f'processed_{date_str}.npz'
            np.savez_compressed(
                str(out_file),
                elevation=data['elevation'],
                u10=data['u10'],
                v10=data['v10'],
                pressure=data['pressure'],
            )
            logger.info(f"  Saved to {out_file}")
            success += 1

            del data
            gc.collect()

        except Exception as e:
            logger.error(f"  Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total dates: {len(dates)}")
    logger.info(f"  Processed: {success}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"\nFiles created:")
    logger.info(f"  - mesh_25k.npz ({len(mesh_data['lon']):,} wet nodes)")
    for f in sorted(output_dir.glob('processed_*.npz')):
        logger.info(f"  - {f.name}")


if __name__ == '__main__':
    main()
