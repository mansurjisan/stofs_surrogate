#!/usr/bin/env python3
"""
Preprocess STOFS CWL (water level) data only - no forcing required.

This script works when only CWL files are available (no wind/pressure forcing).
The model will be trained to predict water level from previous states only.

Usage:
    python scripts/preprocess_cwl_only.py --data-dir /mnt/f/STOFS_TRAINING_DATA/stofs_data
"""

import os
import gc
import time
import argparse
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.spatial import cKDTree
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DATA_DIR = '/mnt/f/STOFS_TRAINING_DATA/stofs_data'
DEFAULT_OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_cwl'

MAX_NODES = 25000

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

# Depth threshold for wet nodes
MIN_DEPTH_THRESHOLD = 0.1  # meters

# Time settings
# STOFS t00z cycle: 6 hours nowcast + 180 hours forecast
# Skip the nowcast period (hours 0-5) to use only forecast data
NOWCAST_HOURS = 6  # Skip first 6 hours (nowcast period)


def select_wet_nodes(cwl_file: str, bbox: dict, max_nodes: int,
                     min_depth: float = MIN_DEPTH_THRESHOLD):
    """
    Select wet nodes within bounding box, filtering out land/dry nodes.
    """
    logger.info(f"Selecting wet nodes from {cwl_file}")
    logger.info(f"  Min depth threshold: {min_depth}m")

    nc = NCDataset(cwl_file, 'r')

    # Get coordinates
    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])

    # Get element connectivity
    element = np.array(nc.variables['element'][:]) - 1  # 0-indexed

    logger.info(f"Full mesh: {len(x):,} nodes")

    # Step 1: Filter by bounding box
    bbox_mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    logger.info(f"Nodes in bbox: {bbox_mask.sum():,}")

    # Step 2: Filter by depth (positive depth = underwater)
    depth_mask = depth >= min_depth
    logger.info(f"Nodes with depth >= {min_depth}m: {depth_mask.sum():,}")

    # Step 3: Validate with actual CWL data
    logger.info("Validating with CWL data...")
    zeta = nc.variables['zeta']

    timesteps_to_check = list(range(NOWCAST_HOURS, min(zeta.shape[0], NOWCAST_HOURS + 10)))

    nan_count = np.zeros(len(x), dtype=np.int32)
    for t in timesteps_to_check:
        zeta_t = np.array(zeta[t, :], dtype=np.float32)
        is_invalid = (zeta_t < -9000) | np.isnan(zeta_t)
        nan_count += is_invalid.astype(np.int32)

    valid_ratio = 1.0 - (nan_count / len(timesteps_to_check))
    data_mask = valid_ratio >= 0.8
    logger.info(f"Nodes with >= 80% valid data: {data_mask.sum():,}")

    nc.close()

    # Combine all masks
    combined_mask = bbox_mask & depth_mask & data_mask
    wet_indices = np.where(combined_mask)[0]
    logger.info(f"Wet nodes in bbox: {len(wet_indices):,}")

    if len(wet_indices) == 0:
        raise ValueError("No wet nodes found!")

    # Subsample if needed
    if len(wet_indices) > max_nodes:
        logger.info(f"Subsampling from {len(wet_indices):,} to {max_nodes:,} nodes")

        wet_coords = np.column_stack([x[wet_indices], y[wet_indices]])

        selected_local = [0]
        remaining = set(range(len(wet_indices)))
        remaining.remove(0)

        np.random.seed(42)

        # Farthest point sampling for initial spread
        while len(selected_local) < min(1000, max_nodes) and remaining:
            selected_coords = wet_coords[selected_local]
            tree = cKDTree(selected_coords)

            remaining_sample = list(remaining)[:5000]
            remaining_coords = wet_coords[remaining_sample]

            dists, _ = tree.query(remaining_coords)
            farthest_local = remaining_sample[np.argmax(dists)]

            selected_local.append(farthest_local)
            remaining.remove(farthest_local)

        # Random sample the rest
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


def preprocess_date(date_str: str, mesh_data: dict, data_dir: Path):
    """Preprocess a single date (CWL only)."""
    cwl_file = data_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'

    if not cwl_file.exists():
        logger.warning(f"  CWL file not found: {cwl_file}")
        return None

    logger.info(f"Processing {date_str}...")
    start_time = time.time()

    global_indices = mesh_data['global_indices']

    # Load CWL
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

    # Check data quality
    nan_pct = 100 * np.isnan(elevation).sum() / elevation.size
    logger.info(f"  Shape: {elevation.shape}, NaN: {nan_pct:.1f}%")
    logger.info(f"  Range: [{np.nanmin(elevation):.3f}, {np.nanmax(elevation):.3f}]m")

    elapsed = time.time() - start_time
    logger.info(f"  Done in {elapsed:.1f}s")

    return {
        'date': date_str,
        'elevation': elevation,
    }


def get_available_dates(data_dir: Path):
    """Get list of dates with CWL data."""
    dates = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8:
            cwl = d / 'stofs_2d_glo.t00z.fields.cwl.nc'
            if cwl.exists():
                dates.append(d.name)
    return dates


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS CWL data (no forcing)')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--min-depth', type=float, default=MIN_DEPTH_THRESHOLD)
    parser.add_argument('--dates', nargs='+', default=None)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("PREPROCESSING STOFS CWL DATA (NO FORCING)")
    logger.info("=" * 70)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Max nodes: {args.max_nodes}")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Get available dates
    if args.dates:
        dates = args.dates
    else:
        dates = get_available_dates(data_dir)

    if not dates:
        logger.error("No dates with CWL data found!")
        return

    logger.info(f"Found {len(dates)} dates with CWL data")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mesh from first date
    first_cwl = data_dir / dates[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(
        str(first_cwl),
        BBOX,
        args.max_nodes,
        min_depth=args.min_depth,
    )

    edge_index = build_edges_for_selected_nodes(
        mesh_data['element'],
        mesh_data['global_indices']
    )

    # Save mesh
    mesh_file = output_dir / 'mesh.npz'
    np.savez_compressed(
        str(mesh_file),
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
    )
    logger.info(f"\nMesh saved: {len(mesh_data['lon']):,} nodes, {edge_index.shape[1]:,} edges")

    # Check existing
    existing = set()
    if args.skip_existing and not args.force:
        for f in output_dir.glob('processed_*.npz'):
            date = f.stem.replace('processed_', '')
            existing.add(date)

    # Process dates
    success = 0
    failed = 0
    skipped = 0

    for i, date_str in enumerate(dates):
        if date_str in existing and not args.force:
            skipped += 1
            continue

        try:
            data = preprocess_date(date_str, mesh_data, data_dir)

            if data is None:
                failed += 1
                continue

            out_file = output_dir / f'processed_{date_str}.npz'
            np.savez_compressed(str(out_file), elevation=data['elevation'])
            success += 1

            del data
            gc.collect()

        except Exception as e:
            logger.error(f"  Error: {e}")
            failed += 1

    logger.info("\n" + "=" * 70)
    logger.info(f"COMPLETE: {success} processed, {skipped} skipped, {failed} failed")
    logger.info(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
