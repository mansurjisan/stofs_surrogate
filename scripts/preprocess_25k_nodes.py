#!/usr/bin/env python3
"""
Preprocess STOFS data for 25,000 nodes (Mid-Atlantic region).
Creates mesh and daily data files for training on A10G.

Usage:
    python scripts/preprocess_25k_nodes.py
"""

import os
import gc
import time
import numpy as np
from netCDF4 import Dataset as NCDataset
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Paths
DATA_DIR = '/mnt/e/Drive2/Good/STOFS_TRAINING_DATA'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_25k'

# Dates to process (Nov 15-20, 6 days)
DATES = ['20251115', '20251116', '20251117', '20251118', '20251119', '20251120']

# Node selection
MAX_NODES = 25000

# Mid-Atlantic bounding box (expanded for better coverage)
BBOX = {
    'lon_min': -77.0,
    'lon_max': -72.0,
    'lat_min': 37.0,
    'lat_max': 42.0
}

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Time settings
NOWCAST_HOURS = 24  # Skip first 24 hours (spin-up)


def select_nodes_in_bbox(cwl_file: str, bbox: dict, max_nodes: int):
    """Select nodes within bounding box, subsample if needed."""
    logger.info(f"Selecting nodes from {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    # Get coordinates
    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])

    # Get element connectivity for edge building
    element = np.array(nc.variables['element'][:]) - 1  # 0-indexed

    nc.close()

    # Filter by bounding box
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )

    bbox_indices = np.where(mask)[0]
    logger.info(f"Nodes in bbox: {len(bbox_indices):,}")

    # Subsample if too many nodes
    if len(bbox_indices) > max_nodes:
        logger.info(f"Subsampling from {len(bbox_indices):,} to {max_nodes:,} nodes")

        # Use KD-tree for spatial subsampling
        bbox_coords = np.column_stack([x[bbox_indices], y[bbox_indices]])

        # Start with first point, greedily add farthest points
        selected_local = [0]
        remaining = set(range(len(bbox_indices)))
        remaining.remove(0)

        # For efficiency, use random sampling after initial spread
        np.random.seed(42)

        # First get ~1000 well-spread points
        while len(selected_local) < min(1000, max_nodes) and remaining:
            # Find point farthest from all selected
            selected_coords = bbox_coords[selected_local]
            tree = cKDTree(selected_coords)

            # Sample subset of remaining for efficiency
            remaining_sample = list(remaining)[:5000]
            remaining_coords = bbox_coords[remaining_sample]

            dists, _ = tree.query(remaining_coords)
            farthest_local = remaining_sample[np.argmax(dists)]

            selected_local.append(farthest_local)
            remaining.remove(farthest_local)

        # Random sample the rest for speed
        if len(selected_local) < max_nodes:
            remaining_list = list(remaining)
            np.random.shuffle(remaining_list)
            selected_local.extend(remaining_list[:max_nodes - len(selected_local)])

        selected_local = np.array(selected_local[:max_nodes])
        global_indices = bbox_indices[selected_local]
    else:
        global_indices = bbox_indices

    logger.info(f"Selected {len(global_indices):,} nodes")

    return {
        'global_indices': global_indices,
        'lon': x[global_indices],
        'lat': y[global_indices],
        'depth': depth[global_indices],
        'element': element,
        'full_lon': x,
        'full_lat': y,
    }


def build_edges_for_selected_nodes(element, global_indices):
    """Build edge connectivity for selected nodes."""
    logger.info("Building edge connectivity...")

    # Create mapping from global to local indices
    global_to_local = {g: l for l, g in enumerate(global_indices)}
    selected_set = set(global_indices)

    edges = set()

    for tri in element:
        # Check which nodes of this triangle are in our selection
        local_nodes = []
        for node in tri:
            if node in selected_set:
                local_nodes.append(global_to_local[node])

        # Add edges for all selected nodes in this triangle
        if len(local_nodes) >= 2:
            for i in range(len(local_nodes)):
                for j in range(i + 1, len(local_nodes)):
                    n1, n2 = local_nodes[i], local_nodes[j]
                    edges.add((min(n1, n2), max(n1, n2)))

    # Convert to edge_index format (both directions)
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

    # Sort grid coordinates
    lat_sort = np.argsort(grid_lat)
    lon_sort = np.argsort(grid_lon)
    grid_lat_s = grid_lat[lat_sort]
    grid_lon_s = grid_lon[lon_sort]

    # Compute fractional indices for nodes
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
        elevation[i] = np.array(zeta[t, global_indices], dtype=np.float32)

    nc_cwl.close()
    logger.info(f"    CWL shape: {elevation.shape}")

    # 2. Load and interpolate forcing
    logger.info("  Loading wind...")
    nc_wind = NCDataset(wind_file, 'r')

    grid_lat = np.array(nc_wind.variables['lat'][:])
    grid_lon = np.array(nc_wind.variables['lon'][:])

    u10_raw = np.array(nc_wind.variables['u10'][time_indices], dtype=np.float32)
    v10_raw = np.array(nc_wind.variables['v10'][time_indices], dtype=np.float32)
    nc_wind.close()

    logger.info("  Interpolating wind...")
    u10 = fast_interpolate_to_nodes(u10_raw, grid_lat, grid_lon, node_lat, node_lon)
    v10 = fast_interpolate_to_nodes(v10_raw, grid_lat, grid_lon, node_lat, node_lon)

    del u10_raw, v10_raw
    gc.collect()

    logger.info("  Loading pressure...")
    nc_pres = NCDataset(pres_file, 'r')
    pres_lat = np.array(nc_pres.variables['lat'][:])
    pres_lon = np.array(nc_pres.variables['lon'][:])
    pres_raw = np.array(nc_pres.variables['pressfc'][time_indices], dtype=np.float32)
    nc_pres.close()

    logger.info("  Interpolating pressure...")
    pressure = fast_interpolate_to_nodes(pres_raw, pres_lat, pres_lon, node_lat, node_lon)

    # Normalize pressure
    pressure = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    del pres_raw
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


def main():
    logger.info("=" * 70)
    logger.info("PREPROCESSING 25K NODES - STOFS DATA")
    logger.info("=" * 70)
    logger.info(f"Dates: {DATES}")
    logger.info(f"MAX_NODES: {MAX_NODES}")
    logger.info(f"BBOX: {BBOX}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build mesh from first date
    first_cwl = f'{DATA_DIR}/{DATES[0]}/stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_nodes_in_bbox(first_cwl, BBOX, MAX_NODES)

    # Build edge connectivity
    edge_index = build_edges_for_selected_nodes(
        mesh_data['element'],
        mesh_data['global_indices']
    )

    # Save mesh
    mesh_file = f'{OUTPUT_DIR}/mesh_25k.npz'
    np.savez_compressed(
        mesh_file,
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
    )
    logger.info(f"\nMesh saved to {mesh_file}")
    logger.info(f"  Nodes: {len(mesh_data['lon']):,}")
    logger.info(f"  Edges: {edge_index.shape[1]:,}")

    # Step 2: Process each date
    for date_str in DATES:
        try:
            data = preprocess_date(date_str, mesh_data)

            # Save
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
            continue

    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Files created:")
    logger.info(f"  - mesh_25k.npz")
    for date_str in DATES:
        logger.info(f"  - processed_{date_str}.npz")


if __name__ == '__main__':
    main()
