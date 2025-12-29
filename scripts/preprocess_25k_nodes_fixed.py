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
from scipy.spatial import cKDTree, Delaunay
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

# Dates to process (15 days: Nov 15-29)
DATES = [
    '20251115', '20251116', '20251117', '20251118', '20251119',
    '20251120', '20251121', '20251122', '20251123', '20251124',
    '20251125', '20251126', '20251127', '20251128', '20251129',
]

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
NOWCAST_HOURS = 5  # Skip first 5 hours (consistent with 15k preprocessing)


def select_nodes_in_bbox(cwl_file: str, bbox: dict, max_nodes: int):
    """Select nodes within bounding box, subsample if needed."""
    logger.info(f"Selecting nodes from {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    # Get coordinates
    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])

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

        # Use random subsampling with fixed seed for reproducibility
        np.random.seed(42)
        selected_local = np.random.choice(len(bbox_indices), size=max_nodes, replace=False)
        selected_local = np.sort(selected_local)
        global_indices = bbox_indices[selected_local]
    else:
        global_indices = bbox_indices

    logger.info(f"Selected {len(global_indices):,} nodes")

    return {
        'global_indices': global_indices,
        'lon': x[global_indices].astype(np.float32),
        'lat': y[global_indices].astype(np.float32),
        'depth': depth[global_indices].astype(np.float32),
    }


def build_edges_delaunay(lon, lat):
    """Build edge connectivity using Delaunay triangulation."""
    logger.info("Building edge connectivity with Delaunay triangulation...")

    points = np.column_stack([lon, lat])
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            n1, n2 = simplex[i], simplex[(i + 1) % 3]
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

    # File paths - your actual file structure
    date_dir = f'{DATA_DIR}/{date_str}'
    cwl_file = f'{date_dir}/stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = f'{date_dir}/stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = f'{date_dir}/stofs_2d_glo.t00z.pressfc.nc'

    logger.info(f"\nProcessing {date_str}...")
    start_time = time.time()

    # Check if files exist
    if not os.path.exists(cwl_file):
        logger.error(f"  CWL file not found: {cwl_file}")
        return None
    if not os.path.exists(wind_file):
        logger.error(f"  Wind file not found: {wind_file}")
        return None
    if not os.path.exists(pres_file):
        logger.error(f"  Pressure file not found: {pres_file}")
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
        elev_t = np.where(elev_t < -9000, np.nan, elev_t)  # Handle missing values
        elevation[i] = elev_t

    nc_cwl.close()
    logger.info(f"    CWL shape: {elevation.shape}")

    # 2. Load wind forcing (uvgrd10m.nc) - same as v3 optimized
    logger.info("  Loading wind...")
    nc_wind = NCDataset(wind_file, 'r')

    # Use grid_xt and grid_yt (1D coordinates)
    grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Subset to bbox region with margin (faster interpolation)
    margin = 2.0
    lon_mask = (grid_lon >= BBOX['lon_min'] - margin) & (grid_lon <= BBOX['lon_max'] + margin)
    lat_mask = (grid_lat >= BBOX['lat_min'] - margin) & (grid_lat <= BBOX['lat_max'] + margin)
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    # Load wind using ugrd10m/vgrd10m variable names
    u_all = np.array(nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    v_all = np.array(nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_wind.close()

    # 3. Load pressure forcing (pressfc.nc)
    logger.info("  Loading pressure...")
    nc_pres = NCDataset(pres_file, 'r')
    p_all = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1], dtype=np.float32)
    nc_pres.close()

    # Use subsetted coordinates for interpolation
    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]

    # Ensure same number of timesteps
    met_times = u_all.shape[0]
    common_times = min(num_times, met_times)
    elevation = elevation[:common_times]
    u_all = u_all[:common_times]
    v_all = v_all[:common_times]
    p_all = p_all[:common_times]

    # 4. Interpolate forcing to mesh nodes (using subsetted grid)
    logger.info("  Interpolating forcing to mesh nodes...")

    u10 = fast_interpolate_to_nodes(u_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    v10 = fast_interpolate_to_nodes(v_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    pressure = fast_interpolate_to_nodes(p_all, grid_lat_sub, grid_lon_sub, node_lat, node_lon)

    # Normalize pressure
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
    logger.info("PREPROCESSING 25K NODES - STOFS DATA")
    logger.info("=" * 70)
    logger.info(f"Dates: {DATES}")
    logger.info(f"MAX_NODES: {MAX_NODES}")
    logger.info(f"BBOX: {BBOX}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build mesh from first date
    # Try both path formats
    first_cwl = f'{DATA_DIR}/stofs_2d_glo.{DATES[0]}/stofs_2d_glo.t00z.fields.cwl.nc'
    if not os.path.exists(first_cwl):
        first_cwl = f'{DATA_DIR}/{DATES[0]}/stofs_2d_glo.t00z.fields.cwl.nc'
    
    if not os.path.exists(first_cwl):
        logger.error(f"Cannot find CWL file. Tried:")
        logger.error(f"  {DATA_DIR}/stofs_2d_glo.{DATES[0]}/stofs_2d_glo.t00z.fields.cwl.nc")
        logger.error(f"  {DATA_DIR}/{DATES[0]}/stofs_2d_glo.t00z.fields.cwl.nc")
        return
    
    mesh_data = select_nodes_in_bbox(first_cwl, BBOX, MAX_NODES)

    # Build edge connectivity using Delaunay (simpler and robust)
    edge_index = build_edges_delaunay(mesh_data['lon'], mesh_data['lat'])

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
    successful_dates = []
    for date_str in DATES:
        try:
            data = preprocess_date(date_str, mesh_data)

            if data is None:
                continue

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
            successful_dates.append(date_str)

            del data
            gc.collect()

        except Exception as e:
            logger.error(f"  Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n" + "=" * 70)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Successfully processed: {len(successful_dates)}/{len(DATES)} dates")
    logger.info(f"Files created:")
    logger.info(f"  - mesh_25k.npz ({len(mesh_data['lon']):,} nodes, {edge_index.shape[1]:,} edges)")
    for date_str in successful_dates:
        logger.info(f"  - processed_{date_str}.npz")


if __name__ == '__main__':
    main()
