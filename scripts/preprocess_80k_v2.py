#!/usr/bin/env python3
"""
Preprocess STOFS CWL data with ENHANCED PHYSICS FEATURES - V2

Domain: Long Island Sound to Southern Maine
  - Latitude: 40-44°N
  - Longitude: 74-69°W

PHYSICS ENHANCEMENTS (Matches STOFS/ADCIRC internal computations):
  1. NONLINEAR WIND FEATURES:
     - u10, v10 (original)
     - wind_speed = sqrt(u10² + v10²)
     - wind_speed_squared (matches τ = ρ × Cd × |U|²)
     - wind_direction = atan2(v10, u10)

  2. PRESSURE GRADIENT:
     - pressure_anomaly (original)
     - dP/dx (matches ADCIRC: -(1/ρ) × ∂P/∂x)
     - dP/dy (matches ADCIRC: -(1/ρ) × ∂P/∂y)

  3. Training script will add EXPANDED TIDAL CONSTITUENTS:
     - M2, S2 (original)
     - N2, K1, O1, M4 (new - 6 total constituents)

Expected improvement: 15-25% RMSE reduction

Usage:
    python scripts/preprocess_80k_v2.py
    python scripts/preprocess_80k_v2.py --dates 20230108 20230109
    python scripts/preprocess_80k_v2.py --output-dir /path/to/output
"""

import numpy as np
from pathlib import Path
import argparse
import logging
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates, sobel
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from netCDF4 import Dataset as NCDataset
except ImportError:
    logger.error("netCDF4 required: pip install netCDF4")
    raise

# ============================================================
# CONFIGURATION - Option A: LI Sound to S. Maine
# ============================================================

# Time settings - VERIFIED: GFS f000 aligns with STOFS Hour 7
NOWCAST_HOURS = 7  # Skip first 7 hours of CWL (nowcast period)
CWL_TOTAL_HOURS = 186  # Total CWL timesteps

# Spatial settings - Option A: Long Island Sound to Southern Maine
BBOX = {
    'lon_min': -74.0,
    'lon_max': -69.0,
    'lat_min': 40.0,
    'lat_max': 44.0
}

# Node configuration
MAX_NODES = 80000
MIN_DEPTH = 0.1

# Edge construction
EDGE_METHOD = 'knn'  # 'knn' or 'triangle'
KNN_K = 6  # Number of nearest neighbors for edge construction

# Normalization constants
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0
WIND_SCALE = 30.0  # Max wind speed for normalization (m/s)

# Grid spacing for gradient computation (GFS is 0.25° resolution)
GFS_DX_DEG = 0.25  # degrees
GFS_DY_DEG = 0.25  # degrees


def build_edges_knn(lon: np.ndarray, lat: np.ndarray, k: int = 6) -> np.ndarray:
    """
    Build edges using K-nearest neighbors.

    This ensures consistent edge density regardless of node subsampling.
    Each node connects to its k nearest neighbors.

    Args:
        lon: Node longitudes
        lat: Node latitudes
        k: Number of nearest neighbors

    Returns:
        edge_index: [2, num_edges] array (bidirectional)
    """
    logger.info(f"Building edges using KNN (k={k})...")

    # Convert to approximate Cartesian coordinates (km)
    lat_center = lat.mean()
    x = lon * np.cos(np.radians(lat_center)) * 111.0  # degrees to km
    y = lat * 111.0

    coords = np.column_stack([x, y])

    # Build KD-tree for efficient neighbor search
    tree = cKDTree(coords)

    # Find k+1 nearest neighbors (includes self)
    distances, indices = tree.query(coords, k=k+1)

    # Build edge set (avoid duplicates)
    edges = set()
    for i in range(len(lon)):
        for j in indices[i, 1:]:  # Skip self (index 0)
            edges.add((min(i, j), max(i, j)))

    # Convert to bidirectional edge_index
    edge_list = list(edges)
    src = [e[0] for e in edge_list] + [e[1] for e in edge_list]
    dst = [e[1] for e in edge_list] + [e[0] for e in edge_list]

    edge_index = np.array([src, dst], dtype=np.int64)

    # Calculate edge statistics
    unique_edges = len(edges)
    avg_edge_length = np.mean([
        np.sqrt((coords[e[0], 0] - coords[e[1], 0])**2 +
                (coords[e[0], 1] - coords[e[1], 1])**2)
        for e in list(edges)[:1000]  # Sample for speed
    ])

    logger.info(f"  Created {unique_edges:,} unique edges ({edge_index.shape[1]:,} bidirectional)")
    logger.info(f"  Edges per node: {edge_index.shape[1] / len(lon):.1f}")
    logger.info(f"  Average edge length: {avg_edge_length:.1f} km")

    return edge_index


def build_edges_triangle(element: np.ndarray, global_indices: np.ndarray) -> np.ndarray:
    """Build edge connectivity from triangular elements."""
    logger.info("Building edges from triangular elements...")

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

    edge_index = np.array([src, dst], dtype=np.int64)
    logger.info(f"  Created {len(edges):,} unique edges ({edge_index.shape[1]:,} bidirectional)")

    return edge_index


def load_gfs_npz(gfs_file: Path, bbox: dict = None):
    """Load GFS data from regional NPZ file."""
    data = np.load(gfs_file)

    result = {
        'u10': data['u10'],
        'v10': data['v10'],
        'sp': data['sp'],
        'fhr': data['fhr'],
        'lat': data['lat'],
        'lon': data['lon'],
    }

    if bbox:
        lat_1d = result['lat'][:, 0]
        lon_1d = result['lon'][0, :]

        lat_mask = (lat_1d >= bbox['lat_min']) & (lat_1d <= bbox['lat_max'])
        lon_mask = (lon_1d >= bbox['lon_min']) & (lon_1d <= bbox['lon_max'])

        lat_idx = np.where(lat_mask)[0]
        lon_idx = np.where(lon_mask)[0]

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
    """Interpolate GFS data temporally to target hourly timesteps."""
    available_hours = gfs_data['fhr']

    logger.info(f"    GFS hours available: {available_hours[0]}-{available_hours[-1]} ({len(available_hours)} timesteps)")
    logger.info(f"    Target hours: {target_hours[0]}-{target_hours[-1]} ({len(target_hours)} timesteps)")

    u10_interp = interp1d(available_hours, gfs_data['u10'], axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    v10_interp = interp1d(available_hours, gfs_data['v10'], axis=0, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    sp_interp = interp1d(available_hours, gfs_data['sp'], axis=0, kind='linear',
                         bounds_error=False, fill_value='extrapolate')

    return {
        'u10': u10_interp(target_hours).astype(np.float32),
        'v10': v10_interp(target_hours).astype(np.float32),
        'sp': sp_interp(target_hours).astype(np.float32),
        'lat': gfs_data['lat'],
        'lon': gfs_data['lon'],
    }


def compute_pressure_gradient(pressure_3d: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple:
    """
    Compute pressure gradient on the GFS grid.

    Uses centered differences to compute dP/dx and dP/dy.
    This matches ADCIRC's pressure gradient term: -(1/ρ) × ∇P

    Args:
        pressure_3d: Pressure field [time, lat, lon] in Pa
        lat: Latitude grid [lat, lon]
        lon: Longitude grid [lat, lon]

    Returns:
        dP_dx: Pressure gradient in x-direction [time, lat, lon] in Pa/m
        dP_dy: Pressure gradient in y-direction [time, lat, lon] in Pa/m
    """
    num_times, nlat, nlon = pressure_3d.shape

    # Get grid spacing in meters
    lat_center = lat.mean()
    dx_m = GFS_DX_DEG * 111000 * np.cos(np.radians(lat_center))  # meters
    dy_m = GFS_DY_DEG * 111000  # meters

    dP_dx = np.zeros_like(pressure_3d)
    dP_dy = np.zeros_like(pressure_3d)

    for t in range(num_times):
        # Use Sobel filter for gradient (more robust than simple diff)
        # Sobel gives gradient in pixel units, need to convert to physical units
        # Sobel kernel sums to 4 in each direction, so divide by 4
        dP_dy_raw = sobel(pressure_3d[t], axis=0, mode='nearest') / 4.0
        dP_dx_raw = sobel(pressure_3d[t], axis=1, mode='nearest') / 4.0

        # Convert to Pa/m
        dP_dx[t] = dP_dx_raw / dx_m
        dP_dy[t] = dP_dy_raw / dy_m

    return dP_dx.astype(np.float32), dP_dy.astype(np.float32)


def interpolate_to_mesh(data_3d: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray,
                        node_lat: np.ndarray, node_lon: np.ndarray) -> np.ndarray:
    """Interpolate gridded data to unstructured mesh nodes."""
    num_times = data_3d.shape[0]
    num_nodes = len(node_lon)

    lat_1d = grid_lat[:, 0] if grid_lat.ndim == 2 else grid_lat
    lon_1d = grid_lon[0, :] if grid_lon.ndim == 2 else grid_lon

    lat_sort = np.argsort(lat_1d)
    lon_sort = np.argsort(lon_1d)
    lat_1d_s = lat_1d[lat_sort]
    lon_1d_s = lon_1d[lon_sort]

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
    logger.info(f"  Domain: {bbox['lat_min']}-{bbox['lat_max']}°N, {abs(bbox['lon_max'])}-{abs(bbox['lon_min'])}°W")

    nc = NCDataset(cwl_file, 'r')

    x = np.array(nc.variables['x'][:])
    y = np.array(nc.variables['y'][:])
    depth = np.array(nc.variables['depth'][:])
    element = np.array(nc.variables['element'][:]) - 1  # Convert to 0-indexed

    logger.info(f"  Full STOFS mesh: {len(x):,} nodes")

    # Filter by bbox and depth
    bbox_mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    depth_mask = depth >= min_depth

    # Validate with CWL data - check for valid values
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
    logger.info(f"  Wet nodes in domain: {len(wet_indices):,}")

    # Subsample if needed
    if len(wet_indices) > max_nodes:
        logger.info(f"  Subsampling to {max_nodes:,} nodes ({100*max_nodes/len(wet_indices):.0f}% utilization)")
        np.random.seed(42)
        selected_idx = np.random.choice(len(wet_indices), max_nodes, replace=False)
        selected_idx = np.sort(selected_idx)
        global_indices = wet_indices[selected_idx]
    else:
        global_indices = wet_indices
        logger.info(f"  Using all {len(wet_indices):,} wet nodes (100% utilization)")

    # Calculate grid spacing
    area_km2 = (bbox['lat_max'] - bbox['lat_min']) * 111 * \
               (bbox['lon_max'] - bbox['lon_min']) * 111 * \
               np.cos(np.radians((bbox['lat_min'] + bbox['lat_max']) / 2))
    grid_spacing = np.sqrt(area_km2 / len(global_indices))
    logger.info(f"  Grid spacing: {grid_spacing:.2f} km")

    return {
        'global_indices': global_indices,
        'lon': x[global_indices],
        'lat': y[global_indices],
        'depth': depth[global_indices],
        'element': element,
    }


def preprocess_date_v2(date_str: str, mesh_data: dict, cwl_dir: Path, gfs_dir: Path):
    """
    Preprocess a single date with ENHANCED PHYSICS FEATURES.

    New features compared to v1:
    - wind_speed: sqrt(u10² + v10²)
    - wind_speed_sq: wind_speed² (matches ADCIRC wind stress)
    - wind_dir: atan2(v10, u10)
    - dP_dx: Pressure gradient x-component
    - dP_dy: Pressure gradient y-component
    """
    cwl_file = cwl_dir / date_str / 'stofs_2d_glo.t00z.fields.cwl.nc'
    gfs_file = gfs_dir / date_str / f'gfs_{date_str}_regional.npz'

    if not cwl_file.exists():
        logger.warning(f"CWL not found: {cwl_file}")
        return None

    if not gfs_file.exists():
        logger.warning(f"GFS not found: {gfs_file}")
        return None

    logger.info(f"Processing {date_str} (V2 - Enhanced Physics)...")

    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']

    # 1. Load CWL (skip nowcast hours)
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(str(cwl_file), 'r')
    zeta = nc_cwl.variables['zeta']

    t_start = NOWCAST_HOURS
    t_end = min(zeta.shape[0], CWL_TOTAL_HOURS)
    num_times = t_end - t_start

    idx_min, idx_max = global_indices.min(), global_indices.max() + 1
    zeta_block = np.array(zeta[t_start:t_end, idx_min:idx_max], dtype=np.float32)
    local_indices = global_indices - idx_min
    elevation = zeta_block[:, local_indices]
    del zeta_block

    elevation = np.where(elevation < -9000, np.nan, elevation)
    nc_cwl.close()
    logger.info(f"    CWL shape: {elevation.shape} (hours {t_start}-{t_end-1})")

    # 2. Load GFS from NPZ
    logger.info("  Loading GFS...")
    gfs_data = load_gfs_npz(gfs_file, BBOX)

    # 3. Interpolate GFS temporally to hourly
    target_gfs_hours = list(range(num_times))
    logger.info("  Interpolating GFS temporally...")
    gfs_hourly = interpolate_gfs_temporal(gfs_data, target_gfs_hours)

    # 4. Compute PRESSURE GRADIENTS on GFS grid BEFORE spatial interpolation
    logger.info("  Computing pressure gradients (matches ADCIRC ∇P term)...")
    dP_dx_grid, dP_dy_grid = compute_pressure_gradient(
        gfs_hourly['sp'], gfs_hourly['lat'], gfs_hourly['lon']
    )

    # 5. Interpolate GFS spatially to mesh nodes
    logger.info("  Interpolating GFS to mesh...")
    u10 = interpolate_to_mesh(gfs_hourly['u10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    v10 = interpolate_to_mesh(gfs_hourly['v10'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    pressure = interpolate_to_mesh(gfs_hourly['sp'], gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    dP_dx = interpolate_to_mesh(dP_dx_grid, gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)
    dP_dy = interpolate_to_mesh(dP_dy_grid, gfs_hourly['lat'], gfs_hourly['lon'], node_lat, node_lon)

    # 6. Compute ENHANCED WIND FEATURES (matches ADCIRC τ = ρ × Cd × |U|²)
    logger.info("  Computing nonlinear wind features (matches ADCIRC wind stress)...")
    wind_speed = np.sqrt(u10**2 + v10**2)
    wind_speed_sq = wind_speed**2  # Key term for wind stress!
    wind_dir = np.arctan2(v10, u10)  # Wind direction in radians

    # 7. Normalize features
    # Pressure anomaly (original)
    pressure_anomaly = (pressure - PRESSURE_MEAN) / PRESSURE_SCALE

    # Pressure gradients - normalize by typical gradient magnitude
    # Typical pressure gradient ~1 Pa/km = 0.001 Pa/m during storms
    PRESSURE_GRAD_SCALE = 0.001  # Pa/m
    dP_dx_norm = dP_dx / PRESSURE_GRAD_SCALE
    dP_dy_norm = dP_dy / PRESSURE_GRAD_SCALE

    # Wind features - normalize
    u10_norm = u10 / WIND_SCALE
    v10_norm = v10 / WIND_SCALE
    wind_speed_norm = wind_speed / WIND_SCALE
    wind_speed_sq_norm = wind_speed_sq / (WIND_SCALE**2)
    # wind_dir is already in radians [-π, π], divide by π to normalize to [-1, 1]
    wind_dir_norm = wind_dir / np.pi

    logger.info(f"    Final shapes:")
    logger.info(f"      elevation: {elevation.shape}")
    logger.info(f"      wind features: u10, v10, speed, speed², dir")
    logger.info(f"      pressure features: anomaly, dP/dx, dP/dy")

    return {
        'date': date_str,
        # Target
        'elevation': elevation.astype(np.float32),
        # Original wind (normalized)
        'u10': u10_norm.astype(np.float32),
        'v10': v10_norm.astype(np.float32),
        # NEW: Enhanced wind features
        'wind_speed': wind_speed_norm.astype(np.float32),
        'wind_speed_sq': wind_speed_sq_norm.astype(np.float32),
        'wind_dir': wind_dir_norm.astype(np.float32),
        # Original pressure (normalized)
        'pressure': pressure_anomaly.astype(np.float32),
        # NEW: Pressure gradients
        'dP_dx': dP_dx_norm.astype(np.float32),
        'dP_dy': dP_dy_norm.astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess STOFS V2 - Enhanced Physics Features')
    parser.add_argument('--cwl-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/stofs_data',
                        help='CWL data directory')
    parser.add_argument('--gfs-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/gfs_forcing',
                        help='GFS data directory')
    parser.add_argument('--output-dir', type=str, default='/mnt/f/STOFS_TRAINING_DATA/processed_80k_v2',
                        help='Output directory')
    parser.add_argument('--dates', nargs='+', help='Specific dates to process')
    parser.add_argument('--max-nodes', type=int, default=MAX_NODES)
    parser.add_argument('--knn-k', type=int, default=KNN_K, help='K for KNN edge construction')
    parser.add_argument('--edge-method', choices=['knn', 'triangle'], default=EDGE_METHOD)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    args = parser.parse_args()

    cwl_dir = Path(args.cwl_dir)
    gfs_dir = Path(args.gfs_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 70)
    logger.info("PREPROCESSING STOFS V2 - ENHANCED PHYSICS FEATURES")
    logger.info("=" * 70)
    logger.info("")
    logger.info("NEW FEATURES (matches STOFS/ADCIRC internal computations):")
    logger.info("  1. Nonlinear wind: speed, speed² (τ = ρ × Cd × |U|²), direction")
    logger.info("  2. Pressure gradient: dP/dx, dP/dy (-(1/ρ) × ∇P)")
    logger.info("  3. Training script will add 6 tidal constituents (M2,S2,N2,K1,O1,M4)")
    logger.info("")
    logger.info(f"Domain: {BBOX['lat_min']}-{BBOX['lat_max']}°N, {abs(BBOX['lon_max'])}-{abs(BBOX['lon_min'])}°W")
    logger.info(f"Max nodes: {args.max_nodes:,}")
    logger.info(f"Edge method: {args.edge_method} (k={args.knn_k})")
    logger.info(f"CWL dir: {cwl_dir}")
    logger.info(f"GFS dir: {gfs_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 70)

    # Get dates
    if args.dates:
        dates = args.dates
    else:
        cwl_dates = set(d.name for d in cwl_dir.iterdir()
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
        gfs_dates = set(d.name for d in gfs_dir.iterdir()
                       if d.is_dir() and d.name.isdigit() and len(d.name) == 8)
        dates = sorted(cwl_dates & gfs_dates)

    if not dates:
        logger.error("No dates found!")
        return

    logger.info(f"Found {len(dates)} dates to process")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mesh from first date
    first_cwl = cwl_dir / dates[0] / 'stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = select_wet_nodes(str(first_cwl), BBOX, args.max_nodes, MIN_DEPTH)

    # Build edges
    if args.edge_method == 'knn':
        edge_index = build_edges_knn(mesh_data['lon'], mesh_data['lat'], k=args.knn_k)
    else:
        edge_index = build_edges_triangle(mesh_data['element'], mesh_data['global_indices'])

    # Save mesh (same as v1 - mesh structure doesn't change)
    mesh_file = output_dir / 'mesh.npz'
    np.savez_compressed(str(mesh_file),
        global_indices=mesh_data['global_indices'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
        depth=mesh_data['depth'],
        edge_index=edge_index,
        bbox=np.array([BBOX['lon_min'], BBOX['lon_max'], BBOX['lat_min'], BBOX['lat_max']]),
        config={'max_nodes': args.max_nodes, 'edge_method': args.edge_method, 'knn_k': args.knn_k,
                'version': 'v2', 'features': ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                               'wind_dir', 'pressure', 'dP_dx', 'dP_dy']},
    )
    logger.info(f"Mesh saved: {len(mesh_data['lon']):,} nodes, {edge_index.shape[1]//2:,} edges")

    # Process dates
    success = 0
    for i, date_str in enumerate(dates):
        out_file = output_dir / f'processed_{date_str}.npz'

        if args.skip_existing and out_file.exists():
            logger.info(f"[{i+1}/{len(dates)}] Skipping {date_str} (exists)")
            success += 1
            continue

        try:
            logger.info(f"[{i+1}/{len(dates)}]")
            data = preprocess_date_v2(date_str, mesh_data, cwl_dir, gfs_dir)

            if data:
                np.savez_compressed(str(out_file),
                    # Target
                    elevation=data['elevation'],
                    # Wind features (5)
                    u10=data['u10'],
                    v10=data['v10'],
                    wind_speed=data['wind_speed'],
                    wind_speed_sq=data['wind_speed_sq'],
                    wind_dir=data['wind_dir'],
                    # Pressure features (3)
                    pressure=data['pressure'],
                    dP_dx=data['dP_dx'],
                    dP_dy=data['dP_dy'],
                )
                success += 1
                logger.info(f"  Saved: {out_file.name}")

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 70)
    logger.info(f"COMPLETED: {success}/{len(dates)} dates")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("OUTPUT FEATURES PER DATE:")
    logger.info("  - elevation: [T, N] water level target")
    logger.info("  - u10, v10: [T, N] wind components (normalized)")
    logger.info("  - wind_speed: [T, N] |U| (normalized)")
    logger.info("  - wind_speed_sq: [T, N] |U|² (normalized) - KEY FOR WIND STRESS")
    logger.info("  - wind_dir: [T, N] wind direction (normalized)")
    logger.info("  - pressure: [T, N] pressure anomaly (normalized)")
    logger.info("  - dP_dx, dP_dy: [T, N] pressure gradients (normalized)")
    logger.info("")
    logger.info("NEXT STEP: Use train_80k_v2.py with 6 tidal constituents")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
