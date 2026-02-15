#!/usr/bin/env python3
"""
Multi-Date CWL GNN Training Script for STOFS 2D Global

This script is adapted to work with the new training data format from:
E:/Drive2/Good/STOFS_TRAINING_DATA/

Each date folder contains:
- stofs_2d_glo.t00z.fields.cwl.nc  (water elevation on unstructured mesh)
- stofs_2d_glo.t00z.pressfc.nc     (surface pressure on regular grid)
- stofs_2d_glo.t00z.uvgrd10m.nc    (wind components on regular grid)

Features:
- Processes multiple dates for more robust training
- Handles time-varying forcing data (wind u/v, pressure)
- Physics-informed loss with mass conservation
- Curriculum learning for multi-step predictions
- Memory optimized for RTX 3050 (4GB VRAM)

Author: Adapted for STOFS operational forecasting
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.checkpoint import checkpoint
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration - Optimized for RTX 3050 (4GB VRAM)
# ============================================================

# Memory settings
USE_FLOAT16_STORAGE = True
USE_GRADIENT_CHECKPOINTING = True

# Domain - Mid-Atlantic bight
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Data paths - NEW TRAINING DATA LOCATION
DATA_DIR = '/mnt/e/Drive2/Good/STOFS_TRAINING_DATA'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'

# Training dates - all available dates in new data
# TRAINING_DATES = [
#     '20251115', '20251116', '20251117', '20251118',
#     '20251119', '20251120', '20251121', '20251122',
#     '20251123', '20251124', '20251125', '20251126',
#     '20251127', '20251128', '20251129', '20251130',
# ]

# TEST: Using only Nov 28-30 for initial testing
TRAINING_DATES = ['20251128', '20251129', '20251130']

# How many dates to use for validation (last N dates)
VAL_DATES = 1  # Using 1 for test with 3 dates (2 train, 1 val)

# Model parameters - VRAM optimized
HIDDEN_DIM = 96
NUM_LAYERS = 6
STATE_DIM = 1             # CWL
STATIC_NODE_FEATURES = 4  # x_norm, y_norm, depth_norm, water_level_norm
FORCING_FEATURES = 3      # u10, v10, pressure
EDGE_FEATURES = 3         # dx, dy, dist

# Training parameters
EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
ETA_SCALE = 2.0
GRAD_CLIP = 1.0

# Mesh parameters
MAX_NODES = 50000  # Subsample mesh to fit in memory (50k keeps ~10% resolution)

# Time alignment parameters
# CWL files have nowcast period before forecast
# For t00z cycle: CWL starts at 19:00 previous day, forecast (00:00) starts at index 5
# Met forcing files only have forecast period (starts at cycle time 00:00)
NOWCAST_HOURS = 5  # Skip first 5 hours of CWL to align with met forcing

# Curriculum learning
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 100
MAX_ROLLOUT_STEPS = 2

# Physics-informed loss
MASS_CONSERVATION_WEIGHT = 0.05
SMOOTHNESS_WEIGHT = 0.01

# Normalization constants
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


# ============================================================
# Physics-Informed Model Architecture (same as original)
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
    """
    Message passing block inspired by Shallow Water Equations.
    """

    def __init__(self, hidden_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.gradient_scale = nn.Parameter(torch.ones(1))

    def _edge_update(self, edge_attr, h_src, h_dst, h_gradient):
        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        return edge_msg

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        h_gradient = h_dst - h_src

        if self.use_checkpointing and self.training:
            edge_msg = checkpoint(
                self._edge_update, edge_attr, h_src, h_dst, h_gradient,
                use_reentrant=False
            )
        else:
            edge_msg = self._edge_update(edge_attr, h_src, h_dst, h_gradient)

        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr


class PhysicsInformedCWLModel(nn.Module):
    """
    GNN for Coastal Water Level prediction with physics-informed design.
    """

    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int = 6,
        use_checkpointing: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing

        node_input_dim = state_dim + static_feature_dim + forcing_feature_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.layers = nn.ModuleList([
            SWEInspiredGraphBlock(hidden_dim, use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PhysicsInformedCWLModel initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")

    def forward(
        self,
        x: torch.Tensor,
        static_features: torch.Tensor,
        forcing_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        node_input = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_input)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        out = self.decoder(h)
        return out


# ============================================================
# Physics-Informed Loss Functions
# ============================================================

class PhysicsInformedLoss(nn.Module):
    """Combined loss with physics constraints."""

    def __init__(
        self,
        mass_weight: float = 0.05,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        self.mass_weight = mass_weight
        self.smoothness_weight = smoothness_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mse_loss = F.mse_loss(pred, target)

        pred_sum = pred.sum()
        target_sum = target.sum()
        mass_loss = ((pred_sum - target_sum) / (target_sum.abs() + 1e-8)) ** 2

        smoothness_loss = torch.tensor(0.0, device=pred.device)
        if edge_index is not None and self.smoothness_weight > 0:
            row, col = edge_index
            pred_diff = (pred[row] - pred[col]).abs()
            target_diff = (target[row] - target[col]).abs()
            smoothness_loss = F.relu(pred_diff - target_diff * 1.5).mean()

        total_loss = (
            mse_loss +
            self.mass_weight * mass_loss +
            self.smoothness_weight * smoothness_loss
        )

        components = {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            'total': total_loss.item(),
        }

        return total_loss, components


# ============================================================
# Curriculum Learning
# ============================================================

class CurriculumScheduler:
    """Curriculum learning scheduler for multi-step predictions."""

    def __init__(
        self,
        max_steps: int = 2,
        warmup_epochs: int = 100,
        min_steps: int = 1,
    ):
        self.max_steps = max_steps
        self.warmup_epochs = warmup_epochs
        self.min_steps = min_steps

    def get_num_steps(self, epoch: int) -> int:
        if epoch >= self.warmup_epochs:
            return self.max_steps
        progress = epoch / self.warmup_epochs
        steps = self.min_steps + int((self.max_steps - self.min_steps) * progress)
        return min(steps, self.max_steps)


# ============================================================
# Data Preprocessing - Subset to Mid-Atlantic Region
# ============================================================

def preprocess_and_save_subset(
    date_str: str,
    source_dir: str,
    output_dir: str,
    bbox: dict,
    max_nodes: int = MAX_NODES,
    nowcast_offset: int = NOWCAST_HOURS,
) -> str:
    """
    Preprocess raw STOFS data by subsetting to Mid-Atlantic region and saving.

    This dramatically reduces file sizes (from ~40GB to ~100MB per date) and
    speeds up subsequent training runs.

    Uses original ADCIRC mesh connectivity instead of Delaunay triangulation
    to preserve proper mesh structure.

    Args:
        date_str: Date string (e.g., '20251115')
        source_dir: Source data directory
        output_dir: Output directory for processed data
        bbox: Bounding box for spatial subset
        max_nodes: Maximum number of mesh nodes
        nowcast_offset: Hours to skip from CWL (nowcast period)

    Returns:
        Path to saved processed file
    """
    source_path = f'{source_dir}/{date_str}'
    output_path = f'{output_dir}/processed_{date_str}.npz'

    if os.path.exists(output_path):
        logger.info(f"  Processed file exists: {output_path}")
        return output_path

    logger.info(f"Preprocessing {date_str}...")

    # Files to process
    cwl_file = f'{source_path}/stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = f'{source_path}/stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = f'{source_path}/stofs_2d_glo.t00z.pressfc.nc'

    # 1. Extract mesh and CWL data
    logger.info("  Loading CWL file...")
    nc_cwl = NCDataset(cwl_file, 'r')

    x = np.array(nc_cwl.variables['x'][:], dtype=np.float32)
    y = np.array(nc_cwl.variables['y'][:], dtype=np.float32)
    depth = np.array(nc_cwl.variables['depth'][:], dtype=np.float32)
    times = np.array(nc_cwl.variables['time'][:])

    # Load original ADCIRC element connectivity
    elements = np.array(nc_cwl.variables['element'][:], dtype=np.int32)  # (nele, 3)
    logger.info(f"    Original mesh: {len(x):,} nodes, {len(elements):,} elements")

    # Filter to bounding box
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    bbox_indices = np.where(mask)[0]
    logger.info(f"    Found {len(bbox_indices):,} nodes in bbox")

    # Subsample if needed (random but reproducible)
    if len(bbox_indices) > max_nodes:
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(bbox_indices, size=max_nodes, replace=False)
        subset_indices = np.sort(subset_indices)
        logger.info(f"    Subsampled to {len(subset_indices):,} nodes")
    else:
        subset_indices = bbox_indices

    # Create mapping from global to local indices
    global_to_local = {g: l for l, g in enumerate(subset_indices)}

    # Extract subset of mesh
    lon_sub = x[subset_indices]
    lat_sub = y[subset_indices]
    depth_sub = depth[subset_indices]

    # 5. Build edge index from original ADCIRC elements
    # Keep only elements where ALL 3 vertices are in our subset
    logger.info("  Building edge connectivity from ADCIRC elements...")
    subset_set = set(subset_indices)
    edges = set()

    for elem in elements:
        # ADCIRC elements are 1-indexed, convert to 0-indexed
        n0, n1, n2 = elem[0] - 1, elem[1] - 1, elem[2] - 1

        # Check if all vertices are in our subset
        if n0 in subset_set and n1 in subset_set and n2 in subset_set:
            # Add edges (using local indices)
            l0, l1, l2 = global_to_local[n0], global_to_local[n1], global_to_local[n2]
            edges.add(tuple(sorted([l0, l1])))
            edges.add(tuple(sorted([l1, l2])))
            edges.add(tuple(sorted([l2, l0])))

    logger.info(f"    Extracted {len(edges):,} unique edges from original mesh")

    # If we lost too many edges due to subsampling, fall back to Delaunay
    if len(edges) < len(subset_indices) * 2:
        logger.warning("    Too few edges from original mesh, using Delaunay triangulation...")
        points = np.column_stack([lon_sub, lat_sub])
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edges.add(tuple(sorted([simplex[i], simplex[j]])))
        logger.info(f"    Delaunay: {len(edges):,} edges")

    edges = np.array(list(edges))
    edge_index = np.vstack([edges, edges[:, ::-1]]).T

    # Extract CWL time series, skipping nowcast
    zeta = nc_cwl.variables['zeta']
    full_times = zeta.shape[0]
    time_indices = list(range(nowcast_offset, full_times))

    logger.info(f"    Loading CWL: skipping {nowcast_offset} nowcast hours, {len(time_indices)} forecast hours")
    elevation = np.zeros((len(time_indices), len(subset_indices)), dtype=np.float16)
    for i, t in enumerate(time_indices):
        elevation[i, :] = zeta[t, subset_indices]
        if i % 50 == 0:
            logger.info(f"      CWL time step {i}/{len(time_indices)}")

    elevation = np.where(elevation < -9000, np.nan, elevation)
    times_subset = times[time_indices]

    nc_cwl.close()

    # 2. Load and subset met forcing
    logger.info("  Loading wind file...")
    nc_wind = NCDataset(wind_file, 'r')
    grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Find met grid subset that covers our bbox (with margin)
    margin = 2.0
    lon_mask = (grid_lon >= bbox['lon_min'] - margin) & (grid_lon <= bbox['lon_max'] + margin)
    lat_mask = (grid_lat >= bbox['lat_min'] - margin) & (grid_lat <= bbox['lat_max'] + margin)

    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]

    logger.info(f"    Met grid subset: lon[{lon_idx[0]}:{lon_idx[-1]+1}], lat[{lat_idx[0]}:{lat_idx[-1]+1}]")

    # Extract subset
    u10_raw = nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    v10_raw = nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    nc_wind.close()

    logger.info("  Loading pressure file...")
    nc_pres = NCDataset(pres_file, 'r')
    pressure_raw = nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    nc_pres.close()

    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]

    met_times = u10_raw.shape[0]

    # 3. Interpolate met to mesh nodes
    logger.info(f"  Interpolating met forcing to {len(subset_indices)} nodes...")

    # Sort grid for interpolation
    lon_sort_idx = np.argsort(grid_lon_sub)
    lat_sort_idx = np.argsort(grid_lat_sub)
    grid_lon_sorted = grid_lon_sub[lon_sort_idx]
    grid_lat_sorted = grid_lat_sub[lat_sort_idx]

    u10_interp = np.zeros((met_times, len(subset_indices)), dtype=np.float16)
    v10_interp = np.zeros_like(u10_interp)
    # Keep pressure in float32 to avoid overflow before normalization
    pressure_interp = np.zeros((met_times, len(subset_indices)), dtype=np.float32)

    for t in range(met_times):
        if t % 50 == 0:
            logger.info(f"    Met time step {t}/{met_times}")

        for var_name, var_data, result_arr in [
            ('u10', u10_raw, u10_interp),
            ('v10', v10_raw, v10_interp),
            ('pressure', pressure_raw, pressure_interp)
        ]:
            data = var_data[t].astype(np.float32)
            data_sorted = data[lat_sort_idx][:, lon_sort_idx]

            interp = RegularGridInterpolator(
                (grid_lat_sorted, grid_lon_sorted),
                data_sorted,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            values = interp(np.column_stack([lat_sub, lon_sub]))
            if np.any(np.isnan(values)):
                values[np.isnan(values)] = np.nanmean(values)

            # Keep pressure in float32, convert others to float16
            if var_name == 'pressure':
                result_arr[t] = values.astype(np.float32)
            else:
                result_arr[t] = values.astype(np.float16)

    # Normalize pressure
    pressure_interp = ((pressure_interp.astype(np.float32) - PRESSURE_MEAN) / PRESSURE_SCALE).astype(np.float16)

    # 4. Align timesteps (CWL may have more forecast hours than met)
    common_times = min(len(time_indices), met_times)
    elevation = elevation[:common_times]
    times_subset = times_subset[:common_times]
    u10_interp = u10_interp[:common_times]
    v10_interp = v10_interp[:common_times]
    pressure_interp = pressure_interp[:common_times]

    logger.info(f"  Final aligned data: {common_times} timesteps, {len(subset_indices)} nodes")

    # 6. Save processed data
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        output_path,
        # Mesh data
        lon=lon_sub,
        lat=lat_sub,
        depth=depth_sub,
        edge_index=edge_index,
        global_indices=subset_indices,
        # Time series data
        elevation=elevation,
        times=times_subset,
        u10=u10_interp,
        v10=v10_interp,
        pressure=pressure_interp,
        # Metadata
        date=date_str,
        bbox=np.array([bbox['lon_min'], bbox['lon_max'], bbox['lat_min'], bbox['lat_max']]),
        nowcast_offset=nowcast_offset,
    )

    file_size_mb = os.path.getsize(output_path) / 1e6
    logger.info(f"  Saved: {output_path} ({file_size_mb:.1f} MB)")

    # Cleanup
    del u10_raw, v10_raw, pressure_raw, elevation
    gc.collect()

    return output_path


def load_preprocessed_data(npz_path: str) -> Dict:
    """
    Load preprocessed data from NPZ file.

    Args:
        npz_path: Path to preprocessed NPZ file

    Returns:
        Dictionary with mesh and time series data
    """
    logger.info(f"  Loading preprocessed: {npz_path}")

    data = np.load(npz_path)

    result = {
        'date': str(data['date']),
        'mesh': {
            'lon': data['lon'],
            'lat': data['lat'],
            'depth': data['depth'],
            'edge_index': data['edge_index'],
            'global_indices': data['global_indices'],
        },
        'elevation': data['elevation'],
        'times': data['times'],
        'forcing': {
            'u10': data['u10'],
            'v10': data['v10'],
            'pressure': data['pressure'],
        },
    }

    logger.info(f"    Loaded: {result['elevation'].shape[0]} timesteps, {len(result['mesh']['lon'])} nodes")

    return result


# ============================================================
# Data Loading for New Format
# ============================================================

def extract_mesh_from_cwl(cwl_file: str, bbox: dict, max_nodes: int = 15000) -> Dict:
    """
    Extract mesh from CWL file for the specified bounding box.

    Args:
        cwl_file: Path to CWL netCDF file
        bbox: Bounding box dict with lon_min, lon_max, lat_min, lat_max
        max_nodes: Maximum number of nodes to include

    Returns:
        Dictionary with mesh data (lon, lat, depth, edge_index, global_indices)
    """
    logger.info(f"Extracting mesh from: {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    x = np.array(nc.variables['x'][:], dtype=np.float32)
    y = np.array(nc.variables['y'][:], dtype=np.float32)
    depth = np.array(nc.variables['depth'][:], dtype=np.float32)

    # Filter to bounding box
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    subset_indices = np.where(mask)[0]

    logger.info(f"  Found {len(subset_indices):,} nodes in bbox")

    # Subsample if needed
    if len(subset_indices) > max_nodes:
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(subset_indices, size=max_nodes, replace=False)
        subset_indices = np.sort(subset_indices)
        logger.info(f"  Subsampled to {len(subset_indices):,} nodes")

    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_sub = depth[subset_indices]

    # Build Delaunay triangulation for edges
    points = np.column_stack([lon, lat])
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))

    edges = np.array(list(edges))
    edge_index = np.vstack([edges, edges[:, ::-1]]).T

    nc.close()

    logger.info(f"  Mesh: {len(lon)} nodes, {edge_index.shape[1]} edges")

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth_sub,
        'edge_index': edge_index,
        'global_indices': subset_indices,
    }


def load_cwl_data(cwl_file: str, global_indices: np.ndarray,
                  nowcast_offset: int = NOWCAST_HOURS,
                  max_times: int = None,
                  temporal_subsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CWL (zeta) time series from file, skipping nowcast period.

    Args:
        cwl_file: Path to CWL netCDF file
        global_indices: Node indices to extract
        nowcast_offset: Number of hours to skip at start (nowcast period)
        max_times: Maximum number of timesteps to load (for alignment with met)
        temporal_subsample: Temporal subsampling factor

    Returns:
        Tuple of (elevation array, time array)
    """
    logger.info(f"  Loading CWL data: {cwl_file}")

    nc = NCDataset(cwl_file, 'r')

    zeta = nc.variables['zeta']
    times = np.array(nc.variables['time'][:])

    full_times = zeta.shape[0]

    # Skip nowcast period and apply temporal subsampling
    start_idx = nowcast_offset
    time_indices = list(range(start_idx, full_times, temporal_subsample))

    # Limit to max_times if specified (for alignment with met forcing)
    if max_times is not None and len(time_indices) > max_times:
        time_indices = time_indices[:max_times]

    num_times = len(time_indices)

    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
    elevation = np.zeros((num_times, len(global_indices)), dtype=dtype)

    for i, t in enumerate(time_indices):
        elevation[i, :] = zeta[t, global_indices]

    # Mask invalid values
    elevation = np.where(elevation < -9000, np.nan, elevation)

    nc.close()

    logger.info(f"    Skipped {nowcast_offset} nowcast hours, loaded {num_times} forecast hours")
    logger.info(f"    Shape: {elevation.shape}, valid: {(~np.isnan(elevation)).sum():,}")

    return elevation, times[time_indices]


def load_met_forcing(date_dir: str, node_lon: np.ndarray,
                     node_lat: np.ndarray, subsample_factor: int = 4) -> Tuple[Dict, int]:
    """
    Load meteorological forcing (wind u/v, pressure) and interpolate to mesh nodes.

    The new data format has:
    - stofs_2d_glo.t00z.uvgrd10m.nc: ugrd10m, vgrd10m on (record, grid_yt, grid_xt)
    - stofs_2d_glo.t00z.pressfc.nc: pressfc on (record, grid_yt, grid_xt)

    Met forcing files only contain forecast period (no nowcast).
    CWL files should be aligned by skipping nowcast hours.

    Args:
        date_dir: Path to date folder
        node_lon: Mesh node longitudes
        node_lat: Mesh node latitudes
        subsample_factor: Spatial subsampling factor for met grid

    Returns:
        Tuple of (dictionary with u10, v10, pressure interpolated to nodes, number of timesteps)
    """
    logger.info(f"  Loading met forcing from: {date_dir}")

    wind_file = f'{date_dir}/stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = f'{date_dir}/stofs_2d_glo.t00z.pressfc.nc'

    # Load wind data
    nc_wind = NCDataset(wind_file, 'r')
    grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)

    # Convert longitude from 0-360 to -180-180 if needed
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    u10_raw = np.array(nc_wind.variables['ugrd10m'][:], dtype=np.float32)
    v10_raw = np.array(nc_wind.variables['vgrd10m'][:], dtype=np.float32)
    nc_wind.close()

    # Load pressure data
    nc_pres = NCDataset(pres_file, 'r')
    pressure_raw = np.array(nc_pres.variables['pressfc'][:], dtype=np.float32)
    nc_pres.close()

    met_times = u10_raw.shape[0]
    logger.info(f"    Met grid: {len(grid_lon)} x {len(grid_lat)}, {met_times} timesteps (forecast only)")

    # Subsample spatial grid
    if subsample_factor > 1:
        u10_raw = u10_raw[:, ::subsample_factor, ::subsample_factor]
        v10_raw = v10_raw[:, ::subsample_factor, ::subsample_factor]
        pressure_raw = pressure_raw[:, ::subsample_factor, ::subsample_factor]
        grid_lon = grid_lon[::subsample_factor]
        grid_lat = grid_lat[::subsample_factor]

    # Sort grid for interpolation
    lon_sort_idx = np.argsort(grid_lon)
    lat_sort_idx = np.argsort(grid_lat)
    grid_lon_sorted = grid_lon[lon_sort_idx]
    grid_lat_sorted = grid_lat[lat_sort_idx]

    # Interpolate to mesh nodes
    num_nodes = len(node_lon)
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32

    result = {
        'u10': np.zeros((met_times, num_nodes), dtype=dtype),
        'v10': np.zeros((met_times, num_nodes), dtype=dtype),
        'pressure': np.zeros((met_times, num_nodes), dtype=dtype),
    }

    logger.info(f"    Interpolating to {num_nodes} nodes...")

    for t in range(met_times):
        if t % 50 == 0:
            logger.info(f"      Time step {t}/{met_times}")

        for var_name, var_data in [('u10', u10_raw), ('v10', v10_raw), ('pressure', pressure_raw)]:
            data = var_data[t].astype(np.float32)
            data_sorted = data[lat_sort_idx][:, lon_sort_idx]

            interp = RegularGridInterpolator(
                (grid_lat_sorted, grid_lon_sorted),
                data_sorted,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            values = interp(np.column_stack([node_lat, node_lon]))

            # Fill NaN with mean
            if np.any(np.isnan(values)):
                values[np.isnan(values)] = np.nanmean(values)

            result[var_name][t] = values.astype(dtype)

    # Normalize pressure
    result['pressure'] = ((result['pressure'].astype(np.float32) - PRESSURE_MEAN) / PRESSURE_SCALE).astype(dtype)

    del u10_raw, v10_raw, pressure_raw
    gc.collect()

    return result, met_times


def load_date_data(date_str: str, mesh_data: Dict, temporal_subsample: int = 1) -> Dict:
    """
    Load all data for a single date with proper time alignment.

    Met forcing files only contain forecast period.
    CWL files contain nowcast + forecast. We skip nowcast hours to align.

    Args:
        date_str: Date string (e.g., '20251115')
        mesh_data: Mesh data dictionary
        temporal_subsample: Temporal subsampling factor

    Returns:
        Dictionary with elevation, times, and forcing data (time-aligned)
    """
    date_dir = f'{DATA_DIR}/{date_str}'
    cwl_file = f'{date_dir}/stofs_2d_glo.t00z.fields.cwl.nc'

    # First load met forcing to get the number of available timesteps
    forcing, met_times = load_met_forcing(
        date_dir,
        mesh_data['lon'], mesh_data['lat'],
        subsample_factor=4
    )

    # Load CWL data, skipping nowcast and limiting to met_times for alignment
    elevation, times = load_cwl_data(
        cwl_file, mesh_data['global_indices'],
        nowcast_offset=NOWCAST_HOURS,
        max_times=met_times,
        temporal_subsample=temporal_subsample
    )

    # Verify alignment
    if elevation.shape[0] != forcing['u10'].shape[0]:
        logger.warning(f"  Time mismatch! CWL: {elevation.shape[0]}, Met: {forcing['u10'].shape[0]}")
        # Use minimum common timesteps
        min_times = min(elevation.shape[0], forcing['u10'].shape[0])
        elevation = elevation[:min_times]
        times = times[:min_times]
        for key in forcing:
            forcing[key] = forcing[key][:min_times]
        logger.info(f"  Aligned to {min_times} common timesteps")

    logger.info(f"  Final aligned data: {elevation.shape[0]} timesteps")

    return {
        'date': date_str,
        'elevation': elevation,
        'times': times,
        'forcing': forcing,
    }


# ============================================================
# Enhanced Dataset
# ============================================================

class MultiDateCWLDataset(Dataset):
    """
    Dataset for multi-date CWL training with physics-informed features.
    """

    def __init__(
        self,
        mesh_data: Dict,
        dates_data: List[Dict],
        eta_scale: float = 2.0,
        max_sequence_length: int = 2,
    ):
        self.eta_scale = np.float32(eta_scale)
        self.max_seq_len = max_sequence_length

        # Store mesh data
        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)

        self.num_nodes = len(self.lon)

        # Process dates
        self.samples = []
        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for date_idx, date_data in enumerate(dates_data):
            elev = date_data['elevation'].astype(np.float32)
            forcing = date_data['forcing']

            # Find valid nodes (no NaN in any timestep)
            valid_mask = np.all(~np.isnan(elev), axis=0)

            if date_idx == 0:
                self.valid_mask = valid_mask
            else:
                self.valid_mask &= valid_mask

            self.elevations.append(elev)
            self.forcings.append(forcing)
            self.date_labels.append(date_data['date'])

            # Create samples (time indices)
            num_times = elev.shape[0]
            for t in range(num_times - max_sequence_length):
                self.samples.append((date_idx, t))

        # Apply valid mask
        valid_indices = np.where(self.valid_mask)[0]
        logger.info(f"Valid nodes: {len(valid_indices):,} / {self.num_nodes:,}")

        self.lon = self.lon[valid_indices]
        self.lat = self.lat[valid_indices]
        self.depth = self.depth[valid_indices]
        self.num_nodes = len(self.lon)

        # Rebuild edge index for valid nodes
        old_to_new = {old: new for new, old in enumerate(valid_indices)}
        new_edges = []
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if src in old_to_new and dst in old_to_new:
                new_edges.append([old_to_new[src], old_to_new[dst]])
        self.edge_index = torch.tensor(np.array(new_edges).T, dtype=torch.long)

        # Filter data arrays
        for i in range(len(self.elevations)):
            self.elevations[i] = self.elevations[i][:, valid_indices]
            self.forcings[i]['u10'] = self.forcings[i]['u10'][:, valid_indices]
            self.forcings[i]['v10'] = self.forcings[i]['v10'][:, valid_indices]
            self.forcings[i]['pressure'] = self.forcings[i]['pressure'][:, valid_indices]

        gc.collect()

        # Compute static features
        self._compute_static_features()
        self._compute_edge_features()

        logger.info(f"Dataset: {len(self.samples)} samples, {self.num_nodes} nodes, {len(dates_data)} dates")

    def _compute_static_features(self):
        """Compute normalized static features."""
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = np.float32(6371000.0)
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)

        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1

        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

        self.static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)

    def _compute_edge_features(self):
        """Compute edge features."""
        src, dst = self.edge_index[0].numpy(), self.edge_index[1].numpy()
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)
        char_length = np.median(dist) + 1e-8

        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_idx, time_idx = self.samples[idx]

        # Get elevation sequence
        elev_sequence = []
        for t in range(self.max_seq_len + 1):
            elev = self.elevations[date_idx][time_idx + t].astype(np.float32)
            elev_sequence.append(elev / self.eta_scale)

        eta_in = elev_sequence[0]
        eta_out = elev_sequence[1]

        # Compute water level feature
        water_level = self.depth + eta_in * self.eta_scale
        water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

        static_features = np.concatenate([
            self.static_base,
            water_level_norm[:, np.newaxis]
        ], axis=1)

        # Get forcing
        forcing = self.forcings[date_idx]
        u10 = forcing['u10'][time_idx].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][time_idx].astype(np.float32) / WIND_SCALE
        pressure = forcing['pressure'][time_idx].astype(np.float32)

        forcing_features = np.stack([u10, v10, pressure], axis=1)

        data = Data(
            x=torch.tensor(eta_in[:, np.newaxis], dtype=torch.float32),
            y=torch.tensor(eta_out[:, np.newaxis], dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            static_features=torch.tensor(static_features, dtype=torch.float32),
            forcing_features=torch.tensor(forcing_features, dtype=torch.float32),
        )

        # For multi-step training
        if self.max_seq_len >= 2:
            u10_next = forcing['u10'][time_idx + 1].astype(np.float32) / WIND_SCALE
            v10_next = forcing['v10'][time_idx + 1].astype(np.float32) / WIND_SCALE
            pressure_next = forcing['pressure'][time_idx + 1].astype(np.float32)

            data.forcing_next = torch.tensor(
                np.stack([u10_next, v10_next, pressure_next], axis=1),
                dtype=torch.float32
            )
            data.y_next = torch.tensor(
                elev_sequence[2][:, np.newaxis],
                dtype=torch.float32
            )

        return data


# ============================================================
# Training Functions
# ============================================================

def train_epoch_curriculum(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    num_steps: int = 1,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Train one epoch with curriculum-based multi-step loss."""
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        accumulated_loss = 0
        current_state = batch.x

        # Step 1
        pred = model(
            current_state,
            batch.static_features,
            batch.forcing_features,
            batch.edge_index,
            batch.edge_attr
        )

        loss1, components1 = criterion(pred, batch.y, batch.edge_index)
        accumulated_loss = loss1

        # Step 2 (if enabled)
        if num_steps >= 2 and hasattr(batch, 'y_next'):
            pred2 = model(
                pred.detach(),
                batch.static_features,
                batch.forcing_next,
                batch.edge_index,
                batch.edge_attr
            )

            loss2, _ = criterion(pred2, batch.y_next, batch.edge_index)
            accumulated_loss = accumulated_loss + 0.5 * loss2

        accumulated_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += accumulated_loss.item()
        for k in total_components:
            total_components[k] += components1.get(k, 0)

        if device.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        del batch, pred, accumulated_loss
        if num_steps >= 2:
            del pred2

    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in total_components.items()}

    return avg_loss, avg_components


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate model."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = len(loader)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(
                batch.x,
                batch.static_features,
                batch.forcing_features,
                batch.edge_index,
                batch.edge_attr
            )

            loss, components = criterion(pred, batch.y, batch.edge_index)
            total_loss += loss.item()
            for k in total_components:
                total_components[k] += components.get(k, 0)

            del batch, pred

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in total_components.items()}

    return avg_loss, avg_components


def rollout_prediction(
    model: nn.Module,
    dataset: MultiDateCWLDataset,
    date_idx: int,
    start_idx: int,
    num_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run multi-step autoregressive prediction."""
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    elev_start = dataset.elevations[date_idx][start_idx].astype(np.float32)
    current = torch.tensor(elev_start / dataset.eta_scale, dtype=torch.float32).to(device)

    predictions = [elev_start.copy()]
    ground_truth = [elev_start.copy()]

    forcing = dataset.forcings[date_idx]

    with torch.no_grad():
        for step in range(num_steps):
            t = start_idx + step

            current_elev = current.cpu().numpy() * dataset.eta_scale
            water_level = dataset.depth + current_elev
            water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

            static_features = np.concatenate([
                dataset.static_base,
                water_level_norm[:, np.newaxis]
            ], axis=1)
            static_features = torch.tensor(static_features, dtype=torch.float32).to(device)

            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pressure = forcing['pressure'][t].astype(np.float32)

            forcing_features = torch.tensor(
                np.stack([u10, v10, pressure], axis=1),
                dtype=torch.float32
            ).to(device)

            x = current.unsqueeze(1)
            next_state = model(
                x, static_features, forcing_features, edge_index, edge_attr
            ).squeeze()

            current = next_state
            predictions.append(current.cpu().numpy() * dataset.eta_scale)

            if t + 1 < len(dataset.elevations[date_idx]):
                gt = dataset.elevations[date_idx][t + 1].astype(np.float32)
                ground_truth.append(gt)

    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Visualization
# ============================================================

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    components: Dict[str, list],
    output_path: str,
):
    """Plot training curves with loss components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax = axes[0]
    ax.semilogy(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.semilogy(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, values in components.items():
        if len(values) > 0:
            ax.semilogy(epochs, values, label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_rollout_comparison(
    lon: np.ndarray,
    lat: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str,
):
    """Plot rollout comparison."""
    timesteps = [0, 6, 12, 24, 48]
    fig, axes = plt.subplots(len(timesteps), 3, figsize=(15, 4*len(timesteps)))

    vmax = 1.0
    s = 2

    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        pred = predictions[t]
        gt = ground_truth[t] if t < len(ground_truth) else pred
        diff = pred - gt
        rmse = np.sqrt(np.mean(diff**2))

        ax = axes[i, 0]
        cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Ground Truth (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        ax = axes[i, 1]
        cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Prediction (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        ax = axes[i, 2]
        cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'Error (RMSE={rmse:.3f}m)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Error (m)')

    plt.suptitle('Multi-Date CWL Model - Rollout', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# ============================================================
# Main Functions
# ============================================================

def preprocess_all_data():
    """
    Preprocess all raw data files to Mid-Atlantic subset.
    Run this once before training to speed up data loading.

    Usage: python train_cwl_gnn_multidate.py --preprocess
    """
    logger.info("=" * 60)
    logger.info("PREPROCESSING RAW DATA TO MID-ATLANTIC SUBSET")
    logger.info("=" * 60)
    logger.info(f"Source: {DATA_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}/data/processed/")
    logger.info(f"Domain: [{BBOX['lon_min']}, {BBOX['lon_max']}] x [{BBOX['lat_min']}, {BBOX['lat_max']}]")

    processed_dir = f'{OUTPUT_DIR}/data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Check available dates
    available_dates = []
    for date in TRAINING_DATES:
        cwl_file = f'{DATA_DIR}/{date}/stofs_2d_glo.t00z.fields.cwl.nc'
        wind_file = f'{DATA_DIR}/{date}/stofs_2d_glo.t00z.uvgrd10m.nc'
        pres_file = f'{DATA_DIR}/{date}/stofs_2d_glo.t00z.pressfc.nc'

        if os.path.exists(cwl_file) and os.path.exists(wind_file) and os.path.exists(pres_file):
            available_dates.append(date)
        else:
            logger.warning(f"  Missing data for {date}")

    logger.info(f"\nAvailable dates: {len(available_dates)}")

    # Process each date
    for i, date in enumerate(available_dates):
        logger.info(f"\n[{i+1}/{len(available_dates)}] Processing {date}...")
        try:
            preprocess_and_save_subset(
                date_str=date,
                source_dir=DATA_DIR,
                output_dir=processed_dir,
                bbox=BBOX,
                max_nodes=MAX_NODES,
                nowcast_offset=NOWCAST_HOURS,
            )
        except Exception as e:
            logger.error(f"  Failed to process {date}: {e}")
            continue

        gc.collect()

    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    # List processed files
    processed_files = sorted(Path(processed_dir).glob('processed_*.npz'))
    total_size = sum(f.stat().st_size for f in processed_files) / 1e6
    logger.info(f"Processed files: {len(processed_files)}")
    logger.info(f"Total size: {total_size:.1f} MB")


def train_from_preprocessed():
    """
    Train model using preprocessed data files.
    Much faster than loading from raw files.

    Usage: python train_cwl_gnn_multidate.py --train
    """
    logger.info("=" * 60)
    logger.info("MULTI-DATE CWL GNN TRAINING (PREPROCESSED DATA)")
    logger.info("=" * 60)
    logger.info(f"Domain: [{BBOX['lon_min']}, {BBOX['lon_max']}] x [{BBOX['lat_min']}, {BBOX['lat_max']}]")
    logger.info(f"Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")

    processed_dir = f'{OUTPUT_DIR}/data/processed'

    # Find preprocessed files
    processed_files = sorted(Path(processed_dir).glob('processed_*.npz'))

    if len(processed_files) == 0:
        logger.error(f"No preprocessed files found in {processed_dir}")
        logger.error("Run with --preprocess first!")
        return

    logger.info(f"\nFound {len(processed_files)} preprocessed files")

    # Load first file for mesh data
    first_data = load_preprocessed_data(str(processed_files[0]))
    mesh_data = first_data['mesh']

    # Split into train/val
    if len(processed_files) < 3:
        logger.error("Need at least 3 preprocessed files for training!")
        return

    train_files = processed_files[:-VAL_DATES]
    val_files = processed_files[-VAL_DATES:]

    logger.info(f"Train files: {len(train_files)}")
    logger.info(f"Val files: {len(val_files)}")

    # Load training data
    logger.info("\nLoading training data...")
    train_data = []
    for f in train_files:
        data = load_preprocessed_data(str(f))
        train_data.append({
            'date': data['date'],
            'elevation': data['elevation'],
            'times': data['times'],
            'forcing': data['forcing'],
        })

    # Load validation data
    logger.info("\nLoading validation data...")
    val_data = []
    for f in val_files:
        data = load_preprocessed_data(str(f))
        val_data.append({
            'date': data['date'],
            'elevation': data['elevation'],
            'times': data['times'],
            'forcing': data['forcing'],
        })

    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = MultiDateCWLDataset(
        mesh_data, train_data,
        eta_scale=ETA_SCALE,
        max_sequence_length=MAX_ROLLOUT_STEPS,
    )

    val_dataset = MultiDateCWLDataset(
        mesh_data, val_data,
        eta_scale=ETA_SCALE,
        max_sequence_length=MAX_ROLLOUT_STEPS,
    )

    del train_data, val_data
    gc.collect()

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = PhysicsInformedCWLModel(
        state_dim=STATE_DIM,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        edge_feature_dim=EDGE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_checkpointing=USE_GRADIENT_CHECKPOINTING,
    ).to(device)

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsInformedLoss(
        mass_weight=MASS_CONSERVATION_WEIGHT,
        smoothness_weight=SMOOTHNESS_WEIGHT,
    )

    curriculum = CurriculumScheduler(
        max_steps=MAX_ROLLOUT_STEPS,
        warmup_epochs=CURRICULUM_WARMUP_EPOCHS,
    ) if CURRICULUM_ENABLED else None

    # Training loop
    logger.info("\nStarting training...")
    os.makedirs(f'{OUTPUT_DIR}/outputs/checkpoints', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/outputs/figures', exist_ok=True)

    train_losses = []
    val_losses = []
    loss_components = {'mse': [], 'mass': [], 'smooth': []}
    best_val_loss = float('inf')

    train_dates = [d.stem.replace('processed_', '') for d in train_files]
    val_dates = [d.stem.replace('processed_', '') for d in val_files]

    for epoch in range(1, EPOCHS + 1):
        num_steps = curriculum.get_num_steps(epoch) if curriculum else 1

        train_loss, train_comp = train_epoch_curriculum(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP
        )

        val_loss, val_comp = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for k in loss_components:
            loss_components[k].append(train_comp.get(k, 0))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'static_features': STATIC_NODE_FEATURES,
                    'forcing_features': FORCING_FEATURES,
                    'eta_scale': ETA_SCALE,
                    'bbox': BBOX,
                    'training_dates': train_dates,
                    'val_dates': val_dates,
                },
            }, f'{OUTPUT_DIR}/outputs/checkpoints/best_multidate_model.pt')

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:3d} | steps={num_steps} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"mse={train_comp['mse']:.6f} | mass={train_comp['mass']:.6f} | "
                f"lr={lr:.2e} | best={best_val_loss:.6f}"
            )

    # Save final model
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'static_features': STATIC_NODE_FEATURES,
            'forcing_features': FORCING_FEATURES,
            'eta_scale': ETA_SCALE,
            'bbox': BBOX,
            'training_dates': train_dates,
            'val_dates': val_dates,
        },
    }, f'{OUTPUT_DIR}/outputs/checkpoints/final_multidate_model.pt')

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, loss_components,
        f'{OUTPUT_DIR}/outputs/figures/multidate_training.png'
    )

    # Load best and do rollout
    logger.info("\nLoading best model for rollout...")
    checkpoint = torch.load(f'{OUTPUT_DIR}/outputs/checkpoints/best_multidate_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Rollout on last validation date
    date_idx = len(val_dataset.elevations) - 1
    start_idx = 100
    predictions, ground_truth = rollout_prediction(
        model, val_dataset, date_idx, start_idx, 48, device
    )

    plot_rollout_comparison(
        val_dataset.lon, val_dataset.lat,
        predictions, ground_truth,
        f'{OUTPUT_DIR}/outputs/figures/multidate_rollout.png'
    )

    # Final metrics
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")
    logger.info(f"Training dates: {len(train_dates)}")
    logger.info(f"Validation dates: {len(val_dates)}")

    for t in [1, 6, 12, 24, 48]:
        if t < len(predictions) and t < len(ground_truth):
            rmse = np.sqrt(np.mean((predictions[t] - ground_truth[t])**2))
            logger.info(f"Rollout t+{t}h RMSE: {rmse:.4f} m")

    logger.info("\nDone!")


def main():
    """
    Main entry point with command-line argument support.

    Usage:
        python train_cwl_gnn_multidate.py --preprocess  # Preprocess raw data first
        python train_cwl_gnn_multidate.py --train       # Train using preprocessed data
        python train_cwl_gnn_multidate.py               # Default: train with preprocessing check
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-Date CWL GNN Training for STOFS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Preprocess raw data (run once, takes ~30min per date)
  python train_cwl_gnn_multidate.py --preprocess

  # Step 2: Train model (much faster with preprocessed data)
  python train_cwl_gnn_multidate.py --train

  # Or run both steps automatically
  python train_cwl_gnn_multidate.py
        """
    )
    parser.add_argument('--preprocess', action='store_true',
                       help='Preprocess raw data to Mid-Atlantic subset')
    parser.add_argument('--train', action='store_true',
                       help='Train model using preprocessed data')

    args = parser.parse_args()

    if args.preprocess:
        preprocess_all_data()
    elif args.train:
        train_from_preprocessed()
    else:
        # Default behavior: check for preprocessed data, preprocess if needed, then train
        processed_dir = f'{OUTPUT_DIR}/data/processed'
        processed_files = list(Path(processed_dir).glob('processed_*.npz')) if os.path.exists(processed_dir) else []

        if len(processed_files) < 3:
            logger.info("Preprocessed data not found or incomplete. Running preprocessing...")
            preprocess_all_data()

        logger.info("\nStarting training...")
        train_from_preprocessed()


if __name__ == '__main__':
    main()
