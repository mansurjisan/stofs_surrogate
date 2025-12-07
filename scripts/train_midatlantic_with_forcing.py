#!/usr/bin/env python3
"""
Train CWL GNN model on Mid-Atlantic region with meteorological forcing.

This script incorporates:
- Wind (u10, v10) and pressure as time-varying node features
- Multi-cycle training support
- Interpolation of met forcing from regular grid to ADCIRC mesh

Domain: [-76, -73] × [38, 41]
Covers: New York, Atlantic City, Delaware Bay, Philadelphia area

Memory-optimized version for systems with limited RAM.
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory optimization: Use float16 where possible
USE_FLOAT16_STORAGE = True  # Store data in float16, compute in float32

# ============================================================
# Configuration
# ============================================================

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Data paths
DATA_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/raw'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'

# Available cycles for training
# NOTE: Reduced to 2 cycles to lower memory usage - add more if RAM allows
CYCLES = [
    ('stofs_2d_glo.20251122', 't00z', 'met_forcing_00z'),
    ('stofs_2d_glo.20251122', 't12z', 'met_forcing_12z'),
    # ('stofs_2d_glo.20251123', 't00z', 'met_forcing_00z'),  # Uncomment if RAM allows
    # ('stofs_2d_glo.20251123', 't12z', 'met_forcing_12z'),  # Uncomment if RAM allows
]

# Model parameters - now with additional forcing features
# NOTE: Tuned for RTX 3050 Laptop (4GB VRAM)
HIDDEN_DIM = 96           # Increased from 64 for more capacity
NUM_LAYERS = 6            # Increased from 4 for deeper network
STATE_DIM = 1             # CWL
STATIC_NODE_FEATURES = 3  # x_norm, y_norm, depth_norm
FORCING_FEATURES = 3      # u10, v10, pressure (normalized)
EDGE_FEATURES = 3         # dx, dy, dist

# Training parameters
EPOCHS = 300              # More epochs for convergence
BATCH_SIZE = 2            # Reduced to fit larger model in VRAM
LEARNING_RATE = 3e-4      # Slightly lower for stability
WEIGHT_DECAY = 1e-5
ETA_SCALE = 2.0

# Forcing normalization (approximate values)
WIND_SCALE = 15.0      # m/s - typical max wind
PRESSURE_MEAN = 101325.0  # Pa (1 atm)
PRESSURE_SCALE = 3000.0   # Pa - typical variation


# ============================================================
# Met Forcing Processing
# ============================================================

def load_met_forcing_for_cycle(date_dir, met_dir, num_cwl_times, subsample_factor=2):
    """
    Load and concatenate met forcing files for a cycle.

    Memory optimization: Subsample the spatial grid before returning.

    Returns arrays of shape (time, lat, lon) for wind u, v, and pressure.
    Also returns the grid coordinates.
    """
    base_path = f'{DATA_DIR}/{date_dir}/{met_dir}'

    # Load coordinate info from any file
    sample_file = f'{base_path}/stofs_2d_glo_ncst.222.nc'
    if not os.path.exists(sample_file):
        sample_file = f'{base_path}/stofs_2d_glo_fcst1.222.nc'

    nc = NCDataset(sample_file)
    grid_lon = np.array(nc.variables['grid_xt'][:], dtype=np.float32)  # 0 to 360
    grid_lat = np.array(nc.variables['grid_yt'][:], dtype=np.float32)
    nc.close()

    # Convert longitude from 0-360 to -180-180 for our domain
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)

    # Load wind files (222) - load one at a time to reduce peak memory
    all_u = []
    all_v = []
    all_p = []

    # Order: ncst first, then fcst1, fcst2
    file_order = ['ncst', 'fcst1', 'fcst2']

    for file_type in file_order:
        wind_file = f'{base_path}/stofs_2d_glo_{file_type}.222.nc'
        pres_file = f'{base_path}/stofs_2d_glo_{file_type}.221.nc'

        if os.path.exists(wind_file):
            nc = NCDataset(wind_file)
            # Load as float32 directly
            u = np.array(nc.variables['ugrd10m'][:], dtype=np.float32)
            v = np.array(nc.variables['vgrd10m'][:], dtype=np.float32)
            nc.close()
            all_u.append(u)
            all_v.append(v)
            del u, v  # Free memory immediately
            gc.collect()

        if os.path.exists(pres_file):
            nc = NCDataset(pres_file)
            p = np.array(nc.variables['pressfc'][:], dtype=np.float32)
            nc.close()
            all_p.append(p)
            del p
            gc.collect()

    # Concatenate
    u_all = np.concatenate(all_u, axis=0)
    del all_u
    v_all = np.concatenate(all_v, axis=0)
    del all_v
    p_all = np.concatenate(all_p, axis=0)
    del all_p
    gc.collect()

    logger.info(f"  Loaded met forcing: {u_all.shape[0]} time steps")

    # Subsample spatial grid to reduce memory (met forcing is smooth, can afford to subsample)
    if subsample_factor > 1:
        u_all = u_all[:, ::subsample_factor, ::subsample_factor]
        v_all = v_all[:, ::subsample_factor, ::subsample_factor]
        p_all = p_all[:, ::subsample_factor, ::subsample_factor]
        grid_lon = grid_lon[::subsample_factor]
        grid_lat = grid_lat[::subsample_factor]
        logger.info(f"  Subsampled met grid by {subsample_factor}x: {u_all.shape}")

    # Interpolate to match CWL time steps if needed
    met_times = u_all.shape[0]
    if met_times != num_cwl_times:
        logger.info(f"  Interpolating met from {met_times} to {num_cwl_times} time steps")

        # Simple linear interpolation along time axis
        old_times = np.linspace(0, 1, met_times)
        new_times = np.linspace(0, 1, num_cwl_times)

        u_interp = np.zeros((num_cwl_times, u_all.shape[1], u_all.shape[2]), dtype=np.float32)
        v_interp = np.zeros_like(u_interp)
        p_interp = np.zeros_like(u_interp)

        for i in range(u_all.shape[1]):
            for j in range(u_all.shape[2]):
                u_interp[:, i, j] = np.interp(new_times, old_times, u_all[:, i, j])
                v_interp[:, i, j] = np.interp(new_times, old_times, v_all[:, i, j])
                p_interp[:, i, j] = np.interp(new_times, old_times, p_all[:, i, j])

        del u_all, v_all, p_all
        u_all, v_all, p_all = u_interp, v_interp, p_interp
        gc.collect()

    # Use float16 for storage if enabled
    # Note: Normalize pressure BEFORE storing to avoid float16 overflow
    # Pressure ~101325 Pa exceeds float16 max (~65504)
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32

    # Normalize pressure to anomaly (centered around 0) before storing
    # This keeps values small enough for float16
    p_normalized = (p_all - PRESSURE_MEAN) / PRESSURE_SCALE  # Now in range ~[-10, 10]

    return {
        'u10': u_all.astype(dtype),
        'v10': v_all.astype(dtype),
        'pressure': p_normalized.astype(dtype),  # Already normalized!
        'pressure_prenormalized': True,  # Flag to indicate normalization done
        'grid_lon': grid_lon,
        'grid_lat': grid_lat,
    }


def interpolate_forcing_to_nodes(forcing_data, node_lon, node_lat):
    """
    Interpolate met forcing from regular grid to ADCIRC mesh nodes.

    Memory optimization: Process in batches and use float16 storage.

    Args:
        forcing_data: dict with u10, v10, pressure (time, lat, lon) and grid coords
        node_lon, node_lat: ADCIRC node coordinates

    Returns:
        Dict with interpolated u10, v10, pressure of shape (time, num_nodes)
    """
    grid_lon = forcing_data['grid_lon']
    grid_lat = forcing_data['grid_lat']

    # Sort grid coordinates for interpolation
    lon_sort_idx = np.argsort(grid_lon)
    lat_sort_idx = np.argsort(grid_lat)

    grid_lon_sorted = grid_lon[lon_sort_idx]
    grid_lat_sorted = grid_lat[lat_sort_idx]

    num_times = forcing_data['u10'].shape[0]
    num_nodes = len(node_lon)

    # Use float16 for storage if enabled
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32

    # Interpolate each variable
    result = {
        'u10': np.zeros((num_times, num_nodes), dtype=dtype),
        'v10': np.zeros((num_times, num_nodes), dtype=dtype),
        'pressure': np.zeros((num_times, num_nodes), dtype=dtype),
    }

    logger.info(f"  Interpolating forcing to {num_nodes} mesh nodes...")

    # Process in time batches to reduce memory
    batch_size = 20
    for t_start in range(0, num_times, batch_size):
        t_end = min(t_start + batch_size, num_times)
        if t_start % 50 == 0:
            logger.info(f"    Time steps {t_start}-{t_end}/{num_times}")

        for t in range(t_start, t_end):
            for var in ['u10', 'v10', 'pressure']:
                # Convert from float16 to float32 for interpolation
                data = forcing_data[var][t].astype(np.float32)
                # Reorder to sorted lat/lon
                data_sorted = data[lat_sort_idx][:, lon_sort_idx]

                # Create interpolator
                interp = RegularGridInterpolator(
                    (grid_lat_sorted, grid_lon_sorted),
                    data_sorted,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )

                # Interpolate (note: interpolator expects (lat, lon) order)
                values = interp(np.column_stack([node_lat, node_lon]))

                # Fill NaN with nearest valid value or mean
                if np.any(np.isnan(values)):
                    nan_mask = np.isnan(values)
                    values[nan_mask] = np.nanmean(values)

                result[var][t] = values.astype(dtype)

        # Periodic garbage collection
        gc.collect()

    return result


# ============================================================
# Model Architecture (with forcing)
# ============================================================

class MeshGraphNetBlock(nn.Module):
    """Message passing block with layer normalization."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index

        # Edge update
        edge_input = torch.cat([edge_attr, h[row], h[col]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)

        # Aggregate messages
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_attr_new)

        # Node update with residual
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr_new


class MidAtlanticGNNWithForcing(nn.Module):
    """
    GNN for Mid-Atlantic CWL prediction with meteorological forcing.

    Node features:
    - State: CWL (1 dim)
    - Static: x_norm, y_norm, depth_norm (3 dims)
    - Forcing: u10_norm, v10_norm, pressure_norm (3 dims)

    Total node input: 1 + 3 + 3 = 7 dims
    """

    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 3,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 8,
    ):
        super().__init__()

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
            MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MidAtlanticGNNWithForcing: {total_params:,} parameters")
        logger.info(f"  Node input dim: {node_input_dim} (state={state_dim}, static={static_feature_dim}, forcing={forcing_feature_dim})")

    def forward(self, x, static_features, forcing_features, edge_index, edge_attr):
        """
        Args:
            x: Current state (CWL), shape (N, 1)
            static_features: Static node features (x, y, depth), shape (N, 3)
            forcing_features: Time-varying forcing (u10, v10, p), shape (N, 3)
            edge_index: Graph connectivity
            edge_attr: Edge features
        """
        # Concatenate all node features
        h = self.node_encoder(torch.cat([x, static_features, forcing_features], dim=-1))
        e = self.edge_encoder(edge_attr)

        # Process
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        # Decode
        return self.decoder(h)


# ============================================================
# Data Processing
# ============================================================

def extract_midatlantic_mesh(nc_file, bbox, max_nodes=25000):  # Higher resolution
    """Extract Mid-Atlantic mesh (one-time operation)."""
    logger.info(f"Opening {nc_file}")
    nc = NCDataset(nc_file, 'r')

    # Load coordinates as float32 to save memory
    x = np.array(nc.variables['x'][:], dtype=np.float32)
    y = np.array(nc.variables['y'][:], dtype=np.float32)
    depth = np.array(nc.variables['depth'][:], dtype=np.float32)

    logger.info(f"Global mesh: {len(x):,} nodes")

    # Filter to bounding box
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    subset_indices = np.where(mask)[0]
    logger.info(f"Nodes in Mid-Atlantic bbox: {len(subset_indices):,}")

    # Subsample if too many nodes
    if len(subset_indices) > max_nodes:
        logger.info(f"Subsampling from {len(subset_indices):,} to {max_nodes:,} nodes")
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(subset_indices, size=max_nodes, replace=False)
        subset_indices = np.sort(subset_indices)

    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_sub = depth[subset_indices]

    # Build connectivity
    logger.info("Building mesh connectivity...")
    points = np.column_stack([lon, lat])
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                e = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(e)

    edges = np.array(list(edges))
    edge_index = np.vstack([edges, edges[:, ::-1]]).T

    logger.info(f"Created {len(edges):,} edges ({edge_index.shape[1]:,} directed)")

    nc.close()

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth_sub,
        'edge_index': edge_index,
        'global_indices': subset_indices,
    }


def extract_cycle_data(nc_file, global_indices, temporal_subsample=1):
    """
    Extract CWL time series for a cycle.

    Memory optimization: Support temporal subsampling and use float16 storage.
    """
    nc = NCDataset(nc_file, 'r')

    zeta = nc.variables['zeta']
    full_times = zeta.shape[0]

    # Subsample temporal dimension
    time_indices = list(range(0, full_times, temporal_subsample))
    num_times = len(time_indices)

    if temporal_subsample > 1:
        logger.info(f"    Subsampling from {full_times} to {num_times} time steps")

    # Use float16 for storage
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
    elevation = np.zeros((num_times, len(global_indices)), dtype=dtype)

    for i, t in enumerate(time_indices):
        if i % 50 == 0:
            logger.info(f"    Timestep {i}/{num_times} (original t={t})")
        elevation[i, :] = zeta[t, global_indices]

    # Handle dry nodes
    elevation = np.where(elevation < -9000, np.nan, elevation)

    times = nc.variables['time'][time_indices]
    nc.close()

    gc.collect()
    return elevation, times


class MultiCycleDataset(Dataset):
    """
    Dataset combining multiple cycles with met forcing.

    Memory optimization: Uses float16 storage and converts to float32 on-the-fly.
    """

    def __init__(self, mesh_data, cycles_data, eta_scale=2.0):
        """
        Args:
            mesh_data: dict with lon, lat, depth, edge_index
            cycles_data: list of dicts, each with 'elevation', 'forcing'
        """
        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.eta_scale = np.float32(eta_scale)

        self.num_nodes = len(self.lon)

        # Combine all cycles
        self.samples = []  # List of (cycle_idx, time_idx) pairs
        self.elevations = []
        self.forcings = []

        for cycle_idx, cycle in enumerate(cycles_data):
            # Convert to float32 for NaN check, then back to storage dtype
            elev = cycle['elevation'].astype(np.float32)
            forcing = cycle['forcing']

            # Filter nodes with valid data
            valid_mask = np.all(~np.isnan(elev), axis=0)
            if cycle_idx == 0:
                self.valid_mask = valid_mask
            else:
                self.valid_mask &= valid_mask

            # Store as float16 for memory efficiency
            dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
            self.elevations.append(elev.astype(dtype))
            self.forcings.append(forcing)

            # Add samples (pairs of consecutive time steps)
            num_times = elev.shape[0]
            for t in range(num_times - 1):
                self.samples.append((cycle_idx, t))

        # Apply valid mask to coordinates
        valid_indices = np.where(self.valid_mask)[0]
        logger.info(f"Valid nodes (all cycles): {len(valid_indices):,} / {self.num_nodes:,}")

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

        # Filter elevations and forcings
        for i in range(len(self.elevations)):
            self.elevations[i] = self.elevations[i][:, valid_indices]
            self.forcings[i]['u10'] = self.forcings[i]['u10'][:, valid_indices]
            self.forcings[i]['v10'] = self.forcings[i]['v10'][:, valid_indices]
            self.forcings[i]['pressure'] = self.forcings[i]['pressure'][:, valid_indices]

        # Force garbage collection after filtering
        gc.collect()

        # Compute static node features
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = np.float32(6371000.0)
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)

        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1

        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

        self.static_features = torch.tensor(
            np.stack([x_norm, y_norm, depth_norm], axis=1),
            dtype=torch.float32
        )

        # Edge features
        src, dst = self.edge_index[0].numpy(), self.edge_index[1].numpy()
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)
        char_length = np.median(dist) + 1e-8

        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        )

        # Log memory usage
        total_elev_bytes = sum(e.nbytes for e in self.elevations)
        total_forcing_bytes = sum(
            f['u10'].nbytes + f['v10'].nbytes + f['pressure'].nbytes
            for f in self.forcings
        )
        logger.info(f"Dataset: {len(self.samples)} samples from {len(cycles_data)} cycles")
        logger.info(f"Nodes: {self.num_nodes}, Edges: {self.edge_index.shape[1]}")
        logger.info(f"Memory: elevations={total_elev_bytes/1e6:.1f}MB, forcing={total_forcing_bytes/1e6:.1f}MB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cycle_idx, time_idx = self.samples[idx]

        # Get elevation - convert float16 to float32 for computation
        eta_in = self.elevations[cycle_idx][time_idx].astype(np.float32) / self.eta_scale
        eta_out = self.elevations[cycle_idx][time_idx + 1].astype(np.float32) / self.eta_scale

        # Get forcing for current time step - convert float16 to float32
        forcing = self.forcings[cycle_idx]
        u10 = forcing['u10'][time_idx].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][time_idx].astype(np.float32) / WIND_SCALE
        # Pressure is already pre-normalized to avoid float16 overflow
        pressure = forcing['pressure'][time_idx].astype(np.float32)

        forcing_features = torch.tensor(
            np.stack([u10, v10, pressure], axis=1),
            dtype=torch.float32
        )

        return Data(
            x=torch.tensor(eta_in[:, np.newaxis], dtype=torch.float32),
            y=torch.tensor(eta_out[:, np.newaxis], dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            static_features=self.static_features,
            forcing_features=forcing_features,
        )


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(
            batch.x,
            batch.static_features,
            batch.forcing_features,
            batch.edge_index,
            batch.edge_attr
        )
        loss = criterion(pred, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        # Clear GPU cache periodically to prevent memory buildup
        if device.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        # Delete batch to free memory
        del batch, pred, loss

    # Final cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
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
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            del batch, pred, loss

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return total_loss / num_batches


def rollout_prediction(model, dataset, cycle_idx, start_idx, num_steps, device):
    """Run multi-step autoregressive prediction."""
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)
    static_features = dataset.static_features.to(device)

    # Convert float16 to float32 for computation
    elev_start = np.asarray(dataset.elevations[cycle_idx][start_idx], dtype=np.float32)
    current = torch.tensor(
        elev_start / dataset.eta_scale,
        dtype=torch.float32
    ).to(device)

    predictions = [current.cpu().numpy() * dataset.eta_scale]
    ground_truth = [elev_start]

    forcing = dataset.forcings[cycle_idx]

    with torch.no_grad():
        for step in range(num_steps):
            t = start_idx + step

            # Get forcing for this time step
            u10 = np.asarray(forcing['u10'][t], dtype=np.float32) / WIND_SCALE
            v10 = np.asarray(forcing['v10'][t], dtype=np.float32) / WIND_SCALE
            # Pressure is already pre-normalized
            pressure = np.asarray(forcing['pressure'][t], dtype=np.float32)

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
            if t + 1 < len(dataset.elevations[cycle_idx]):
                gt = np.asarray(dataset.elevations[cycle_idx][t + 1], dtype=np.float32)
                ground_truth.append(gt)

    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Visualization
# ============================================================

def plot_training_curves(train_losses, val_losses, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, 'b-', label='Train Loss')
    ax.semilogy(epochs, val_losses, 'r-', label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Mid-Atlantic CWL Model with Forcing - Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_domain_with_forcing(lon, lat, depth, elevation, forcing, output_path):
    """Plot domain with sample forcing."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    t = len(elevation) // 2

    # Bathymetry
    ax = axes[0, 0]
    cf = ax.scatter(lon, lat, c=depth, s=1, cmap='Blues_r', vmin=0, vmax=100)
    ax.set_title(f'Bathymetry\n{len(lon):,} nodes')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Depth (m)')

    # CWL
    ax = axes[0, 1]
    cf = ax.scatter(lon, lat, c=elevation[t], s=1, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'CWL at t={t}')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    # Wind speed
    ax = axes[1, 0]
    wind_speed = np.sqrt(forcing['u10'][t]**2 + forcing['v10'][t]**2)
    cf = ax.scatter(lon, lat, c=wind_speed, s=1, cmap='viridis', vmin=0, vmax=15)
    ax.set_title(f'Wind Speed at t={t}')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Wind (m/s)')

    # Pressure
    ax = axes[1, 1]
    pressure_hpa = forcing['pressure'][t] / 100
    cf = ax.scatter(lon, lat, c=pressure_hpa, s=1, cmap='coolwarm', vmin=1000, vmax=1025)
    ax.set_title(f'Surface Pressure at t={t}')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Pressure (hPa)')

    plt.suptitle('Mid-Atlantic Domain with Met Forcing', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_rollout(lon, lat, predictions, ground_truth, output_path):
    """Plot rollout comparison."""
    timesteps = [0, 6, 12, 24, 48]
    num_times = len(timesteps)

    fig, axes = plt.subplots(num_times, 3, figsize=(15, 4*num_times))

    vmax = 1.0
    s = 2

    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        pred = predictions[t]
        gt = ground_truth[t] if t < len(ground_truth) else predictions[t]
        diff = pred - gt

        # Ground truth
        ax = axes[i, 0]
        cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Ground Truth (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Prediction
        ax = axes[i, 1]
        cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Prediction (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Error
        ax = axes[i, 2]
        rmse = np.sqrt(np.mean(diff**2))
        cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'Error (RMSE={rmse:.3f}m)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Error (m)')

    plt.suptitle('Mid-Atlantic CWL Model with Forcing - Rollout', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    logger.info("="*60)
    logger.info("Mid-Atlantic CWL GNN Training WITH MET FORCING")
    logger.info("(Memory-optimized version)")
    logger.info("="*60)
    logger.info(f"Domain: [{BBOX['lon_min']}, {BBOX['lon_max']}] x [{BBOX['lat_min']}, {BBOX['lat_max']}]")
    logger.info(f"Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")
    logger.info(f"Forcing: u10, v10, pressure (normalized)")
    logger.info(f"Float16 storage: {USE_FLOAT16_STORAGE}")

    # Check for existing processed data
    # Use v5 for higher resolution (25000 nodes)
    mesh_path = f'{OUTPUT_DIR}/data/processed/midatlantic_mesh_v5.npz'

    # First, extract mesh (once)
    if os.path.exists(mesh_path):
        logger.info("\nLoading existing mesh...")
        mesh_data_np = np.load(mesh_path)
        mesh_data = {
            'lon': mesh_data_np['lon'],
            'lat': mesh_data_np['lat'],
            'depth': mesh_data_np['depth'],
            'edge_index': mesh_data_np['edge_index'],
            'global_indices': mesh_data_np['global_indices'],
        }
        mesh_data_np.close()  # Close the file handle
    else:
        logger.info("\nExtracting Mid-Atlantic mesh...")
        # Use first cycle file for mesh
        first_cwl = f'{DATA_DIR}/{CYCLES[0][0]}/stofs_2d_glo.{CYCLES[0][1]}.fields.cwl.nc'
        mesh_data = extract_midatlantic_mesh(first_cwl, BBOX)

        os.makedirs(f'{OUTPUT_DIR}/data/processed', exist_ok=True)
        np.savez(mesh_path, **mesh_data)
        logger.info(f"Saved mesh to {mesh_path}")

    gc.collect()

    # Load data for each cycle
    logger.info("\nLoading cycle data with met forcing...")
    cycles_data = []

    for date_dir, cycle, met_dir in CYCLES:
        logger.info(f"\nProcessing {date_dir} {cycle}...")

        cwl_file = f'{DATA_DIR}/{date_dir}/stofs_2d_glo.{cycle}.fields.cwl.nc'

        # Extract CWL with temporal subsampling to reduce memory
        logger.info("  Extracting CWL...")
        elevation, times = extract_cycle_data(
            cwl_file,
            mesh_data['global_indices'],
            temporal_subsample=1  # Set to 2 to halve time steps if needed
        )
        num_cwl_times = elevation.shape[0]

        # Load met forcing with spatial subsampling
        logger.info("  Loading met forcing...")
        forcing_raw = load_met_forcing_for_cycle(
            date_dir, met_dir, num_cwl_times,
            subsample_factor=2  # Subsample met grid to save memory
        )

        # Interpolate to mesh nodes
        logger.info("  Interpolating to mesh nodes...")
        forcing = interpolate_forcing_to_nodes(
            forcing_raw,
            mesh_data['lon'],
            mesh_data['lat']
        )

        # Delete raw forcing to free memory
        del forcing_raw
        gc.collect()

        cycles_data.append({
            'elevation': elevation,
            'times': times,
            'forcing': forcing,
        })

        logger.info(f"  Done: {num_cwl_times} time steps")

        # Force garbage collection after each cycle
        gc.collect()

    # Create dataset
    logger.info("\nCreating multi-cycle dataset...")
    dataset = MultiCycleDataset(mesh_data, cycles_data, eta_scale=ETA_SCALE)

    # Free cycles_data - it's been copied into the dataset
    del cycles_data
    gc.collect()

    # Plot domain
    os.makedirs(f'{OUTPUT_DIR}/outputs/figures', exist_ok=True)
    plot_domain_with_forcing(
        dataset.lon, dataset.lat,
        dataset.depth, dataset.elevations[0],
        dataset.forcings[0],
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_domain_forcing.png'
    )

    # Train/val split
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train: {train_size}, Val: {val_size}")

    # Data loaders - pin_memory=True for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model = MidAtlanticGNNWithForcing(
        state_dim=STATE_DIM,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        edge_feature_dim=EDGE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Training loop
    logger.info("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    os.makedirs(f'{OUTPUT_DIR}/outputs/checkpoints', exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'eta_scale': ETA_SCALE,
                'bbox': BBOX,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'forcing_features': FORCING_FEATURES,
            }, f'{OUTPUT_DIR}/outputs/checkpoints/best_midatlantic_forcing_model.pt')

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.2e}, best={best_val_loss:.6f}")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_forcing_training.png'
    )

    # Load best model and do rollout
    logger.info("\nLoading best model for rollout...")
    checkpoint = torch.load(f'{OUTPUT_DIR}/outputs/checkpoints/best_midatlantic_forcing_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Rollout on last cycle (validation)
    cycle_idx = len(dataset.elevations) - 1
    start_idx = 100
    predictions, ground_truth = rollout_prediction(
        model, dataset, cycle_idx, start_idx, 48, device
    )

    plot_rollout(
        dataset.lon, dataset.lat,
        predictions, ground_truth,
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_forcing_rollout.png'
    )

    # Print final metrics
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")
    logger.info(f"Cycles used: {len(dataset.elevations)}")
    logger.info(f"Total samples: {len(dataset)}")

    for t in [1, 6, 12, 24, 48]:
        if t < len(predictions) and t < len(ground_truth):
            rmse = np.sqrt(np.mean((predictions[t] - ground_truth[t])**2))
            logger.info(f"Rollout t+{t}h RMSE: {rmse:.4f} m")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
