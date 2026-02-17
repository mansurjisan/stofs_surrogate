#!/usr/bin/env python3
"""
OPTIMIZED Multi-Date CWL GNN Training Script

Key optimizations over original:
1. MAX_NODES = 15,000 (vs 50,000) - 5-7x faster
2. Reduced MASS_CONSERVATION_WEIGHT = 0.01 - stable training
3. Tighter GRAD_CLIP = 0.5 - prevents gradient explosions
4. Lower LEARNING_RATE = 1e-4 - smoother convergence
5. Vectorized interpolation - faster preprocessing
6. Better progress logging

Expected training time:
- 3 days: ~3-4 hours
- 15 days: ~27-32 hours

Usage:
    python train_cwl_gnn_optimized.py --preprocess   # Preprocess data
    python train_cwl_gnn_optimized.py --train        # Train model
    python train_cwl_gnn_optimized.py                # Both steps
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.checkpoint import checkpoint
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, Tuple, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# OPTIMIZED CONFIGURATION
# ============================================================

# Paths
DATA_DIR = '/mnt/e/Drive2/Good/STOFS_TRAINING_DATA'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'

# Domain
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Training dates - modify as needed
TRAINING_DATES = ['20251128', '20251129', '20251130']  # 3 days for testing
# TRAINING_DATES = ['20251115', '20251116', ..., '20251130']  # 15 days for full run

VAL_DATES = 1  # Last N dates for validation

# ============================================================
# OPTIMIZED PARAMETERS (vs original)
# ============================================================

# Mesh - REDUCED for speed (original: 50000)
MAX_NODES = 15000  # ~3x faster training, minimal accuracy loss

# Model - keep same
HIDDEN_DIM = 96
NUM_LAYERS = 6
STATE_DIM = 1
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3
EDGE_FEATURES = 3

# Training - OPTIMIZED
EPOCHS = 150                    # Reduced from 200 (more data = faster convergence)
BATCH_SIZE = 1                  # Use 1 - avoids slow sequential loop
LEARNING_RATE = 1e-4            # Reduced from 3e-4 (more stable)
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5                 # Tighter than 1.0 (prevents spikes)

# Curriculum - OPTIMIZED
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 30   # Reduced from 100 (20% of epochs)
MAX_ROLLOUT_STEPS = 2

# Physics loss - REDUCED (original caused instability)
MASS_CONSERVATION_WEIGHT = 0.01  # Reduced from 0.05
SMOOTHNESS_WEIGHT = 0.01

# Normalization
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Time alignment
NOWCAST_HOURS = 5

# Storage
USE_FLOAT16_STORAGE = True
USE_GRADIENT_CHECKPOINTING = True


# ============================================================
# Model Architecture (unchanged)
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
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
    
    def forward(self, h, edge_index, edge_attr):
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
        logger.info(f"Model: {total_params:,} parameters, hidden={hidden_dim}, layers={num_layers}")
    
    def forward(self, x, static_features, forcing_features, edge_index, edge_attr):
        node_features = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        
        return self.decoder(h)


# ============================================================
# Stable Physics Loss (fixed mass loss formulation)
# ============================================================

class StablePhysicsLoss(nn.Module):
    """Physics-informed loss with stable mass conservation."""
    
    def __init__(self, mass_weight: float = 0.01, smoothness_weight: float = 0.01):
        super().__init__()
        self.mass_weight = mass_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, pred, target, edge_index=None):
        # MSE loss
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Stable mass conservation (absolute difference, clamped)
        if self.mass_weight > 0:
            pred_sum = pred.sum()
            target_sum = target.sum()
            mass_diff = (pred_sum - target_sum).abs() / (target.shape[0] + 1e-8)
            mass_loss = torch.clamp(mass_diff, max=10.0)  # Prevent explosion
        else:
            mass_loss = torch.tensor(0.0, device=pred.device)
        
        # Smoothness loss
        if self.smoothness_weight > 0 and edge_index is not None:
            row, col = edge_index
            diff = pred[row] - pred[col]
            smooth_loss = (diff ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=pred.device)
        
        total = mse_loss + self.mass_weight * mass_loss + self.smoothness_weight * smooth_loss
        
        return total, {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smooth_loss.item(),
        }


# ============================================================
# Dataset
# ============================================================

class MultiDateCWLDataset(Dataset):
    def __init__(self, mesh_data: Dict, date_data_list: List[Dict], eta_scale: float = 2.0):
        self.eta_scale = eta_scale
        
        # Mesh info
        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        
        # Compute static features
        self._compute_static_features()
        self._compute_edge_features()
        
        # Load all date data
        self.elevations = []
        self.forcings = []
        self.date_labels = []
        
        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])
        
        # Build sample index
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(num_times - 2):  # Need t, t+1, t+2
                self.samples.append((date_idx, t))
        
        logger.info(f"Dataset: {len(self.samples)} samples from {len(date_data_list)} dates")
    
    def _compute_static_features(self):
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = 6371000.0
        
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)
        
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1
        
        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
        
        self.static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)
    
    def _compute_edge_features(self):
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
        date_idx, t = self.samples[idx]
        
        elev = self.elevations[date_idx]
        forcing = self.forcings[date_idx]
        
        # Current state
        cwl_t = elev[t].astype(np.float32)
        cwl_t = np.nan_to_num(cwl_t, nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        
        # Water level feature
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)
        
        # Forcing
        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        pres = forcing['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)
        
        # Targets
        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale
        
        # Next forcing
        u10_next = forcing['u10'][t+1].astype(np.float32) / WIND_SCALE
        v10_next = forcing['v10'][t+1].astype(np.float32) / WIND_SCALE
        pres_next = forcing['pressure'][t+1].astype(np.float32)
        forcing_next = np.stack([u10_next, v10_next, pres_next], axis=1)
        
        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_arr, dtype=torch.float32),
            'y': torch.tensor(cwl_t1[:, np.newaxis], dtype=torch.float32),
            'y_next': torch.tensor(cwl_t2[:, np.newaxis], dtype=torch.float32),
            'forcing_next': torch.tensor(forcing_next, dtype=torch.float32),
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
        }


# ============================================================
# Fast Interpolation (vectorized)
# ============================================================

def fast_interpolate_to_nodes(data_3d, grid_lat, grid_lon, node_lat, node_lon):
    """
    Vectorized interpolation using scipy.ndimage.map_coordinates.
    Much faster than per-timestep RegularGridInterpolator.
    
    Args:
        data_3d: [time, lat, lon] array
        grid_lat, grid_lon: 1D coordinate arrays
        node_lat, node_lon: Target node coordinates
    
    Returns:
        [time, nodes] interpolated array
    """
    num_times = data_3d.shape[0]
    num_nodes = len(node_lon)
    
    # Sort grid coordinates
    lat_sort = np.argsort(grid_lat)
    lon_sort = np.argsort(grid_lon)
    grid_lat_s = grid_lat[lat_sort]
    grid_lon_s = grid_lon[lon_sort]
    
    # Compute fractional indices for nodes (do once)
    lat_frac = np.interp(node_lat, grid_lat_s, np.arange(len(grid_lat_s)))
    lon_frac = np.interp(node_lon, grid_lon_s, np.arange(len(grid_lon_s)))
    
    coords = np.array([lat_frac, lon_frac])
    
    result = np.zeros((num_times, num_nodes), dtype=np.float32)
    
    for t in range(num_times):
        data = data_3d[t][lat_sort][:, lon_sort].astype(np.float32)
        result[t] = map_coordinates(data, coords, order=1, mode='nearest')
    
    return result


# ============================================================
# Data Preprocessing (optimized)
# ============================================================

def preprocess_date(date_str: str, mesh_data: Dict) -> Dict:
    """Preprocess a single date with optimized interpolation."""
    
    date_dir = f'{DATA_DIR}/{date_str}'
    cwl_file = f'{date_dir}/stofs_2d_glo.t00z.fields.cwl.nc'
    wind_file = f'{date_dir}/stofs_2d_glo.t00z.uvgrd10m.nc'
    pres_file = f'{date_dir}/stofs_2d_glo.t00z.pressfc.nc'
    
    logger.info(f"Processing {date_str}...")
    start_time = time.time()
    
    global_indices = mesh_data['global_indices']
    node_lon = mesh_data['lon']
    node_lat = mesh_data['lat']
    
    # 1. Load CWL
    logger.info("  Loading CWL...")
    nc_cwl = NCDataset(cwl_file, 'r')
    zeta = nc_cwl.variables['zeta']
    times = np.array(nc_cwl.variables['time'][:])
    
    full_times = zeta.shape[0]
    time_indices = list(range(NOWCAST_HOURS, full_times))
    
    elevation = np.zeros((len(time_indices), len(global_indices)), dtype=np.float16)
    for i, t in enumerate(time_indices):
        elevation[i] = zeta[t, global_indices]
    
    elevation = np.where(elevation < -9000, np.nan, elevation)
    nc_cwl.close()
    logger.info(f"    CWL: {elevation.shape[0]} timesteps")
    
    # 2. Load met forcing
    logger.info("  Loading wind...")
    nc_wind = NCDataset(wind_file, 'r')
    grid_lon = np.array(nc_wind.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc_wind.variables['grid_yt'][:], dtype=np.float32)
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    # Subset to region with margin
    margin = 2.0
    lon_mask = (grid_lon >= BBOX['lon_min'] - margin) & (grid_lon <= BBOX['lon_max'] + margin)
    lat_mask = (grid_lat >= BBOX['lat_min'] - margin) & (grid_lat <= BBOX['lat_max'] + margin)
    lon_idx = np.where(lon_mask)[0]
    lat_idx = np.where(lat_mask)[0]
    
    u10_raw = np.array(nc_wind.variables['ugrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
    v10_raw = np.array(nc_wind.variables['vgrd10m'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
    nc_wind.close()
    
    logger.info("  Loading pressure...")
    nc_pres = NCDataset(pres_file, 'r')
    pres_raw = np.array(nc_pres.variables['pressfc'][:, lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1])
    nc_pres.close()
    
    grid_lon_sub = grid_lon[lon_idx]
    grid_lat_sub = grid_lat[lat_idx]
    
    # 3. Fast interpolation
    logger.info("  Interpolating to mesh (vectorized)...")
    u10_interp = fast_interpolate_to_nodes(u10_raw, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    v10_interp = fast_interpolate_to_nodes(v10_raw, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    pres_interp = fast_interpolate_to_nodes(pres_raw, grid_lat_sub, grid_lon_sub, node_lat, node_lon)
    
    # Normalize pressure
    pres_interp = (pres_interp - PRESSURE_MEAN) / PRESSURE_SCALE
    
    # 4. Align timesteps
    met_times = u10_interp.shape[0]
    common_times = min(len(time_indices), met_times)
    
    elevation = elevation[:common_times]
    u10_interp = u10_interp[:common_times]
    v10_interp = v10_interp[:common_times]
    pres_interp = pres_interp[:common_times]
    
    elapsed = time.time() - start_time
    logger.info(f"  Done: {common_times} timesteps in {elapsed:.1f}s")
    
    return {
        'date': date_str,
        'elevation': elevation.astype(np.float16),
        'forcing': {
            'u10': u10_interp.astype(np.float16),
            'v10': v10_interp.astype(np.float16),
            'pressure': pres_interp.astype(np.float16),
        },
    }


def extract_mesh(cwl_file: str, bbox: dict, max_nodes: int) -> Dict:
    """Extract mesh from CWL file."""
    logger.info(f"Extracting mesh (max {max_nodes} nodes)...")
    
    nc = NCDataset(cwl_file, 'r')
    x = np.array(nc.variables['x'][:], dtype=np.float32)
    y = np.array(nc.variables['y'][:], dtype=np.float32)
    depth = np.array(nc.variables['depth'][:], dtype=np.float32)
    
    # Filter to bbox
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    subset_indices = np.where(mask)[0]
    logger.info(f"  Found {len(subset_indices)} nodes in bbox")
    
    # Subsample if needed
    if len(subset_indices) > max_nodes:
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(subset_indices, size=max_nodes, replace=False)
        subset_indices = np.sort(subset_indices)
        logger.info(f"  Subsampled to {len(subset_indices)} nodes")
    
    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_sub = depth[subset_indices]
    
    # Build edges with Delaunay
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


def preprocess_all():
    """Preprocess all training dates."""
    logger.info("=" * 60)
    logger.info("PREPROCESSING")
    logger.info("=" * 60)
    
    processed_dir = f'{OUTPUT_DIR}/data/processed_optimized'
    os.makedirs(processed_dir, exist_ok=True)
    
    # Extract mesh from first date
    first_date = TRAINING_DATES[0]
    cwl_file = f'{DATA_DIR}/{first_date}/stofs_2d_glo.t00z.fields.cwl.nc'
    mesh_data = extract_mesh(cwl_file, BBOX, MAX_NODES)
    
    # Save mesh
    np.savez_compressed(
        f'{processed_dir}/mesh_optimized.npz',
        **mesh_data
    )
    
    # Process each date
    total_start = time.time()
    for date_str in TRAINING_DATES:
        data = preprocess_date(date_str, mesh_data)
        
        np.savez_compressed(
            f'{processed_dir}/processed_{date_str}.npz',
            date=data['date'],
            elevation=data['elevation'],
            u10=data['forcing']['u10'],
            v10=data['forcing']['v10'],
            pressure=data['forcing']['pressure'],
        )
        gc.collect()
    
    total_time = time.time() - total_start
    logger.info(f"\nPreprocessing complete: {total_time/60:.1f} minutes")


# ============================================================
# Training Functions
# ============================================================

class CurriculumScheduler:
    def __init__(self, max_steps: int, warmup_epochs: int):
        self.max_steps = max_steps
        self.warmup_epochs = warmup_epochs
    
    def get_num_steps(self, epoch: int) -> int:
        if epoch <= self.warmup_epochs:
            return 1
        return self.max_steps


def train_epoch(model, loader, optimizer, criterion, device, num_steps, grad_clip):
    """Train one epoch with batch_size=1 (simpler and faster)."""
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    
    for batch in loader:
        optimizer.zero_grad()
        
        # With batch_size=1, squeeze removes the batch dimension
        x = batch['x'].squeeze(0).to(device)
        static = batch['static'].squeeze(0).to(device)
        forcing = batch['forcing'].squeeze(0).to(device)
        y = batch['y'].squeeze(0).to(device)
        edge_index = batch['edge_index'].squeeze(0).to(device)
        edge_attr = batch['edge_attr'].squeeze(0).to(device)
        
        # Step 1
        pred = model(x, static, forcing, edge_index, edge_attr)
        loss, components = criterion(pred, y, edge_index)
        
        # Step 2 (curriculum)
        if num_steps >= 2:
            y_next = batch['y_next'].squeeze(0).to(device)
            forcing_next = batch['forcing_next'].squeeze(0).to(device)
            
            # Update static with new water level
            depth = static[:, 2:3]
            wl_new = depth + pred * ETA_SCALE
            wl_norm = (wl_new - wl_new.mean()) / (wl_new.std() + 1e-8)
            static_new = torch.cat([static[:, :3], wl_norm], dim=1)
            
            pred2 = model(pred.detach(), static_new, forcing_next, edge_index, edge_attr)
            loss2, _ = criterion(pred2, y_next, edge_index)
            loss = loss + 0.5 * loss2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        for k in total_components:
            total_components[k] += components[k]
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_components.items()}


def validate(model, loader, criterion, device):
    """Validate with batch_size=1."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].squeeze(0).to(device)
            static = batch['static'].squeeze(0).to(device)
            forcing = batch['forcing'].squeeze(0).to(device)
            y = batch['y'].squeeze(0).to(device)
            edge_index = batch['edge_index'].squeeze(0).to(device)
            edge_attr = batch['edge_attr'].squeeze(0).to(device)
            
            pred = model(x, static, forcing, edge_index, edge_attr)
            loss, components = criterion(pred, y, edge_index)
            
            total_loss += loss.item()
            for k in total_components:
                total_components[k] += components[k]
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in total_components.items()}


def train():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("OPTIMIZED TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dates: {TRAINING_DATES}")
    logger.info(f"MAX_NODES: {MAX_NODES}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"MASS_WEIGHT: {MASS_CONSERVATION_WEIGHT}")
    logger.info(f"GRAD_CLIP: {GRAD_CLIP}")
    
    processed_dir = f'{OUTPUT_DIR}/data/processed_optimized'
    
    # Load mesh
    mesh_data = dict(np.load(f'{processed_dir}/mesh_optimized.npz'))
    logger.info(f"Mesh: {len(mesh_data['lon'])} nodes")
    
    # Load data
    train_dates = TRAINING_DATES[:-VAL_DATES]
    val_dates = TRAINING_DATES[-VAL_DATES:]
    
    logger.info(f"Train dates: {train_dates}")
    logger.info(f"Val dates: {val_dates}")
    
    train_data = []
    for date_str in train_dates:
        data = dict(np.load(f'{processed_dir}/processed_{date_str}.npz'))
        train_data.append({
            'date': str(data['date']),
            'elevation': data['elevation'],
            'forcing': {
                'u10': data['u10'],
                'v10': data['v10'],
                'pressure': data['pressure'],
            }
        })
    
    val_data = []
    for date_str in val_dates:
        data = dict(np.load(f'{processed_dir}/processed_{date_str}.npz'))
        val_data.append({
            'date': str(data['date']),
            'elevation': data['elevation'],
            'forcing': {
                'u10': data['u10'],
                'v10': data['v10'],
                'pressure': data['pressure'],
            }
        })
    
    # Create datasets
    train_dataset = MultiDateCWLDataset(mesh_data, train_data, eta_scale=ETA_SCALE)
    val_dataset = MultiDateCWLDataset(mesh_data, val_data, eta_scale=ETA_SCALE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    model = PhysicsInformedCWLModel(
        state_dim=STATE_DIM,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_checkpointing=USE_GRADIENT_CHECKPOINTING,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = StablePhysicsLoss(
        mass_weight=MASS_CONSERVATION_WEIGHT,
        smoothness_weight=SMOOTHNESS_WEIGHT,
    )
    
    curriculum = CurriculumScheduler(MAX_ROLLOUT_STEPS, CURRICULUM_WARMUP_EPOCHS)
    
    # Training loop
    os.makedirs(f'{OUTPUT_DIR}/outputs/checkpoints', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/outputs/figures', exist_ok=True)
    
    train_losses, val_losses = [], []
    loss_components = {'mse': [], 'mass': [], 'smooth': []}
    best_val_loss = float('inf')
    
    logger.info("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        num_steps = curriculum.get_num_steps(epoch)
        
        train_loss, train_comp = train_epoch(
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
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'static_features': STATIC_NODE_FEATURES,
                    'forcing_features': FORCING_FEATURES,
                    'eta_scale': ETA_SCALE,
                    'max_nodes': MAX_NODES,
                },
            }, f'{OUTPUT_DIR}/outputs/checkpoints/best_optimized_model.pt')
        
        epoch_time = time.time() - epoch_start
        
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | steps={num_steps} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"mse={train_comp['mse']:.4f} | mass={train_comp['mass']:.4f} | "
                f"best={best_val_loss:.4f} | {epoch_time:.1f}s"
            )
    
    total_time = time.time() - start_time
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].semilogy(train_losses, label='Train', alpha=0.7)
    axes[0].semilogy(val_losses, label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(loss_components['mse'], label='MSE')
    axes[1].semilogy(loss_components['mass'], label='Mass')
    axes[1].semilogy(loss_components['smooth'], label='Smooth')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/outputs/figures/optimized_training.png', dpi=150)
    plt.close()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Time per epoch: {total_time/EPOCHS:.1f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final MSE: {loss_components['mse'][-1]:.6f}")
    
    # Run rollout evaluation
    logger.info("\nRunning rollout evaluation...")
    run_rollout_evaluation(model, val_dataset, device)


def run_rollout_evaluation(model, dataset, device, num_steps=48, start_idx=50):
    """Run 48-hour rollout and generate evaluation plots."""
    model.eval()
    
    # Get initial condition
    sample = dataset[start_idx]
    x = sample['x'].unsqueeze(0).to(device)
    static = sample['static'].to(device)
    edge_index = sample['edge_index'].to(device)
    edge_attr = sample['edge_attr'].to(device)
    
    predictions = [x.squeeze().cpu().numpy() * ETA_SCALE]
    ground_truth = [sample['y'].squeeze().numpy() * ETA_SCALE]
    
    current_x = x.squeeze(0)
    
    with torch.no_grad():
        for t in range(num_steps):
            # Get forcing for this timestep
            if start_idx + t < len(dataset):
                future_sample = dataset[start_idx + t]
                forcing = future_sample['forcing'].to(device)
                gt = future_sample['y'].squeeze().numpy() * ETA_SCALE
            else:
                forcing = sample['forcing'].to(device)
                gt = np.zeros_like(predictions[0])
            
            # Update water level in static features
            depth = static[:, 2:3]
            wl_new = depth + current_x * ETA_SCALE
            wl_norm = (wl_new - wl_new.mean()) / (wl_new.std() + 1e-8)
            static_updated = torch.cat([static[:, :3], wl_norm], dim=1)
            
            # Predict
            pred = model(current_x, static_updated, forcing, edge_index, edge_attr)
            
            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(gt)
            
            current_x = pred
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate RMSE at different lead times
    lead_times = [1, 6, 12, 24, 48]
    rmse_values = {}
    
    for lt in lead_times:
        if lt < len(predictions):
            rmse = np.sqrt(np.mean((predictions[lt] - ground_truth[lt])**2))
            rmse_values[lt] = rmse
            logger.info(f"  Rollout t+{lt}h RMSE: {rmse:.4f} m")
    
    # Plot rollout comparison
    fig, axes = plt.subplots(len(lead_times), 3, figsize=(15, 4*len(lead_times)))
    
    lon, lat = dataset.lon, dataset.lat
    
    for i, lt in enumerate(lead_times):
        if lt >= len(predictions):
            continue
            
        pred = predictions[lt]
        truth = ground_truth[lt]
        error = pred - truth
        rmse = rmse_values.get(lt, 0)
        
        # Predicted
        sc1 = axes[i, 0].scatter(lon, lat, c=pred, s=1, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[i, 0].set_title(f't+{lt}h Predicted')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Latitude')
        plt.colorbar(sc1, ax=axes[i, 0], label='Elevation (m)')
        
        # Ground Truth
        sc2 = axes[i, 1].scatter(lon, lat, c=truth, s=1, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[i, 1].set_title(f't+{lt}h Ground Truth')
        axes[i, 1].set_xlabel('Longitude')
        plt.colorbar(sc2, ax=axes[i, 1], label='Elevation (m)')
        
        # Error
        sc3 = axes[i, 2].scatter(lon, lat, c=error, s=1, cmap='RdBu_r', vmin=-1.5, vmax=1.5)
        axes[i, 2].set_title(f't+{lt}h Error (RMSE: {rmse:.3f}m)')
        axes[i, 2].set_xlabel('Longitude')
        plt.colorbar(sc3, ax=axes[i, 2], label='Error (m)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/outputs/figures/optimized_rollout.png', dpi=150)
    plt.close()
    
    # Plot time series
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Domain average
    pred_mean = predictions.mean(axis=1)
    truth_mean = ground_truth.mean(axis=1)
    
    axes[0, 0].plot(pred_mean, 'b-', label='Predicted', linewidth=2)
    axes[0, 0].plot(truth_mean, 'r--', label='Ground Truth', linewidth=2)
    axes[0, 0].set_xlabel('Forecast Hour')
    axes[0, 0].set_ylabel('Mean Elevation (m)')
    axes[0, 0].set_title('Domain-Average Water Surface Elevation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # RMSE vs time
    rmse_curve = [np.sqrt(np.mean((predictions[t] - ground_truth[t])**2)) 
                  for t in range(min(len(predictions), len(ground_truth)))]
    axes[0, 1].plot(rmse_curve, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Forecast Hour')
    axes[0, 1].set_ylabel('RMSE (m)')
    axes[0, 1].set_title('RMSE vs Forecast Hour')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scatter t+6h
    if 6 < len(predictions):
        axes[1, 0].scatter(ground_truth[6], predictions[6], s=1, alpha=0.5)
        axes[1, 0].plot([-1.5, 1.5], [-1.5, 1.5], 'r--', label='1:1')
        axes[1, 0].set_xlabel('Ground Truth (m)')
        axes[1, 0].set_ylabel('Predicted (m)')
        r = np.corrcoef(ground_truth[6].flatten(), predictions[6].flatten())[0, 1]
        axes[1, 0].set_title(f't+6h (RMSE: {rmse_values.get(6, 0):.3f}m, R={r:.3f})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter t+24h
    if 24 < len(predictions):
        axes[1, 1].scatter(ground_truth[24], predictions[24], s=1, alpha=0.5)
        axes[1, 1].plot([-1.5, 1.5], [-1.5, 1.5], 'r--', label='1:1')
        axes[1, 1].set_xlabel('Ground Truth (m)')
        axes[1, 1].set_ylabel('Predicted (m)')
        r = np.corrcoef(ground_truth[24].flatten(), predictions[24].flatten())[0, 1]
        axes[1, 1].set_title(f't+24h (RMSE: {rmse_values.get(24, 0):.3f}m, R={r:.3f})')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/outputs/figures/optimized_rollout_timeseries.png', dpi=150)
    plt.close()
    
    logger.info(f"Rollout plots saved to {OUTPUT_DIR}/outputs/figures/")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized CWL GNN Training')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('--train', action='store_true', help='Train model')
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_all()
    elif args.train:
        train()
    else:
        # Check for preprocessed data
        processed_dir = f'{OUTPUT_DIR}/data/processed_optimized'
        if not os.path.exists(f'{processed_dir}/mesh_optimized.npz'):
            logger.info("Preprocessed data not found, running preprocessing...")
            preprocess_all()
        
        train()


if __name__ == '__main__':
    main()
