#!/usr/bin/env python3
"""
Training Script for STOFS Surrogate - Option A (80k nodes, KNN edges)

Domain: Long Island Sound to Southern Maine (40-44°N, 74-69°W)
Coverage: NY, CT, RI, MA, Southern ME

Configuration:
- 80,000 nodes (42% utilization)
- ~580,000 edges (KNN k=6, ~7.3 edges/node)
- 1.5 km grid spacing
- Temporal memory model with tidal harmonics

Training Estimates:
- 30-day pilot: ~4-5 days
- 300-day full: ~45 days

Based on: train_midatlantic_40k_pilot.py

Usage:
    python scripts/train_80k_option_a.py

    # With custom paths (for ParallelWorks)
    STOFS_DATA_DIR=/path/to/data STOFS_OUTPUT_DIR=/path/to/output python scripts/train_80k_option_a.py
"""

import os
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from contextlib import nullcontext

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CUDA OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIGURATION - Option A: LI Sound to S. Maine (80k nodes)
# ============================================================

# Paths
# For ParallelWorks: /home/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a
# For local: /mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a
DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/home/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/home/Mansur.Jisan/stofs_surrogate'))

# Domain info
DOMAIN_NAME = "Option_A_LI_Sound_to_S_Maine"
DOMAIN_BBOX = "40-44°N, 74-69°W"
NUM_NODES = 80000
GRID_SPACING_KM = 1.5

# Training dates - auto-discover from data directory
# Set to None to auto-discover, or provide explicit list
TRAINING_DATES = None  # Will be populated from DATA_DIR

# Validation split: use 2025 dates for validation, 2023-2024 for training
VAL_YEAR = '2025'  # Dates starting with this year go to validation
VAL_RATIO = 0.15   # Fallback: use last 15% if no year-based split

# ============================================================
# MODEL & TRAINING PARAMETERS
# ============================================================

# Model architecture
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1           # eta(t) output
TEMPORAL_FEATURES = 6   # eta(t-1), deta/dt, + 4 tidal harmonics (sin/cos for M2, S2)
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

# Training - adjusted for 80k nodes with dense edges
EPOCHS = 100
BATCH_SIZE = 2          # Reduced from 4 due to more nodes/edges
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 0         # Must be 0 for lazy loading (cache not picklable)
USE_AMP = True          # Mixed precision for memory efficiency
RESUME_FROM_CHECKPOINT = True  # Auto-resume from latest checkpoint if available
CHECKPOINT_INTERVAL = 10       # Save checkpoint every N epochs
LAZY_CACHE_SIZE = 20    # Number of files to keep in memory (~4.6GB for 16GB RAM system)

# Curriculum learning
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 15   # 15% of epochs with 1-step
MAX_ROLLOUT_STEPS = 3           # Cap at 3 steps

# Physics loss weights
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0  # 1-hour timesteps

# Tidal harmonic periods (hours)
M2_PERIOD = 12.42  # Principal lunar semi-diurnal
S2_PERIOD = 12.00  # Principal solar semi-diurnal

# Reference epoch for tidal harmonics (2023-01-01 00:00:00 UTC)
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Pressure constants (already normalized in preprocessing)
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


# ============================================================
# Model Architecture with Temporal Memory
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
    """Message passing block with SWE-inspired gradient awareness."""

    def __init__(self, hidden_dim: int):
        super().__init__()

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

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        h_gradient = h_dst - h_src

        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr


class TemporalMemoryGNN(nn.Module):
    """
    GNN with temporal memory and tidal harmonics for resolving phase ambiguity.

    Input features:
    - eta(t): current water level
    - eta(t-1): previous water level
    - deta/dt: rate of change (tells if rising/falling)
    - tidal_harmonics: sin/cos of M2, S2 tidal phases (global clock)
    - static features: x, y, depth, total water level
    - forcing: u10, v10, pressure
    """

    def __init__(
        self,
        state_dim: int = 1,
        temporal_dim: int = 6,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        node_input_dim = state_dim + temporal_dim + static_feature_dim + forcing_feature_dim

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

        self.gnn_layers = nn.ModuleList([
            SWEInspiredGraphBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)

        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        delta = self.decoder(h)
        output = x + delta

        return output


# ============================================================
# Loss Function
# ============================================================

class PhysicsLoss(nn.Module):
    def __init__(self, mass_weight=0.01, smooth_weight=0.01):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, edge_index):
        mse_loss = ((pred - target) ** 2).mean()

        pred_sum = pred.sum()
        target_sum = target.sum()
        mass_diff = (pred_sum - target_sum).abs() / (pred.shape[0] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        row, col = edge_index
        smooth_loss = ((pred[row] - pred[col]) ** 2).mean()

        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss

        return total, {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smooth_loss.item()
        }


# ============================================================
# Dataset with Temporal Memory - LAZY LOADING VERSION
# ============================================================

class LazyTemporalMemoryDataset(Dataset):
    """
    Dataset with lazy loading - loads data on-demand to save memory.

    Uses an LRU cache to keep recently accessed files in memory.
    This allows training on datasets much larger than available RAM.
    """

    def __init__(self, mesh_data: Dict, date_file_paths: List[tuple],
                 eta_scale: float = 2.0, dt_hours: float = 1.0, cache_size: int = 10):
        """
        Args:
            mesh_data: Dict with lon, lat, depth, edge_index
            date_file_paths: List of (date_str, file_path) tuples
            eta_scale: Normalization scale for water levels
            dt_hours: Time step in hours
            cache_size: Number of files to keep in memory cache
        """
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours
        self.cache_size = cache_size

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.num_nodes = len(self.lon)

        self._compute_static_features()
        self._compute_edge_features()

        # Store file paths instead of loading data
        self.date_labels = [d[0] for d in date_file_paths]
        self.file_paths = [d[1] for d in date_file_paths]

        # LRU cache for loaded files: {date_idx: (elevation, forcing)}
        self._cache = {}
        self._cache_order = []

        self._compute_global_times()

        # Build sample index - we need to know timesteps per file
        # Load first file to get timestep count (assume all files have same structure)
        self.samples = []
        self._timesteps_per_file = {}

        for date_idx in range(len(self.date_labels)):
            # Peek at file to get timestep count without keeping data
            data = np.load(self.file_paths[date_idx])
            num_times = data['elevation'].shape[0]
            self._timesteps_per_file[date_idx] = num_times
            data.close()  # Close the file handle

            for t in range(1, num_times - 3):
                self.samples.append((date_idx, t))

        logger.info(f"LazyDataset: {len(self.samples):,} samples from {len(date_file_paths)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}")
        logger.info(f"  Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Cache size: {cache_size} files (~{cache_size * 230 / 1000:.1f} GB)")
        logger.info(f"  Temporal memory: using eta(t-1), deta/dt, and tidal harmonics (M2, S2)")

    def _compute_global_times(self):
        """Compute global hours from epoch for each date."""
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours_from_epoch = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours_from_epoch)

    def _compute_tidal_harmonics(self, global_hour: float) -> np.ndarray:
        phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
        sin_m2 = np.sin(phase_m2)
        cos_m2 = np.cos(phase_m2)

        phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
        sin_s2 = np.sin(phase_s2)
        cos_s2 = np.cos(phase_s2)

        return np.array([sin_m2, cos_m2, sin_s2, cos_s2], dtype=np.float32)

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

    def _get_file_data(self, date_idx: int):
        """Load file data from cache or disk with LRU eviction."""
        if date_idx in self._cache:
            # Move to end of order (most recently used)
            self._cache_order.remove(date_idx)
            self._cache_order.append(date_idx)
            return self._cache[date_idx]

        # Load from disk
        data = np.load(self.file_paths[date_idx])
        elevation = data['elevation']
        forcing = {
            'u10': data['u10'],
            'v10': data['v10'],
            'pressure': data['pressure'],
        }
        data.close()

        # Add to cache
        self._cache[date_idx] = (elevation, forcing)
        self._cache_order.append(date_idx)

        # Evict oldest if cache is full
        while len(self._cache_order) > self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
            gc.collect()

        return self._cache[date_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_idx, t = self.samples[idx]

        # Load data on-demand from cache
        elev, forcing = self._get_file_data(date_idx)

        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale

        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale

        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_harmonics_t = self._compute_tidal_harmonics(global_hour_t)
        tidal_harmonics = np.tile(tidal_harmonics_t, (self.num_nodes, 1))

        global_hour_t1 = global_hour_t + self.dt_hours
        global_hour_t2 = global_hour_t + 2 * self.dt_hours
        tidal_harmonics_t1 = np.tile(self._compute_tidal_harmonics(global_hour_t1), (self.num_nodes, 1))
        tidal_harmonics_t2 = np.tile(self._compute_tidal_harmonics(global_hour_t2), (self.num_nodes, 1))

        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        pres = forcing['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)

        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t3 = np.nan_to_num(elev[t+3].astype(np.float32), nan=0.0) / self.eta_scale

        forcing_t1 = np.stack([
            forcing['u10'][t+1].astype(np.float32) / WIND_SCALE,
            forcing['v10'][t+1].astype(np.float32) / WIND_SCALE,
            forcing['pressure'][t+1].astype(np.float32),
        ], axis=1)

        forcing_t2 = np.stack([
            forcing['u10'][t+2].astype(np.float32) / WIND_SCALE,
            forcing['v10'][t+2].astype(np.float32) / WIND_SCALE,
            forcing['pressure'][t+2].astype(np.float32),
        ], axis=1)

        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'x_prev': torch.tensor(cwl_prev_norm[:, np.newaxis], dtype=torch.float32),
            'dxdt': torch.tensor(dxdt[:, np.newaxis], dtype=torch.float32),
            'tidal_harmonics': torch.tensor(tidal_harmonics, dtype=torch.float32),
            'tidal_harmonics_t1': torch.tensor(tidal_harmonics_t1, dtype=torch.float32),
            'tidal_harmonics_t2': torch.tensor(tidal_harmonics_t2, dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_arr, dtype=torch.float32),
            'y': torch.tensor(cwl_t1[:, np.newaxis], dtype=torch.float32),
            'y_t2': torch.tensor(cwl_t2[:, np.newaxis], dtype=torch.float32),
            'y_t3': torch.tensor(cwl_t3[:, np.newaxis], dtype=torch.float32),
            'forcing_t1': torch.tensor(forcing_t1, dtype=torch.float32),
            'forcing_t2': torch.tensor(forcing_t2, dtype=torch.float32),
            'raw_depth': torch.tensor(self.depth[:, np.newaxis], dtype=torch.float32),
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
        }


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, num_steps, grad_clip, scaler=None, amp_ctx=None):
    """Training with multi-step rollout using temporal memory and tidal harmonics."""
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0

    if amp_ctx is None:
        amp_ctx = lambda: nullcontext()

    for batch in loader:
        optimizer.zero_grad()

        batch_size = batch['x'].shape[0]
        batch_loss = 0
        batch_comp = {'mse': 0.0, 'mass': 0.0, 'smooth': 0.0}

        edge_index = batch['edge_index'][0].to(device)
        edge_attr = batch['edge_attr'][0].to(device)

        for i in range(batch_size):
            x = batch['x'][i].to(device)
            x_prev = batch['x_prev'][i].to(device)
            dxdt = batch['dxdt'][i].to(device)
            tidal_harmonics = batch['tidal_harmonics'][i].to(device)
            static = batch['static'][i].to(device)
            forcing = batch['forcing'][i].to(device)
            y = batch['y'][i].to(device)
            raw_depth = batch['raw_depth'][i].to(device)

            with amp_ctx():
                pred = model(x, x_prev, dxdt, tidal_harmonics, static, forcing, edge_index, edge_attr)
                loss, components = criterion(pred, y, edge_index)

                if num_steps >= 2:
                    y_t2 = batch['y_t2'][i].to(device)
                    forcing_t1 = batch['forcing_t1'][i].to(device)
                    tidal_harmonics_t1 = batch['tidal_harmonics_t1'][i].to(device)

                    pred_detach = pred.detach()
                    dxdt_new = (pred_detach - x) / DT_HOURS

                    pred_meters = pred_detach * ETA_SCALE
                    wl_physical = raw_depth + pred_meters
                    wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)
                    static_new = torch.cat([static[:, :3], wl_norm], dim=1)

                    pred2 = model(pred_detach, x, dxdt_new, tidal_harmonics_t1, static_new, forcing_t1, edge_index, edge_attr)
                    loss2, _ = criterion(pred2, y_t2, edge_index)
                    loss = loss + 0.5 * loss2

                if num_steps >= 3:
                    y_t3 = batch['y_t3'][i].to(device)
                    forcing_t2 = batch['forcing_t2'][i].to(device)
                    tidal_harmonics_t2 = batch['tidal_harmonics_t2'][i].to(device)

                    pred2_detach = pred2.detach()
                    dxdt_new2 = (pred2_detach - pred_detach) / DT_HOURS

                    pred2_meters = pred2_detach * ETA_SCALE
                    wl_physical2 = raw_depth + pred2_meters
                    wl_norm2 = (wl_physical2 - wl_physical2.mean()) / (wl_physical2.std() + 1e-8)
                    static_new2 = torch.cat([static[:, :3], wl_norm2], dim=1)

                    pred3 = model(pred2_detach, pred_detach, dxdt_new2, tidal_harmonics_t2, static_new2, forcing_t2, edge_index, edge_attr)
                    loss3, _ = criterion(pred3, y_t3, edge_index)
                    loss = loss + 0.25 * loss3

                scaled_loss = loss / batch_size

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            batch_loss += loss.item()
            for k in components:
                batch_comp[k] += components[k]

        for k in total_components:
            total_components[k] += batch_comp[k] / batch_size

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += batch_loss
        num_samples += batch_size

    num_batches = len(loader)
    return total_loss / num_samples, {k: v / num_batches for k, v in total_components.items()}


def validate(model, loader, criterion, device, amp_ctx=None):
    """Validation with temporal memory model and tidal harmonics."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0

    if amp_ctx is None:
        amp_ctx = lambda: nullcontext()

    with torch.no_grad():
        for batch in loader:
            batch_size = batch['x'].shape[0]
            batch_comp = {'mse': 0.0, 'mass': 0.0, 'smooth': 0.0}
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            for i in range(batch_size):
                x = batch['x'][i].to(device)
                x_prev = batch['x_prev'][i].to(device)
                dxdt = batch['dxdt'][i].to(device)
                tidal_harmonics = batch['tidal_harmonics'][i].to(device)
                static = batch['static'][i].to(device)
                forcing = batch['forcing'][i].to(device)
                y = batch['y'][i].to(device)

                with amp_ctx():
                    pred = model(x, x_prev, dxdt, tidal_harmonics, static, forcing, edge_index, edge_attr)
                    loss, components = criterion(pred, y, edge_index)

                total_loss += loss.item()
                for k in components:
                    batch_comp[k] += components[k]

            for k in total_components:
                total_components[k] += batch_comp[k] / batch_size

            num_samples += batch_size

    num_batches = len(loader)
    return total_loss / num_samples, {k: v / num_batches for k, v in total_components.items()}


def evaluate_rollout(model, dataset, device, num_steps=48, amp_ctx=None):
    """Evaluate autoregressive rollout with temporal memory and tidal harmonics."""
    model.eval()

    if amp_ctx is None:
        amp_ctx = lambda: nullcontext()

    eval_date = dataset.date_labels[0]
    logger.info(f"Rollout evaluated on validation date: {eval_date}")

    # Load first validation file (works with lazy dataset)
    elev, forcing = dataset._get_file_data(0)
    global_hours_offset = dataset.global_hours_offset[0]

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    predictions = []
    ground_truth = []

    cwl_prev = np.nan_to_num(elev[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elev[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        for t in range(1, min(num_steps + 1, len(elev) - 1)):
            dxdt = (current_cwl - current_prev) / DT_HOURS

            global_hour_t = global_hours_offset + t * DT_HOURS
            tidal_harmonics = dataset._compute_tidal_harmonics(global_hour_t)
            tidal_harmonics = np.tile(tidal_harmonics, (dataset.num_nodes, 1))
            tidal_harmonics_tensor = torch.tensor(tidal_harmonics, dtype=torch.float32).to(device)

            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = dataset.depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([dataset.static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).to(device)

            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pres = forcing['pressure'][t].astype(np.float32)
            forcing_arr = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).to(device)

            with amp_ctx():
                pred = model(current_cwl, current_prev, dxdt, tidal_harmonics_tensor, static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elev[t + 1].astype(np.float32), nan=0.0))

            current_prev = current_cwl
            current_cwl = pred

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    results = {}
    for lead_hours in [1, 6, 12, 24, 48]:
        step_index = int(lead_hours / DT_HOURS) - 1
        if 0 <= step_index < len(predictions):
            rmse = np.sqrt(np.mean((predictions[step_index] - ground_truth[step_index])**2))
            results[f't+{lead_hours}h'] = rmse

    return results, predictions, ground_truth


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 70)
    logger.info("STOFS SURROGATE - OPTION A (80k nodes, KNN edges)")
    logger.info("=" * 70)

    logger.info(f"\nDomain: {DOMAIN_BBOX} (LI Sound to S. Maine)")
    logger.info(f"Coverage: NY, CT, RI, MA, Southern ME")
    logger.info(f"Resolution: {NUM_NODES:,} nodes (~{GRID_SPACING_KM} km grid spacing)")
    logger.info(f"Edges: ~580k (KNN k=6, ~7.3 edges/node)")

    logger.info(f"\nConfiguration:")
    logger.info(f"  DATA_DIR: {DATA_DIR}")
    logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"  TRAINING_DATES: auto-discover from data directory")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    logger.info(f"  NUM_LAYERS: {NUM_LAYERS}")
    logger.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"  EPOCHS: {EPOCHS}")
    logger.info(f"  MAX_ROLLOUT_STEPS: {MAX_ROLLOUT_STEPS}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_option_a'
    figure_dir = OUTPUT_DIR / 'outputs' / 'figures_80k_option_a'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    if not mesh_path.exists():
        logger.error(f"Mesh file not found: {mesh_path}")
        logger.error(f"Run preprocessing first: python scripts/preprocess_80k_option_a.py")
        return

    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    logger.info(f"\nMesh loaded: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")
    logger.info(f"  Edges/node: {mesh_data['edge_index'].shape[1] / len(mesh_data['lon']):.1f}")

    # Discover available dates from data directory
    available_dates = sorted([
        f.stem.replace('processed_', '')
        for f in DATA_DIR.glob('processed_*.npz')
        if f.stem != 'mesh'
    ])
    logger.info(f"\nFound {len(available_dates)} processed dates in {DATA_DIR}")

    if len(available_dates) == 0:
        logger.error("No processed data files found!")
        return

    # Split into train/val based on year
    train_dates = [d for d in available_dates if not d.startswith(VAL_YEAR)]
    val_dates = [d for d in available_dates if d.startswith(VAL_YEAR)]

    # Fallback to ratio-based split if no year-based split possible
    if len(val_dates) == 0:
        split_idx = int(len(available_dates) * (1 - VAL_RATIO))
        train_dates = available_dates[:split_idx]
        val_dates = available_dates[split_idx:]
        logger.info(f"Using ratio-based split ({VAL_RATIO*100:.0f}% validation)")

    logger.info(f"\nTrain dates: {len(train_dates)} days")
    logger.info(f"Val dates: {len(val_dates)} days")

    # Collect file paths (LAZY LOADING - don't load data into memory)
    logger.info("\nCollecting data file paths (lazy loading enabled)...")

    train_file_paths = []
    for date_str in train_dates:
        data_path = DATA_DIR / f'processed_{date_str}.npz'
        if data_path.exists():
            train_file_paths.append((date_str, str(data_path)))
        else:
            logger.warning(f"  Missing: {data_path}")

    val_file_paths = []
    for date_str in val_dates:
        data_path = DATA_DIR / f'processed_{date_str}.npz'
        if data_path.exists():
            val_file_paths.append((date_str, str(data_path)))
        else:
            logger.warning(f"  Missing: {data_path}")

    logger.info(f"  Train files: {len(train_file_paths)}")
    logger.info(f"  Val files: {len(val_file_paths)}")

    if not train_file_paths or not val_file_paths:
        logger.error("Insufficient data!")
        logger.error("Ensure preprocessing has completed for the required dates.")
        return

    # Create lazy datasets (data loaded on-demand during training)
    # Cache size controls memory usage: 10 files = ~2.3 GB RAM
    logger.info(f"\nCreating lazy datasets with cache_size={LAZY_CACHE_SIZE}...")
    train_dataset = LazyTemporalMemoryDataset(mesh_data, train_file_paths, eta_scale=ETA_SCALE, dt_hours=DT_HOURS, cache_size=LAZY_CACHE_SIZE)
    val_dataset = LazyTemporalMemoryDataset(mesh_data, val_file_paths, eta_scale=ETA_SCALE, dt_hours=DT_HOURS, cache_size=LAZY_CACHE_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info(f"\nTrain samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model
    model = TemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {num_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)

    # AMP setup
    use_amp = USE_AMP and device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    amp_ctx = lambda: autocast('cuda') if use_amp else nullcontext()

    # Training
    history = {'train_loss': [], 'val_loss': [], 'mse': [], 'mass': [], 'lr': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Check for checkpoint to resume from
    if RESUME_FROM_CHECKPOINT:
        checkpoint_files = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            logger.info(f"\nResuming from checkpoint: {latest_checkpoint.name}")
            checkpoint = torch.load(latest_checkpoint, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'history' in checkpoint:
                history = checkpoint['history']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            logger.info(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING - 80k Option A (Full Dataset)")
    logger.info(f"Training: {len(train_dates)} dates | Validation: {len(val_dates)} dates")
    logger.info("=" * 70)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        if epoch <= CURRICULUM_WARMUP_EPOCHS:
            num_steps = 1
        elif epoch <= CURRICULUM_WARMUP_EPOCHS * 2:
            num_steps = 2
        else:
            num_steps = 3

        train_loss, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP, scaler=scaler, amp_ctx=amp_ctx
        )

        val_loss, val_comp = validate(model, val_loader, criterion, device, amp_ctx=amp_ctx)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mse'].append(train_comp['mse'])
        history['mass'].append(train_comp['mass'])
        history['lr'].append(current_lr)

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
                    'temporal_features': TEMPORAL_FEATURES,
                    'num_nodes': len(mesh_data['lon']),
                    'num_edges': mesh_data['edge_index'].shape[1],
                    'model_type': 'TemporalMemoryGNN',
                    'domain': DOMAIN_NAME,
                    'domain_bbox': DOMAIN_BBOX,
                    'grid_spacing_km': GRID_SPACING_KM,
                }
            }, checkpoint_dir / 'best_model.pt')

        epoch_time = time.time() - epoch_start

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | steps={num_steps} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"mse={train_comp['mse']:.5f} | best={best_val_loss:.5f} | "
                f"lr={current_lr:.2e} | {epoch_time:.1f}s"
            )

        # Periodic checkpoint save
        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
            logger.info(f"  Checkpoint saved: checkpoint_epoch_{epoch}.pt")

            # Keep only last 3 checkpoints to save disk space
            old_checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))[:-3]
            for old_ckpt in old_checkpoints:
                old_ckpt.unlink()
                logger.info(f"  Removed old checkpoint: {old_ckpt.name}")

    total_time = time.time() - total_start

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Evaluate rollout
    logger.info("\nEvaluating rollout...")
    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pt', weights_only=True)['model_state_dict'])
    rollout_results, predictions, ground_truth = evaluate_rollout(model, val_dataset, device, num_steps=96, amp_ctx=amp_ctx)

    logger.info("\nRollout RMSE:")
    for lead_time, rmse in rollout_results.items():
        logger.info(f"  {lead_time}: {rmse:.4f} m")

    # Save plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].semilogy(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].semilogy(history['val_loss'], label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training Progress - 80k Option A\n({DOMAIN_BBOX})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(history['mse'], label='MSE', alpha=0.8)
    axes[0, 1].semilogy(history['mass'], label='Mass', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Component')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history['lr'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)

    lead_times = [int(k.split('+')[1].replace('h', '')) for k in rollout_results.keys()]
    rmse_values = list(rollout_results.values())
    axes[1, 1].bar(range(len(lead_times)), rmse_values, tick_label=[f't+{lt}h' for lt in lead_times])
    axes[1, 1].set_xlabel('Lead Time')
    axes[1, 1].set_ylabel('RMSE (m)')
    axes[1, 1].set_title('Rollout Performance')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figure_dir / 'training_summary_80k_option_a.png', dpi=150)
    plt.close()

    logger.info(f"\nModel saved to: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"Figures saved to: {figure_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
