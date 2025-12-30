#!/usr/bin/env python3
"""
Training Script for STOFS Surrogate - 80k nodes with FULL DATA per epoch

This script uses ALL training dates every epoch (no subsampling).
Requires sufficient RAM to cache data files.

RAM Requirements:
- g5.xlarge (16 GB):  LAZY_CACHE_SIZE=20  (~6 GB cache)   - SLOW (lots of I/O)
- g5.2xlarge (32 GB): LAZY_CACHE_SIZE=60  (~20 GB cache)  - MODERATE
- g5.4xlarge (64 GB): LAZY_CACHE_SIZE=150 (~50 GB cache)  - FAST
- g5.8xlarge (128 GB): LAZY_CACHE_SIZE=300 (all in RAM)   - FASTEST

Usage:
    # Auto-detect RAM and set cache size
    python scripts/train_80k_full_data.py

    # Or set manually via environment
    LAZY_CACHE_SIZE=60 python scripts/train_80k_full_data.py
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
from functools import lru_cache
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# CUDA OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/home/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/home/Mansur.Jisan/stofs_surrogate'))

DOMAIN_NAME = "Option_A_LI_Sound_to_S_Maine"
NUM_NODES = 80000
VAL_YEAR = '2025'

# Model architecture
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 6
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

# Training parameters
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 0
USE_AMP = True
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 200

# Auto-detect RAM and set cache size
def get_optimal_cache_size():
    """Determine cache size based on available RAM."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)

    # Reserve ~8 GB for system + PyTorch + model
    available_for_cache = total_ram_gb - 8

    # Each file is ~330 MB
    file_size_gb = 0.33
    optimal_cache = int(available_for_cache / file_size_gb)

    # Clamp to reasonable range
    optimal_cache = max(20, min(optimal_cache, 300))

    logger.info(f"System RAM: {total_ram_gb:.1f} GB")
    logger.info(f"Optimal cache size: {optimal_cache} files (~{optimal_cache * 0.33:.1f} GB)")

    return optimal_cache

# Allow override via environment variable
LAZY_CACHE_SIZE = int(os.environ.get('LAZY_CACHE_SIZE', 0)) or get_optimal_cache_size()

# Curriculum learning
CURRICULUM_WARMUP_EPOCHS = 20
MAX_ROLLOUT_STEPS = 3

# Physics loss weights
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Normalization
ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0

# Tidal harmonics
M2_PERIOD = 12.42
S2_PERIOD = 12.00
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)


# ============================================================
# Model Architecture
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
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
    def __init__(self, state_dim=1, temporal_dim=6, static_feature_dim=4,
                 forcing_feature_dim=3, edge_feature_dim=3, hidden_dim=128, num_layers=6):
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
            SWEInspiredGraphBlock(hidden_dim) for _ in range(num_layers)
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
        return x + delta


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
        return total, {'mse': mse_loss.item(), 'mass': mass_loss.item(), 'smooth': smooth_loss.item()}


# ============================================================
# Dataset with LRU Cache - Uses ALL dates per epoch
# ============================================================

class LazyFullDataDataset(Dataset):
    """
    Dataset that uses ALL training dates every epoch.
    Uses LRU cache for memory efficiency.
    """

    def __init__(self, mesh_data: Dict, date_files: List[tuple],
                 cache_size: int, eta_scale: float = 2.0, dt_hours: float = 1.0):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours
        self.date_files = date_files
        self.cache_size = cache_size

        # Store mesh data
        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.num_nodes = len(self.lon)

        # Compute features
        self._compute_static_features()
        self._compute_edge_features()
        self._compute_global_times()

        # Build sample index (date_idx, timestep, date_str)
        self.samples = []
        for date_idx, (date_str, _) in enumerate(self.date_files):
            # Assume each file has ~175 valid timesteps (24 hours - 4 for rollout)
            # We'll verify actual count when loading
            for t in range(1, 175):
                self.samples.append((date_idx, t, date_str))

        # Create cached loader
        self._create_cache()

        logger.info(f"LazyFullDataDataset: {len(self.date_files)} dates, ~{len(self.samples)} samples")
        logger.info(f"  Cache size: {self.cache_size} files")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")

    def _create_cache(self):
        """Create LRU cached data loader."""
        @lru_cache(maxsize=self.cache_size)
        def load_date_data(date_idx):
            date_str, file_path = self.date_files[date_idx]
            data = np.load(file_path)
            return {
                'elevation': data['elevation'],
                'u10': data['u10'],
                'v10': data['v10'],
                'pressure': data['pressure'],
            }
        self._load_cached = load_date_data

    def _compute_global_times(self):
        self.global_hours = {}
        for date_str, _ in self.date_files:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            self.global_hours[date_str] = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    def _compute_tidal_harmonics(self, global_hour: float) -> np.ndarray:
        phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
        phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
        return np.array([np.sin(phase_m2), np.cos(phase_m2),
                        np.sin(phase_s2), np.cos(phase_s2)], dtype=np.float32)

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
        date_idx, t, date_str = self.samples[idx]

        # Load data (cached)
        data = self._load_cached(date_idx)
        elev = data['elevation']

        # Check bounds
        if t >= len(elev) - 3:
            # Return last valid sample if out of bounds
            t = len(elev) - 4

        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale
        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        global_hour_t = self.global_hours[date_str] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))
        tidal_t1 = np.tile(self._compute_tidal_harmonics(global_hour_t + self.dt_hours), (self.num_nodes, 1))
        tidal_t2 = np.tile(self._compute_tidal_harmonics(global_hour_t + 2*self.dt_hours), (self.num_nodes, 1))

        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        u10 = data['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = data['v10'][t].astype(np.float32) / WIND_SCALE
        pres = data['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)

        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t3 = np.nan_to_num(elev[t+3].astype(np.float32), nan=0.0) / self.eta_scale

        forcing_t1 = np.stack([
            data['u10'][t+1].astype(np.float32) / WIND_SCALE,
            data['v10'][t+1].astype(np.float32) / WIND_SCALE,
            data['pressure'][t+1].astype(np.float32),
        ], axis=1)
        forcing_t2 = np.stack([
            data['u10'][t+2].astype(np.float32) / WIND_SCALE,
            data['v10'][t+2].astype(np.float32) / WIND_SCALE,
            data['pressure'][t+2].astype(np.float32),
        ], axis=1)

        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'x_prev': torch.tensor(cwl_prev_norm[:, np.newaxis], dtype=torch.float32),
            'dxdt': torch.tensor(dxdt[:, np.newaxis], dtype=torch.float32),
            'tidal_harmonics': torch.tensor(tidal_harmonics, dtype=torch.float32),
            'tidal_harmonics_t1': torch.tensor(tidal_t1, dtype=torch.float32),
            'tidal_harmonics_t2': torch.tensor(tidal_t2, dtype=torch.float32),
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

def train_epoch(model, loader, optimizer, criterion, device, num_steps,
                grad_clip, scaler, amp_ctx, log_every):
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch_size = batch['x'].shape[0]
        batch_loss = 0
        batch_comp = {'mse': 0.0, 'mass': 0.0, 'smooth': 0.0}

        edge_index = batch['edge_index'][0].to(device, non_blocking=True)
        edge_attr = batch['edge_attr'][0].to(device, non_blocking=True)

        x_batch = batch['x'].to(device, non_blocking=True)
        x_prev_batch = batch['x_prev'].to(device, non_blocking=True)
        dxdt_batch = batch['dxdt'].to(device, non_blocking=True)
        tidal_batch = batch['tidal_harmonics'].to(device, non_blocking=True)
        static_batch = batch['static'].to(device, non_blocking=True)
        forcing_batch = batch['forcing'].to(device, non_blocking=True)
        y_batch = batch['y'].to(device, non_blocking=True)
        raw_depth_batch = batch['raw_depth'].to(device, non_blocking=True)

        for i in range(batch_size):
            optimizer.zero_grad()

            x = x_batch[i]
            x_prev = x_prev_batch[i]
            dxdt = dxdt_batch[i]
            tidal = tidal_batch[i]
            static = static_batch[i]
            forcing = forcing_batch[i]
            y = y_batch[i]
            raw_depth = raw_depth_batch[i]

            with amp_ctx():
                pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                loss, components = criterion(pred, y, edge_index)

                if num_steps >= 2:
                    y_t2 = batch['y_t2'][i].to(device, non_blocking=True)
                    forcing_t1 = batch['forcing_t1'][i].to(device, non_blocking=True)
                    tidal_t1 = batch['tidal_harmonics_t1'][i].to(device, non_blocking=True)

                    pred_detach = pred.detach()
                    dxdt_new = (pred_detach - x) / DT_HOURS
                    pred_meters = pred_detach * ETA_SCALE
                    wl_physical = raw_depth + pred_meters
                    wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)
                    static_new = torch.cat([static[:, :3], wl_norm], dim=1)

                    pred2 = model(pred_detach, x, dxdt_new, tidal_t1, static_new, forcing_t1, edge_index, edge_attr)
                    loss2, _ = criterion(pred2, y_t2, edge_index)
                    loss = loss + 0.5 * loss2

                if num_steps >= 3:
                    y_t3 = batch['y_t3'][i].to(device, non_blocking=True)
                    forcing_t2 = batch['forcing_t2'][i].to(device, non_blocking=True)
                    tidal_t2 = batch['tidal_harmonics_t2'][i].to(device, non_blocking=True)

                    pred2_detach = pred2.detach()
                    dxdt_new2 = (pred2_detach - pred_detach) / DT_HOURS
                    pred2_meters = pred2_detach * ETA_SCALE
                    wl_physical2 = raw_depth + pred2_meters
                    wl_norm2 = (wl_physical2 - wl_physical2.mean()) / (wl_physical2.std() + 1e-8)
                    static_new2 = torch.cat([static[:, :3], wl_norm2], dim=1)

                    pred3 = model(pred2_detach, pred_detach, dxdt_new2, tidal_t2, static_new2, forcing_t2, edge_index, edge_attr)
                    loss3, _ = criterion(pred3, y_t3, edge_index)
                    loss = loss + 0.25 * loss3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_loss += loss.item()
            for k in components:
                batch_comp[k] += components[k]

        total_loss += batch_loss
        for k in total_components:
            total_components[k] += batch_comp[k] / batch_size
        num_samples += batch_size

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            samples_per_sec = num_samples / elapsed
            eta_minutes = (len(loader) - batch_idx - 1) / (batch_idx + 1) * elapsed / 60
            logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                       f"Loss: {batch_loss/batch_size:.5f} | "
                       f"Speed: {samples_per_sec:.1f} samples/s | "
                       f"ETA: {eta_minutes:.1f} min")

    num_batches = len(loader)
    return total_loss / num_samples, {k: v / num_batches for k, v in total_components.items()}


def validate(model, loader, criterion, device, amp_ctx):
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = batch['x'].shape[0]
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            for i in range(batch_size):
                x = batch['x'][i].to(device)
                x_prev = batch['x_prev'][i].to(device)
                dxdt = batch['dxdt'][i].to(device)
                tidal = batch['tidal_harmonics'][i].to(device)
                static = batch['static'][i].to(device)
                forcing = batch['forcing'][i].to(device)
                y = batch['y'][i].to(device)

                with amp_ctx():
                    pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                    loss, _ = criterion(pred, y, edge_index)

                total_loss += loss.item()
                num_samples += 1

    return total_loss / num_samples


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 70)
    logger.info("FULL DATA TRAINING - 80k nodes (All dates per epoch)")
    logger.info("=" * 70)

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Cache size: {LAZY_CACHE_SIZE} files")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Epochs: {EPOCHS}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_full'
    figure_dir = OUTPUT_DIR / 'outputs' / 'figures_80k_full'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    if not mesh_path.exists():
        logger.error(f"Mesh not found: {mesh_path}")
        return
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    logger.info(f"\nMesh: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    # Discover dates
    available_dates = sorted([
        (f.stem.replace('processed_', ''), f)
        for f in DATA_DIR.glob('processed_*.npz')
        if 'mesh' not in f.stem
    ])

    train_dates = [(d, p) for d, p in available_dates if not d.startswith(VAL_YEAR)]
    val_dates = [(d, p) for d, p in available_dates if d.startswith(VAL_YEAR)]

    logger.info(f"\nTraining dates: {len(train_dates)}")
    logger.info(f"Validation dates: {len(val_dates)}")

    # Create datasets
    logger.info("\nCreating training dataset...")
    train_dataset = LazyFullDataDataset(
        mesh_data, train_dates,
        cache_size=LAZY_CACHE_SIZE,
        eta_scale=ETA_SCALE, dt_hours=DT_HOURS
    )

    logger.info("\nCreating validation dataset...")
    val_dataset = LazyFullDataDataset(
        mesh_data, val_dates[:20],  # Use first 20 val dates
        cache_size=20,
        eta_scale=ETA_SCALE, dt_hours=DT_HOURS
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"\nTrain samples/epoch: ~{len(train_dataset):,}")
    logger.info(f"Train batches/epoch: ~{len(train_dataset)//BATCH_SIZE:,}")
    logger.info(f"Val samples: ~{len(val_dataset):,}")

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
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    amp_ctx = lambda: autocast('cuda') if USE_AMP and device.type == 'cuda' else nullcontext()

    history = {'train_loss': [], 'val_loss': [], 'mse': [], 'mass': [], 'lr': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume checkpoint
    if RESUME_FROM_CHECKPOINT:
        ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if ckpts:
            latest = ckpts[-1]
            logger.info(f"\nResuming from: {latest.name}")
            ckpt = torch.load(latest, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            if 'history' in ckpt:
                history = ckpt['history']
            if 'best_val_loss' in ckpt:
                best_val_loss = ckpt['best_val_loss']

    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # Curriculum
        if epoch <= CURRICULUM_WARMUP_EPOCHS:
            num_steps = 1
        elif epoch <= CURRICULUM_WARMUP_EPOCHS * 2:
            num_steps = 2
        else:
            num_steps = 3

        logger.info(f"\nEpoch {epoch}/{EPOCHS} | rollout_steps={num_steps}")

        train_loss, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP, scaler=scaler,
            amp_ctx=amp_ctx, log_every=LOG_EVERY_N_BATCHES
        )

        val_loss = validate(model, val_loader, criterion, device, amp_ctx)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mse'].append(train_comp['mse'])
        history['mass'].append(train_comp['mass'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_nodes': len(mesh_data['lon']),
                }
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  ★ New best model saved!")

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | "
                   f"mse={train_comp['mse']:.5f} | best={best_val_loss:.5f} | "
                   f"lr={current_lr:.2e} | {epoch_time/60:.1f} min")

        # Checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

            # Clean old checkpoints
            old_ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))[:-3]
            for old in old_ckpts:
                old.unlink()

    total_time = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
