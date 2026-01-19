#!/usr/bin/env python3
"""
Training Script for STOFS Surrogate - 25k nodes V2 with ENHANCED PHYSICS FEATURES
Optimized for URSA H100

V2 Changes:
- 8 forcing features instead of 3:
  * u10, v10 (wind components)
  * wind_speed, wind_speed_sq (nonlinear wind stress)
  * wind_dir (wind direction)
  * pressure (surface pressure anomaly)
  * dP_dx, dP_dy (pressure gradients)

- 6 tidal constituents instead of 2:
  * M2, S2 (original)
  * N2, K1, O1, M4 (new)

- Extended rollout training (up to 12 steps)
- Dynamic batch sizing to prevent OOM

Expected improvement: 15-25% RMSE reduction from physics-informed features

Usage:
    STOFS_DATA_DIR=/path/to/data python scripts/train_25k_ursa_h100_v2.py
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# H100 optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Note: removed torch.set_float32_matmul_precision('high') - can cause cuBLAS issues

# ============================================================
# CONFIGURATION - V2 with Enhanced Physics
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate'))

VAL_YEAR = '2025'
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1

# V2: 12 temporal features (6 tidal constituents × 2 for sin/cos)
TEMPORAL_FEATURES = 12  # Was 6 (M2, S2) -> Now 12 (M2, S2, N2, K1, O1, M4)

STATIC_NODE_FEATURES = 4

# V2: 8 forcing features instead of 3
FORCING_FEATURES = 8  # u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure, dP_dx, dP_dy

EPOCHS = 100
BASE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16  # Increased to compensate for smaller batch sizes (effective batch=64)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 8  # Match old working script
USE_AMP = True
USE_COMPILE = False
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 50

# Extended rollout schedule with dynamic batch sizing
# Format: {num_steps: (start_epoch, end_epoch, batch_size)}
# NOTE: This uses TRUE batching (entire batch in one forward pass), so batch sizes
# must be much smaller than the old script which processed samples one-at-a-time.
# With 25k nodes and [B, N, F] tensors, memory scales linearly with batch size.
ROLLOUT_SCHEDULE = {
    1:  (1, 15, 4),     # Epochs 1-15: 1-step, batch=4
    2:  (16, 30, 4),    # Epochs 16-30: 2-step, batch=4
    3:  (31, 50, 2),    # Epochs 31-50: 3-step, batch=2
    6:  (51, 75, 2),    # Epochs 51-75: 6-step, batch=2 (6 hours)
    12: (76, 100, 1),   # Epochs 76-100: 12-step, batch=1 (12 hours)
}

MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Tidal constituent periods (hours)
TIDAL_PERIODS = {
    'M2': 12.4206,   # Principal lunar semidiurnal
    'S2': 12.0000,   # Principal solar semidiurnal
    'N2': 12.6583,   # Larger lunar elliptic semidiurnal
    'K1': 23.9345,   # Lunar diurnal
    'O1': 25.8193,   # Lunar diurnal
    'M4': 6.2103,    # Shallow water overtide of M2
}


# ============================================================
# TRUE BATCHED Model Architecture
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    """TRUE batched GNN block that processes [B, N, F] tensors."""
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
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]

        h_src = h[:, row, :]
        h_dst = h[:, col, :]
        h_gradient = h_dst - h_src

        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1)

        edge_input_flat = edge_input.reshape(B * E, -1)
        edge_msg_flat = self.edge_mlp(edge_input_flat)
        edge_msg = edge_msg_flat.reshape(B, E, F)

        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, E, F)
        aggr.scatter_add_(1, row_expanded, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        node_input_flat = node_input.reshape(B * N, -1)
        node_out_flat = self.node_mlp(node_input_flat)
        node_out = node_out_flat.reshape(B, N, F)

        h_new = h + node_out
        return h_new, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    """TRUE batched GNN model - V2 with enhanced physics features."""
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Input: x(state_dim) + x_prev(state_dim) + dxdt(state_dim) + tidal + static + forcing
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim

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
            BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        """
        Args:
            x: [B, N, 1] current state
            x_prev: [B, N, 1] previous state
            dxdt: [B, N, 1] rate of change
            tidal_harmonics: [B, N, 12] tidal features (6 constituents × 2)
            static_features: [B, N, 4] static node features
            forcing: [B, N, 8] V2 forcing features
            edge_index: [2, E] edge connectivity
            edge_attr: [E, 3] edge features
        """
        B = x.shape[0]
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)

        B, N, F_in = node_features.shape
        node_flat = node_features.reshape(B * N, F_in)
        h_flat = self.node_encoder(node_flat)
        h = h_flat.reshape(B, N, self.hidden_dim)

        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        h_flat = h.reshape(B * N, self.hidden_dim)
        delta_flat = self.decoder(h_flat)
        delta = delta_flat.reshape(B, N, -1)

        return x + delta


class PhysicsLoss(nn.Module):
    def __init__(self, mass_weight=0.01, smooth_weight=0.01):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, edge_index):
        mse_loss = ((pred - target) ** 2).mean()

        pred_sum = pred.sum(dim=(1, 2))
        target_sum = target.sum(dim=(1, 2))
        mass_diff = (pred_sum - target_sum).abs().mean() / (pred.shape[1] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        row, col = edge_index
        smooth_loss = ((pred[:, row, :] - pred[:, col, :]) ** 2).mean()

        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss
        return total, {'mse': mse_loss.item(), 'mass': mass_loss.item(), 'smooth': smooth_loss.item()}


# ============================================================
# V2 Dataset with Enhanced Physics Features
# ============================================================

class InMemoryDatasetV2(Dataset):
    """V2 Dataset with 8 forcing features and 6 tidal constituents."""

    def __init__(self, mesh_data: Dict, date_data_list: List[Dict],
                 eta_scale: float = 2.0, dt_hours: float = 1.0):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.num_nodes = len(self.lon)

        self._compute_static_features()
        self._compute_edge_features()

        # Store all data in memory
        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])

        self._compute_global_times()

        # Build sample index (need enough future timesteps for 12-step rollout)
        self.max_rollout = 12
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(1, num_times - self.max_rollout - 1):
                self.samples.append((date_idx, t))

        logger.info(f"InMemoryDatasetV2: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Forcing features: 8 (u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure, dP_dx, dP_dy)")
        logger.info(f"  Tidal constituents: 6 (M2, S2, N2, K1, O1, M4)")

    def _compute_global_times(self):
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours)

    def _compute_tidal_harmonics_v2(self, global_hour: float) -> np.ndarray:
        """Compute 6 tidal constituents (12 features: sin/cos for each)."""
        harmonics = []
        for name, period in TIDAL_PERIODS.items():
            phase = 2.0 * np.pi * global_hour / period
            harmonics.extend([np.sin(phase), np.cos(phase)])
        return np.array(harmonics, dtype=np.float32)

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

    def _get_forcing_v2(self, forcing: Dict, t: int) -> np.ndarray:
        """Get all 8 forcing features for timestep t."""
        return np.stack([
            forcing['u10'][t],
            forcing['v10'][t],
            forcing['wind_speed'][t],
            forcing['wind_speed_sq'][t],
            forcing['wind_dir'][t],
            forcing['pressure'][t],
            forcing['dP_dx'][t],
            forcing['dP_dy'][t],
        ], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        date_idx, t = self.samples[idx]

        elev = self.elevations[date_idx]
        forcing = self.forcings[date_idx]

        # Current and previous state
        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale
        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        # V2: 6 tidal constituents (12 features)
        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics_v2(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))
        tidal_t1 = np.tile(self._compute_tidal_harmonics_v2(global_hour_t + self.dt_hours), (self.num_nodes, 1))
        tidal_t2 = np.tile(self._compute_tidal_harmonics_v2(global_hour_t + 2*self.dt_hours), (self.num_nodes, 1))

        # Static features
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        # V2: 8 forcing features for current timestep
        forcing_arr = self._get_forcing_v2(forcing, t)

        # Future forcing and targets (up to max_rollout steps)
        future_forcing = []
        future_targets = []
        future_tidal = []

        for k in range(1, self.max_rollout + 1):
            future_forcing.append(self._get_forcing_v2(forcing, t + k))
            cwl_k = np.nan_to_num(elev[t + k].astype(np.float32), nan=0.0) / self.eta_scale
            future_targets.append(cwl_k)
            future_tidal.append(np.tile(
                self._compute_tidal_harmonics_v2(global_hour_t + k * self.dt_hours),
                (self.num_nodes, 1)
            ))

        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'x_prev': torch.tensor(cwl_prev_norm[:, np.newaxis], dtype=torch.float32),
            'dxdt': torch.tensor(dxdt[:, np.newaxis], dtype=torch.float32),
            'tidal_harmonics': torch.tensor(tidal_harmonics, dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_arr, dtype=torch.float32),
            'future_forcing': torch.tensor(np.stack(future_forcing), dtype=torch.float32),  # [max_rollout, N, 8]
            'future_targets': torch.tensor(np.stack(future_targets), dtype=torch.float32),  # [max_rollout, N]
            'future_tidal': torch.tensor(np.stack(future_tidal), dtype=torch.float32),      # [max_rollout, N, 12]
            'raw_depth': torch.tensor(self.depth[:, np.newaxis], dtype=torch.float32),
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
        }


# ============================================================
# Training Functions - V2 with Extended Rollout
# ============================================================

def get_rollout_config(epoch):
    """Get number of rollout steps and batch size for current epoch based on schedule."""
    for num_steps, (start, end, batch_size) in ROLLOUT_SCHEDULE.items():
        if start <= epoch <= end:
            return num_steps, batch_size
    # Default to max rollout with smallest batch
    max_steps = max(ROLLOUT_SCHEDULE.keys())
    return max_steps, ROLLOUT_SCHEDULE[max_steps][2]


def train_epoch_batched(model, loader, optimizer, criterion, device, num_steps,
                        grad_clip, scaler, use_amp, grad_accum_steps, log_every):
    """
    V2 Training with extended rollout (up to 12 steps).
    Uses exponentially decaying loss weights for later timesteps.
    """
    model.train()
    total_loss = 0
    total_comp = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = 0
    start_time = time.time()

    amp_ctx = autocast('cuda', enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    accum_comp = {'mse': 0, 'mass': 0, 'smooth': 0}

    # Loss weights decay exponentially: 1.0, 0.7, 0.5, 0.35, ...
    loss_weights = [0.7 ** i for i in range(num_steps)]
    loss_weights = [w / sum(loss_weights) for w in loss_weights]  # Normalize

    for batch_idx, batch in enumerate(loader):
        edge_index = batch['edge_index'][0].to(device, non_blocking=True)
        edge_attr = batch['edge_attr'][0].to(device, non_blocking=True)

        x = batch['x'].to(device, non_blocking=True)
        x_prev = batch['x_prev'].to(device, non_blocking=True)
        dxdt = batch['dxdt'].to(device, non_blocking=True)
        tidal = batch['tidal_harmonics'].to(device, non_blocking=True)
        static = batch['static'].to(device, non_blocking=True)
        forcing = batch['forcing'].to(device, non_blocking=True)
        raw_depth = batch['raw_depth'].to(device, non_blocking=True)

        # Future data: [B, max_rollout, N, F]
        future_forcing = batch['future_forcing'].to(device, non_blocking=True)
        future_targets = batch['future_targets'].to(device, non_blocking=True)
        future_tidal = batch['future_tidal'].to(device, non_blocking=True)

        with amp_ctx:
            # Step 1: Initial prediction
            pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
            y = future_targets[:, 0, :].unsqueeze(-1)  # [B, N, 1]
            loss, comp = criterion(pred, y, edge_index)
            loss = loss * loss_weights[0]

            # Extended rollout steps
            prev_prev = x
            prev = pred.detach()

            for step in range(1, num_steps):
                # Get target and forcing for this step
                y_step = future_targets[:, step, :].unsqueeze(-1)
                forcing_step = future_forcing[:, step - 1, :, :]  # forcing for step
                tidal_step = future_tidal[:, step - 1, :, :]

                # Update state derivatives
                dxdt_step = (prev - prev_prev) / DT_HOURS

                # Update water level in static features
                wl = raw_depth + prev * ETA_SCALE
                wl_mean = wl.mean(dim=1, keepdim=True)
                wl_std = wl.std(dim=1, keepdim=True) + 1e-8
                wl_norm = (wl - wl_mean) / wl_std
                static_step = torch.cat([static[:, :, :3], wl_norm], dim=-1)

                # Forward pass
                pred_step = model(prev, prev_prev, dxdt_step, tidal_step, static_step,
                                  forcing_step, edge_index, edge_attr)

                # Add weighted loss
                loss_step, _ = criterion(pred_step, y_step, edge_index)
                loss = loss + loss_weights[step] * loss_step

                # Update for next step
                prev_prev = prev
                prev = pred_step.detach()

            scaled_loss = loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()

        accumulated_loss += loss.item()
        for k in comp:
            accum_comp[k] += comp[k]

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += accumulated_loss
            num_batches += grad_accum_steps
            for k in accum_comp:
                total_comp[k] += accum_comp[k]

            if ((batch_idx + 1) // grad_accum_steps) % log_every == 0:
                elapsed = time.time() - start_time
                effective_batches = (batch_idx + 1) // grad_accum_steps
                batches_per_sec = effective_batches / elapsed
                samples_per_sec = (batch_idx + 1) * loader.batch_size / elapsed
                remaining = (len(loader) - batch_idx - 1) / grad_accum_steps
                eta_min = remaining / batches_per_sec / 60 if batches_per_sec > 0 else 0

                avg_loss = accumulated_loss / grad_accum_steps
                logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                           f"Loss: {avg_loss:.5f} ({num_steps}-step) | "
                           f"Speed: {samples_per_sec:.1f} samp/s | "
                           f"ETA: {eta_min:.1f} min")

            accumulated_loss = 0
            accum_comp = {'mse': 0, 'mass': 0, 'smooth': 0}

    if accumulated_loss > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += accumulated_loss
        remaining_batches = len(loader) % grad_accum_steps
        num_batches += remaining_batches
        for k in accum_comp:
            total_comp[k] += accum_comp[k]

    avg_loss = total_loss / max(num_batches, 1)
    avg_comp = {k: v / max(num_batches / grad_accum_steps, 1) for k, v in total_comp.items()}
    return avg_loss, avg_comp


def validate(model, loader, criterion, device, use_amp, num_steps=6):
    """Validate with multi-step rollout (default 6 steps = 6 hours)."""
    model.eval()
    total_loss = 0
    num_batches = 0

    amp_ctx = autocast('cuda', enabled=use_amp)

    with torch.no_grad():
        for batch in loader:
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            x = batch['x'].to(device)
            x_prev = batch['x_prev'].to(device)
            dxdt = batch['dxdt'].to(device)
            tidal = batch['tidal_harmonics'].to(device)
            static = batch['static'].to(device)
            forcing = batch['forcing'].to(device)
            raw_depth = batch['raw_depth'].to(device)

            future_forcing = batch['future_forcing'].to(device)
            future_targets = batch['future_targets'].to(device)
            future_tidal = batch['future_tidal'].to(device)

            with amp_ctx:
                # Multi-step validation rollout
                pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                y = future_targets[:, 0, :].unsqueeze(-1)
                loss, _ = criterion(pred, y, edge_index)

                prev_prev = x
                prev = pred

                for step in range(1, min(num_steps, future_targets.shape[1])):
                    y_step = future_targets[:, step, :].unsqueeze(-1)
                    forcing_step = future_forcing[:, step - 1, :, :]
                    tidal_step = future_tidal[:, step - 1, :, :]

                    dxdt_step = (prev - prev_prev) / DT_HOURS
                    wl = raw_depth + prev * ETA_SCALE
                    wl_mean = wl.mean(dim=1, keepdim=True)
                    wl_std = wl.std(dim=1, keepdim=True) + 1e-8
                    wl_norm = (wl - wl_mean) / wl_std
                    static_step = torch.cat([static[:, :, :3], wl_norm], dim=-1)

                    pred_step = model(prev, prev_prev, dxdt_step, tidal_step, static_step,
                                      forcing_step, edge_index, edge_attr)
                    loss_step, _ = criterion(pred_step, y_step, edge_index)
                    loss = loss + loss_step

                    prev_prev = prev
                    prev = pred_step

                loss = loss / num_steps

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main
# ============================================================

def main():
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)

    logger.info("=" * 70)
    logger.info("V2 TRAINING - 25k nodes - ENHANCED PHYSICS FEATURES")
    logger.info("=" * 70)
    logger.info(f"System RAM: {ram_gb:.1f} GB")
    logger.info(f"FORCING_FEATURES: {FORCING_FEATURES} (u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure, dP_dx, dP_dy)")
    logger.info(f"TEMPORAL_FEATURES: {TEMPORAL_FEATURES} (6 tidal constituents: M2, S2, N2, K1, O1, M4)")
    logger.info(f"BASE_BATCH_SIZE: {BASE_BATCH_SIZE} (dynamic per rollout phase)")
    logger.info(f"GRAD_ACCUM_STEPS: {GRAD_ACCUM_STEPS}")
    logger.info(f"Rollout schedule: {ROLLOUT_SCHEDULE}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_25k_v2'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    logger.info(f"Mesh: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    # Discover dates
    available_dates = sorted([
        (f.stem.replace('processed_', ''), f)
        for f in DATA_DIR.glob('processed_*.npz')
        if 'mesh' not in f.stem
    ])

    train_files = [(d, p) for d, p in available_dates if not d.startswith(VAL_YEAR)]
    val_files = [(d, p) for d, p in available_dates if d.startswith(VAL_YEAR)]

    logger.info(f"Training dates: {len(train_files)}")
    logger.info(f"Validation dates: {len(val_files)}")

    # Load ALL training data with V2 features
    logger.info("\nLoading ALL training data (V2 features)...")
    train_data = []
    for i, (date_str, file_path) in enumerate(train_files):
        if (i + 1) % 50 == 0:
            logger.info(f"  Loaded {i+1}/{len(train_files)} dates")
        data = np.load(file_path)
        train_data.append({
            'date': date_str,
            'elevation': data['elevation'],
            'forcing': {
                'u10': data['u10'],
                'v10': data['v10'],
                'wind_speed': data['wind_speed'],
                'wind_speed_sq': data['wind_speed_sq'],
                'wind_dir': data['wind_dir'],
                'pressure': data['pressure'],
                'dP_dx': data['dP_dx'],
                'dP_dy': data['dP_dy'],
            }
        })
    logger.info(f"  Loaded {len(train_files)} training dates")

    # Load validation data
    logger.info("Loading validation data...")
    val_data = []
    for date_str, file_path in val_files[:30]:
        data = np.load(file_path)
        val_data.append({
            'date': date_str,
            'elevation': data['elevation'],
            'forcing': {
                'u10': data['u10'],
                'v10': data['v10'],
                'wind_speed': data['wind_speed'],
                'wind_speed_sq': data['wind_speed_sq'],
                'wind_dir': data['wind_dir'],
                'pressure': data['pressure'],
                'dP_dx': data['dP_dx'],
                'dP_dy': data['dP_dy'],
            }
        })
    logger.info(f"  Loaded {len(val_data)} validation dates")

    # Create V2 datasets (loaders created dynamically per epoch for dynamic batch sizing)
    train_dataset = InMemoryDatasetV2(mesh_data, train_data, ETA_SCALE, DT_HOURS)
    val_dataset = InMemoryDatasetV2(mesh_data, val_data, ETA_SCALE, DT_HOURS)

    logger.info(f"\nTrain samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Track current batch size to recreate loaders when it changes
    current_batch_size = None
    train_loader = None
    val_loader = None

    # Device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else GradScaler('cpu')

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume from checkpoint
    if RESUME_FROM_CHECKPOINT:
        ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'),
                       key=lambda x: int(x.stem.split('_')[-1]))  # Sort numerically by epoch
        if ckpts:
            ckpt = torch.load(ckpts[-1], weights_only=False)
            state_dict = ckpt['model_state_dict']
            try:
                model.load_state_dict(state_dict)
            except:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('_orig_mod.', '')
                    new_state_dict[new_key] = v
                model.load_state_dict(new_state_dict, strict=False)

            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt.get('scheduler_state_dict', scheduler.state_dict()))
            start_epoch = ckpt['epoch'] + 1
            history = ckpt.get('history', history)
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            logger.info(f"Resumed from epoch {ckpt['epoch']}")

    logger.info("\n" + "=" * 70)
    logger.info("STARTING V2 TRAINING")
    logger.info("=" * 70)

    # CUDA warmup - initialize cuBLAS before training
    if device.type == 'cuda':
        logger.info("Warming up CUDA...")
        dummy = torch.randn(256, 256, device=device)
        _ = torch.mm(dummy, dummy)
        del dummy
        torch.cuda.synchronize()
        logger.info("CUDA warmup complete")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # V2: Extended rollout schedule with dynamic batch sizing
        num_steps, batch_size = get_rollout_config(epoch)

        # Recreate data loaders if batch size changed (to prevent OOM at higher rollout steps)
        if batch_size != current_batch_size:
            # Clean up old loaders and clear GPU cache (only if we have old loaders)
            if train_loader is not None:
                del train_loader, val_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            current_batch_size = batch_size
            logger.info(f"  Adjusting batch size to {batch_size} for {num_steps}-step rollout")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True,
                                      persistent_workers=False)  # Disabled - can cause CUDA issues
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True,
                                    persistent_workers=False)

        logger.info(f"\nEpoch {epoch}/{EPOCHS} | rollout_steps={num_steps} | batch_size={batch_size}")

        train_loss, train_comp = train_epoch_batched(
            model, train_loader, optimizer, criterion,
            device, num_steps, GRAD_CLIP, scaler, USE_AMP, GRAD_ACCUM_STEPS, LOG_EVERY_N_BATCHES
        )
        # Validate with same number of rollout steps as training
        val_loss = validate(model, val_loader, criterion, device, USE_AMP, num_steps=num_steps)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_state_dict,
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_nodes': len(mesh_data['lon']),
                    'forcing_features': FORCING_FEATURES,
                    'temporal_features': TEMPORAL_FEATURES,
                    'version': 'v2'
                }
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  ★ New best!")

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | "
                   f"best={best_val_loss:.5f} | lr={lr:.2e} | {epoch_time/60:.1f} min")

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    logger.info("\nV2 TRAINING COMPLETE")
    logger.info(f"Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
