#!/usr/bin/env python3
"""
Fine-tune STOFS-GNN with Long-Range Enhanced Mesh

Uses the pretrained 25k model and fine-tunes with additional long-range edges.
This is simpler than multi-scale and tests if information propagation is the bottleneck.

Key changes from original training:
1. Uses enhanced mesh with long-range edges
2. Starts from pretrained checkpoint
3. Lower learning rate for fine-tuning
4. Shorter training (model already knows local dynamics)

FIXED: Proper multi-step rollout with correct targets/forcing/tidal for each step
"""

import os
import gc
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# H100 optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# Configuration
# ============================================================

# Data from original V2, but mesh from long-range enhanced
DATA_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2')
MESH_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2_longrange')  # Enhanced mesh
OUTPUT_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/outputs/checkpoints_25k_longrange')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume from pretrained model
RESUME_FROM = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/outputs/checkpoints_25k_v2/checkpoint_epoch_60.pt')

# Model config (must match pretrained)
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12   # 6 tidal constituents × 2 (sin/cos)
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8     # u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure, dP_dx, dP_dy

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Training config (fine-tuning settings)
NUM_EPOCHS = 20            # Shorter since fine-tuning
BASE_BATCH_SIZE = 1        # Must be 1 due to 2.4x more edges (447k vs 185k)
GRAD_ACCUM_STEPS = 32      # Increased to compensate for batch=1
LEARNING_RATE = 2e-5       # Lower LR for fine-tuning (10x lower than original)
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 4
USE_AMP = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 100

# Fine-tuning rollout schedule - all batch=1 due to 447k edges
ROLLOUT_SCHEDULE = {
    6:  (1, 8, 1),     # Epochs 1-8: 6-step, batch=1
    12: (9, 15, 1),    # Epochs 9-15: 12-step, batch=1
    24: (16, 20, 1),   # Epochs 16-20: 24-step, batch=1
}

MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Tidal periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}


# ============================================================
# Model Architecture (same as V2)
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
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

        return h + node_out, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
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
# Dataset with PROPER Multi-Step Rollout Support
# ============================================================

class InMemoryDatasetLongRange(Dataset):
    """Dataset that returns future targets, forcing, and tidal for proper multi-step rollout."""

    def __init__(self, mesh_data, date_data_list, eta_scale=2.0, dt_hours=1.0, max_rollout=24):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours
        self.max_rollout = max_rollout

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.num_nodes = len(self.lon)

        self._compute_static_features()
        self._compute_edge_features(mesh_data)

        # Store all data in memory
        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])

        self._compute_global_times()

        # Build sample index (need enough future timesteps for max rollout)
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(1, num_times - self.max_rollout - 1):
                self.samples.append((date_idx, t))

        logger.info(f"InMemoryDatasetLongRange: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Max rollout: {self.max_rollout} steps")

    def _compute_global_times(self):
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours)

    def _compute_tidal_harmonics(self, global_hour):
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

    def _compute_edge_features(self, mesh_data):
        """Compute edge features, using original edges for normalization."""
        src, dst = self.edge_index[0].numpy(), self.edge_index[1].numpy()
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)

        # Use original edges for characteristic length (if available)
        n_original = mesh_data.get('n_original_edges', len(src))
        if isinstance(n_original, np.ndarray):
            n_original = int(n_original)
        char_length = np.median(dist[:n_original]) + 1e-8

        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        )

    def _get_forcing(self, forcing, t):
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

    def __len__(self):
        return len(self.samples)

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

        # Tidal harmonics for current timestep
        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))

        # Static features (with water level)
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        # Forcing for current timestep
        forcing_arr = self._get_forcing(forcing, t)

        # Future forcing, targets, and tidal (for proper multi-step rollout)
        future_forcing = []
        future_targets = []
        future_tidal = []

        for k in range(1, self.max_rollout + 1):
            future_forcing.append(self._get_forcing(forcing, t + k))
            cwl_k = np.nan_to_num(elev[t + k].astype(np.float32), nan=0.0) / self.eta_scale
            future_targets.append(cwl_k)
            future_tidal.append(np.tile(
                self._compute_tidal_harmonics(global_hour_t + k * self.dt_hours),
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
# Training Functions with CORRECT Multi-Step Rollout
# ============================================================

def get_rollout_config(epoch):
    """Get number of rollout steps and batch size for current epoch."""
    for num_steps, (start, end, batch_size) in ROLLOUT_SCHEDULE.items():
        if start <= epoch <= end:
            return num_steps, batch_size
    max_steps = max(ROLLOUT_SCHEDULE.keys())
    return max_steps, ROLLOUT_SCHEDULE[max_steps][2]


def train_epoch(model, loader, optimizer, criterion, device, num_steps,
                grad_clip, scaler, use_amp, grad_accum_steps, log_every):
    """Training with PROPER multi-step rollout - each step uses correct target/forcing/tidal."""
    model.train()
    total_loss = 0
    total_comp = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = 0
    start_time = time.time()

    amp_ctx = autocast('cuda', enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0

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
            # Step 1: Initial prediction (t -> t+1)
            pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
            y = future_targets[:, 0, :].unsqueeze(-1)  # [B, N, 1] - target for step 1
            loss, comp = criterion(pred, y, edge_index)
            loss = loss * loss_weights[0]

            # Extended rollout steps (step 2 onwards)
            prev_prev = x
            prev = pred.detach()

            for step in range(1, num_steps):
                # Get CORRECT target, forcing, and tidal for this step
                y_step = future_targets[:, step, :].unsqueeze(-1)           # [B, N, 1]
                forcing_step = future_forcing[:, step - 1, :, :]            # [B, N, 8]
                tidal_step = future_tidal[:, step - 1, :, :]                # [B, N, 12]

                # Update state derivatives
                dxdt_step = (prev - prev_prev) / DT_HOURS

                # Update water level in static features
                wl = raw_depth + prev * ETA_SCALE
                wl_mean = wl.mean(dim=1, keepdim=True)
                wl_std = wl.std(dim=1, keepdim=True) + 1e-8
                wl_norm = (wl - wl_mean) / wl_std
                static_step = torch.cat([static[:, :, :3], wl_norm], dim=-1)

                # Forward pass with correct inputs for this step
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

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            total_loss += accumulated_loss
            num_batches += grad_accum_steps
            for k in comp:
                total_comp[k] += comp[k]

            if ((batch_idx + 1) // grad_accum_steps) % log_every == 0:
                elapsed = time.time() - start_time
                effective_batches = (batch_idx + 1) // grad_accum_steps
                samples_per_sec = (batch_idx + 1) * loader.batch_size / elapsed

                avg_loss = accumulated_loss / grad_accum_steps
                logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                           f"Loss: {avg_loss:.5f} ({num_steps}-step) | "
                           f"Speed: {samples_per_sec:.1f} samp/s")

            accumulated_loss = 0

    if accumulated_loss > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += accumulated_loss
        remaining_batches = len(loader) % grad_accum_steps
        num_batches += remaining_batches

    return total_loss / max(num_batches, 1)


def validate(model, loader, criterion, device, use_amp, num_steps=12):
    """Validate with multi-step rollout using correct targets/forcing/tidal."""
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
    logger.info("="*70)
    logger.info("STOFS-GNN FINE-TUNING WITH LONG-RANGE EDGES")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ========================================
    # Load enhanced mesh with long-range edges
    # ========================================
    mesh_path = MESH_DIR / 'mesh.npz'

    if not mesh_path.exists():
        logger.error(f"Long-range mesh not found at {mesh_path}")
        logger.error("Run create_longrange_mesh.py first!")
        return

    mesh_data = dict(np.load(mesh_path, allow_pickle=True))

    n_nodes = len(mesh_data['lon'])
    n_edges = mesh_data['edge_index'].shape[1]
    n_original = mesh_data.get('n_original_edges', n_edges)
    if isinstance(n_original, np.ndarray):
        n_original = int(n_original)
    n_longrange = n_edges - n_original

    logger.info(f"Enhanced mesh loaded:")
    logger.info(f"  Nodes: {n_nodes:,}")
    logger.info(f"  Original edges: {n_original:,}")
    logger.info(f"  Long-range edges: {n_longrange:,}")
    logger.info(f"  Total edges: {n_edges:,} (+{100*n_longrange/max(n_original,1):.1f}%)")

    # ========================================
    # Load data (from original directory)
    # ========================================
    logger.info("\nLoading training data...")
    train_files = sorted([f for f in DATA_DIR.glob('processed_202[34]*.npz') if 'mesh' not in f.stem])
    val_files = sorted([f for f in DATA_DIR.glob('processed_2025*.npz') if 'mesh' not in f.stem])

    train_data = []
    for i, f in enumerate(train_files):
        if (i + 1) % 50 == 0:
            logger.info(f"  Loaded {i+1}/{len(train_files)} training dates")
        data = np.load(f)
        train_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    val_data = []
    for f in val_files[:30]:  # Limit val dates
        data = np.load(f)
        val_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    logger.info(f"Training dates: {len(train_data)}")
    logger.info(f"Validation dates: {len(val_data)}")

    # Create datasets with proper multi-step support
    train_dataset = InMemoryDatasetLongRange(mesh_data, train_data, ETA_SCALE, DT_HOURS, max_rollout=24)
    val_dataset = InMemoryDatasetLongRange(mesh_data, val_data, ETA_SCALE, DT_HOURS, max_rollout=24)

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # ========================================
    # Create model and load weights
    # ========================================
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ========================================
    # Optimizer (lower LR for fine-tuning)
    # ========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    criterion = PhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else GradScaler('cpu')

    # ========================================
    # Check for existing longrange checkpoints (for resumption)
    # ========================================
    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    existing_ckpts = sorted(OUTPUT_DIR.glob('checkpoint_longrange_epoch_*.pt'),
                           key=lambda x: int(x.stem.split('_')[-1]))

    if existing_ckpts:
        # Resume from latest longrange checkpoint
        latest_ckpt = existing_ckpts[-1]
        logger.info(f"\nResuming from existing checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)

        state_dict = ckpt.get('model_state_dict', ckpt)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])

        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history = ckpt.get('history', history)

        logger.info(f"Resumed from epoch {start_epoch - 1}, best_val={best_val_loss:.5f}")

    elif RESUME_FROM.exists():
        # Load from pretrained V2 model (first time)
        logger.info(f"\nLoading pretrained model from {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('_orig_mod.', '')
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)
        logger.info("Pretrained weights loaded successfully")
    else:
        logger.warning(f"No pretrained model found at {RESUME_FROM}")
        logger.warning("Training from scratch!")

    # ========================================
    # Training loop
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("STARTING FINE-TUNING WITH LONG-RANGE EDGES")
    logger.info("="*70)

    # CUDA warmup
    if device.type == 'cuda':
        logger.info("Warming up CUDA...")
        dummy = torch.randn(256, 256, device=device)
        _ = torch.mm(dummy, dummy)
        del dummy
        torch.cuda.synchronize()
        logger.info("CUDA warmup complete")

    current_batch_size = None
    train_loader = None
    val_loader = None

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start = time.time()

        num_steps, batch_size = get_rollout_config(epoch)

        # Recreate loaders if batch size changed
        if batch_size != current_batch_size:
            if train_loader is not None:
                del train_loader, val_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            current_batch_size = batch_size
            logger.info(f"  Adjusting batch size to {batch_size} for {num_steps}-step rollout")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)

        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS} | rollout={num_steps} | batch={batch_size}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, num_steps, GRAD_CLIP, scaler, USE_AMP, GRAD_ACCUM_STEPS, LOG_EVERY_N_BATCHES
        )

        val_loss = validate(model, val_loader, criterion, device, USE_AMP, num_steps=num_steps)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | lr={lr:.2e} | {epoch_time/60:.1f} min")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_dict,
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_nodes': n_nodes,
                    'n_longrange_edges': n_longrange,
                    'version': 'longrange'
                }
            }, OUTPUT_DIR / 'best_model_longrange.pt')
            logger.info(f"  ★ New best model saved! (val={val_loss:.5f})")

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, OUTPUT_DIR / f'checkpoint_longrange_epoch_{epoch}.pt')
            logger.info(f"  Saved checkpoint at epoch {epoch}")

    logger.info("\n" + "="*70)
    logger.info("FINE-TUNING COMPLETE!")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")
    logger.info(f"Checkpoints saved to: {OUTPUT_DIR}")
    logger.info("="*70)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
