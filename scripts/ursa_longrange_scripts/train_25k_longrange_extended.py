#!/usr/bin/env python3
"""
Extended Long-Range Training: 36-step and 48-step rollouts

Resume from epoch 50 checkpoint to continue training with even longer rollouts.
This tests the hypothesis that long-range edges help with 36h-48h forecasts.

Training Schedule:
- Epochs 1-15:  6-step  (from original script)
- Epochs 16-30: 12-step (from original script)
- Epochs 31-50: 24-step (from original script)
- Epochs 51-65: 36-step (NEW - this script)
- Epochs 66-80: 48-step (NEW - this script)

Usage:
    python scripts/ursa_longrange_scripts/train_25k_longrange_extended.py
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

DATA_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2')
MESH_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_25k_v2_longrange')
OUTPUT_DIR = Path('/scratch5/purged/Mansur.Jisan/stofs_surrogate/outputs/checkpoints_25k_longrange')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume from epoch 50 checkpoint (after 24-step training completes)
RESUME_FROM = OUTPUT_DIR / 'checkpoint_longrange_epoch_50.pt'

# Model config (must match pretrained)
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Extended training config
NUM_EPOCHS = 80            # Extended from 50 to 80
BASE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 32
LEARNING_RATE = 1e-5       # Even lower LR for extended fine-tuning
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 4
USE_AMP = True
CHECKPOINT_INTERVAL = 1  # Save every epoch
LOG_EVERY_N_BATCHES = 100

# Extended rollout schedule - includes original + new phases
ROLLOUT_SCHEDULE = {
    6:  (1, 15, 1),      # Epochs 1-15: 6-step (6h)
    12: (16, 30, 1),     # Epochs 16-30: 12-step (12h)
    24: (31, 50, 1),     # Epochs 31-50: 24-step (24h)
    36: (51, 65, 1),     # Epochs 51-65: 36-step (36h) - NEW
    48: (66, 80, 1),     # Epochs 66-80: 48-step (48h) - NEW
}

MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Tidal periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}


# ============================================================
# Model Architecture (same as original)
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

        h_new = h + node_out
        return h_new, edge_attr


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
        mse_loss = F.mse_loss(pred, target)

        pred_sum = pred.sum(dim=(1, 2))
        target_sum = target.sum(dim=(1, 2))
        mass_diff = (pred_sum - target_sum).abs().mean() / (pred.shape[1] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        row, col = edge_index
        smooth_loss = ((pred[:, row, :] - pred[:, col, :]) ** 2).mean()

        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss
        return total, {'mse': mse_loss.item(), 'mass': mass_loss.item(), 'smooth': smooth_loss.item()}


# ============================================================
# Dataset (same as original)
# ============================================================

class InMemoryDatasetV2(Dataset):
    def __init__(self, mesh_data, date_data_list, eta_scale=2.0, dt_hours=1.0):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.edge_attr = torch.tensor(mesh_data['edge_attr'], dtype=torch.float32)
        self.num_nodes = len(self.lon)

        self._compute_static_features()

        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])

        self._compute_global_times()

        # Extended max_rollout for 48-step training
        self.max_rollout = 48
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(1, num_times - self.max_rollout - 1):
                self.samples.append((date_idx, t))

        logger.info(f"InMemoryDatasetV2: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Max rollout: {self.max_rollout} steps (48h)")

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

    def _compute_global_times(self):
        self.global_times = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours_since_epoch = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_times.append(hours_since_epoch)

    def _compute_tidal_harmonics(self, global_hour):
        harmonics = []
        for name, period in TIDAL_PERIODS.items():
            omega = 2.0 * np.pi / period
            phase = omega * global_hour
            harmonics.extend([np.sin(phase), np.cos(phase)])
        return np.array(harmonics, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_idx, t = self.samples[idx]

        elevation = self.elevations[date_idx]
        forcing = self.forcings[date_idx]
        global_time_base = self.global_times[date_idx]

        cwl_prev = np.nan_to_num(elevation[t-1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t = np.nan_to_num(elevation[t].astype(np.float32), nan=0.0) / self.eta_scale
        dxdt = (cwl_t - cwl_prev) / self.dt_hours

        global_hour = global_time_base + t * self.dt_hours
        tidal = self._compute_tidal_harmonics(global_hour)

        water_level = self.depth + cwl_t * self.eta_scale
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        forcing_t = forcing[t]

        # Multi-step targets (up to 48 steps)
        targets = []
        future_forcings = []
        future_tidals = []
        for step in range(1, self.max_rollout + 1):
            future_t = t + step
            if future_t < len(elevation):
                target = np.nan_to_num(elevation[future_t].astype(np.float32), nan=0.0) / self.eta_scale
                targets.append(target)
                future_forcings.append(forcing[future_t])
                future_global_hour = global_time_base + future_t * self.dt_hours
                future_tidals.append(self._compute_tidal_harmonics(future_global_hour))
            else:
                targets.append(np.zeros(self.num_nodes, dtype=np.float32))
                future_forcings.append(np.zeros((self.num_nodes, FORCING_FEATURES), dtype=np.float32))
                future_tidals.append(np.zeros(TEMPORAL_FEATURES, dtype=np.float32))

        return {
            'x': torch.tensor(cwl_t, dtype=torch.float32).unsqueeze(-1),
            'x_prev': torch.tensor(cwl_prev, dtype=torch.float32).unsqueeze(-1),
            'dxdt': torch.tensor(dxdt, dtype=torch.float32).unsqueeze(-1),
            'tidal': torch.tensor(np.tile(tidal, (self.num_nodes, 1)), dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_t, dtype=torch.float32),
            'targets': torch.tensor(np.stack(targets), dtype=torch.float32).unsqueeze(-1),
            'future_forcings': torch.tensor(np.stack(future_forcings), dtype=torch.float32),
            'future_tidals': torch.tensor(np.stack(future_tidals), dtype=torch.float32),
            'depth': torch.tensor(self.depth, dtype=torch.float32),
            'static_base': torch.tensor(self.static_base, dtype=torch.float32),
        }


# ============================================================
# Training Functions
# ============================================================

def get_rollout_config(epoch):
    for num_steps, (start, end, batch_size) in ROLLOUT_SCHEDULE.items():
        if start <= epoch <= end:
            return num_steps, batch_size
    max_steps = max(ROLLOUT_SCHEDULE.keys())
    return max_steps, ROLLOUT_SCHEDULE[max_steps][2]


def train_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, edge_index, edge_attr):
    model.train()
    total_loss = 0.0
    num_batches = 0

    num_steps, _ = get_rollout_config(epoch)

    for batch_idx, batch in enumerate(train_loader):
        x = batch['x'].to(device)
        x_prev = batch['x_prev'].to(device)
        dxdt = batch['dxdt'].to(device)
        tidal = batch['tidal'].to(device)
        static = batch['static'].to(device)
        forcing = batch['forcing'].to(device)
        targets = batch['targets'].to(device)
        future_forcings = batch['future_forcings'].to(device)
        future_tidals = batch['future_tidals'].to(device)
        depth = batch['depth'].to(device)
        static_base = batch['static_base'].to(device)

        B, N, _ = x.shape

        with autocast(device_type='cuda', enabled=USE_AMP):
            loss_accum = 0.0
            current = x
            current_prev = x_prev
            current_dxdt = dxdt
            current_tidal = tidal
            current_forcing = forcing
            current_static = static

            for step in range(num_steps):
                pred = model(current, current_prev, current_dxdt, current_tidal,
                            current_static, current_forcing, edge_index, edge_attr)

                target = targets[:, step, :, :]
                step_loss, _ = criterion(pred, target, edge_index)
                loss_accum = loss_accum + step_loss

                with torch.no_grad():
                    new_dxdt = (pred - current) / DT_HOURS

                    next_tidal = future_tidals[:, step, :]
                    next_tidal_expanded = next_tidal.unsqueeze(1).expand(B, N, -1)

                    cwl_pred = pred.squeeze(-1) * ETA_SCALE
                    water_level = depth + cwl_pred
                    wl_norm = (water_level - water_level.mean(dim=1, keepdim=True)) / (water_level.std(dim=1, keepdim=True) + 1e-8)
                    next_static = torch.cat([static_base, wl_norm.unsqueeze(-1)], dim=-1)

                    next_forcing = future_forcings[:, step, :, :]

                current_prev = current
                current = pred
                current_dxdt = new_dxdt
                current_tidal = next_tidal_expanded
                current_static = next_static
                current_forcing = next_forcing

            loss = loss_accum / num_steps

        loss_scaled = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % LOG_EVERY_N_BATCHES == 0:
            logger.info(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, edge_index, edge_attr):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    num_steps, _ = get_rollout_config(epoch)

    for batch in val_loader:
        x = batch['x'].to(device)
        x_prev = batch['x_prev'].to(device)
        dxdt = batch['dxdt'].to(device)
        tidal = batch['tidal'].to(device)
        static = batch['static'].to(device)
        forcing = batch['forcing'].to(device)
        targets = batch['targets'].to(device)
        future_forcings = batch['future_forcings'].to(device)
        future_tidals = batch['future_tidals'].to(device)
        depth = batch['depth'].to(device)
        static_base = batch['static_base'].to(device)

        B, N, _ = x.shape

        with autocast(device_type='cuda', enabled=USE_AMP):
            loss_accum = 0.0
            current = x
            current_prev = x_prev
            current_dxdt = dxdt
            current_tidal = tidal
            current_forcing = forcing
            current_static = static

            for step in range(num_steps):
                pred = model(current, current_prev, current_dxdt, current_tidal,
                            current_static, current_forcing, edge_index, edge_attr)

                target = targets[:, step, :, :]
                step_loss, _ = criterion(pred, target, edge_index)
                loss_accum = loss_accum + step_loss

                new_dxdt = (pred - current) / DT_HOURS

                next_tidal = future_tidals[:, step, :]
                next_tidal_expanded = next_tidal.unsqueeze(1).expand(B, N, -1)

                cwl_pred = pred.squeeze(-1) * ETA_SCALE
                water_level = depth + cwl_pred
                wl_norm = (water_level - water_level.mean(dim=1, keepdim=True)) / (water_level.std(dim=1, keepdim=True) + 1e-8)
                next_static = torch.cat([static_base, wl_norm.unsqueeze(-1)], dim=-1)

                next_forcing = future_forcings[:, step, :, :]

                current_prev = current
                current = pred
                current_dxdt = new_dxdt
                current_tidal = next_tidal_expanded
                current_static = next_static
                current_forcing = next_forcing

            loss = loss_accum / num_steps

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("EXTENDED LONG-RANGE TRAINING: 36-step and 48-step")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load mesh
    logger.info(f"\nLoading mesh from {MESH_DIR}")
    mesh_data = dict(np.load(MESH_DIR / 'mesh.npz', allow_pickle=True))
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(mesh_data['edge_attr'], dtype=torch.float32).to(device)
    logger.info(f"Mesh: {len(mesh_data['lon']):,} nodes, {edge_index.shape[1]:,} edges")

    # Load data
    logger.info(f"\nLoading data from {DATA_DIR}")
    train_data = []
    val_data = []

    for f in sorted(DATA_DIR.glob('processed_*.npz')):
        if 'mesh' in f.stem:
            continue
        date_str = f.stem.replace('processed_', '')
        data = np.load(f)

        item = {
            'date': date_str,
            'elevation': data['elevation'],
            'forcing': np.stack([
                data['u10'], data['v10'], data['wind_speed'], data['wind_speed_sq'],
                data['wind_dir'], data['pressure'], data['dP_dx'], data['dP_dy']
            ], axis=-1)
        }

        if date_str.startswith('2025'):
            val_data.append(item)
        else:
            train_data.append(item)

    logger.info(f"Train dates: {len(train_data)}, Val dates: {len(val_data)}")

    # Create datasets
    train_dataset = InMemoryDatasetV2(mesh_data, train_data, eta_scale=ETA_SCALE)
    val_dataset = InMemoryDatasetV2(mesh_data, val_data, eta_scale=ETA_SCALE)

    train_loader = DataLoader(train_dataset, batch_size=BASE_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BASE_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Create model
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    scaler = GradScaler()
    criterion = PhysicsLoss(mass_weight=MASS_CONSERVATION_WEIGHT, smooth_weight=SMOOTHNESS_WEIGHT)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    # Check for existing checkpoints
    existing_ckpts = sorted(OUTPUT_DIR.glob('checkpoint_longrange_epoch_*.pt'))
    best_model_path = OUTPUT_DIR / 'best_model_longrange.pt'

    if existing_ckpts:
        latest_ckpt = existing_ckpts[-1]
        ckpt_epoch = int(latest_ckpt.stem.split('_')[-1].replace('.pt', ''))

        logger.info(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        history = ckpt.get('history', {'train_loss': [], 'val_loss': []})
        logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
    elif RESUME_FROM.exists():
        logger.info(f"Loading base checkpoint from {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded model weights, starting fresh optimizer")

    # Training loop
    logger.info(f"\nStarting training from epoch {start_epoch}")
    logger.info(f"Extended schedule: 36-step (ep51-65), 48-step (ep66-80)")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        num_steps, batch_size = get_rollout_config(epoch)
        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS} | rollout={num_steps}-step | batch={batch_size}")

        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, device, epoch, edge_index, edge_attr)
        val_loss = validate(model, val_loader, criterion, device, epoch, edge_index, edge_attr)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        logger.info(f"  train={train_loss:.6f}, val={val_loss:.6f} {'★' if is_best else ''}")

        # Save checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0 or is_best:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'history': history,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'rollout_steps': num_steps,
                }
            }

            if epoch % CHECKPOINT_INTERVAL == 0:
                torch.save(ckpt, OUTPUT_DIR / f'checkpoint_longrange_epoch_{epoch}.pt')
                logger.info(f"  Saved checkpoint_longrange_epoch_{epoch}.pt")

            if is_best:
                torch.save(ckpt, best_model_path)
                logger.info(f"  Saved best_model_longrange.pt (val={val_loss:.6f})")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\n" + "=" * 60)
    logger.info("EXTENDED TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
