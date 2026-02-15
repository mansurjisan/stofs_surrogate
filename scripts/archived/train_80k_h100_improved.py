#!/usr/bin/env python3
"""
IMPROVED Training Script for STOFS Surrogate - 80k nodes
Extended rollout training for better long-term forecasts

Key Improvements over train_80k_h100_fixed.py:
1. Extended rollout: 1 -> 2 -> 3 -> 6 -> 12 steps (vs max 3)
2. Scheduled sampling: Gradually use model predictions during training
3. Enhanced loss: Amplitude preservation + temporal consistency
4. Per-step loss tracking for monitoring
5. Gradient checkpointing for memory efficiency with long rollouts

Usage:
    python scripts/train_80k_h100_improved.py

    # Resume from checkpoint
    python scripts/train_80k_h100_improved.py --resume

    # Start fresh (ignore checkpoints)
    python scripts/train_80k_h100_improved.py --fresh
"""

import os
import gc
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# H100 optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ============================================================
# IMPROVED CONFIGURATION
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate'))

VAL_YEAR = '2025'
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 6
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

EPOCHS = 150  # Extended for longer rollout training

# IMPROVED: Extended rollout schedule
# Format: {rollout_steps: (start_epoch, end_epoch, batch_size, grad_accum)}
ROLLOUT_SCHEDULE = {
    1:  (1, 15, 4, 8),      # Epochs 1-15: 1-step, batch=4, accum=8 (eff=32)
    2:  (16, 30, 2, 16),    # Epochs 16-30: 2-step, batch=2, accum=16 (eff=32)
    3:  (31, 50, 1, 32),    # Epochs 31-50: 3-step, batch=1, accum=32 (eff=32)
    6:  (51, 80, 1, 32),    # Epochs 51-80: 6-step, batch=1, accum=32
    12: (81, 150, 1, 32),   # Epochs 81-150: 12-step, batch=1, accum=32
}

# IMPROVED: Scheduled sampling (teacher forcing decay)
TEACHER_FORCING_START = 1.0      # Start with 100% ground truth
TEACHER_FORCING_END = 0.5        # End with 50% ground truth
TEACHER_FORCING_DECAY_EPOCHS = 100  # Decay over this many epochs

# IMPROVED: Enhanced loss weights
MSE_WEIGHT = 1.0
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01
AMPLITUDE_WEIGHT = 0.05          # NEW: Preserve tidal amplitude
TEMPORAL_CONSISTENCY_WEIGHT = 0.02  # NEW: Smooth predictions over time

# Step-wise loss decay (later steps weighted less)
STEP_LOSS_DECAY = 0.7  # Each step's weight = previous * decay

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 4
USE_AMP = True
USE_GRAD_CHECKPOINT = True  # NEW: Enable gradient checkpointing for long rollouts
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 50

ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0
M2_PERIOD = 12.42
S2_PERIOD = 12.00
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Maximum timesteps to load per date (for extended rollout)
MAX_FUTURE_STEPS = 15  # Need t+1 through t+12, plus buffer


# ============================================================
# Model Architecture (same as before, with optional grad checkpoint)
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
    """TRUE batched GNN model with optional gradient checkpointing."""
    def __init__(self, state_dim=1, temporal_dim=6, static_feature_dim=4,
                 forcing_feature_dim=3, edge_feature_dim=3, hidden_dim=128,
                 num_layers=6, use_checkpoint=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_checkpoint = use_checkpoint
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
            BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def _forward_gnn_layer(self, layer, h, edge_index, edge_attr):
        """Helper for gradient checkpointing."""
        h_new, e_new = layer(h, edge_index, edge_attr)
        return h_new, e_new

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        B = x.shape[0]
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)

        B, N, F_in = node_features.shape
        node_flat = node_features.reshape(B * N, F_in)
        h_flat = self.node_encoder(node_flat)
        h = h_flat.reshape(B, N, self.hidden_dim)

        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            if self.use_checkpoint and self.training:
                # Gradient checkpointing to save memory
                h, e = grad_checkpoint(self._forward_gnn_layer, layer, h, edge_index, e,
                                       use_reentrant=False)
            else:
                h, e = layer(h, edge_index, e)

        h_flat = h.reshape(B * N, self.hidden_dim)
        delta_flat = self.decoder(h_flat)
        delta = delta_flat.reshape(B, N, -1)

        return x + delta


# ============================================================
# IMPROVED Physics Loss with Amplitude Preservation
# ============================================================

class ImprovedPhysicsLoss(nn.Module):
    """Enhanced loss function for better long-term predictions."""

    def __init__(self, mass_weight=0.01, smooth_weight=0.01,
                 amplitude_weight=0.05, temporal_weight=0.02):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight
        self.amplitude_weight = amplitude_weight
        self.temporal_weight = temporal_weight

    def forward(self, pred, target, edge_index, prev_pred=None):
        """
        Compute loss with multiple physics-informed terms.

        Args:
            pred: [B, N, 1] predicted state
            target: [B, N, 1] target state
            edge_index: [2, E] edge connectivity
            prev_pred: [B, N, 1] previous prediction (for temporal consistency)
        """
        # Base MSE loss
        mse_loss = ((pred - target) ** 2).mean()

        # Mass conservation
        pred_sum = pred.sum(dim=(1, 2))
        target_sum = target.sum(dim=(1, 2))
        mass_diff = (pred_sum - target_sum).abs().mean() / (pred.shape[1] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        # Spatial smoothness
        row, col = edge_index
        smooth_loss = ((pred[:, row, :] - pred[:, col, :]) ** 2).mean()

        # NEW: Amplitude preservation (preserve variance/range)
        pred_std = pred.std(dim=1).mean()
        target_std = target.std(dim=1).mean()
        amplitude_loss = ((pred_std - target_std) / (target_std + 1e-8)) ** 2

        # NEW: Temporal consistency (smooth predictions over time)
        temporal_loss = torch.tensor(0.0, device=pred.device)
        if prev_pred is not None:
            # Penalize sudden jumps in predictions
            pred_change = pred - prev_pred
            target_change = target - prev_pred  # Approximate expected change
            temporal_loss = ((pred_change - target_change) ** 2).mean()

        total = (mse_loss +
                 self.mass_weight * mass_loss +
                 self.smooth_weight * smooth_loss +
                 self.amplitude_weight * amplitude_loss +
                 self.temporal_weight * temporal_loss)

        components = {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smooth_loss.item(),
            'amplitude': amplitude_loss.item(),
            'temporal': temporal_loss.item() if prev_pred is not None else 0.0,
        }

        return total, components


# ============================================================
# Extended Dataset (supports up to 12-step rollout)
# ============================================================

class ExtendedInMemoryDataset(Dataset):
    """Dataset supporting extended rollout (up to 12 steps)."""

    def __init__(self, mesh_data: Dict, date_data_list: List[Dict],
                 eta_scale: float = 2.0, dt_hours: float = 1.0,
                 max_future_steps: int = 15):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours
        self.max_future_steps = max_future_steps

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        self.num_nodes = len(self.lon)

        self._compute_static_features()
        self._compute_edge_features()

        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])

        self._compute_global_times()

        # Build sample index (ensure enough future timesteps)
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            # Need t-1, t, and t+1 through t+max_future_steps
            for t in range(1, num_times - max_future_steps):
                self.samples.append((date_idx, t))

        logger.info(f"ExtendedDataset: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Max future steps: {max_future_steps}")

    def _compute_global_times(self):
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours)

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
        date_idx, t = self.samples[idx]

        elev = self.elevations[date_idx]
        forcing = self.forcings[date_idx]

        # Current and previous state
        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale
        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))

        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        pres = forcing['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)

        # Future targets and forcing (up to max_future_steps)
        future_targets = []
        future_forcings = []
        future_tidals = []

        for step in range(1, self.max_future_steps + 1):
            ft = t + step
            cwl_future = np.nan_to_num(elev[ft].astype(np.float32), nan=0.0) / self.eta_scale
            future_targets.append(cwl_future)

            future_forcings.append(np.stack([
                forcing['u10'][ft].astype(np.float32) / WIND_SCALE,
                forcing['v10'][ft].astype(np.float32) / WIND_SCALE,
                forcing['pressure'][ft].astype(np.float32),
            ], axis=1))

            future_tidals.append(np.tile(
                self._compute_tidal_harmonics(global_hour_t + step * self.dt_hours),
                (self.num_nodes, 1)
            ))

        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'x_prev': torch.tensor(cwl_prev_norm[:, np.newaxis], dtype=torch.float32),
            'dxdt': torch.tensor(dxdt[:, np.newaxis], dtype=torch.float32),
            'tidal_harmonics': torch.tensor(tidal_harmonics, dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_arr, dtype=torch.float32),
            'raw_depth': torch.tensor(self.depth[:, np.newaxis], dtype=torch.float32),
            'future_targets': torch.stack([torch.tensor(ft[:, np.newaxis], dtype=torch.float32)
                                           for ft in future_targets]),  # [max_steps, N, 1]
            'future_forcings': torch.stack([torch.tensor(ff, dtype=torch.float32)
                                            for ff in future_forcings]),  # [max_steps, N, 3]
            'future_tidals': torch.stack([torch.tensor(ft, dtype=torch.float32)
                                          for ft in future_tidals]),  # [max_steps, N, 4]
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
        }


# ============================================================
# IMPROVED Training Loop with Extended Rollout
# ============================================================

def get_teacher_forcing_ratio(epoch: int) -> float:
    """Compute teacher forcing ratio based on current epoch."""
    if epoch >= TEACHER_FORCING_DECAY_EPOCHS:
        return TEACHER_FORCING_END
    progress = epoch / TEACHER_FORCING_DECAY_EPOCHS
    return TEACHER_FORCING_START - progress * (TEACHER_FORCING_START - TEACHER_FORCING_END)


def get_rollout_config(epoch: int) -> Tuple[int, int, int]:
    """Get rollout steps, batch size, and gradient accumulation for current epoch."""
    for num_steps, (start, end, batch_size, grad_accum) in ROLLOUT_SCHEDULE.items():
        if start <= epoch <= end:
            return num_steps, batch_size, grad_accum
    # Default to maximum rollout
    max_steps = max(ROLLOUT_SCHEDULE.keys())
    _, _, batch_size, grad_accum = ROLLOUT_SCHEDULE[max_steps]
    return max_steps, batch_size, grad_accum


def train_epoch_extended(model, loader, optimizer, criterion, device, num_steps,
                         grad_clip, scaler, use_amp, grad_accum_steps,
                         teacher_forcing_ratio, log_every):
    """
    Training with extended rollout and scheduled sampling.
    """
    model.train()
    total_loss = 0
    total_comp = {'mse': 0, 'mass': 0, 'smooth': 0, 'amplitude': 0, 'temporal': 0}
    step_losses = [0.0] * num_steps  # Track per-step losses
    num_batches = 0
    start_time = time.time()

    amp_ctx = autocast('cuda', enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    accum_comp = {k: 0 for k in total_comp}

    for batch_idx, batch in enumerate(loader):
        # Shared graph structure
        edge_index = batch['edge_index'][0].to(device, non_blocking=True)
        edge_attr = batch['edge_attr'][0].to(device, non_blocking=True)

        # Current state
        x = batch['x'].to(device, non_blocking=True)
        x_prev = batch['x_prev'].to(device, non_blocking=True)
        dxdt = batch['dxdt'].to(device, non_blocking=True)
        tidal = batch['tidal_harmonics'].to(device, non_blocking=True)
        static = batch['static'].to(device, non_blocking=True)
        forcing = batch['forcing'].to(device, non_blocking=True)
        raw_depth = batch['raw_depth'].to(device, non_blocking=True)

        # Future data
        future_targets = batch['future_targets'].to(device, non_blocking=True)  # [B, max_steps, N, 1]
        future_forcings = batch['future_forcings'].to(device, non_blocking=True)
        future_tidals = batch['future_tidals'].to(device, non_blocking=True)

        B = x.shape[0]

        with amp_ctx:
            total_step_loss = 0
            step_weight = 1.0

            # Current state for rollout
            current_x = x
            current_x_prev = x_prev
            prev_pred = None

            for step in range(num_steps):
                # Compute temporal features
                current_dxdt = (current_x - current_x_prev) / DT_HOURS

                # Update static features with current water level
                if step > 0:
                    wl = raw_depth + current_x * ETA_SCALE
                    wl_mean = wl.mean(dim=1, keepdim=True)
                    wl_std = wl.std(dim=1, keepdim=True) + 1e-8
                    wl_norm = (wl - wl_mean) / wl_std
                    current_static = torch.cat([static[:, :, :3], wl_norm], dim=-1)
                else:
                    current_static = static

                # Get forcing and tidal for this step
                if step == 0:
                    current_forcing = forcing
                    current_tidal = tidal
                else:
                    current_forcing = future_forcings[:, step-1]  # step-1 because future starts at t+1
                    current_tidal = future_tidals[:, step-1]

                # Forward pass
                pred = model(current_x, current_x_prev, current_dxdt, current_tidal,
                           current_static, current_forcing, edge_index, edge_attr)

                # Get target for this step
                target = future_targets[:, step]  # target for t+step+1

                # Compute loss with temporal consistency
                loss, comp = criterion(pred, target, edge_index, prev_pred)

                # Weight loss by step (earlier steps more important)
                weighted_loss = step_weight * loss
                total_step_loss = total_step_loss + weighted_loss
                step_weight *= STEP_LOSS_DECAY

                # Track per-step loss
                step_losses[step] += loss.item()

                # Scheduled sampling: decide whether to use prediction or ground truth
                if step < num_steps - 1:  # Not the last step
                    use_ground_truth = random.random() < teacher_forcing_ratio

                    if use_ground_truth:
                        # Use ground truth for next step
                        next_x = target.detach()
                    else:
                        # Use model prediction for next step
                        next_x = pred.detach()

                    # Update states for next step
                    current_x_prev = current_x.detach()
                    current_x = next_x
                    prev_pred = pred.detach()

                # Clean up intermediate tensors
                if step > 0:
                    del wl, wl_mean, wl_std, wl_norm, current_static

            # Scale for gradient accumulation
            scaled_loss = total_step_loss / grad_accum_steps / num_steps

        # Backward pass
        scaler.scale(scaled_loss).backward()

        accumulated_loss += total_step_loss.item() / num_steps
        for k in comp:
            accum_comp[k] += comp[k]

        # Step optimizer after accumulating
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
                samples_per_sec = (batch_idx + 1) * loader.batch_size / elapsed
                remaining = (len(loader) - batch_idx - 1) / grad_accum_steps
                eta_min = remaining / (samples_per_sec / loader.batch_size) / 60 if samples_per_sec > 0 else 0

                avg_loss = accumulated_loss / grad_accum_steps
                logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                           f"Loss: {avg_loss:.5f} | "
                           f"Speed: {samples_per_sec:.1f} samp/s | "
                           f"ETA: {eta_min:.1f} min")

            accumulated_loss = 0
            accum_comp = {k: 0 for k in total_comp}

    # Handle remaining batches
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
    avg_step_losses = [sl / max(num_batches, 1) for sl in step_losses]

    return avg_loss, avg_comp, avg_step_losses


def validate(model, loader, criterion, device, use_amp):
    """Validation with 1-step prediction."""
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
            y = batch['future_targets'][:, 0].to(device)  # First future target

            with amp_ctx:
                pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                loss, _ = criterion(pred, y, edge_index)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()

    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)

    logger.info("=" * 70)
    logger.info("IMPROVED TRAINING - Extended Rollout + Scheduled Sampling")
    logger.info("=" * 70)
    logger.info(f"System RAM: {ram_gb:.1f} GB")
    logger.info(f"Rollout schedule: {list(ROLLOUT_SCHEDULE.keys())} steps")
    logger.info(f"Teacher forcing: {TEACHER_FORCING_START} -> {TEACHER_FORCING_END}")
    logger.info(f"Max epochs: {EPOCHS}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_improved'
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

    # Load ALL training data
    logger.info("\nLoading ALL training data into memory...")
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
                'pressure': data['pressure'],
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
                'pressure': data['pressure'],
            }
        })
    logger.info(f"  Loaded {len(val_data)} validation dates")

    # Create datasets with extended future steps
    train_dataset = ExtendedInMemoryDataset(mesh_data, train_data, ETA_SCALE, DT_HOURS, MAX_FUTURE_STEPS)
    val_dataset = ExtendedInMemoryDataset(mesh_data, val_data, ETA_SCALE, DT_HOURS, MAX_FUTURE_STEPS)

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
        use_checkpoint=USE_GRAD_CHECKPOINT,
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = ImprovedPhysicsLoss(
        MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT,
        AMPLITUDE_WEIGHT, TEMPORAL_CONSISTENCY_WEIGHT
    )
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else GradScaler('cpu')

    history = {'train_loss': [], 'val_loss': [], 'step_losses': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume from checkpoint
    if not args.fresh:
        ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'),
                       key=lambda x: int(x.stem.split('_')[-1]))
        if ckpts:
            ckpt = torch.load(ckpts[-1], weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt.get('scheduler_state_dict', scheduler.state_dict()))
            start_epoch = ckpt['epoch'] + 1
            history = ckpt.get('history', history)
            best_val_loss = ckpt.get('best_val_loss', best_val_loss)
            logger.info(f"Resumed from epoch {ckpt['epoch']}")

    logger.info("\n" + "=" * 70)
    logger.info("STARTING IMPROVED TRAINING")
    logger.info("=" * 70)

    current_batch_size = None
    train_loader = None
    val_loader = None

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # Get rollout configuration
        num_steps, batch_size, grad_accum_steps = get_rollout_config(epoch)
        teacher_forcing_ratio = get_teacher_forcing_ratio(epoch)

        # Recreate data loaders if batch size changed
        if batch_size != current_batch_size:
            logger.info(f"  Adjusting batch_size: {current_batch_size} -> {batch_size}")

            if train_loader is not None:
                del train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      num_workers=NUM_WORKERS, pin_memory=True,
                                      persistent_workers=True if NUM_WORKERS > 0 else False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS, pin_memory=True,
                                    persistent_workers=True if NUM_WORKERS > 0 else False)
            current_batch_size = batch_size

        logger.info(f"\nEpoch {epoch}/{EPOCHS} | steps={num_steps} | batch={batch_size} | "
                   f"tf_ratio={teacher_forcing_ratio:.2f}")

        train_loss, train_comp, step_losses = train_epoch_extended(
            model, train_loader, optimizer, criterion,
            device, num_steps, GRAD_CLIP, scaler, USE_AMP, grad_accum_steps,
            teacher_forcing_ratio, LOG_EVERY_N_BATCHES
        )
        val_loss = validate(model, val_loader, criterion, device, USE_AMP)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['step_losses'].append(step_losses)

        epoch_time = time.time() - epoch_start

        # Log per-step losses
        step_loss_str = " | ".join([f"s{i+1}={sl:.5f}" for i, sl in enumerate(step_losses)])
        logger.info(f"  Step losses: {step_loss_str}")

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
                    'max_rollout_steps': max(ROLLOUT_SCHEDULE.keys()),
                }
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  ★ New best!")

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | "
                   f"best={best_val_loss:.5f} | lr={lr:.2e} | {epoch_time/60:.1f} min")

        if epoch % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    logger.info("\nTRAINING COMPLETE")
    logger.info(f"Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
