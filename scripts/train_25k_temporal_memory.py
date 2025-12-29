#!/usr/bin/env python3
"""
Training Script with Temporal Memory for Phase Error Fix

Key changes from train_25k_15day.py:
1. Model receives η(t-1) and dη/dt as additional input features
2. Increased multi-step training horizon (4-6 steps)
3. Dataset provides temporal context (previous timesteps)

The phase lag issue occurs because the model only sees η(t) and cannot
distinguish whether the tide is rising or falling. Adding temporal
memory resolves this ambiguity.

Usage:
    python train_25k_temporal_memory.py
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

# ============================================================
# CUDA OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIGURATION
# ============================================================

# Paths - Configurable (set for ParallelWorks)
DATA_DIR = Path('/home/Mansur.Jisan/stofs_surrogate/data/processed_25k')
OUTPUT_DIR = Path('/home/Mansur.Jisan/stofs_surrogate')

# Training dates (15 days: Nov 15-29)
TRAINING_DATES = [
    '20251115', '20251116', '20251117', '20251118', '20251119',
    '20251120', '20251121', '20251122', '20251123', '20251124',
    '20251125', '20251126', '20251127', '20251128', '20251129',
]

VAL_DATES = 2  # Last 2 days for validation

# ============================================================
# MODEL & TRAINING PARAMETERS
# ============================================================

# Model architecture
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1           # η(t) output
TEMPORAL_FEATURES = 2   # NEW: η(t-1) and dη/dt
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

# Training
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 2
USE_AMP = True

# Curriculum learning - INCREASED for phase error fix
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 15   # 15% of epochs with 1-step
MAX_ROLLOUT_STEPS = 3           # Cap at 3 (trainer only implements up to 3 steps)

# Physics loss weights
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0  # 1-hour timesteps

# Pressure constants (for reference)
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
    GNN with temporal memory for resolving phase ambiguity.

    Input features:
    - η(t): current water level
    - η(t-1): previous water level
    - dη/dt: rate of change (tells if rising/falling)
    - static features: x, y, depth, total water level
    - forcing: u10, v10, pressure

    This allows the model to distinguish whether the tide is:
    - Rising (dη/dt > 0, η(t) > η(t-1))
    - Falling (dη/dt < 0, η(t) < η(t-1))
    - At peak (dη/dt ≈ 0, transitioning sign)
    """

    def __init__(
        self,
        state_dim: int = 1,
        temporal_dim: int = 2,          # η(t-1), dη/dt
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Input: η(t) + η(t-1) + dη/dt + static + forcing
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

    def forward(self, x, x_prev, dxdt, static_features, forcing, edge_index, edge_attr):
        """
        Forward pass with temporal memory.

        Args:
            x: Current elevation η(t), shape [N, 1]
            x_prev: Previous elevation η(t-1), shape [N, 1]
            dxdt: Rate of change dη/dt, shape [N, 1]
            static_features: Static node features [N, 4]
            forcing: Atmospheric forcing [N, 3]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, 3]

        Returns:
            Predicted elevation η(t+1), shape [N, 1]
        """
        # Concatenate all node features including temporal memory
        node_features = torch.cat([x, x_prev, dxdt, static_features, forcing], dim=-1)

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
# Dataset with Temporal Memory
# ============================================================

class TemporalMemoryDataset(Dataset):
    """
    Dataset that provides temporal context for each sample.

    For each sample at time t, provides:
    - x: η(t) - current elevation
    - x_prev: η(t-1) - previous elevation
    - dxdt: (η(t) - η(t-1)) / Δt - rate of change
    - y: η(t+1) - target (next timestep)
    - y_next: η(t+2) for multi-step training
    """

    def __init__(self, mesh_data: Dict, date_data_list: List[Dict],
                 eta_scale: float = 2.0, dt_hours: float = 0.5):
        self.eta_scale = eta_scale
        self.dt_hours = dt_hours

        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)

        self._compute_static_features()
        self._compute_edge_features()

        self.elevations = []
        self.forcings = []
        self.date_labels = []

        for data in date_data_list:
            self.elevations.append(data['elevation'])
            self.forcings.append(data['forcing'])
            self.date_labels.append(data['date'])

        # Build sample list - need at least t-1 and t+2 available
        # So we start from t=1 and end at t=num_times-3
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            # Need: t-1, t, t+1, t+2 for multi-step training
            # So t ranges from 1 to num_times-3
            for t in range(1, num_times - 3):
                self.samples.append((date_idx, t))

        logger.info(f"Dataset: {len(self.samples)} samples from {len(date_data_list)} dates")
        logger.info(f"  Temporal memory: using η(t-1) and dη/dt as inputs")

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

        # Current timestep: η(t)
        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale

        # Previous timestep: η(t-1)
        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale

        # Rate of change: dη/dt = (η(t) - η(t-1)) / Δt
        # Normalized: already in normalized units, divide by dt_hours for rate
        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        # Total water level (static feature)
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        # Forcing at time t
        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        pres = forcing['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)

        # Targets: η(t+1), η(t+2), η(t+3) for multi-step training
        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t3 = np.nan_to_num(elev[t+3].astype(np.float32), nan=0.0) / self.eta_scale

        # Forcing for future timesteps (for multi-step rollout)
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

def train_epoch(model, loader, optimizer, criterion, device, num_steps, grad_clip, scaler=None):
    """Training with multi-step rollout using temporal memory."""
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0

    for batch in loader:
        optimizer.zero_grad()

        batch_size = batch['x'].shape[0]
        batch_loss = 0
        batch_comp = {'mse': 0.0, 'mass': 0.0, 'smooth': 0.0}  # Track per-batch components

        edge_index = batch['edge_index'][0].to(device)
        edge_attr = batch['edge_attr'][0].to(device)

        for i in range(batch_size):
            x = batch['x'][i].to(device)
            x_prev = batch['x_prev'][i].to(device)
            dxdt = batch['dxdt'][i].to(device)
            static = batch['static'][i].to(device)
            forcing = batch['forcing'][i].to(device)
            y = batch['y'][i].to(device)
            raw_depth = batch['raw_depth'][i].to(device)

            if scaler is not None:
                with autocast('cuda'):
                    # Step 1: Predict η(t+1) from η(t), η(t-1), dη/dt
                    pred = model(x, x_prev, dxdt, static, forcing, edge_index, edge_attr)
                    loss, components = criterion(pred, y, edge_index)

                    # Multi-step rollout
                    if num_steps >= 2:
                        y_t2 = batch['y_t2'][i].to(device)
                        forcing_t1 = batch['forcing_t1'][i].to(device)

                        # For step 2: x_new = pred, x_prev_new = x, dxdt_new = (pred - x)/dt
                        pred_detach = pred.detach()
                        dxdt_new = (pred_detach - x) / DT_HOURS

                        # Update static features with new water level
                        pred_meters = pred_detach * ETA_SCALE
                        wl_physical = raw_depth + pred_meters
                        wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)
                        static_new = torch.cat([static[:, :3], wl_norm], dim=1)

                        pred2 = model(pred_detach, x, dxdt_new, static_new, forcing_t1, edge_index, edge_attr)
                        loss2, _ = criterion(pred2, y_t2, edge_index)
                        loss = loss + 0.5 * loss2

                    if num_steps >= 3:
                        y_t3 = batch['y_t3'][i].to(device)
                        forcing_t2 = batch['forcing_t2'][i].to(device)

                        # For step 3: x_new = pred2, x_prev_new = pred, dxdt_new = (pred2 - pred)/dt
                        pred2_detach = pred2.detach()
                        dxdt_new2 = (pred2_detach - pred_detach) / DT_HOURS

                        pred2_meters = pred2_detach * ETA_SCALE
                        wl_physical2 = raw_depth + pred2_meters
                        wl_norm2 = (wl_physical2 - wl_physical2.mean()) / (wl_physical2.std() + 1e-8)
                        static_new2 = torch.cat([static[:, :3], wl_norm2], dim=1)

                        pred3 = model(pred2_detach, pred_detach, dxdt_new2, static_new2, forcing_t2, edge_index, edge_attr)
                        loss3, _ = criterion(pred3, y_t3, edge_index)
                        loss = loss + 0.25 * loss3

                    scaled_loss = loss / batch_size

                scaler.scale(scaled_loss).backward()
            else:
                # Non-AMP path
                pred = model(x, x_prev, dxdt, static, forcing, edge_index, edge_attr)
                loss, components = criterion(pred, y, edge_index)

                if num_steps >= 2:
                    y_t2 = batch['y_t2'][i].to(device)
                    forcing_t1 = batch['forcing_t1'][i].to(device)

                    pred_detach = pred.detach()
                    dxdt_new = (pred_detach - x) / DT_HOURS

                    pred_meters = pred_detach * ETA_SCALE
                    wl_physical = raw_depth + pred_meters
                    wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)
                    static_new = torch.cat([static[:, :3], wl_norm], dim=1)

                    pred2 = model(pred_detach, x, dxdt_new, static_new, forcing_t1, edge_index, edge_attr)
                    loss2, _ = criterion(pred2, y_t2, edge_index)
                    loss = loss + 0.5 * loss2

                if num_steps >= 3:
                    y_t3 = batch['y_t3'][i].to(device)
                    forcing_t2 = batch['forcing_t2'][i].to(device)

                    pred2_detach = pred2.detach()
                    dxdt_new2 = (pred2_detach - pred_detach) / DT_HOURS

                    pred2_meters = pred2_detach * ETA_SCALE
                    wl_physical2 = raw_depth + pred2_meters
                    wl_norm2 = (wl_physical2 - wl_physical2.mean()) / (wl_physical2.std() + 1e-8)
                    static_new2 = torch.cat([static[:, :3], wl_norm2], dim=1)

                    pred3 = model(pred2_detach, pred_detach, dxdt_new2, static_new2, forcing_t2, edge_index, edge_attr)
                    loss3, _ = criterion(pred3, y_t3, edge_index)
                    loss = loss + 0.25 * loss3

                (loss / batch_size).backward()

            batch_loss += loss.item()
            for k in components:
                batch_comp[k] += components[k]

        # Average batch components and add to total
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


def validate(model, loader, criterion, device):
    """Validation with temporal memory model."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = batch['x'].shape[0]
            batch_comp = {'mse': 0.0, 'mass': 0.0, 'smooth': 0.0}  # Track per-batch
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            for i in range(batch_size):
                x = batch['x'][i].to(device)
                x_prev = batch['x_prev'][i].to(device)
                dxdt = batch['dxdt'][i].to(device)
                static = batch['static'][i].to(device)
                forcing = batch['forcing'][i].to(device)
                y = batch['y'][i].to(device)

                with autocast('cuda'):
                    pred = model(x, x_prev, dxdt, static, forcing, edge_index, edge_attr)
                    loss, components = criterion(pred, y, edge_index)

                total_loss += loss.item()
                for k in components:
                    batch_comp[k] += components[k]

            # Average batch components and add to total
            for k in total_components:
                total_components[k] += batch_comp[k] / batch_size

            num_samples += batch_size

    num_batches = len(loader)
    return total_loss / num_samples, {k: v / num_batches for k, v in total_components.items()}


def evaluate_rollout(model, dataset, device, num_steps=48):
    """Evaluate autoregressive rollout with temporal memory."""
    model.eval()

    elev = dataset.elevations[0]
    forcing = dataset.forcings[0]

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    predictions = []
    ground_truth = []

    # Initialize with first two timesteps
    cwl_prev = np.nan_to_num(elev[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elev[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        for t in range(1, min(num_steps + 1, len(elev) - 1)):
            # Compute temporal features
            dxdt = (current_cwl - current_prev) / DT_HOURS

            # Static features
            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = dataset.depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([dataset.static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).to(device)

            # Forcing at current timestep
            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pres = forcing['pressure'][t].astype(np.float32)
            forcing_arr = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).to(device)

            with autocast('cuda'):
                pred = model(current_cwl, current_prev, dxdt, static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elev[t + 1].astype(np.float32), nan=0.0))

            # Update temporal state
            current_prev = current_cwl
            current_cwl = pred

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    results = {}
    for lead_time in [1, 6, 12, 24, 48]:
        if lead_time <= len(predictions):
            rmse = np.sqrt(np.mean((predictions[lead_time-1] - ground_truth[lead_time-1])**2))
            results[f't+{lead_time}h'] = rmse

    return results, predictions, ground_truth


# ============================================================
# Main
# ============================================================

def main():
    logger.info("=" * 70)
    logger.info("TEMPORAL MEMORY GNN TRAINING - PHASE ERROR FIX")
    logger.info("=" * 70)

    logger.info(f"\nKey changes from baseline:")
    logger.info(f"  1. Added temporal memory: η(t-1) and dη/dt as inputs")
    logger.info(f"  2. Increased multi-step horizon: {MAX_ROLLOUT_STEPS} steps")
    logger.info(f"  3. Model can now distinguish rising vs falling tide")

    logger.info(f"\nConfiguration:")
    logger.info(f"  DATA_DIR: {DATA_DIR}")
    logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"  TRAINING_DATES: {len(TRAINING_DATES)} days")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    logger.info(f"  TEMPORAL_FEATURES: {TEMPORAL_FEATURES}")
    logger.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"  EPOCHS: {EPOCHS}")
    logger.info(f"  MAX_ROLLOUT_STEPS: {MAX_ROLLOUT_STEPS}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints'
    figure_dir = OUTPUT_DIR / 'outputs' / 'figures'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh_path = DATA_DIR / 'mesh_25k.npz'
    if not mesh_path.exists():
        logger.error(f"Mesh file not found: {mesh_path}")
        return

    mesh_data = dict(np.load(mesh_path))
    logger.info(f"\nMesh loaded: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    # Split dates
    train_dates = TRAINING_DATES[:-VAL_DATES]
    val_dates = TRAINING_DATES[-VAL_DATES:]

    logger.info(f"\nTrain dates: {len(train_dates)} days")
    logger.info(f"Val dates: {len(val_dates)} days")

    # Load data
    logger.info("\nLoading preprocessed data...")

    train_data = []
    for date_str in train_dates:
        data_path = DATA_DIR / f'processed_{date_str}.npz'
        if data_path.exists():
            data = dict(np.load(data_path))
            train_data.append({
                'date': date_str,
                'elevation': data['elevation'],
                'forcing': {
                    'u10': data['u10'],
                    'v10': data['v10'],
                    'pressure': data['pressure'],
                }
            })
            logger.info(f"  Loaded {date_str}: {data['elevation'].shape[0]} timesteps")
        else:
            logger.warning(f"  Missing: {data_path}")

    val_data = []
    for date_str in val_dates:
        data_path = DATA_DIR / f'processed_{date_str}.npz'
        if data_path.exists():
            data = dict(np.load(data_path))
            val_data.append({
                'date': date_str,
                'elevation': data['elevation'],
                'forcing': {
                    'u10': data['u10'],
                    'v10': data['v10'],
                    'pressure': data['pressure'],
                }
            })
            logger.info(f"  Loaded {date_str}: {data['elevation'].shape[0]} timesteps (val)")
        else:
            logger.warning(f"  Missing: {data_path}")

    if not train_data or not val_data:
        logger.error("Insufficient data!")
        return

    # Create datasets
    train_dataset = TemporalMemoryDataset(mesh_data, train_data, eta_scale=ETA_SCALE, dt_hours=DT_HOURS)
    val_dataset = TemporalMemoryDataset(mesh_data, val_data, eta_scale=ETA_SCALE, dt_hours=DT_HOURS)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info(f"\nTrain samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model with temporal memory
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
    logger.info(f"  Input features: η(t) + η(t-1) + dη/dt + static(4) + forcing(3) = {STATE_DIM + TEMPORAL_FEATURES + STATIC_NODE_FEATURES + FORCING_FEATURES}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)
    scaler = GradScaler('cuda') if USE_AMP else None

    # Training
    history = {'train_loss': [], 'val_loss': [], 'mse': [], 'mass': [], 'lr': []}
    best_val_loss = float('inf')

    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Curriculum learning: gradually increase rollout steps
        # Epoch 1-15: 1 step, Epoch 16-30: 2 steps, Epoch 31+: 3 steps
        if epoch <= CURRICULUM_WARMUP_EPOCHS:
            num_steps = 1
        elif epoch <= CURRICULUM_WARMUP_EPOCHS * 2:
            num_steps = 2
        else:
            num_steps = 3  # MAX_ROLLOUT_STEPS (trainer only implements up to 3)

        train_loss, train_comp = train_epoch(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP, scaler=scaler
        )

        val_loss, val_comp = validate(model, val_loader, criterion, device)

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
                    'model_type': 'TemporalMemoryGNN',
                }
            }, checkpoint_dir / 'best_temporal_memory_model.pt')

        epoch_time = time.time() - epoch_start

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | steps={num_steps} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"mse={train_comp['mse']:.5f} | best={best_val_loss:.5f} | "
                f"lr={current_lr:.2e} | {epoch_time:.1f}s"
            )

    total_time = time.time() - total_start

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    # Evaluate rollout
    logger.info("\nEvaluating rollout with temporal memory...")
    model.load_state_dict(torch.load(checkpoint_dir / 'best_temporal_memory_model.pt', weights_only=True)['model_state_dict'])
    rollout_results, predictions, ground_truth = evaluate_rollout(model, val_dataset, device, num_steps=48)

    logger.info("\nRollout RMSE:")
    for lead_time, rmse in rollout_results.items():
        logger.info(f"  {lead_time}: {rmse:.4f} m")

    # Save plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].semilogy(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].semilogy(history['val_loss'], label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress (Temporal Memory GNN)')
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
    plt.savefig(figure_dir / 'temporal_memory_training_summary.png', dpi=150)
    plt.close()

    logger.info(f"\nModel saved to: {checkpoint_dir / 'best_temporal_memory_model.pt'}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
