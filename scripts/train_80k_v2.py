#!/usr/bin/env python3
"""
Training Script for STOFS Surrogate - 80k nodes V2 WITH PHYSICS ENHANCEMENTS

Enhanced Physics Features:
1. 6 Tidal Constituents: M2, S2, N2, K1, O1, M4 (12 temporal features)
2. Enhanced Wind: u10, v10, wind_speed, wind_speed_sq, wind_dir (5 features)
3. Enhanced Pressure: pressure_anomaly, dP_dx, dP_dy (3 features)

Total forcing features: 8 (was 3)
Total temporal features: 12 (was 4)

This script is designed to work with data from preprocess_80k_v2.py

Usage:
    STOFS_DATA_DIR=/path/to/processed_80k_v2 python scripts/train_80k_v2.py
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
torch.set_float32_matmul_precision('high')

# ============================================================
# CONFIGURATION - V2 Enhanced Physics
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate/data/processed_80k_v2'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/scratch5/purged/Mansur.Jisan/stofs_surrogate'))

VAL_YEAR = '2025'
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1

# V2: Enhanced feature dimensions
TEMPORAL_FEATURES = 12  # 6 tidal constituents * 2 (sin/cos) = 12
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8  # u10, v10, wind_speed, wind_speed_sq, wind_dir, pressure_anomaly, dP_dx, dP_dy

EPOCHS = 100
BATCH_SIZE = 4  # Small batch size due to 80k nodes
GRAD_ACCUM_STEPS = 8  # Effective batch = 4 * 8 = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 4
USE_AMP = True
USE_COMPILE = False  # Disabled - Triton doesn't support 80k nodes
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 50

CURRICULUM_WARMUP_EPOCHS = 15
MAX_ROLLOUT_STEPS = 3
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Tidal constituent periods (hours) - STOFS uses 8+ but we use 6 major ones
TIDAL_PERIODS = {
    'M2': 12.42,   # Principal lunar semidiurnal
    'S2': 12.00,   # Principal solar semidiurnal
    'N2': 12.66,   # Larger lunar elliptic semidiurnal
    'K1': 23.93,   # Lunisolar diurnal
    'O1': 25.82,   # Principal lunar diurnal
    'M4': 6.21,    # Shallow water quarter-diurnal
}


# ============================================================
# TRUE BATCHED Model Architecture - Processes [B, N, F] tensors
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    """
    TRUE batched GNN block that processes [B, N, F] tensors in ONE forward pass.
    Edge messages are computed for all batch samples simultaneously.
    """
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
        """
        Args:
            h: [B, N, hidden_dim] - batched node features
            edge_index: [2, E] - shared edge indices
            edge_attr: [E, hidden_dim] - shared edge features
        Returns:
            h_new: [B, N, hidden_dim]
        """
        B, N, F = h.shape
        row, col = edge_index  # [E]
        E = row.shape[0]

        # Gather source and destination node features for all edges and all batches
        h_src = h[:, row, :]  # [B, E, F]
        h_dst = h[:, col, :]  # [B, E, F]

        # Compute gradient
        h_gradient = h_dst - h_src  # [B, E, F]

        # Expand edge_attr for batch: [E, F] -> [B, E, F]
        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, F]

        # Concatenate edge inputs
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1)  # [B, E, 4*F]

        # Process through edge MLP (reshape for batch processing)
        edge_input_flat = edge_input.reshape(B * E, -1)  # [B*E, 4*F]
        edge_msg_flat = self.edge_mlp(edge_input_flat)  # [B*E, F]
        edge_msg = edge_msg_flat.reshape(B, E, F)  # [B, E, F]

        # Apply gradient gating
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)  # [B, E, F]
        edge_msg = edge_msg * (1.0 + gradient_gate)  # [B, E, F]

        # Normalize edge messages
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        # Aggregate messages to nodes using scatter_add
        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, E, F)  # [B, E, F]
        aggr.scatter_add_(1, row_expanded, edge_msg)  # [B, N, F]

        # Node update
        node_input = torch.cat([h, aggr], dim=-1)  # [B, N, 2*F]
        node_input_flat = node_input.reshape(B * N, -1)  # [B*N, 2*F]
        node_out_flat = self.node_mlp(node_input_flat)  # [B*N, F]
        node_out = node_out_flat.reshape(B, N, F)  # [B, N, F]

        h_new = h + node_out  # Residual connection
        return h_new, edge_attr


class BatchedTemporalMemoryGNN_V2(nn.Module):
    """
    V2: Enhanced GNN model with expanded tidal and forcing features.

    Input features:
    - state: [B, N, 1] current elevation
    - x_prev: [B, N, 1] previous elevation
    - dxdt: [B, N, 1] rate of change
    - tidal_harmonics: [B, N, 12] - 6 constituents * 2 (sin/cos)
    - static_features: [B, N, 4] - x, y, depth, water_level
    - forcing: [B, N, 8] - enhanced forcing

    Total input: 1 + 1 + 1 + 12 + 4 + 8 = 27 features per node
    """
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # V2: x (1) + x_prev (1) + dxdt (1) + tidal (12) + static (4) + forcing (8) = 27
        node_input_dim = state_dim * 3 + temporal_dim + static_feature_dim + forcing_feature_dim

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
        TRUE batched forward pass - processes all B samples in ONE pass.

        Args:
            x: [B, N, 1] current state
            x_prev: [B, N, 1] previous state
            dxdt: [B, N, 1] rate of change
            tidal_harmonics: [B, N, 12] tidal features (6 constituents)
            static_features: [B, N, 4] static node features
            forcing: [B, N, 8] enhanced forcing features
            edge_index: [2, E] edge connectivity (shared across batch)
            edge_attr: [E, 3] edge features (shared across batch)

        Returns:
            pred: [B, N, 1] predicted next state
        """
        B = x.shape[0]

        # Concatenate all node features: [B, N, total_features]
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)

        # Encode nodes: reshape to [B*N, F], process, reshape back
        B, N, F_in = node_features.shape
        node_flat = node_features.reshape(B * N, F_in)
        h_flat = self.node_encoder(node_flat)
        h = h_flat.reshape(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        # Encode edges (shared across batch)
        e = self.edge_encoder(edge_attr)  # [E, hidden_dim]

        # Process through GNN layers
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        # Decode to state
        h_flat = h.reshape(B * N, self.hidden_dim)
        delta_flat = self.decoder(h_flat)
        delta = delta_flat.reshape(B, N, -1)  # [B, N, state_dim]

        return x + delta


class PhysicsLoss(nn.Module):
    def __init__(self, mass_weight=0.01, smooth_weight=0.01):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, edge_index):
        """
        Batched physics loss.
        Args:
            pred: [B, N, 1]
            target: [B, N, 1]
            edge_index: [2, E]
        """
        mse_loss = ((pred - target) ** 2).mean()

        # Mass conservation (per batch)
        pred_sum = pred.sum(dim=(1, 2))  # [B]
        target_sum = target.sum(dim=(1, 2))  # [B]
        mass_diff = (pred_sum - target_sum).abs().mean() / (pred.shape[1] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)

        # Smoothness
        row, col = edge_index
        smooth_loss = ((pred[:, row, :] - pred[:, col, :]) ** 2).mean()

        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss
        return total, {'mse': mse_loss.item(), 'mass': mass_loss.item(), 'smooth': smooth_loss.item()}


# ============================================================
# V2 In-Memory Dataset with Enhanced Features
# ============================================================

class InMemoryDatasetV2(Dataset):
    """Load ALL data into RAM with V2 enhanced features."""

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

        # Build sample index
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(1, num_times - 3):
                self.samples.append((date_idx, t))

        logger.info(f"InMemoryDatasetV2: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Features: {TEMPORAL_FEATURES} tidal + {FORCING_FEATURES} forcing")

    def _compute_global_times(self):
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours)

    def _compute_tidal_harmonics_v2(self, global_hour: float) -> np.ndarray:
        """
        V2: Compute harmonics for all 6 tidal constituents.
        Returns [12] array: [sin_M2, cos_M2, sin_S2, cos_S2, sin_N2, cos_N2,
                            sin_K1, cos_K1, sin_O1, cos_O1, sin_M4, cos_M4]
        """
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

        # V2: Enhanced tidal harmonics (6 constituents = 12 features)
        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics_v2(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))  # [N, 12]

        # Future tidal harmonics for rollout
        tidal_t1 = np.tile(self._compute_tidal_harmonics_v2(global_hour_t + self.dt_hours), (self.num_nodes, 1))
        tidal_t2 = np.tile(self._compute_tidal_harmonics_v2(global_hour_t + 2*self.dt_hours), (self.num_nodes, 1))

        # Static features with water level
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)

        # V2: Enhanced forcing (8 features)
        # Note: v2 preprocessing outputs normalized features
        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        wind_speed = forcing['wind_speed'][t].astype(np.float32) / WIND_SCALE
        wind_speed_sq = forcing['wind_speed_sq'][t].astype(np.float32) / (WIND_SCALE ** 2)
        wind_dir = forcing['wind_dir'][t].astype(np.float32) / np.pi  # Normalize to [-1, 1]
        pres_anom = forcing['pressure_anomaly'][t].astype(np.float32)  # Already normalized
        dP_dx = forcing['dP_dx'][t].astype(np.float32)
        dP_dy = forcing['dP_dy'][t].astype(np.float32)

        forcing_arr = np.stack([u10, v10, wind_speed, wind_speed_sq, wind_dir,
                                pres_anom, dP_dx, dP_dy], axis=1)  # [N, 8]

        # Target states
        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t3 = np.nan_to_num(elev[t+3].astype(np.float32), nan=0.0) / self.eta_scale

        # Future forcing for rollout
        forcing_t1 = np.stack([
            forcing['u10'][t+1].astype(np.float32) / WIND_SCALE,
            forcing['v10'][t+1].astype(np.float32) / WIND_SCALE,
            forcing['wind_speed'][t+1].astype(np.float32) / WIND_SCALE,
            forcing['wind_speed_sq'][t+1].astype(np.float32) / (WIND_SCALE ** 2),
            forcing['wind_dir'][t+1].astype(np.float32) / np.pi,
            forcing['pressure_anomaly'][t+1].astype(np.float32),
            forcing['dP_dx'][t+1].astype(np.float32),
            forcing['dP_dy'][t+1].astype(np.float32),
        ], axis=1)

        forcing_t2 = np.stack([
            forcing['u10'][t+2].astype(np.float32) / WIND_SCALE,
            forcing['v10'][t+2].astype(np.float32) / WIND_SCALE,
            forcing['wind_speed'][t+2].astype(np.float32) / WIND_SCALE,
            forcing['wind_speed_sq'][t+2].astype(np.float32) / (WIND_SCALE ** 2),
            forcing['wind_dir'][t+2].astype(np.float32) / np.pi,
            forcing['pressure_anomaly'][t+2].astype(np.float32),
            forcing['dP_dx'][t+2].astype(np.float32),
            forcing['dP_dy'][t+2].astype(np.float32),
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
# TRUE BATCHED Training with Gradient Accumulation
# ============================================================

def train_epoch_batched(model, loader, optimizer, criterion, device, num_steps,
                        grad_clip, scaler, use_amp, grad_accum_steps, log_every):
    """
    TRUE BATCHED training with gradient accumulation.
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

    for batch_idx, batch in enumerate(loader):
        # Shared graph structure
        edge_index = batch['edge_index'][0].to(device, non_blocking=True)
        edge_attr = batch['edge_attr'][0].to(device, non_blocking=True)

        # Move batch data to GPU - shape [B, N, F]
        x = batch['x'].to(device, non_blocking=True)
        x_prev = batch['x_prev'].to(device, non_blocking=True)
        dxdt = batch['dxdt'].to(device, non_blocking=True)
        tidal = batch['tidal_harmonics'].to(device, non_blocking=True)
        static = batch['static'].to(device, non_blocking=True)
        forcing = batch['forcing'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        raw_depth = batch['raw_depth'].to(device, non_blocking=True)

        with amp_ctx:
            # TRUE BATCHED forward - ONE call for entire batch!
            pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
            loss, comp = criterion(pred, y, edge_index)

            # Multi-step rollout (still batched!)
            if num_steps >= 2:
                y_t2 = batch['y_t2'].to(device, non_blocking=True)
                forcing_t1 = batch['forcing_t1'].to(device, non_blocking=True)
                tidal_t1 = batch['tidal_harmonics_t1'].to(device, non_blocking=True)

                pred_d = pred.detach()
                dxdt_new = (pred_d - x) / DT_HOURS
                wl = raw_depth + pred_d * ETA_SCALE
                wl_mean = wl.mean(dim=1, keepdim=True)
                wl_std = wl.std(dim=1, keepdim=True) + 1e-8
                wl_norm = (wl - wl_mean) / wl_std
                static_new = torch.cat([static[:, :, :3], wl_norm], dim=-1)

                pred2 = model(pred_d, x, dxdt_new, tidal_t1, static_new, forcing_t1, edge_index, edge_attr)
                loss2, _ = criterion(pred2, y_t2, edge_index)
                loss = loss + 0.5 * loss2

            if num_steps >= 3:
                y_t3 = batch['y_t3'].to(device, non_blocking=True)
                forcing_t2 = batch['forcing_t2'].to(device, non_blocking=True)
                tidal_t2 = batch['tidal_harmonics_t2'].to(device, non_blocking=True)

                pred2_d = pred2.detach()
                dxdt_new2 = (pred2_d - pred_d) / DT_HOURS
                wl2 = raw_depth + pred2_d * ETA_SCALE
                wl2_mean = wl2.mean(dim=1, keepdim=True)
                wl2_std = wl2.std(dim=1, keepdim=True) + 1e-8
                wl_norm2 = (wl2 - wl2_mean) / wl2_std
                static_new2 = torch.cat([static[:, :, :3], wl_norm2], dim=-1)

                pred3 = model(pred2_d, pred_d, dxdt_new2, tidal_t2, static_new2, forcing_t2, edge_index, edge_attr)
                loss3, _ = criterion(pred3, y_t3, edge_index)
                loss = loss + 0.25 * loss3

            # Scale for gradient accumulation
            scaled_loss = loss / grad_accum_steps

        # Backward pass (accumulates gradients)
        scaler.scale(scaled_loss).backward()

        accumulated_loss += loss.item()
        for k in comp:
            accum_comp[k] += comp[k]

        # Step optimizer after accumulating enough gradients
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
                           f"Loss: {avg_loss:.5f} | "
                           f"Speed: {samples_per_sec:.1f} samp/s | "
                           f"ETA: {eta_min:.1f} min")

            accumulated_loss = 0
            accum_comp = {'mse': 0, 'mass': 0, 'smooth': 0}

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
    return avg_loss, avg_comp


def validate(model, loader, criterion, device, use_amp):
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
            y = batch['y'].to(device)

            with amp_ctx:
                pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                loss, _ = criterion(pred, y, edge_index)

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
    logger.info("V2 PHYSICS-ENHANCED TRAINING - 80k nodes")
    logger.info("=" * 70)
    logger.info(f"System RAM: {ram_gb:.1f} GB")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"GRAD_ACCUM_STEPS: {GRAD_ACCUM_STEPS}")
    logger.info(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    logger.info(f"TEMPORAL_FEATURES: {TEMPORAL_FEATURES} (6 tidal constituents)")
    logger.info(f"FORCING_FEATURES: {FORCING_FEATURES} (enhanced wind + pressure)")
    logger.info(f"USE_COMPILE: {USE_COMPILE}")
    logger.info(f"NUM_WORKERS: {NUM_WORKERS}")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_v2'
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
    logger.info("\nLoading ALL training data into memory (V2 features)...")
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
                'pressure_anomaly': data['pressure_anomaly'],
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
                'pressure_anomaly': data['pressure_anomaly'],
                'dP_dx': data['dP_dx'],
                'dP_dy': data['dP_dy'],
            }
        })
    logger.info(f"  Loaded {len(val_data)} validation dates")

    # Create datasets
    train_dataset = InMemoryDatasetV2(mesh_data, train_data, ETA_SCALE, DT_HOURS)
    val_dataset = InMemoryDatasetV2(mesh_data, val_data, ETA_SCALE, DT_HOURS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True if NUM_WORKERS > 0 else False)

    logger.info(f"\nTrain samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")
    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Effective optimizer steps per epoch: {len(train_loader) // GRAD_ACCUM_STEPS}")

    # Device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = BatchedTemporalMemoryGNN_V2(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # torch.compile disabled for 80k nodes (Triton limit)
    if USE_COMPILE and device.type == 'cuda':
        logger.info("Applying torch.compile...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("  torch.compile applied successfully!")
        except Exception as e:
            logger.warning(f"  torch.compile failed, continuing without: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else GradScaler('cpu')

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume from checkpoint
    if RESUME_FROM_CHECKPOINT:
        ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if ckpts:
            ckpt = torch.load(ckpts[-1], weights_only=False)
            state_dict = ckpt['model_state_dict']
            try:
                model.load_state_dict(state_dict)
            except:
                # Handle compiled model state dict
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
    logger.info("STARTING V2 PHYSICS-ENHANCED TRAINING")
    logger.info("=" * 70)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        # Curriculum: rollout steps
        if epoch <= CURRICULUM_WARMUP_EPOCHS:
            num_steps = 1
        elif epoch <= 2 * CURRICULUM_WARMUP_EPOCHS:
            num_steps = 2
        else:
            num_steps = 3

        logger.info(f"\nEpoch {epoch}/{EPOCHS} | rollout_steps={num_steps}")

        train_loss, train_comp = train_epoch_batched(
            model, train_loader, optimizer, criterion,
            device, num_steps, GRAD_CLIP, scaler, USE_AMP, GRAD_ACCUM_STEPS, LOG_EVERY_N_BATCHES
        )
        val_loss = validate(model, val_loader, criterion, device, USE_AMP)

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
                    'temporal_dim': TEMPORAL_FEATURES,
                    'forcing_dim': FORCING_FEATURES,
                    'version': 'v2_physics_enhanced',
                }
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  New best!")

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
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'num_nodes': len(mesh_data['lon']),
                    'temporal_dim': TEMPORAL_FEATURES,
                    'forcing_dim': FORCING_FEATURES,
                    'version': 'v2_physics_enhanced',
                }
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
