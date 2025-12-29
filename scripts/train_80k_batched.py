#!/usr/bin/env python3
"""
TRUE BATCHED Training Script for 80k nodes

Key optimization: Process entire batches in ONE forward/backward pass
instead of looping over samples.

Expected speedup: 5-10x faster than per-sample processing
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
# CONFIGURATION
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/home/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/home/Mansur.Jisan/stofs_surrogate'))

DOMAIN_NAME = "Option_A_LI_Sound_to_S_Maine"
DOMAIN_BBOX = "40-44°N, 74-69°W"
NUM_NODES = 80000

VAL_YEAR = '2025'

# Model architecture
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 6
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

# Training parameters - TRUE BATCHING
BATCH_SIZE = 2              # Limited by GPU memory (581k edges × batch × hidden)
EPOCHS = 100
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 0
USE_AMP = True
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 100      # Log frequently to see progress

# Curriculum
CURRICULUM_WARMUP_EPOCHS = 15
MAX_ROLLOUT_STEPS = 3

# Physics loss weights
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Normalization
ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0

# Tidal
M2_PERIOD = 12.42
S2_PERIOD = 12.00
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Lazy loading cache
LAZY_CACHE_SIZE = 20


# ============================================================
# BATCHED Model - Processes [B, N, F] tensors
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    """
    Message passing block that handles BATCHED node features.
    Input shape: [B, N, hidden_dim]
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
            edge_index: [2, E] - shared edge structure
            edge_attr: [E, hidden_dim] - edge features (shared across batch)
        Returns:
            h_new: [B, N, hidden_dim]
        """
        B, N, H = h.shape
        row, col = edge_index  # [E]
        E = row.shape[0]

        # Gather source and destination node features for all edges
        # h[:, row, :] -> [B, E, H]
        h_src = h[:, row, :]  # [B, E, H]
        h_dst = h[:, col, :]  # [B, E, H]

        # Compute gradient (destination - source)
        h_gradient = h_dst - h_src  # [B, E, H]

        # Expand edge_attr for batch: [E, H] -> [B, E, H]
        edge_attr_expanded = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, H]

        # Edge input: concat along feature dimension
        edge_input = torch.cat([edge_attr_expanded, h_src, h_dst, h_gradient], dim=-1)  # [B, E, 4*H]

        # Process edges - reshape for MLP
        edge_input_flat = edge_input.reshape(B * E, -1)  # [B*E, 4*H]
        edge_msg_flat = self.edge_mlp(edge_input_flat)   # [B*E, H]
        edge_msg = edge_msg_flat.reshape(B, E, H)        # [B, E, H]

        # Apply gradient gating
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)  # [B, E, H]
        edge_msg = edge_msg * (1.0 + gradient_gate)

        # Normalize edge messages
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        # Aggregate messages to nodes using scatter_add
        # We need to scatter [B, E, H] to [B, N, H] using row indices
        aggr = torch.zeros(B, N, H, device=h.device, dtype=h.dtype)

        # Expand row indices for batched scatter
        row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, -1, H)  # [B, E, H]
        aggr.scatter_add_(1, row_expanded, edge_msg)

        # Node update
        node_input = torch.cat([h, aggr], dim=-1)  # [B, N, 2*H]
        node_input_flat = node_input.reshape(B * N, -1)  # [B*N, 2*H]
        node_update_flat = self.node_mlp(node_input_flat)  # [B*N, H]
        node_update = node_update_flat.reshape(B, N, H)    # [B, N, H]

        h_new = h + node_update

        return h_new


class BatchedTemporalMemoryGNN(nn.Module):
    """
    GNN that processes BATCHED inputs for efficient training.
    All inputs have shape [B, N, F] instead of [N, F].
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
            BatchedSWEGraphBlock(hidden_dim)
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
        """
        Args:
            x: [B, N, 1] - current state
            x_prev: [B, N, 1] - previous state
            dxdt: [B, N, 1] - rate of change
            tidal_harmonics: [B, N, 4] - tidal features
            static_features: [B, N, 4] - static node features
            forcing: [B, N, 3] - forcing features
            edge_index: [2, E] - edge structure (shared)
            edge_attr: [E, 3] - edge features (shared)
        Returns:
            output: [B, N, 1] - predicted next state
        """
        B, N, _ = x.shape

        # Concatenate all node features
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)  # [B, N, 14]

        # Encode nodes - reshape for MLP
        node_features_flat = node_features.reshape(B * N, -1)  # [B*N, 14]
        h_flat = self.node_encoder(node_features_flat)          # [B*N, H]
        h = h_flat.reshape(B, N, self.hidden_dim)               # [B, N, H]

        # Encode edges (shared across batch)
        e = self.edge_encoder(edge_attr)  # [E, H]

        # Message passing layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index, e)

        # Decode
        h_flat = h.reshape(B * N, -1)        # [B*N, H]
        delta_flat = self.decoder(h_flat)    # [B*N, 1]
        delta = delta_flat.reshape(B, N, 1)  # [B, N, 1]

        output = x + delta

        return output


# ============================================================
# BATCHED Physics Loss
# ============================================================

class BatchedPhysicsLoss(nn.Module):
    def __init__(self, mass_weight=0.01, smooth_weight=0.01):
        super().__init__()
        self.mass_weight = mass_weight
        self.smooth_weight = smooth_weight

    def forward(self, pred, target, edge_index):
        """
        Args:
            pred: [B, N, 1]
            target: [B, N, 1]
            edge_index: [2, E]
        """
        B, N, _ = pred.shape
        row, col = edge_index

        # MSE loss
        mse_loss = ((pred - target) ** 2).mean()

        # Mass conservation (per sample, then average)
        pred_sum = pred.sum(dim=1)      # [B, 1]
        target_sum = target.sum(dim=1)  # [B, 1]
        mass_diff = (pred_sum - target_sum).abs() / (N + 1e-8)
        mass_loss = torch.clamp(mass_diff.mean(), max=10.0)

        # Smoothness loss
        pred_src = pred[:, row, :]  # [B, E, 1]
        pred_dst = pred[:, col, :]  # [B, E, 1]
        smooth_loss = ((pred_src - pred_dst) ** 2).mean()

        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss

        return total, {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smooth_loss.item()
        }


# ============================================================
# Lazy Dataset (same as before)
# ============================================================

class LazyTemporalMemoryDataset(Dataset):
    """Lazy loading dataset - loads files on demand."""

    def __init__(self, mesh_data: Dict, date_file_paths: List[tuple],
                 eta_scale: float = 2.0, dt_hours: float = 1.0, cache_size: int = 20):
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

        self.date_labels = [d[0] for d in date_file_paths]
        self.file_paths = [d[1] for d in date_file_paths]

        self._cache = {}
        self._cache_order = []

        self._compute_global_times()

        self.samples = []
        self._timesteps_per_file = {}

        for date_idx in range(len(self.date_labels)):
            data = np.load(self.file_paths[date_idx])
            num_times = data['elevation'].shape[0]
            self._timesteps_per_file[date_idx] = num_times
            data.close()

            for t in range(1, num_times - 3):
                self.samples.append((date_idx, t))

        logger.info(f"LazyDataset: {len(self.samples):,} samples from {len(date_file_paths)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")
        logger.info(f"  Cache: {cache_size} files")

    def _compute_global_times(self):
        self.global_hours_offset = []
        for date_str in self.date_labels:
            date_dt = datetime.strptime(date_str, '%Y%m%d')
            hours_from_epoch = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
            self.global_hours_offset.append(hours_from_epoch)

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

    def _get_file_data(self, date_idx: int):
        if date_idx in self._cache:
            self._cache_order.remove(date_idx)
            self._cache_order.append(date_idx)
            return self._cache[date_idx]

        data = np.load(self.file_paths[date_idx])
        elevation = data['elevation']
        forcing = {
            'u10': data['u10'],
            'v10': data['v10'],
            'pressure': data['pressure'],
        }
        data.close()

        self._cache[date_idx] = (elevation, forcing)
        self._cache_order.append(date_idx)

        while len(self._cache_order) > self.cache_size:
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
            gc.collect()

        return self._cache[date_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date_idx, t = self.samples[idx]
        elev, forcing = self._get_file_data(date_idx)

        cwl_t = np.nan_to_num(elev[t].astype(np.float32), nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        cwl_prev = np.nan_to_num(elev[t-1].astype(np.float32), nan=0.0)
        cwl_prev_norm = cwl_prev / self.eta_scale
        dxdt = (cwl_norm - cwl_prev_norm) / self.dt_hours

        global_hour_t = self.global_hours_offset[date_idx] + t * self.dt_hours
        tidal_t = self._compute_tidal_harmonics(global_hour_t)
        tidal_harmonics = np.tile(tidal_t, (self.num_nodes, 1))
        tidal_t1 = np.tile(self._compute_tidal_harmonics(global_hour_t + self.dt_hours), (self.num_nodes, 1))
        tidal_t2 = np.tile(self._compute_tidal_harmonics(global_hour_t + 2*self.dt_hours), (self.num_nodes, 1))

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
# TRUE BATCHED Training Functions
# ============================================================

def train_epoch_batched(model, loader, optimizer, criterion, device, num_steps,
                        grad_clip, scaler, amp_ctx, log_every):
    """
    TRUE BATCHED training - ONE forward/backward pass per batch.
    """
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()

        batch_size = batch['x'].shape[0]

        # Move entire batch to GPU at once
        x = batch['x'].to(device)                           # [B, N, 1]
        x_prev = batch['x_prev'].to(device)                 # [B, N, 1]
        dxdt = batch['dxdt'].to(device)                     # [B, N, 1]
        tidal = batch['tidal_harmonics'].to(device)         # [B, N, 4]
        static = batch['static'].to(device)                 # [B, N, 4]
        forcing = batch['forcing'].to(device)               # [B, N, 3]
        y = batch['y'].to(device)                           # [B, N, 1]
        raw_depth = batch['raw_depth'].to(device)           # [B, N, 1]

        edge_index = batch['edge_index'][0].to(device)      # [2, E] - shared
        edge_attr = batch['edge_attr'][0].to(device)        # [E, 3] - shared

        with amp_ctx():
            # ONE forward pass for entire batch!
            pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
            loss, components = criterion(pred, y, edge_index)

            # Multi-step rollout
            if num_steps >= 2:
                y_t2 = batch['y_t2'].to(device)
                forcing_t1 = batch['forcing_t1'].to(device)
                tidal_t1 = batch['tidal_harmonics_t1'].to(device)

                pred_detach = pred.detach()
                dxdt_new = (pred_detach - x) / DT_HOURS
                pred_meters = pred_detach * ETA_SCALE
                wl_physical = raw_depth + pred_meters
                wl_mean = wl_physical.mean(dim=1, keepdim=True)
                wl_std = wl_physical.std(dim=1, keepdim=True) + 1e-8
                wl_norm = (wl_physical - wl_mean) / wl_std
                static_new = torch.cat([static[:, :, :3], wl_norm], dim=-1)

                pred2 = model(pred_detach, x, dxdt_new, tidal_t1, static_new, forcing_t1, edge_index, edge_attr)
                loss2, _ = criterion(pred2, y_t2, edge_index)
                loss = loss + 0.5 * loss2

            if num_steps >= 3:
                y_t3 = batch['y_t3'].to(device)
                forcing_t2 = batch['forcing_t2'].to(device)
                tidal_t2 = batch['tidal_harmonics_t2'].to(device)

                pred2_detach = pred2.detach()
                dxdt_new2 = (pred2_detach - pred_detach) / DT_HOURS
                pred2_meters = pred2_detach * ETA_SCALE
                wl_physical2 = raw_depth + pred2_meters
                wl_mean2 = wl_physical2.mean(dim=1, keepdim=True)
                wl_std2 = wl_physical2.std(dim=1, keepdim=True) + 1e-8
                wl_norm2 = (wl_physical2 - wl_mean2) / wl_std2
                static_new2 = torch.cat([static[:, :, :3], wl_norm2], dim=-1)

                pred3 = model(pred2_detach, pred_detach, dxdt_new2, tidal_t2, static_new2, forcing_t2, edge_index, edge_attr)
                loss3, _ = criterion(pred3, y_t3, edge_index)
                loss = loss + 0.25 * loss3

        # ONE backward pass for entire batch!
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        for k in components:
            total_components[k] += components[k]
        num_batches += 1

        # Progress logging
        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            batches_per_sec = (batch_idx + 1) / elapsed
            samples_per_sec = batches_per_sec * batch_size
            eta_minutes = (len(loader) - batch_idx - 1) / batches_per_sec / 60
            logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                       f"Loss: {loss.item():.5f} | "
                       f"Speed: {samples_per_sec:.1f} samples/s | "
                       f"ETA: {eta_minutes:.1f} min")

        # Clear GPU cache periodically to prevent fragmentation
        if (batch_idx + 1) % 500 == 0:
            torch.cuda.empty_cache()

    return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}


def validate_batched(model, loader, criterion, device, amp_ctx):
    """Batched validation."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            x_prev = batch['x_prev'].to(device)
            dxdt = batch['dxdt'].to(device)
            tidal = batch['tidal_harmonics'].to(device)
            static = batch['static'].to(device)
            forcing = batch['forcing'].to(device)
            y = batch['y'].to(device)

            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            with amp_ctx():
                pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)
                loss, components = criterion(pred, y, edge_index)

            total_loss += loss.item()
            for k in components:
                total_components[k] += components[k]
            num_batches += 1

    return total_loss / num_batches, {k: v / num_batches for k, v in total_components.items()}


def evaluate_rollout(model, dataset, device, num_steps=48, amp_ctx=None):
    """Evaluate autoregressive rollout."""
    model.eval()

    if amp_ctx is None:
        amp_ctx = lambda: nullcontext()

    eval_date = dataset.date_labels[0]
    logger.info(f"Rollout on: {eval_date}")

    elev, forcing = dataset._get_file_data(0)
    global_hours_offset = dataset.global_hours_offset[0]

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    predictions = []
    ground_truth = []

    cwl_prev = np.nan_to_num(elev[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elev[1].astype(np.float32), nan=0.0)

    # Add batch dimension for model
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, N, 1]
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)      # [1, N, 1]

    with torch.no_grad():
        for t in range(1, min(num_steps + 1, len(elev) - 1)):
            dxdt = (current_cwl - current_prev) / DT_HOURS

            global_hour_t = global_hours_offset + t * DT_HOURS
            tidal_t = dataset._compute_tidal_harmonics(global_hour_t)
            tidal_harmonics = np.tile(tidal_t, (dataset.num_nodes, 1))
            tidal_tensor = torch.tensor(tidal_harmonics, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 4]

            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = dataset.depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([dataset.static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 4]

            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pres = forcing['pressure'][t].astype(np.float32)
            forcing_arr = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 3]

            with amp_ctx():
                pred = model(current_cwl, current_prev, dxdt, tidal_tensor, static_tensor, forcing_tensor, edge_index, edge_attr)

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
    logger.info("TRUE BATCHED TRAINING - 80k nodes")
    logger.info("One forward/backward pass per batch (5-10x faster)")
    logger.info("=" * 70)

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_batched'
    figure_dir = OUTPUT_DIR / 'outputs' / 'figures_80k_batched'
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
        (f.stem.replace('processed_', ''), str(f))
        for f in DATA_DIR.glob('processed_*.npz')
        if 'mesh' not in f.stem
    ])

    train_dates = [(d, p) for d, p in available_dates if not d.startswith(VAL_YEAR)]
    val_dates = [(d, p) for d, p in available_dates if d.startswith(VAL_YEAR)]

    logger.info(f"\nTrain dates: {len(train_dates)}")
    logger.info(f"Val dates: {len(val_dates)}")

    # Create datasets
    train_dataset = LazyTemporalMemoryDataset(
        mesh_data, train_dates,
        eta_scale=ETA_SCALE, dt_hours=DT_HOURS, cache_size=LAZY_CACHE_SIZE
    )
    val_dataset = LazyTemporalMemoryDataset(
        mesh_data, val_dates,
        eta_scale=ETA_SCALE, dt_hours=DT_HOURS, cache_size=LAZY_CACHE_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    logger.info(f"\nTrain: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    logger.info(f"Val: {len(val_dataset):,} samples, {len(val_loader):,} batches")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Model - BATCHED VERSION
    model = BatchedTemporalMemoryGNN(
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
    criterion = BatchedPhysicsLoss(MASS_CONSERVATION_WEIGHT, SMOOTHNESS_WEIGHT)

    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    amp_ctx = lambda: autocast('cuda') if USE_AMP and device.type == 'cuda' else nullcontext()

    history = {'train_loss': [], 'val_loss': [], 'mse': [], 'mass': [], 'lr': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume
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
    logger.info(f"Batches per epoch: {len(train_loader):,} (vs {len(train_dataset):,} with per-sample)")
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

        train_loss, train_comp = train_epoch_batched(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP, scaler=scaler,
            amp_ctx=amp_ctx, log_every=LOG_EVERY_N_BATCHES
        )

        val_loss, val_comp = validate_batched(model, val_loader, criterion, device, amp_ctx)

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
                    'batch_type': 'true_batched',
                }
            }, checkpoint_dir / 'best_model.pt')
            logger.info(f"  ★ New best model!")

        logger.info(f"  EPOCH {epoch} DONE | train={train_loss:.5f} | val={val_loss:.5f} | "
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
            logger.info(f"  Checkpoint saved: epoch {epoch}")

            # Clean old
            old_ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))[:-3]
            for old in old_ckpts:
                old.unlink()

    total_time = time.time() - total_start

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best val loss: {best_val_loss:.6f}")

    # Rollout evaluation
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
    axes[0, 0].set_title('Training Progress - 80k Batched')
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

    if rollout_results:
        lead_times = [int(k.split('+')[1].replace('h', '')) for k in rollout_results.keys()]
        rmse_values = list(rollout_results.values())
        axes[1, 1].bar(range(len(lead_times)), rmse_values, tick_label=[f't+{lt}h' for lt in lead_times])
        axes[1, 1].set_xlabel('Lead Time')
        axes[1, 1].set_ylabel('RMSE (m)')
        axes[1, 1].set_title('Rollout Performance')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(figure_dir / 'training_summary_80k_batched.png', dpi=150)
    plt.close()

    logger.info(f"\nModel saved to: {checkpoint_dir / 'best_model.pt'}")
    logger.info(f"Figures saved to: {figure_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
