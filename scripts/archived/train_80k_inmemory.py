#!/usr/bin/env python3
"""
Training Script for STOFS Surrogate - 80k nodes with IN-MEMORY data loading

This is the FASTEST approach - loads ALL data into RAM upfront.
No I/O during training = maximum GPU utilization.

RAM Requirements:
- 253 training dates × ~330 MB = ~83 GB
- Plus model/PyTorch overhead = ~10 GB
- TOTAL: ~95 GB RAM minimum

Recommended instances:
- g5.8xlarge:  128 GB RAM, 1 A10G, $2.45/hr  ← RECOMMENDED
- g5.12xlarge: 192 GB RAM, 4 A10G, $5.67/hr
- g5.16xlarge: 256 GB RAM, 1 A10G, $4.10/hr

Usage:
    python scripts/train_80k_inmemory.py
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/home/Mansur.Jisan/stofs_surrogate/data/processed_80k_option_a'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', '/home/Mansur.Jisan/stofs_surrogate'))

VAL_YEAR = '2025'
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 6
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 0
USE_AMP = True
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
LOG_EVERY_N_BATCHES = 500

CURRICULUM_WARMUP_EPOCHS = 20
MAX_ROLLOUT_STEPS = 3
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0
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
        mass_diff = (pred.sum() - target.sum()).abs() / (pred.shape[0] + 1e-8)
        mass_loss = torch.clamp(mass_diff, max=10.0)
        row, col = edge_index
        smooth_loss = ((pred[row] - pred[col]) ** 2).mean()
        total = mse_loss + self.mass_weight * mass_loss + self.smooth_weight * smooth_loss
        return total, {'mse': mse_loss.item(), 'mass': mass_loss.item(), 'smooth': smooth_loss.item()}


# ============================================================
# In-Memory Dataset - FASTEST (like 40k pilot)
# ============================================================

class InMemoryDataset(Dataset):
    """Load ALL data into RAM for maximum speed."""

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

        logger.info(f"InMemoryDataset: {len(self.samples):,} samples from {len(date_data_list)} dates")
        logger.info(f"  Nodes: {self.num_nodes:,}, Edges: {self.edge_index.shape[1]:,}")

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
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, num_steps,
                grad_clip, scaler, amp_ctx, log_every):
    model.train()
    total_loss = 0
    total_comp = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        batch_size = batch['x'].shape[0]
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

        batch_loss = 0
        for i in range(batch_size):
            optimizer.zero_grad()

            with amp_ctx():
                pred = model(x_batch[i], x_prev_batch[i], dxdt_batch[i],
                           tidal_batch[i], static_batch[i], forcing_batch[i],
                           edge_index, edge_attr)
                loss, comp = criterion(pred, y_batch[i], edge_index)

                if num_steps >= 2:
                    y_t2 = batch['y_t2'][i].to(device)
                    forcing_t1 = batch['forcing_t1'][i].to(device)
                    tidal_t1 = batch['tidal_harmonics_t1'][i].to(device)

                    pred_d = pred.detach()
                    dxdt_new = (pred_d - x_batch[i]) / DT_HOURS
                    wl = raw_depth_batch[i] + pred_d * ETA_SCALE
                    wl_norm = (wl - wl.mean()) / (wl.std() + 1e-8)
                    static_new = torch.cat([static_batch[i][:, :3], wl_norm], dim=1)

                    pred2 = model(pred_d, x_batch[i], dxdt_new, tidal_t1,
                                static_new, forcing_t1, edge_index, edge_attr)
                    loss2, _ = criterion(pred2, y_t2, edge_index)
                    loss = loss + 0.5 * loss2

                if num_steps >= 3:
                    y_t3 = batch['y_t3'][i].to(device)
                    forcing_t2 = batch['forcing_t2'][i].to(device)
                    tidal_t2 = batch['tidal_harmonics_t2'][i].to(device)

                    pred2_d = pred2.detach()
                    dxdt_new2 = (pred2_d - pred_d) / DT_HOURS
                    wl2 = raw_depth_batch[i] + pred2_d * ETA_SCALE
                    wl_norm2 = (wl2 - wl2.mean()) / (wl2.std() + 1e-8)
                    static_new2 = torch.cat([static_batch[i][:, :3], wl_norm2], dim=1)

                    pred3 = model(pred2_d, pred_d, dxdt_new2, tidal_t2,
                                static_new2, forcing_t2, edge_index, edge_attr)
                    loss3, _ = criterion(pred3, y_t3, edge_index)
                    loss = loss + 0.25 * loss3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_loss += loss.item()
            for k in comp:
                total_comp[k] += comp[k]

        total_loss += batch_loss
        num_samples += batch_size

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - start_time
            speed = num_samples / elapsed
            eta = (len(loader) - batch_idx - 1) / (batch_idx + 1) * elapsed / 60
            logger.info(f"    Batch {batch_idx+1}/{len(loader)} | "
                       f"Loss: {batch_loss/batch_size:.5f} | "
                       f"Speed: {speed:.1f} s/s | ETA: {eta:.1f} min")

    n = len(loader)
    return total_loss / num_samples, {k: v / n for k, v in total_comp.items()}


def validate(model, loader, criterion, device, amp_ctx):
    model.eval()
    total_loss = 0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)

            for i in range(batch['x'].shape[0]):
                with amp_ctx():
                    pred = model(batch['x'][i].to(device),
                               batch['x_prev'][i].to(device),
                               batch['dxdt'][i].to(device),
                               batch['tidal_harmonics'][i].to(device),
                               batch['static'][i].to(device),
                               batch['forcing'][i].to(device),
                               edge_index, edge_attr)
                    loss, _ = criterion(pred, batch['y'][i].to(device), edge_index)
                total_loss += loss.item()
                num_samples += 1

    return total_loss / num_samples


# ============================================================
# Main
# ============================================================

def main():
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)

    logger.info("=" * 70)
    logger.info("IN-MEMORY TRAINING - 80k nodes (All data in RAM)")
    logger.info("=" * 70)
    logger.info(f"System RAM: {ram_gb:.1f} GB")

    if ram_gb < 90:
        logger.warning(f"WARNING: {ram_gb:.1f} GB RAM may not be enough!")
        logger.warning("Recommended: 128 GB (g5.8xlarge) or higher")

    checkpoint_dir = OUTPUT_DIR / 'outputs' / 'checkpoints_80k_inmemory'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    logger.info(f"Mesh: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    # Discover and load ALL dates
    available_dates = sorted([
        (f.stem.replace('processed_', ''), f)
        for f in DATA_DIR.glob('processed_*.npz')
        if 'mesh' not in f.stem
    ])

    train_files = [(d, p) for d, p in available_dates if not d.startswith(VAL_YEAR)]
    val_files = [(d, p) for d, p in available_dates if d.startswith(VAL_YEAR)]

    logger.info(f"Training dates: {len(train_files)}")
    logger.info(f"Validation dates: {len(val_files)}")

    # Load ALL training data into memory
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
    for date_str, file_path in val_files[:30]:  # Use first 30 val dates
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

    # Create datasets
    train_dataset = InMemoryDataset(mesh_data, train_data, ETA_SCALE, DT_HOURS)
    val_dataset = InMemoryDataset(mesh_data, val_data, ETA_SCALE, DT_HOURS)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    logger.info(f"\nTrain samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model = TemporalMemoryGNN(
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
    scaler = GradScaler('cuda') if USE_AMP and device.type == 'cuda' else None
    amp_ctx = lambda: autocast('cuda') if USE_AMP and device.type == 'cuda' else nullcontext()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    start_epoch = 1

    # Resume
    if RESUME_FROM_CHECKPOINT:
        ckpts = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
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
    logger.info("STARTING TRAINING")
    logger.info("=" * 70)

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()

        num_steps = 1 if epoch <= CURRICULUM_WARMUP_EPOCHS else (2 if epoch <= 2*CURRICULUM_WARMUP_EPOCHS else 3)

        logger.info(f"\nEpoch {epoch}/{EPOCHS} | rollout_steps={num_steps}")

        train_loss, train_comp = train_epoch(model, train_loader, optimizer, criterion,
                                             device, num_steps, GRAD_CLIP, scaler, amp_ctx,
                                             LOG_EVERY_N_BATCHES)
        val_loss = validate(model, val_loader, criterion, device, amp_ctx)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {'hidden_dim': HIDDEN_DIM, 'num_layers': NUM_LAYERS, 'num_nodes': len(mesh_data['lon'])}
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
