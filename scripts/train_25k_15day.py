#!/usr/bin/env python3
"""
A10G Training Script for 15-Day / 25K Node Dataset

Optimized for:
- 15 days of training data (~1,650 samples)
- 25,000 nodes (higher resolution)
- NVIDIA A10G (24GB VRAM)

Expected training time: ~3-4 hours on A10G

Usage:
    python train_25k_15day.py
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
# A10G OPTIMIZATIONS
# ============================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIGURATION
# ============================================================

# Paths - UPDATE FOR YOUR SETUP
DATA_DIR = '/home/Mansur.Jisan/stofs_surrogate/data/processed_25k'
OUTPUT_DIR = '/home/Mansur.Jisan/stofs_surrogate'

# Training dates (15 days: Nov 15-29)
TRAINING_DATES = [
    '20251115', '20251116', '20251117', '20251118', '20251119',
    '20251120', '20251121', '20251122', '20251123', '20251124',
    '20251125', '20251126', '20251127', '20251128', '20251129',
]

VAL_DATES = 2  # Last 2 days for validation (Nov 28-29)

# ============================================================
# MODEL & TRAINING PARAMETERS (Optimized for 25k nodes)
# ============================================================

# Model architecture
HIDDEN_DIM = 128        # Increased from 96 (more capacity for larger mesh)
NUM_LAYERS = 6
STATE_DIM = 1
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 3

# Training - Optimized for 25k nodes on A10G
EPOCHS = 100            # Reduced (more data = faster convergence)
BATCH_SIZE = 4          # Reduced from 8 (25k nodes use more memory)
LEARNING_RATE = 2e-4    # Slightly lower (more data, larger model)
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
NUM_WORKERS = 2         # Reduced to avoid memory issues
USE_AMP = True

# Curriculum learning
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 20  # 20% of epochs
MAX_ROLLOUT_STEPS = 2

# Physics loss weights
MASS_CONSERVATION_WEIGHT = 0.01
SMOOTHNESS_WEIGHT = 0.01

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0

# Pressure was already normalized during preprocessing as:
#   p_norm = (p_raw_Pa - PRESSURE_MEAN) / PRESSURE_SCALE
# These constants are kept for reference/diagnostics only - DO NOT re-normalize in training!
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


# ============================================================
# Model Architecture
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


class PhysicsInformedCWLModel(nn.Module):
    """Physics-informed GNN for coastal water level prediction."""
    
    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        node_input_dim = state_dim + static_feature_dim + forcing_feature_dim
        
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
    
    def forward(self, x, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, static_features, forcing], dim=-1)
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
# Dataset
# ============================================================

class MultiDateCWLDataset(Dataset):
    def __init__(self, mesh_data: Dict, date_data_list: List[Dict], eta_scale: float = 2.0):
        self.eta_scale = eta_scale
        
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
        
        self.samples = []
        for date_idx, elev in enumerate(self.elevations):
            num_times = elev.shape[0]
            for t in range(num_times - 2):
                self.samples.append((date_idx, t))
        
        logger.info(f"Dataset: {len(self.samples)} samples from {len(date_data_list)} dates")
    
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
        
        cwl_t = elev[t].astype(np.float32)
        cwl_t = np.nan_to_num(cwl_t, nan=0.0)
        cwl_norm = cwl_t / self.eta_scale
        
        water_level = self.depth + cwl_t
        wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        
        static = np.concatenate([self.static_base, wl_norm[:, np.newaxis]], axis=1)
        
        u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
        # Pressure is already normalized during preprocessing (z-score with PRESSURE_MEAN/SCALE)
        pres = forcing['pressure'][t].astype(np.float32)
        forcing_arr = np.stack([u10, v10, pres], axis=1)

        cwl_t1 = np.nan_to_num(elev[t+1].astype(np.float32), nan=0.0) / self.eta_scale
        cwl_t2 = np.nan_to_num(elev[t+2].astype(np.float32), nan=0.0) / self.eta_scale

        u10_next = forcing['u10'][t+1].astype(np.float32) / WIND_SCALE
        v10_next = forcing['v10'][t+1].astype(np.float32) / WIND_SCALE
        # Pressure is already normalized during preprocessing
        pres_next = forcing['pressure'][t+1].astype(np.float32)
        forcing_next = np.stack([u10_next, v10_next, pres_next], axis=1)
        
        return {
            'x': torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'forcing': torch.tensor(forcing_arr, dtype=torch.float32),
            'y': torch.tensor(cwl_t1[:, np.newaxis], dtype=torch.float32),
            'y_next': torch.tensor(cwl_t2[:, np.newaxis], dtype=torch.float32),
            'forcing_next': torch.tensor(forcing_next, dtype=torch.float32),
            'raw_depth': torch.tensor(self.depth[:, np.newaxis], dtype=torch.float32),  # Physical depth (meters)
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
        }


# ============================================================
# Training Functions
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, num_steps, grad_clip, scaler=None):
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        batch_size = batch['x'].shape[0]
        batch_loss = 0
        
        edge_index = batch['edge_index'][0].to(device)
        edge_attr = batch['edge_attr'][0].to(device)
        
        for i in range(batch_size):
            x = batch['x'][i].to(device)
            static = batch['static'][i].to(device)
            forcing = batch['forcing'][i].to(device)
            y = batch['y'][i].to(device)
            
            if scaler is not None:
                with autocast('cuda'):
                    pred = model(x, static, forcing, edge_index, edge_attr)
                    loss, components = criterion(pred, y, edge_index)

                    if num_steps >= 2:
                        y_next = batch['y_next'][i].to(device)
                        forcing_next = batch['forcing_next'][i].to(device)
                        raw_depth = batch['raw_depth'][i].to(device)  # Physical depth (meters)

                        # FIX: Use physical units for Total Water Level calculation
                        # 1. Convert predicted surge back to meters
                        pred_surge_meters = pred.detach() * ETA_SCALE

                        # 2. Calculate physical Total Water Level (meters)
                        wl_physical = raw_depth + pred_surge_meters

                        # 3. Normalize TWL (matching __getitem__ logic)
                        wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)

                        # 4. Create static features for t+1 (keep x_norm, y_norm, depth_norm unchanged)
                        static_new = torch.cat([static[:, :3], wl_norm], dim=1)

                        pred2 = model(pred.detach(), static_new, forcing_next, edge_index, edge_attr)
                        loss2, _ = criterion(pred2, y_next, edge_index)
                        loss = loss + 0.5 * loss2

                    scaled_loss = loss / batch_size

                scaler.scale(scaled_loss).backward()
            else:
                pred = model(x, static, forcing, edge_index, edge_attr)
                loss, components = criterion(pred, y, edge_index)

                if num_steps >= 2:
                    y_next = batch['y_next'][i].to(device)
                    forcing_next = batch['forcing_next'][i].to(device)
                    raw_depth = batch['raw_depth'][i].to(device)  # Physical depth (meters)

                    # FIX: Use physical units for Total Water Level calculation
                    pred_surge_meters = pred.detach() * ETA_SCALE
                    wl_physical = raw_depth + pred_surge_meters
                    wl_norm = (wl_physical - wl_physical.mean()) / (wl_physical.std() + 1e-8)
                    static_new = torch.cat([static[:, :3], wl_norm], dim=1)
                    
                    pred2 = model(pred.detach(), static_new, forcing_next, edge_index, edge_attr)
                    loss2, _ = criterion(pred2, y_next, edge_index)
                    loss = loss + 0.5 * loss2
                
                (loss / batch_size).backward()
            
            batch_loss += loss.item()
            for k in components:
                total_components[k] += components[k]
        
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
    
    return total_loss / num_samples, {k: v / num_samples for k, v in total_components.items()}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch_size = batch['x'].shape[0]
            edge_index = batch['edge_index'][0].to(device)
            edge_attr = batch['edge_attr'][0].to(device)
            
            for i in range(batch_size):
                x = batch['x'][i].to(device)
                static = batch['static'][i].to(device)
                forcing = batch['forcing'][i].to(device)
                y = batch['y'][i].to(device)
                
                with autocast('cuda'):
                    pred = model(x, static, forcing, edge_index, edge_attr)
                    loss, components = criterion(pred, y, edge_index)
                
                total_loss += loss.item()
                for k in components:
                    total_components[k] += components[k]
            
            num_samples += batch_size
    
    return total_loss / num_samples, {k: v / num_samples for k, v in total_components.items()}


def evaluate_rollout(model, dataset, device, num_steps=48):
    model.eval()
    
    elev = dataset.elevations[0]
    forcing = dataset.forcings[0]
    
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)
    
    predictions = []
    ground_truth = []
    
    cwl_t = elev[0].astype(np.float32)
    cwl_t = np.nan_to_num(cwl_t, nan=0.0)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        for t in range(min(num_steps, len(elev) - 1)):
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
            
            with autocast('cuda'):
                pred = model(current_cwl, static_tensor, forcing_tensor, edge_index, edge_attr)
            
            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elev[t + 1].astype(np.float32), nan=0.0))
            
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
    logger.info("15-DAY / 25K NODE TRAINING - STOFS GNN SURROGATE")
    logger.info("=" * 70)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  DATA_DIR: {DATA_DIR}")
    logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"  TRAINING_DATES: {len(TRAINING_DATES)} days")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  HIDDEN_DIM: {HIDDEN_DIM}")
    logger.info(f"  LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"  EPOCHS: {EPOCHS}")
    
    checkpoint_dir = Path(OUTPUT_DIR) / 'outputs' / 'checkpoints'
    figure_dir = Path(OUTPUT_DIR) / 'outputs' / 'figures'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mesh (25k nodes)
    mesh_path = Path(DATA_DIR) / 'mesh_25k.npz'
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
        data_path = Path(DATA_DIR) / f'processed_{date_str}.npz'
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
        data_path = Path(DATA_DIR) / f'processed_{date_str}.npz'
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
    train_dataset = MultiDateCWLDataset(mesh_data, train_data, eta_scale=ETA_SCALE)
    val_dataset = MultiDateCWLDataset(mesh_data, val_data, eta_scale=ETA_SCALE)
    
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
    
    # Model
    model = PhysicsInformedCWLModel(
        state_dim=STATE_DIM,
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
        
        num_steps = 1 if epoch <= CURRICULUM_WARMUP_EPOCHS else MAX_ROLLOUT_STEPS
        
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
                    'num_nodes': len(mesh_data['lon']),
                }
            }, checkpoint_dir / 'best_25k_15day_model.pt')
        
        epoch_time = time.time() - epoch_start
        
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | steps={num_steps} | "
                f"train={train_loss:.5f} | val={val_loss:.5f} | "
                f"mse={train_comp['mse']:.5f} | mass={train_comp['mass']:.4f} | "
                f"best={best_val_loss:.5f} | lr={current_lr:.2e} | {epoch_time:.1f}s"
            )
    
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    # Evaluate rollout
    logger.info("\nEvaluating rollout...")
    model.load_state_dict(torch.load(checkpoint_dir / 'best_25k_15day_model.pt', weights_only=True)['model_state_dict'])
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
    axes[0, 0].set_title('Training Progress (15-day / 25k nodes)')
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
    plt.savefig(figure_dir / '25k_15day_training_summary.png', dpi=150)
    plt.close()
    
    logger.info(f"\nModel saved to: {checkpoint_dir / 'best_25k_15day_model.pt'}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
