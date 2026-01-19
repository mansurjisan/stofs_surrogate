#!/usr/bin/env python3
"""
STOFS-GNN 25K V2 Training with Scheduled Sampling

Scheduled sampling gradually reduces teacher forcing during training,
helping the model learn to recover from its own prediction errors.

Key addition:
- Teacher forcing ratio starts at 1.0 (always use ground truth)
- Gradually decreases to 0.0 (always use model predictions)
- Model learns to handle error accumulation during rollout

This is a drop-in enhancement to the existing V2 training.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2_ss')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume from existing checkpoint (your trained model)
RESUME_FROM = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2/checkpoint_epoch_55.pt')

# Model config (must match existing)
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Training config
NUM_EPOCHS = 30  # Additional epochs with scheduled sampling
BASE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-5

# Scheduled Sampling Configuration
SS_START_EPOCH = 1      # When to start reducing teacher forcing
SS_END_EPOCH = 20       # When teacher forcing reaches minimum
SS_MIN_RATIO = 0.2      # Minimum teacher forcing ratio (keep some for stability)
SS_SCHEDULE = 'linear'  # 'linear', 'exponential', or 'inverse_sigmoid'

# Rollout schedule (start with longer rollouts since model is pretrained)
ROLLOUT_SCHEDULE = {
    6: (1, 10, 2),    # 6-step: epochs 1-10
    12: (11, 20, 1),  # 12-step: epochs 11-20
    24: (21, 30, 1),  # 24-step: epochs 21-30
}

# Tidal periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}


# ============================================================
# Scheduled Sampling Functions
# ============================================================

def get_teacher_forcing_ratio(epoch, schedule='linear'):
    """
    Compute teacher forcing ratio for current epoch.

    Teacher forcing = using ground truth as input for next step
    As training progresses, we reduce this to force model to use its own predictions.

    Args:
        epoch: Current epoch number
        schedule: 'linear', 'exponential', or 'inverse_sigmoid'

    Returns:
        ratio: Probability of using ground truth (1.0 = always GT, 0.0 = always prediction)
    """
    if epoch < SS_START_EPOCH:
        return 1.0

    if epoch >= SS_END_EPOCH:
        return SS_MIN_RATIO

    progress = (epoch - SS_START_EPOCH) / (SS_END_EPOCH - SS_START_EPOCH)

    if schedule == 'linear':
        # Linear decay
        ratio = 1.0 - progress * (1.0 - SS_MIN_RATIO)

    elif schedule == 'exponential':
        # Exponential decay (slower start, faster end)
        ratio = (1.0 - SS_MIN_RATIO) * (0.9 ** (progress * 20)) + SS_MIN_RATIO

    elif schedule == 'inverse_sigmoid':
        # Inverse sigmoid (smooth transition)
        k = 10  # Steepness
        ratio = (1.0 - SS_MIN_RATIO) / (1 + np.exp(k * (progress - 0.5))) + SS_MIN_RATIO

    else:
        ratio = 1.0 - progress * (1.0 - SS_MIN_RATIO)

    return max(SS_MIN_RATIO, min(1.0, ratio))


def scheduled_sampling_step(pred, ground_truth, teacher_forcing_ratio):
    """
    Apply scheduled sampling: randomly choose between prediction and ground truth.

    Args:
        pred: Model prediction (B, N, 1)
        ground_truth: Ground truth (B, N, 1)
        teacher_forcing_ratio: Probability of using ground truth

    Returns:
        next_input: Either pred or ground_truth based on sampling
    """
    if teacher_forcing_ratio >= 1.0:
        return ground_truth

    if teacher_forcing_ratio <= 0.0:
        return pred

    # Per-sample decision (not per-node)
    B = pred.shape[0]
    mask = torch.rand(B, 1, 1, device=pred.device) < teacher_forcing_ratio

    return torch.where(mask, ground_truth, pred)


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


# ============================================================
# Dataset
# ============================================================

class InMemoryDatasetV2(Dataset):
    def __init__(self, data_list, mesh_data):
        self.data_list = data_list
        self.mesh_data = mesh_data
        self.samples = []

        for data in data_list:
            date_str = data['date']
            T = data['elevation'].shape[0]
            for t in range(1, T - 1):
                self.samples.append((data, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, t = self.samples[idx]

        elevation = data['elevation']
        forcing = data['forcing']
        date_str = data['date']

        date_dt = datetime.strptime(date_str, '%Y%m%d')
        global_hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0 + t * DT_HOURS

        harmonics = []
        for name, period in TIDAL_PERIODS.items():
            phase = 2.0 * np.pi * global_hours / period
            harmonics.extend([np.sin(phase), np.cos(phase)])
        tidal = np.array(harmonics, dtype=np.float32)

        cwl_t = np.nan_to_num(elevation[t], nan=0.0).astype(np.float32) / ETA_SCALE
        cwl_prev = np.nan_to_num(elevation[t-1], nan=0.0).astype(np.float32) / ETA_SCALE
        cwl_next = np.nan_to_num(elevation[t+1], nan=0.0).astype(np.float32) / ETA_SCALE

        forcing_t = np.stack([
            forcing['u10'][t],
            forcing['v10'][t],
            forcing['wind_speed'][t],
            forcing['wind_speed_sq'][t],
            forcing['wind_dir'][t],
            forcing['pressure'][t],
            forcing['dP_dx'][t],
            forcing['dP_dy'][t],
        ], axis=1).astype(np.float32)

        return {
            'state': cwl_t,
            'state_prev': cwl_prev,
            'target': cwl_next,
            'forcing': forcing_t,
            'tidal': tidal,
        }


def collate_fn(batch):
    return {
        'state': torch.tensor(np.stack([b['state'] for b in batch])),
        'state_prev': torch.tensor(np.stack([b['state_prev'] for b in batch])),
        'target': torch.tensor(np.stack([b['target'] for b in batch])),
        'forcing': torch.tensor(np.stack([b['forcing'] for b in batch])),
        'tidal': torch.tensor(np.stack([b['tidal'] for b in batch])),
    }


# ============================================================
# Training with Scheduled Sampling
# ============================================================

def train_epoch_with_scheduled_sampling(model, dataloader, optimizer, device, mesh_tensors,
                                        rollout_steps, grad_accum_steps, teacher_forcing_ratio):
    """
    Train for one epoch with scheduled sampling.

    Key difference from regular training:
    - During rollout, randomly choose between prediction and ground truth for next input
    - This helps model learn to recover from its own errors
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        state = batch['state'].unsqueeze(-1).to(device)
        state_prev = batch['state_prev'].unsqueeze(-1).to(device)
        target = batch['target'].unsqueeze(-1).to(device)
        forcing = batch['forcing'].to(device)
        tidal = batch['tidal'].to(device)

        B, N = state.shape[:2]
        tidal_expanded = tidal.unsqueeze(1).expand(B, N, -1)

        static = mesh_tensors['static'].unsqueeze(0).expand(B, -1, -1).to(device)
        edge_index = mesh_tensors['edge_index'].to(device)
        edge_attr = mesh_tensors['edge_attr'].to(device)

        # Rollout with scheduled sampling
        current = state
        current_prev = state_prev
        loss = 0.0

        for step in range(rollout_steps):
            dxdt = (current - current_prev) / DT_HOURS

            # Forward pass
            pred = model(current, current_prev, dxdt, tidal_expanded,
                        static, forcing, edge_index, edge_attr)

            # Loss
            step_loss = F.mse_loss(pred, target)
            loss = loss + step_loss

            # SCHEDULED SAMPLING: Choose next input
            # With probability teacher_forcing_ratio, use ground truth
            # Otherwise, use model's prediction
            next_state = scheduled_sampling_step(pred.detach(), target, teacher_forcing_ratio)

            # Update for next step
            current_prev = current
            current = next_state

        loss = loss / rollout_steps
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if batch_idx % 400 == 0:
            logger.info(f"    Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item()*grad_accum_steps:.5f} | TF: {teacher_forcing_ratio:.2f}")

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, mesh_tensors, rollout_steps=6):
    """Validate without teacher forcing (use only predictions)"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].unsqueeze(-1).to(device)
            state_prev = batch['state_prev'].unsqueeze(-1).to(device)
            target = batch['target'].unsqueeze(-1).to(device)
            forcing = batch['forcing'].to(device)
            tidal = batch['tidal'].to(device)

            B, N = state.shape[:2]
            tidal_expanded = tidal.unsqueeze(1).expand(B, N, -1)
            static = mesh_tensors['static'].unsqueeze(0).expand(B, -1, -1).to(device)
            edge_index = mesh_tensors['edge_index'].to(device)
            edge_attr = mesh_tensors['edge_attr'].to(device)

            current = state
            current_prev = state_prev
            loss = 0.0

            for step in range(rollout_steps):
                dxdt = (current - current_prev) / DT_HOURS
                pred = model(current, current_prev, dxdt, tidal_expanded,
                            static, forcing, edge_index, edge_attr)
                loss = loss + F.mse_loss(pred, target)

                # Always use prediction for validation (no teacher forcing)
                current_prev = current
                current = pred

            total_loss += (loss / rollout_steps).item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main
# ============================================================

def main():
    logger.info("="*70)
    logger.info("STOFS-GNN V2 TRAINING WITH SCHEDULED SAMPLING")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================
    # Load mesh
    # ========================================
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))

    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = mesh_data['edge_index']
    n_nodes = len(lon)

    logger.info(f"Mesh: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges")

    # Static features
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    static_features = np.stack([x_norm, y_norm, depth_norm, np.zeros_like(depth_norm)], axis=1).astype(np.float32)

    # Edge features
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    mesh_tensors = {
        'static': torch.tensor(static_features),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_attr),
    }

    # ========================================
    # Load data
    # ========================================
    logger.info("\nLoading training data...")
    train_files = sorted([f for f in DATA_DIR.glob('processed_202[34]*.npz') if 'mesh' not in f.stem])
    val_files = sorted([f for f in DATA_DIR.glob('processed_2025*.npz') if 'mesh' not in f.stem])

    train_data = []
    for f in train_files:
        data = np.load(f)
        train_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    val_data = []
    for f in val_files:
        data = np.load(f)
        val_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    logger.info(f"Training dates: {len(train_data)}")
    logger.info(f"Validation dates: {len(val_data)}")

    train_dataset = InMemoryDatasetV2(train_data, mesh_data)
    val_dataset = InMemoryDatasetV2(val_data, mesh_data)

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # ========================================
    # Create model
    # ========================================
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    # Load pretrained weights
    if RESUME_FROM.exists():
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
        logger.warning(f"No pretrained model found at {RESUME_FROM}, training from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # ========================================
    # Optimizer
    # ========================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ========================================
    # Training loop with scheduled sampling
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING WITH SCHEDULED SAMPLING")
    logger.info("="*70)
    logger.info(f"Schedule: {SS_SCHEDULE}")
    logger.info(f"Teacher forcing: {1.0} -> {SS_MIN_RATIO} over epochs {SS_START_EPOCH}-{SS_END_EPOCH}")

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # Get rollout steps
        rollout_steps = 6
        batch_mult = 2
        for steps, (start, end, mult) in ROLLOUT_SCHEDULE.items():
            if start <= epoch <= end:
                rollout_steps = steps
                batch_mult = mult
                break

        batch_size = BASE_BATCH_SIZE * batch_mult

        # Get teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(epoch, schedule=SS_SCHEDULE)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS} | rollout={rollout_steps} | TF_ratio={tf_ratio:.2f}")

        # Train with scheduled sampling
        train_loss = train_epoch_with_scheduled_sampling(
            model, train_loader, optimizer, device, mesh_tensors,
            rollout_steps, GRAD_ACCUM_STEPS, tf_ratio
        )

        # Validate (no teacher forcing)
        val_loss = validate(model, val_loader, device, mesh_tensors, rollout_steps=12)

        scheduler.step()

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | TF={tf_ratio:.2f} | lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % 5 == 0 or val_loss < best_val_loss:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'teacher_forcing_ratio': tf_ratio,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, OUTPUT_DIR / 'best_model_ss.pt')
                logger.info(f"  Saved best model (val={val_loss:.5f})")

            if epoch % 5 == 0:
                torch.save(ckpt, OUTPUT_DIR / f'checkpoint_ss_epoch_{epoch}.pt')

    logger.info("\nTraining with scheduled sampling complete!")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    main()
