#!/usr/bin/env python3
"""
Station Time Series Visualization for STOFS-GNN V2 Model

Plots predicted vs ground truth water levels at key tide gauge stations.

Usage:
    python scripts/visualize_stations_v2.py --date 20250120 --checkpoint checkpoint_epoch_60.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
CHECKPOINT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures_25k_v2')

# Model config
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Tidal constituent periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}

# Key tide gauge stations (coordinates tuned to best mesh nodes)
# Focused on Mid-Atlantic and Chesapeake Bay (avoiding LIS boundary issues)
STATIONS = {
    # New York / New Jersey Coast
    'The_Battery': (-74.003, 40.704),         # Node 12730, R24=0.95
    'Sandy_Hook': (-74.025, 40.474),          # Node 18576, R24=0.90
    'Atlantic_City': (-74.406, 39.349),       # Node 23128, R24=0.93
    # Delaware Bay / River
    'Philadelphia_PA': (-75.225, 39.856),     # Node 1517
    'Cape_May': (-74.972, 38.974),            # Node 17008, R24=0.68
    'Lewes_DE': (-75.131, 38.797),            # Node 18636, R24=0.85
    # Chesapeake Bay (North to South)
    'Baltimore': (-76.578, 39.267),           # Node 1045, R24=0.99
    'Annapolis': (-76.480, 38.983),           # Node 3499, R24=0.95
    'Cambridge_MD': (-76.090, 38.653),        # Node 2995
    'Solomons_Island_MD': (-76.505, 38.361),  # Node 6621
    # Coastal (North to South)
    'Ocean_City_MD': (-75.093, 38.343),       # Node 20981, R24=0.76
    'Kiptopeke_VA': (-75.857, 37.201),        # Node 20845
}

MAX_ROLLOUT_HOURS = 48


# ============================================================
# Model Architecture (same as training)
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


# ============================================================
# Helper Functions
# ============================================================

def compute_tidal_harmonics(global_hour: float) -> np.ndarray:
    harmonics = []
    for name, period in TIDAL_PERIODS.items():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


def get_forcing(forcing_dict, t: int, num_nodes: int) -> np.ndarray:
    return np.stack([
        forcing_dict['u10'][t], forcing_dict['v10'][t],
        forcing_dict['wind_speed'][t], forcing_dict['wind_speed_sq'][t],
        forcing_dict['wind_dir'][t], forcing_dict['pressure'][t],
        forcing_dict['dP_dx'][t], forcing_dict['dP_dy'][t],
    ], axis=1).astype(np.float32)


def find_nearest_node(lon, lat, target_lon, target_lat):
    """Find nearest mesh node to target coordinates."""
    dist = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
    return np.argmin(dist)


def load_model(checkpoint_path, device):
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM, temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES, forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def run_rollout(model, mesh_data, data, device, max_hours=48):
    """Run rollout and return full time series."""
    elevation = data['elevation']
    forcing = data['forcing']
    date_str = data['date']

    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_base = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)
    num_nodes = len(lon)

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
    static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)

    # Edge features
    src, dst = mesh_data['edge_index'][0], mesh_data['edge_index'][1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    ).to(device)

    # Initialize
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = []
    ground_truth = []
    num_steps = min(max_hours, elevation.shape[0] - 2)

    with torch.no_grad():
        for t in range(1, num_steps + 1):
            dxdt = (current - current_prev) / DT_HOURS
            global_hour = global_hours_base + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)

            forcing_arr = get_forcing(forcing, t, num_nodes)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            current_prev = current
            current = pred

    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Station time series visualization')
    parser.add_argument('--date', type=str, default=None, help='Date (YYYYMMDD), defaults to first 2025 file')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_60.pt', help='Checkpoint filename')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    logger.info(f"Mesh: {len(lon):,} nodes")

    # Find station indices
    station_indices = {}
    for name, (slon, slat) in STATIONS.items():
        idx = find_nearest_node(lon, lat, slon, slat)
        station_indices[name] = idx
        logger.info(f"  {name}: node {idx} at ({lon[idx]:.3f}, {lat[idx]:.3f})")

    # Find validation file
    if args.date:
        val_file = DATA_DIR / f'processed_{args.date}.npz'
        if not val_file.exists():
            logger.error(f"Data file not found: {val_file}")
            return
        date_str = args.date
    else:
        val_files = sorted([f for f in DATA_DIR.glob('processed_2025*.npz') if 'mesh' not in f.stem])
        if not val_files:
            logger.error("No validation files found!")
            return
        val_file = val_files[0]
        date_str = val_file.stem.replace('processed_', '')

    logger.info(f"\nUsing validation date: {date_str}")

    # Load data
    val_data_raw = np.load(val_file)
    val_data = {
        'date': date_str,
        'elevation': val_data_raw['elevation'],
        'forcing': {k: val_data_raw[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                                   'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
    }

    # Checkpoint to use
    ckpt_file = args.checkpoint
    epoch_num = ckpt_file.replace('checkpoint_epoch_', '').replace('.pt', '')
    label = f'Epoch {epoch_num}'
    checkpoints = [(label, ckpt_file)]

    # Run rollout for checkpoint
    for label, ckpt_file in checkpoints:
        ckpt_path = CHECKPOINT_DIR / ckpt_file
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info(f"\nRunning rollout for {label}...")
        model = load_model(ckpt_path, device)
        predictions, ground_truth = run_rollout(model, mesh_data, val_data, device, MAX_ROLLOUT_HOURS)

        # Create station time series plot
        n_stations = len(STATIONS)
        n_cols = 3
        n_rows = (n_stations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        hours = np.arange(1, len(predictions) + 1)

        for i, (name, idx) in enumerate(station_indices.items()):
            ax = axes[i]

            pred_ts = predictions[:, idx]
            truth_ts = ground_truth[:, idx]

            # Compute metrics
            rmse = np.sqrt(np.mean((pred_ts - truth_ts)**2))
            try:
                r, _ = pearsonr(pred_ts, truth_ts)
            except:
                r = np.nan

            ax.plot(hours, truth_ts, 'g-', linewidth=2, label='STOFS')
            ax.plot(hours, pred_ts, 'b--', linewidth=2, label='GNN Prediction')

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Water Level (m, MSL)')
            ax.set_title(f'{name}\nRMSE vs STOFS: {rmse:.3f}m, R: {r:.2f}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, MAX_ROLLOUT_HOURS)

        # Hide unused subplots
        for i in range(len(STATIONS), len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'25K V2 Model - {MAX_ROLLOUT_HOURS}h Rollout ({date_str})\nModel: {label}',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'station_timeseries_{date_str}.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {OUTPUT_DIR / f'station_timeseries_{date_str}.png'}")

    # Now create comparison plot with multiple checkpoints (optimized - cache rollouts first)
    logger.info("\nCreating multi-checkpoint comparison...")

    all_checkpoints = [
        ('2-step (ep30)', 'checkpoint_epoch_30.pt', '#1f77b4'),
        ('3-step (ep50)', 'checkpoint_epoch_50.pt', '#ff7f0e'),
        ('6-step (ep55)', 'checkpoint_epoch_55.pt', '#ff9896'),
        ('6-step (ep60)', 'checkpoint_epoch_60.pt', '#2ca02c'),
    ]

    # Run all rollouts first and cache results (3 rollouts instead of 12)
    cached_predictions = {}
    for ckpt_label, ckpt_file, color in all_checkpoints:
        ckpt_path = CHECKPOINT_DIR / ckpt_file
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info(f"  Running rollout for {ckpt_label}...")
        model_ckpt = load_model(ckpt_path, device)
        preds, gt = run_rollout(model_ckpt, mesh_data, val_data, device, MAX_ROLLOUT_HOURS)
        cached_predictions[ckpt_label] = {'preds': preds, 'gt': gt, 'color': color}
        del model_ckpt  # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Select key stations for comparison (representative from each region)
    key_stations = [
        'The_Battery',        # NY Harbor
        'Sandy_Hook',         # NJ Coast
        'Atlantic_City',      # NJ Coast
        'Philadelphia_PA',    # Delaware River
        'Lewes_DE',           # Delaware Bay
        'Baltimore',          # Upper Chesapeake
        'Solomons_Island_MD', # Mid Chesapeake
        'Kiptopeke_VA',       # Southern domain
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    hours = np.arange(1, MAX_ROLLOUT_HOURS + 1)

    for i, name in enumerate(key_stations):
        ax = axes[i]
        idx = station_indices[name]

        # Plot ground truth (from any cached result)
        gt = list(cached_predictions.values())[0]['gt']
        truth_ts = gt[:len(hours), idx]
        ax.plot(hours[:len(truth_ts)], truth_ts, 'k-', linewidth=2.5, label='STOFS (truth)', alpha=0.8)

        # Plot each checkpoint from cache
        for ckpt_label, data in cached_predictions.items():
            pred_ts = data['preds'][:len(hours), idx]
            color = data['color']

            rmse = np.sqrt(np.mean((pred_ts - truth_ts[:len(pred_ts)])**2))
            ax.plot(hours[:len(pred_ts)], pred_ts, '--', color=color, linewidth=1.5,
                   label=f'{ckpt_label} (RMSE: {rmse*100:.1f}cm)')

        ax.set_xlabel('Time (hours)', fontsize=11)
        ax.set_ylabel('Water Level (m, MSL)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, MAX_ROLLOUT_HOURS)

    plt.suptitle(f'V2 Model Checkpoint Comparison - {date_str}\nStation Time Series (48h Rollout)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'station_comparison_{date_str}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {OUTPUT_DIR / f'station_comparison_{date_str}.png'}")

    logger.info("\nStation visualization complete!")


if __name__ == '__main__':
    main()
