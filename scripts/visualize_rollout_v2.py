#!/usr/bin/env python3
"""
Rollout Visualization for STOFS-GNN V2 Model

Compares different checkpoints by running autoregressive rollout
and plotting RMSE vs forecast lead time.

Usage:
    python scripts/visualize_rollout_v2.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/e/STOFS_TRAINING_DATA/processed_25k_v2')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(os.environ.get('STOFS_CHECKPOINT_DIR', PROJECT_ROOT / 'outputs/checkpoints_25k_v2'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', PROJECT_ROOT / 'outputs/figures_25k_v2'))

# Model config (must match training)
HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12  # 6 tidal constituents x 2
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Tidal constituent periods (hours)
TIDAL_PERIODS = {
    'M2': 12.4206,
    'S2': 12.0000,
    'N2': 12.6583,
    'K1': 23.9345,
    'O1': 25.8193,
    'M4': 6.2103,
}

# Checkpoints to compare (based on rollout schedule)
# 1-step: epochs 1-15, 2-step: 16-30, 3-step: 31-50, 6-step: 51-75, 12-step: 76-100
CHECKPOINTS_TO_COMPARE = [
    ('best_model', 'best_model.pt', 'Best (ep13)'),
    ('epoch_30', 'checkpoint_epoch_30.pt', '2-step (ep30)'),
    ('epoch_50', 'checkpoint_epoch_50.pt', '3-step (ep50)'),
    ('epoch_60', 'checkpoint_epoch_60.pt', '6-step (ep60)'),
    ('epoch_80', 'checkpoint_epoch_80.pt', '12-step (ep80)'),
    ('epoch_95', 'checkpoint_epoch_95.pt', '12-step (ep95)'),
]

# Rollout settings
MAX_ROLLOUT_HOURS = 48
VAL_DATE = '20250115'  # Validation date to use


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
    """Compute 6 tidal constituents (12 features)."""
    harmonics = []
    for name, period in TIDAL_PERIODS.items():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


def get_forcing(forcing_dict, t: int, num_nodes: int) -> np.ndarray:
    """Get 8 forcing features for timestep t."""
    return np.stack([
        forcing_dict['u10'][t],
        forcing_dict['v10'][t],
        forcing_dict['wind_speed'][t],
        forcing_dict['wind_speed_sq'][t],
        forcing_dict['wind_dir'][t],
        forcing_dict['pressure'][t],
        forcing_dict['dP_dx'][t],
        forcing_dict['dP_dy'][t],
    ], axis=1).astype(np.float32)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM,
        temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Remove _orig_mod prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def run_rollout(model, mesh_data, data, device, max_hours=48):
    """Run autoregressive rollout and return predictions and ground truth."""

    elevation = data['elevation']
    forcing = data['forcing']
    date_str = data['date']

    # Compute date offset
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_base = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    # Mesh data
    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)
    num_nodes = len(lon)

    # Compute static features
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

    # Compute edge features
    src, dst = mesh_data['edge_index'][0], mesh_data['edge_index'][1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    ).to(device)

    # Initialize from timestep 1
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = []
    ground_truth = []

    num_steps = min(max_hours, elevation.shape[0] - 2)

    with torch.no_grad():
        for t in range(1, num_steps + 1):
            # Rate of change
            dxdt = (current - current_prev) / DT_HOURS

            # Tidal harmonics
            global_hour = global_hours_base + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            # Static features with water level
            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)

            # Forcing
            forcing_arr = get_forcing(forcing, t, num_nodes)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass
            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            # Store results
            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            # Update state
            current_prev = current
            current = pred

    return np.array(predictions), np.array(ground_truth)


def compute_rmse_by_lead_time(predictions, ground_truth):
    """Compute RMSE for each lead time."""
    rmse = []
    for t in range(len(predictions)):
        err = predictions[t] - ground_truth[t]
        rmse.append(np.sqrt(np.mean(err**2)))
    return np.array(rmse)


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Rollout Visualization for STOFS-GNN V2')
    parser.add_argument('--date', type=str, default=None, help='Validation date (YYYYMMDD)')
    parser.add_argument('--snapshots', action='store_true', help='Generate individual spatial snapshots')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    logger.info(f"Mesh: {len(mesh_data['lon']):,} nodes")

    # Find a validation date
    if args.date:
        val_file = DATA_DIR / f'processed_{args.date}.npz'
        date_str = args.date
    else:
        val_files = sorted([f for f in DATA_DIR.glob('processed_2025*.npz') if 'mesh' not in f.stem])
        if not val_files:
            logger.error("No validation files found!")
            return
        val_file = val_files[0]
        date_str = val_file.stem.replace('processed_', '')
    logger.info(f"Using validation date: {date_str}")

    # Load validation data
    val_data_raw = np.load(val_file)
    val_data = {
        'date': date_str,
        'elevation': val_data_raw['elevation'],
        'forcing': {
            'u10': val_data_raw['u10'],
            'v10': val_data_raw['v10'],
            'wind_speed': val_data_raw['wind_speed'],
            'wind_speed_sq': val_data_raw['wind_speed_sq'],
            'wind_dir': val_data_raw['wind_dir'],
            'pressure': val_data_raw['pressure'],
            'dP_dx': val_data_raw['dP_dx'],
            'dP_dy': val_data_raw['dP_dy'],
        }
    }
    logger.info(f"Loaded {val_data['elevation'].shape[0]} timesteps")

    # Run rollout for each checkpoint
    results = {}

    for name, ckpt_file, label in CHECKPOINTS_TO_COMPARE:
        ckpt_path = CHECKPOINT_DIR / ckpt_file
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info(f"\nRunning rollout for {label}...")
        model = load_model(ckpt_path, device)

        predictions, ground_truth = run_rollout(model, mesh_data, val_data, device, MAX_ROLLOUT_HOURS)
        rmse = compute_rmse_by_lead_time(predictions, ground_truth)

        results[name] = {
            'label': label,
            'rmse': rmse,
            'predictions': predictions,
            'ground_truth': ground_truth,
        }

        # Print key metrics
        for h in [1, 6, 12, 24, 48]:
            if h <= len(rmse):
                logger.info(f"  t+{h}h RMSE: {rmse[h-1]*100:.1f} cm")

    # ========================================
    # Plot 1: RMSE vs Lead Time
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, data) in enumerate(results.items()):
        hours = np.arange(1, len(data['rmse']) + 1)
        ax.plot(hours, data['rmse'] * 100, label=data['label'],
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=12)
    ax.set_ylabel('RMSE (cm)', fontsize=12)
    ax.set_title(f'STOFS-GNN V2 Rollout Performance\nValidation Date: {date_str}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ROLLOUT_HOURS)
    ax.set_ylim(0, None)

    # Add reference lines
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10 cm')
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='20 cm')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'rollout_rmse_comparison_{date_str}.png', dpi=150)
    plt.close()
    logger.info(f"\nSaved: {OUTPUT_DIR / f'rollout_rmse_comparison_{date_str}.png'}")

    # ========================================
    # Plot 2: RMSE Table
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Create table data
    lead_times = [1, 3, 6, 12, 24, 48]
    table_data = [['Checkpoint'] + [f't+{h}h' for h in lead_times]]

    for name, data in results.items():
        row = [data['label']]
        for h in lead_times:
            if h <= len(data['rmse']):
                row.append(f"{data['rmse'][h-1]*100:.1f} cm")
            else:
                row.append('-')
        table_data.append(row)

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.title(f'RMSE by Lead Time - Validation Date: {date_str}', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'rollout_rmse_table_{date_str}.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {OUTPUT_DIR / f'rollout_rmse_table_{date_str}.png'}")

    # ========================================
    # Plot 3: Spatial Error Map at t+6h and t+12h
    # ========================================
    if results:
        # Use latest checkpoint for spatial plot
        latest_name = list(results.keys())[-1]
        latest_data = results[latest_name]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, (ax, hour) in enumerate(zip(axes, [6, 12])):
            if hour <= len(latest_data['predictions']):
                pred = latest_data['predictions'][hour - 1]
                truth = latest_data['ground_truth'][hour - 1]
                error = pred - truth

                scatter = ax.scatter(mesh_data['lon'], mesh_data['lat'],
                                    c=error * 100, cmap='RdBu_r',
                                    s=1, vmin=-30, vmax=30)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_title(f't+{hour}h Error (cm) - {latest_data["label"]}')
                plt.colorbar(scatter, ax=ax, label='Error (cm)')

        plt.suptitle(f'Spatial Error Distribution - {date_str}', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'rollout_spatial_error_{date_str}.png', dpi=150)
        plt.close()
        logger.info(f"Saved: {OUTPUT_DIR / f'rollout_spatial_error_{date_str}.png'}")

    # ========================================
    # Plot 4: Individual Spatial Snapshots (6-hourly)
    # ========================================
    if args.snapshots and results:
        snapshot_dir = OUTPUT_DIR / 'spatial_snapshots'
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        latest_name = list(results.keys())[-1]
        latest_data = results[latest_name]

        # Get color scale
        all_preds = np.concatenate(latest_data['predictions'])
        all_truth = np.concatenate(latest_data['ground_truth'])
        vmax = np.percentile(np.abs(np.concatenate([all_preds, all_truth])), 98)
        vmin = -vmax

        logger.info(f"\nGenerating individual spatial snapshots...")
        logger.info(f"  Color scale: [{vmin:.2f}, {vmax:.2f}] m")

        for h in range(0, MAX_ROLLOUT_HOURS + 1, 6):
            if h == 0 or h > len(latest_data['predictions']):
                continue

            pred = latest_data['predictions'][h - 1]
            truth = latest_data['ground_truth'][h - 1]
            rmse_h = np.sqrt(np.mean((pred - truth)**2))
            corr_h = np.corrcoef(pred, truth)[0, 1]

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))

            # STOFS ground truth
            sc1 = axes[0].scatter(mesh_data['lon'], mesh_data['lat'],
                                  c=truth, s=3, cmap='RdBu_r',
                                  vmin=vmin, vmax=vmax, marker='.')
            axes[0].set_title(f'STOFS Ground Truth - t+{h}h', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Longitude')
            axes[0].set_ylabel('Latitude')
            axes[0].set_aspect('equal')
            plt.colorbar(sc1, ax=axes[0], label='Water Level (m MSL)', shrink=0.8)

            # GNN prediction
            sc2 = axes[1].scatter(mesh_data['lon'], mesh_data['lat'],
                                  c=pred, s=3, cmap='RdBu_r',
                                  vmin=vmin, vmax=vmax, marker='.')
            axes[1].set_title(f'GNN ({latest_data["label"]}) - t+{h}h | RMSE: {rmse_h*100:.1f}cm, R: {corr_h:.3f}',
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            axes[1].set_aspect('equal')
            plt.colorbar(sc2, ax=axes[1], label='Water Level (m MSL)', shrink=0.8)

            plt.suptitle(f'Water Elevation Comparison - {date_str} - Forecast Hour {h}',
                        fontsize=16, fontweight='bold')
            plt.tight_layout()

            out_path = snapshot_dir / f'snapshot_{date_str}_h{h:02d}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  Saved: {out_path.name} (RMSE: {rmse_h*100:.1f}cm)")

        logger.info(f"  Snapshots saved to: {snapshot_dir}")

    logger.info("\nRollout visualization complete!")


if __name__ == '__main__':
    main()
