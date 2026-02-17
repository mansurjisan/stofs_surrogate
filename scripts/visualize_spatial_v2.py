#!/usr/bin/env python3
"""
Spatial Rollout Visualization for 25K V2 Model

Generates scatter plot visualizations showing ground truth vs predictions
and spatial error maps for specified forecast hours.

Usage:
    python scripts/visualize_spatial_v2.py --date 20250115 --hours 6 12 24 48
    python scripts/visualize_spatial_v2.py --date 20250101 --hours 6 12 24 36 48
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(os.environ.get('STOFS_CHECKPOINT_DIR', PROJECT_ROOT / 'outputs/checkpoints_25k_v2'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', PROJECT_ROOT / 'outputs/figures_25k_v2'))

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

TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}


# ============================================================
# Model Architecture
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.gradient_scale = nn.Parameter(torch.ones(1))

    def forward(self, h, edge_index, edge_attr):
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]
        h_src, h_dst = h[:, row, :], h[:, col, :]
        h_gradient = h_dst - h_src
        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1).reshape(B * E, -1)
        edge_msg = self.edge_mlp(edge_input).reshape(B, E, F)
        edge_msg = edge_msg * (1.0 + torch.tanh(self.gradient_scale * h_gradient))
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        aggr = torch.zeros(B, N, F, device=h.device).scatter_add_(
            1, row.unsqueeze(0).unsqueeze(-1).expand(B, E, F), edge_msg
        )
        return h + self.node_mlp(torch.cat([h, aggr], dim=-1).reshape(B * N, -1)).reshape(B, N, F), edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.gnn_layers = nn.ModuleList([BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

    def forward(self, x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr):
        B, N, _ = x.shape
        node_features = torch.cat([x, x_prev, dxdt, tidal, static, forcing], dim=-1).reshape(B * N, -1)
        h = self.node_encoder(node_features).reshape(B, N, self.hidden_dim)
        e = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)
        return x + self.decoder(h.reshape(B * N, self.hidden_dim)).reshape(B, N, -1)


# ============================================================
# Helper Functions
# ============================================================

def compute_tidal_harmonics(global_hour):
    harmonics = []
    for period in TIDAL_PERIODS.values():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


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
    return model, ckpt


def run_rollout(model, mesh_data, val_data, device, max_hours, date_str):
    """Run autoregressive rollout."""
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
    depth_log = np.log10(np.maximum(np.abs(depth), 0.1))
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
    elevation = val_data['elevation']
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_base = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    cwl_prev = np.nan_to_num(elevation[0], nan=0.0)
    cwl_t = np.nan_to_num(elevation[1], nan=0.0)
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = [cwl_prev, cwl_t]
    ground_truth = [elevation[0], elevation[1]]

    with torch.no_grad():
        for t in range(1, max_hours + 1):
            dxdt = (current - current_prev) / DT_HOURS
            global_hour = global_hours_base + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)

            forcing_arr = np.stack([
                val_data['u10'][t], val_data['v10'][t], val_data['wind_speed'][t],
                val_data['wind_speed_sq'][t], val_data['wind_dir'][t], val_data['pressure'][t],
                val_data['dP_dx'][t], val_data['dP_dy'][t]
            ], axis=1).astype(np.float32)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1], nan=0.0))

            current_prev = current
            current = pred

    coords = np.stack([lon, lat], axis=1)
    return np.array(predictions), np.array(ground_truth), coords


def plot_spatial_rollout(predictions, ground_truth, coords, hours, output_path, date_str, model_name):
    """Generate side-by-side ground truth vs prediction scatter plots."""
    num_hours = len(hours)
    fig, axes = plt.subplots(num_hours, 2, figsize=(14, 4 * num_hours))

    if num_hours == 1:
        axes = axes.reshape(1, 2)

    lon = coords[:, 0]
    lat = coords[:, 1]

    # Get global min/max for consistent colorbar
    all_data = np.concatenate([ground_truth.flatten(), predictions.flatten()])
    valid_data = all_data[~np.isnan(all_data)]
    vmin, vmax = np.percentile(valid_data, [2, 98])

    for i, h in enumerate(hours):
        timestep = h + 1  # offset for initial conditions

        if timestep >= len(predictions):
            print(f"Warning: Hour {h} exceeds available data, skipping")
            continue

        gt = ground_truth[timestep]
        pred = predictions[timestep]

        valid = ~np.isnan(gt) & ~np.isnan(pred)
        rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))

        # Ground truth
        sc1 = axes[i, 0].scatter(lon, lat, c=gt, s=0.5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'STOFS Ground Truth (t+{h}h)', fontsize=11)
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Latitude')
        plt.colorbar(sc1, ax=axes[i, 0], label='Water Level (m)')

        # Prediction
        sc2 = axes[i, 1].scatter(lon, lat, c=pred, s=0.5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'GNN Prediction (t+{h}h), RMSE: {rmse*100:.1f}cm', fontsize=11)
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Latitude')
        plt.colorbar(sc2, ax=axes[i, 1], label='Water Level (m)')

    plt.suptitle(f'25K V2 Model Spatial Rollout - {date_str}\n{model_name}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_spatial_error(predictions, ground_truth, coords, hours, output_path, date_str):
    """Generate spatial error scatter plots."""
    num_plots = len(hours)
    cols = min(2, num_plots)
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))

    if num_plots == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    lon = coords[:, 0]
    lat = coords[:, 1]

    # Compute error limits
    all_errors = []
    for h in hours:
        timestep = h + 1
        if timestep < len(predictions):
            error = predictions[timestep] - ground_truth[timestep]
            valid = ~np.isnan(error)
            all_errors.append(error[valid])

    if all_errors:
        all_errors = np.concatenate(all_errors)
        vmax_err = np.percentile(np.abs(all_errors), 98)
    else:
        vmax_err = 0.5

    for idx, h in enumerate(hours):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        timestep = h + 1
        if timestep >= len(predictions):
            ax.axis('off')
            continue

        error = predictions[timestep] - ground_truth[timestep]
        valid = ~np.isnan(error)
        rmse = np.sqrt(np.mean(error[valid]**2))
        bias = np.mean(error[valid])

        sc = ax.scatter(lon, lat, c=error, s=0.5, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
        ax.set_title(f't+{h}h Error\nRMSE: {rmse*100:.1f}cm, Bias: {bias*100:.1f}cm', fontsize=11)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(sc, ax=ax, label='Error (m)')

    # Hide unused subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle(f'25K V2 Model Spatial Error Maps - {date_str}', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate spatial rollout visualizations for 25K V2 model')
    parser.add_argument('--date', type=str, default='20250115', help='Validation date (YYYYMMDD)')
    parser.add_argument('--hours', type=int, nargs='+', default=[6, 12, 24, 48],
                        help='Forecast hours to visualize')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_35.pt',
                        help='Checkpoint filename')
    parser.add_argument('--scatter-only', action='store_true', help='Only generate rollout plots')
    parser.add_argument('--error-only', action='store_true', help='Only generate error maps')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading model: {checkpoint_path}")
    model, ckpt = load_model(checkpoint_path, device)
    epoch = ckpt.get('epoch', 'unknown')
    model_name = f"Epoch {epoch} (3-step trained)"
    print(f"  Epoch: {epoch}")

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    print(f"Mesh: {len(mesh_data['lon']):,} nodes")

    # Load validation data
    data_path = DATA_DIR / f'processed_{args.date}.npz'
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    val_data = dict(np.load(data_path))
    print(f"Validation date: {args.date}")

    # Run rollout
    max_hour = max(args.hours)
    print(f"Running {max_hour}-hour rollout...")
    predictions, ground_truth, coords = run_rollout(model, mesh_data, val_data, device, max_hour, args.date)
    print(f"Rollout complete: {len(predictions)} timesteps")

    # Generate plots
    hours_str = '_'.join(map(str, args.hours))

    if not args.error_only:
        rollout_path = OUTPUT_DIR / f'spatial_rollout_scatter_temporal_{args.date}_h{hours_str}.png'
        plot_spatial_rollout(predictions, ground_truth, coords, args.hours, rollout_path, args.date, model_name)

    if not args.scatter_only:
        error_path = OUTPUT_DIR / f'spatial_error_scatter_temporal_{args.date}_h{hours_str}.png'
        plot_spatial_error(predictions, ground_truth, coords, args.hours, error_path, args.date)

    print("\nDone!")


if __name__ == '__main__':
    main()
