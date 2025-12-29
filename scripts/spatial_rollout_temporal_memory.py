#!/usr/bin/env python3
"""
Spatial Rollout Visualization for Temporal Memory GNN Model

Generates scatter plot visualizations showing ground truth vs predictions
and spatial error maps for specified forecast hours.

Usage:
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 12 24 48
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 12 24 36 48 --scatter-only
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 --error-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt

# Constants
WIND_SCALE = 15.0
ETA_SCALE = 2.0
DT_HOURS = 1.0  # 1-hour timesteps

# Tidal harmonic periods (hours) - must match training
M2_PERIOD = 12.42  # Principal lunar semi-diurnal
S2_PERIOD = 12.00  # Principal solar semi-diurnal

# Global epoch for continuous time (must match training)
from datetime import datetime
EPOCH_DATETIME = datetime(2025, 1, 1, 0, 0, 0)


# Model architecture (from train_25k_temporal_memory.py)
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
    """GNN with temporal memory and tidal harmonics for resolving phase ambiguity."""

    def __init__(
        self,
        state_dim: int = 1,
        temporal_dim: int = 6,  # η(t-1), dη/dt, + 4 tidal harmonics
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

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        delta = self.decoder(h)
        output = x + delta

        return output


def compute_tidal_harmonics(global_hour: float, num_nodes: int) -> np.ndarray:
    """Compute tidal harmonic features for a given global hour."""
    # M2 tidal constituent (12.42 hour period)
    phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
    sin_m2 = np.sin(phase_m2)
    cos_m2 = np.cos(phase_m2)

    # S2 tidal constituent (12.00 hour period)
    phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
    sin_s2 = np.sin(phase_s2)
    cos_s2 = np.cos(phase_s2)

    # Broadcast to all nodes [4] -> [num_nodes, 4]
    tidal_harmonics = np.array([sin_m2, cos_m2, sin_s2, cos_s2], dtype=np.float32)
    return np.tile(tidal_harmonics, (num_nodes, 1))


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = TemporalMemoryGNN(
        state_dim=1,
        temporal_dim=config.get('temporal_features', 2),
        static_feature_dim=config.get('static_features', 4),
        forcing_feature_dim=config.get('forcing_features', 3),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def run_rollout(model, mesh, data, device, max_hours: int = 120, date_str: str = None):
    """Run autoregressive rollout with temporal memory and tidal harmonics."""
    # Extract mesh data
    lon = mesh['lon'].astype(np.float32)
    lat = mesh['lat'].astype(np.float32)
    depth = mesh['depth'].astype(np.float32)
    edge_index = mesh['edge_index']
    num_nodes = len(lon)

    # Compute edge features
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    src, dst = edge_index
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    ).to(device)

    edge_index_t = torch.tensor(edge_index, dtype=torch.long).to(device)

    # Static features
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
    static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)

    # Extract time series data
    elevation = data['elevation']
    forcing = {
        'u10': data['u10'],
        'v10': data['v10'],
        'pressure': data['pressure'],
    }

    # Initialize with first two timesteps
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    predictions = [cwl_prev, cwl_t]  # t=0 and t=1h
    ground_truth = [elevation[0], elevation[1]]

    # Compute base time for tidal harmonics
    if date_str:
        base_dt = datetime.strptime(date_str, '%Y%m%d')
    else:
        base_dt = EPOCH_DATETIME
    base_hours = (base_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    # 1 step per hour for hourly data
    num_steps = max_hours
    num_steps = min(num_steps, len(elevation) - 2)

    with torch.no_grad():
        for t in range(1, num_steps + 1):
            # Compute temporal features
            dxdt = (current_cwl - current_prev) / DT_HOURS

            # Compute tidal harmonics for this timestep
            global_hour = base_hours + t
            tidal = compute_tidal_harmonics(global_hour, num_nodes)
            tidal_tensor = torch.tensor(tidal, dtype=torch.float32).to(device)

            # Static features with water level
            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).to(device)

            # Forcing
            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pres = forcing['pressure'][t].astype(np.float32)
            forcing_arr = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).to(device)

            with autocast('cuda'):
                pred = model(current_cwl, current_prev, dxdt, tidal_tensor, static_tensor, forcing_tensor, edge_index_t, edge_attr)

            pred_cwl = pred.squeeze().cpu().numpy() * ETA_SCALE
            predictions.append(pred_cwl)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            # Update temporal state
            current_prev = current_cwl
            current_cwl = pred

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
    vmin, vmax = np.percentile(all_data[~np.isnan(all_data)], [2, 98])

    for i, h in enumerate(hours):
        # Convert hours to timestep index (2 steps per hour, offset by initial 2 timesteps)
        timestep = h  # 1-hour timesteps

        if timestep >= len(predictions):
            print(f"Warning: Hour {h} exceeds available data, skipping")
            continue

        gt = ground_truth[timestep]
        pred = predictions[timestep]

        # Handle NaN
        valid = ~np.isnan(gt) & ~np.isnan(pred)
        rmse = np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2))

        # Ground truth
        sc1 = axes[i, 0].scatter(lon, lat, c=gt, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'STOFS Ground Truth (t={h}h)')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Latitude')
        plt.colorbar(sc1, ax=axes[i, 0], label='Water Level (m)')

        # Prediction
        sc2 = axes[i, 1].scatter(lon, lat, c=pred, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'GNN Prediction (t={h}h), RMSE: {rmse:.3f}m')
        axes[i, 1].set_xlabel('Longitude')
        axes[i, 1].set_ylabel('Latitude')
        plt.colorbar(sc2, ax=axes[i, 1], label='Water Level (m)')

    plt.suptitle(f'Temporal Memory GNN Spatial Rollout (Scatter) - {date_str}\nModel: {model_name}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rollout plot: {output_path}")


def plot_spatial_error(predictions, ground_truth, coords, hours, output_path, date_str):
    """Generate spatial error scatter plots."""
    num_plots = len(hours)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    if num_plots == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    lon = coords[:, 0]
    lat = coords[:, 1]

    # Compute all errors for consistent colorbar
    all_errors = []
    for h in hours:
        timestep = h  # 1-hour timesteps
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

        timestep = h  # 1-hour timesteps
        if timestep >= len(predictions):
            ax.axis('off')
            continue

        error = predictions[timestep] - ground_truth[timestep]
        valid = ~np.isnan(error)
        rmse = np.sqrt(np.mean(error[valid] ** 2))
        bias = np.mean(error[valid])

        sc = ax.scatter(lon, lat, c=error, s=1, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
        ax.set_title(f'Error (t={h}h)\nRMSE: {rmse:.3f}m, Bias: {bias:.3f}m')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(sc, ax=ax, label='Error (m)')

    # Hide unused subplots
    for idx in range(num_plots, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle(f'Temporal Memory GNN Spatial Error Maps (Scatter) - {date_str}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error plot: {output_path}")


def plot_hourly_snapshots(predictions, ground_truth, coords, output_dir, date_str, max_hours=120):
    """Generate individual hourly snapshot plots (GT vs Prediction vs Error)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lon = coords[:, 0]
    lat = coords[:, 1]

    # Get global min/max for consistent colorbar across all hours
    all_data = np.concatenate([ground_truth.flatten(), predictions.flatten()])
    vmin, vmax = np.percentile(all_data[~np.isnan(all_data)], [2, 98])

    # Compute error limits for consistent error colorbar
    all_errors = predictions - ground_truth
    valid_errors = all_errors[~np.isnan(all_errors)]
    vmax_err = np.percentile(np.abs(valid_errors), 98)

    num_hours = min(max_hours, len(predictions) - 1)

    for h in range(num_hours + 1):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        gt = ground_truth[h]
        pred = predictions[h]
        error = pred - gt

        # Handle NaN
        valid = ~np.isnan(gt) & ~np.isnan(pred)
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2))
            bias = np.mean(pred[valid] - gt[valid])
        else:
            rmse = 0.0
            bias = 0.0

        # Ground truth
        sc1 = axes[0].scatter(lon, lat, c=gt, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'STOFS Ground Truth', fontsize=12)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(sc1, ax=axes[0], label='Water Level (m)')

        # Prediction
        sc2 = axes[1].scatter(lon, lat, c=pred, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'GNN Prediction', fontsize=12)
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        plt.colorbar(sc2, ax=axes[1], label='Water Level (m)')

        # Error (Prediction - Ground Truth)
        sc3 = axes[2].scatter(lon, lat, c=error, s=1, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)
        axes[2].set_title(f'Error (RMSE: {rmse:.3f}m, Bias: {bias:.3f}m)', fontsize=12)
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        plt.colorbar(sc3, ax=axes[2], label='Error (m)')

        # Parse date for title
        try:
            dt = datetime.strptime(date_str, '%Y%m%d')
            from datetime import timedelta
            current_dt = dt + timedelta(hours=h)
            time_str = current_dt.strftime('%Y-%m-%d %H:%M UTC')
        except:
            time_str = f't+{h}h'

        plt.suptitle(f'Temporal Memory GNN - {date_str} - Hour {h:02d}\n{time_str}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f'spatial_hour_{h:02d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        if h % 6 == 0:
            print(f"  Saved hour {h:02d}: {output_path}")

    print(f"Generated {num_hours + 1} hourly snapshots in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate spatial rollout visualizations for Temporal Memory GNN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 12 24 48
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 12 24 36 48 --scatter-only
    python scripts/spatial_rollout_temporal_memory.py --date 20251128 --hours 6 --error-only
        """
    )

    parser.add_argument('--date', type=str, required=True,
                        help='Date string for data file (e.g., 20251128)')
    parser.add_argument('--hours', type=int, nargs='+', default=[6, 12, 24, 48, 72, 96, 120],
                        help='Forecast hours to visualize (default: 6 12 24 48 72 96 120)')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/checkpoints/best_temporal_memory_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--mesh', type=str, default='data/processed_25k/mesh_25k.npz',
                        help='Path to mesh file')
    parser.add_argument('--data-dir', type=str, default='data/processed_25k',
                        help='Directory containing processed data')
    parser.add_argument('--output-dir', type=str, default='outputs/figures',
                        help='Directory for output figures')
    parser.add_argument('--scatter-only', action='store_true',
                        help='Only generate rollout scatter plots (skip error maps)')
    parser.add_argument('--error-only', action='store_true',
                        help='Only generate error maps (skip rollout plots)')
    parser.add_argument('--hourly', action='store_true',
                        help='Generate individual hourly snapshot plots')
    parser.add_argument('--hourly-only', action='store_true',
                        help='Only generate hourly snapshots (skip other plots)')

    args = parser.parse_args()

    # Validate hours
    valid_hours = [h for h in args.hours if h > 0 and h <= 120]
    if not valid_hours:
        print("Error: Hours must be between 1 and 120")
        sys.exit(1)

    if len(valid_hours) != len(args.hours):
        print(f"Warning: Some hours filtered. Using: {valid_hours}")

    # Setup paths
    project_root = Path(__file__).parent.parent
    checkpoint_path = project_root / args.checkpoint
    mesh_path = project_root / args.mesh
    data_path = project_root / args.data_dir / f'processed_{args.date}.npz'
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check files exist
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not mesh_path.exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model, checkpoint = load_model(str(checkpoint_path), device)
    epoch = checkpoint.get('epoch', 'unknown')
    model_name = checkpoint_path.stem
    print(f"Model loaded (Epoch {epoch})")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Config: {checkpoint.get('config', {})}")

    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = dict(np.load(str(mesh_path)))
    print(f"Mesh: {len(mesh['lon'])} nodes, {mesh['edge_index'].shape[1]} edges")

    # Load date data
    print(f"Loading data from: {data_path}")
    data = dict(np.load(str(data_path)))
    print(f"Data loaded: {data['elevation'].shape[0]} timesteps")

    # Run rollout
    max_hour = max(valid_hours)
    print(f"Running rollout for {max_hour} hours...")
    predictions, ground_truth, coords = run_rollout(model, mesh, data, device, max_hour, args.date)
    print(f"Rollout complete: {len(predictions)} timesteps ({len(predictions):.0f} hours)")

    # Generate hours string for filenames
    hours_str = '_'.join(map(str, valid_hours))

    # Generate plots
    if args.hourly_only:
        # Only generate hourly snapshots
        hourly_dir = output_dir / f'hourly_{args.date}'
        print(f"Generating hourly snapshots...")
        plot_hourly_snapshots(predictions, ground_truth, coords, hourly_dir, args.date, max_hour)
    else:
        if not args.error_only:
            rollout_path = output_dir / f'spatial_rollout_scatter_temporal_{args.date}_h{hours_str}.png'
            plot_spatial_rollout(predictions, ground_truth, coords, valid_hours,
                                str(rollout_path), args.date, f'{model_name} (Epoch {epoch})')

        if not args.scatter_only:
            error_path = output_dir / f'spatial_error_scatter_temporal_{args.date}_h{hours_str}.png'
            plot_spatial_error(predictions, ground_truth, coords, valid_hours,
                              str(error_path), args.date)

        if args.hourly:
            hourly_dir = output_dir / f'hourly_{args.date}'
            print(f"Generating hourly snapshots...")
            plot_hourly_snapshots(predictions, ground_truth, coords, hourly_dir, args.date, max_hour)

    print("Done!")


if __name__ == '__main__':
    main()
