#!/usr/bin/env python3
"""
Spatial Rollout Visualization for 25K Node Model

Generates scatter plot visualizations showing ground truth vs predictions
and spatial error maps for specified forecast hours.

Usage:
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 12 24 48
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 12 24 36 48 --scatter-only
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 12 24 --error-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Constants
WIND_SCALE = 30.0
ETA_SCALE = 2.0


# Model architecture (from train_25k_15day.py)
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


class PhysicsInformedCWLModel(nn.Module):
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


def compute_static_features(lon, lat, depth):
    """Compute static node features."""
    # Normalize lon/lat to [0, 1]
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    x_norm = (lon - lon_min) / (lon_max - lon_min)
    y_norm = (lat - lat_min) / (lat_max - lat_min)

    # Normalize depth (log scale)
    depth_norm = np.log1p(np.clip(depth, 0, 1000)) / np.log1p(1000)

    # Cartesian coords for edge features
    R = 6371  # km
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x_cart = R * np.cos(lat_rad) * np.cos(lon_rad)
    y_cart = R * np.cos(lat_rad) * np.sin(lon_rad)

    static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)
    return static_base, x_cart, y_cart


def compute_edge_features(x_cart, y_cart, edge_index):
    """Compute normalized edge features."""
    src = edge_index[0]
    dst = edge_index[1]

    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1)
    return torch.tensor(edge_attr, dtype=torch.float32)


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = PhysicsInformedCWLModel(
        state_dim=1,
        static_feature_dim=config['static_features'],
        forcing_feature_dim=config['forcing_features'],
        edge_feature_dim=3,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def run_rollout(model, mesh, data, device, max_hours: int = 48):
    """Run autoregressive rollout."""
    # Extract mesh data
    lon = mesh['lon']
    lat = mesh['lat']
    depth = mesh['depth']
    edge_index = mesh['edge_index']

    # Extract time series data
    elevation = data['elevation']  # (T, N)
    u10 = data['u10']
    v10 = data['v10']
    pressure = data['pressure']

    # Compute features
    static_base, x_cart, y_cart = compute_static_features(lon, lat, depth)
    edge_attr = compute_edge_features(x_cart, y_cart, edge_index)

    # Convert to tensors
    edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_t = edge_attr.to(device)
    static_base_t = torch.tensor(static_base, dtype=torch.float32, device=device)
    depth_t = torch.tensor(depth, dtype=torch.float32, device=device)

    # Initialize from t=0
    cwl_t = elevation[0].astype(np.float32)
    cwl_t = np.nan_to_num(cwl_t, nan=0.0)
    x = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(-1).to(device)

    predictions = [cwl_t.copy()]
    ground_truth = [elevation[0]]

    # 2 steps per hour for 30-min data
    num_steps = max_hours * 2
    num_steps = min(num_steps, len(elevation) - 1)

    with torch.no_grad():
        for t in range(num_steps):
            # Update water level feature (4th static feature)
            current_cwl = x.squeeze(-1) * ETA_SCALE
            water_level = depth_t + current_cwl
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

            # Full static features [x_norm, y_norm, depth_norm, wl_norm]
            static_full = torch.cat([static_base_t, wl_norm.unsqueeze(-1)], dim=-1)

            # Get forcing for current timestep
            u10_t = u10[t].astype(np.float32) / WIND_SCALE
            v10_t = v10[t].astype(np.float32) / WIND_SCALE
            pres_t = pressure[t].astype(np.float32)

            forcing_t = torch.tensor(
                np.stack([u10_t, v10_t, pres_t], axis=1), dtype=torch.float32
            ).to(device)

            # Model forward pass
            x_next = model(x, static_full, forcing_t, edge_index_t, edge_attr_t)

            # Store prediction and ground truth at hourly intervals (every 2 steps)
            if (t + 1) % 2 == 0:  # At 1h, 2h, 3h, etc.
                pred_cwl = x_next.squeeze(-1).cpu().numpy() * ETA_SCALE
                predictions.append(pred_cwl)
                ground_truth.append(elevation[t + 1])

            # Update state for next step
            x = x_next

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
        timestep = h  # predictions are hourly

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

    plt.suptitle(f'25K Node Model Spatial Rollout (Scatter) - {date_str}\nModel: {model_name}', y=1.02)
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
        timestep = h
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

        timestep = h
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

    plt.suptitle(f'25K Node Model Spatial Error Maps (Scatter) - {date_str}', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved error plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate spatial rollout visualizations for 25K node model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 12 24 48
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 12 24 36 48 --scatter-only
    python scripts/spatial_rollout_25k.py --date 20251128 --hours 6 --error-only
        """
    )

    parser.add_argument('--date', type=str, required=True,
                        help='Date string for data file (e.g., 20251128)')
    parser.add_argument('--hours', type=int, nargs='+', default=[6, 12, 24, 36, 48],
                        help='Forecast hours to visualize (default: 6 12 24 36 48)')
    parser.add_argument('--checkpoint', type=str,
                        default='outputs/checkpoints/best_25k_15day_model.pt',
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

    args = parser.parse_args()

    # Validate hours
    valid_hours = [h for h in args.hours if h > 0 and h <= 48]
    if not valid_hours:
        print("Error: Hours must be between 1 and 48")
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

    # Load mesh
    print(f"Loading mesh from: {mesh_path}")
    mesh = np.load(str(mesh_path))
    print(f"Mesh: {len(mesh['lon'])} nodes, {mesh['edge_index'].shape[1]} edges")

    # Load date data
    print(f"Loading data from: {data_path}")
    data = np.load(str(data_path))
    print(f"Data loaded: {data['elevation'].shape[0]} timesteps")

    # Run rollout
    max_hour = max(valid_hours)
    print(f"Running rollout for {max_hour} hours...")
    predictions, ground_truth, coords = run_rollout(model, mesh, data, device, max_hour)
    print(f"Rollout complete: {len(predictions)} hourly timesteps")

    # Generate hours string for filenames
    hours_str = '_'.join(map(str, valid_hours))

    # Generate plots
    if not args.error_only:
        rollout_path = output_dir / f'spatial_rollout_scatter_25k_{args.date}_h{hours_str}.png'
        plot_spatial_rollout(predictions, ground_truth, coords, valid_hours,
                            str(rollout_path), args.date, f'{model_name} (Epoch {epoch})')

    if not args.scatter_only:
        error_path = output_dir / f'spatial_error_scatter_25k_{args.date}_h{hours_str}.png'
        plot_spatial_error(predictions, ground_truth, coords, valid_hours,
                          str(error_path), args.date)

    print("Done!")


if __name__ == '__main__':
    main()
