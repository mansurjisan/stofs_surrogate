#!/usr/bin/env python3
"""
Rollout Visualization for 80k Node STOFS-GNN Model

Generates multi-step rollout plots showing model predictions vs ground truth.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.tri as mtri
from datetime import datetime, timedelta
import argparse
import os

# ============================================================
# Model Definition (must match training)
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


class BatchedTemporalMemoryGNN(nn.Module):
    """TRUE batched GNN model."""
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
# Data Loading
# ============================================================

def load_mesh(mesh_path):
    """Load mesh data."""
    mesh = np.load(mesh_path)
    return {
        'lon': mesh['lon'].astype(np.float32),
        'lat': mesh['lat'].astype(np.float32),
        'depth': mesh['depth'].astype(np.float32),
        'edge_index': mesh['edge_index'],
    }


def load_date_data(data_path, date_str):
    """Load data for a specific date."""
    filepath = Path(data_path) / f"processed_{date_str}.npz"
    data = np.load(filepath)
    return {
        'elevation': data['elevation'].astype(np.float32),
        'u10': data['u10'].astype(np.float32),
        'v10': data['v10'].astype(np.float32),
        'pressure': data['pressure'].astype(np.float32),
    }


def compute_static_base(mesh_data):
    """Compute base static node features (without water level)."""
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']

    # Convert to Cartesian coordinates
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    # Normalize
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1

    # Log-scaled depth
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)
    return static_base, depth


def compute_static_features_with_wl(static_base, depth, elevation):
    """Compute full static features including water level normalization."""
    water_level = depth + elevation
    wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
    static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
    return static.astype(np.float32)


def compute_edge_features(mesh_data):
    """Compute edge features matching training."""
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = mesh_data['edge_index']

    # Convert to Cartesian coordinates
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    dist_norm = dist / (dist.mean() + 1e-8)
    angle = np.arctan2(dy, dx) / np.pi

    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_diff = (depth_log[dst] - depth_log[src])
    depth_diff = depth_diff / (np.abs(depth_diff).max() + 1e-8)

    return np.stack([dist_norm, angle, depth_diff], axis=-1).astype(np.float32)


def compute_tidal_harmonics(timestep, num_nodes, dt_hours=1.0):
    """Compute tidal harmonic features."""
    hours = timestep * dt_hours
    M2_period = 12.42
    S2_period = 12.0
    omega_M2 = 2 * np.pi / M2_period
    omega_S2 = 2 * np.pi / S2_period

    sin_M2 = np.sin(omega_M2 * hours)
    cos_M2 = np.cos(omega_M2 * hours)
    sin_S2 = np.sin(omega_S2 * hours)
    cos_S2 = np.cos(omega_S2 * hours)

    return np.tile([sin_M2, cos_M2, sin_S2, cos_S2], (num_nodes, 1)).astype(np.float32)


# ============================================================
# Rollout
# ============================================================

def run_rollout(model, mesh_data, date_data, start_timestep, num_steps, eta_scale=2.0, device='cpu'):
    """
    Run multi-step rollout starting from a specific timestep.

    Args:
        model: The GNN model
        mesh_data: Mesh dictionary
        date_data: Data dictionary for the date
        start_timestep: Starting timestep index
        num_steps: Number of steps to rollout
        eta_scale: Normalization scale for elevation
        device: Computing device

    Returns:
        predictions: [num_steps, num_nodes] array of predictions
        ground_truth: [num_steps, num_nodes] array of ground truth
    """
    model.eval()

    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(compute_edge_features(mesh_data), dtype=torch.float32).to(device)
    static_base, depth = compute_static_base(mesh_data)

    num_nodes = len(mesh_data['lon'])

    # Get elevation and forcing data (handle NaN by replacing with 0)
    elevation = np.nan_to_num(date_data['elevation'], nan=0.0)
    u10 = np.nan_to_num(date_data['u10'], nan=0.0)
    v10 = np.nan_to_num(date_data['v10'], nan=0.0)
    pressure = np.nan_to_num(date_data['pressure'], nan=0.0)

    # Normalize wind (match training WIND_SCALE = 15.0)
    u10_norm = u10 / 15.0
    v10_norm = v10 / 15.0
    # Pressure is used raw (not normalized) in training

    # Initialize with ground truth
    wl = elevation[start_timestep] / eta_scale
    wl_prev = elevation[start_timestep - 1] / eta_scale
    cwl_t = elevation[start_timestep]  # Raw elevation for static features

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for step in range(num_steps):
            current_t = start_timestep + step

            if current_t + 1 >= elevation.shape[0]:
                break

            # Prepare inputs
            x = torch.tensor(wl, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # [1, N, 1]
            x_prev = torch.tensor(wl_prev, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            dxdt = x - x_prev

            tidal = compute_tidal_harmonics(current_t, num_nodes)
            tidal_t = torch.tensor(tidal, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 4]

            # Forcing: u10/WIND_SCALE, v10/WIND_SCALE, pressure (raw)
            forcing = np.stack([u10_norm[current_t], v10_norm[current_t], pressure[current_t]], axis=-1)
            forcing_t = torch.tensor(forcing, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 3]

            # Static features include water level normalization
            static = compute_static_features_with_wl(static_base, depth, cwl_t)
            static_t = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 4]

            # Forward pass
            pred = model(x, x_prev, dxdt, tidal_t, static_t, forcing_t, edge_index, edge_attr)

            # Store results (denormalize prediction)
            pred_np = pred.squeeze().cpu().numpy() * eta_scale
            predictions.append(pred_np)
            ground_truth.append(elevation[current_t + 1])

            # Update state for next step
            wl_prev = wl.copy()
            wl = pred.squeeze().cpu().numpy()  # Normalized prediction
            cwl_t = wl * eta_scale  # Raw elevation for static features

    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Visualization
# ============================================================

def plot_spatial_comparison(mesh_data, predictions, ground_truth, timesteps, output_dir):
    """Create spatial comparison plots."""
    lon = mesh_data['lon']
    lat = mesh_data['lat']

    # Create triangulation for plotting
    triang = mtri.Triangulation(lon, lat)

    for idx, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        pred = predictions[t]
        truth = ground_truth[t]
        error = pred - truth

        # Common color scale
        vmin = min(pred.min(), truth.min())
        vmax = max(pred.max(), truth.max())

        # Ground truth
        ax = axes[0]
        tcf = ax.tricontourf(triang, truth, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label='Water Level (m)')
        ax.set_title(f'Ground Truth (t+{t+1}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Prediction
        ax = axes[1]
        tcf = ax.tricontourf(triang, pred, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label='Water Level (m)')
        ax.set_title(f'GNN Prediction (t+{t+1}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Error
        ax = axes[2]
        err_max = max(abs(error.min()), abs(error.max()), 0.01)
        norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)
        tcf = ax.tricontourf(triang, error, levels=50, cmap='RdBu_r', norm=norm)
        plt.colorbar(tcf, ax=ax, label='Error (m)')
        ax.set_title(f'Error (Pred - Truth) | RMSE: {np.sqrt((error**2).mean()):.4f}m')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.tight_layout()
        plt.savefig(output_dir / f'spatial_comparison_t{t+1:02d}h.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved spatial_comparison_t{t+1:02d}h.png")


def plot_error_over_time(predictions, ground_truth, output_dir):
    """Plot RMSE and MAE over rollout steps."""
    num_steps = len(predictions)

    rmse = []
    mae = []
    max_err = []

    for t in range(num_steps):
        error = predictions[t] - ground_truth[t]
        rmse.append(np.sqrt((error**2).mean()))
        mae.append(np.abs(error).mean())
        max_err.append(np.abs(error).max())

    hours = np.arange(1, num_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, rmse, 'b-o', label='RMSE', linewidth=2, markersize=6)
    ax.plot(hours, mae, 'g-s', label='MAE', linewidth=2, markersize=6)
    ax.plot(hours, max_err, 'r-^', label='Max Error', linewidth=2, markersize=6)

    ax.set_xlabel('Forecast Hour', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('Rollout Error vs Forecast Hour', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, num_steps + 1)
    ax.set_ylim(0, None)

    # Add cm scale on right axis
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: x*100, lambda x: x/100))
    ax2.set_ylabel('Error (cm)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved error_over_time.png")

    return rmse, mae


def plot_station_timeseries(mesh_data, predictions, ground_truth, station_indices, station_names, output_dir):
    """Plot time series at specific stations."""
    num_steps = len(predictions)
    hours = np.arange(1, num_steps + 1)

    num_stations = len(station_indices)
    fig, axes = plt.subplots(num_stations, 1, figsize=(12, 3*num_stations), sharex=True)
    if num_stations == 1:
        axes = [axes]

    for ax, idx, name in zip(axes, station_indices, station_names):
        truth_series = [ground_truth[t][idx] for t in range(num_steps)]
        pred_series = [predictions[t][idx] for t in range(num_steps)]

        ax.plot(hours, truth_series, 'b-', label='Ground Truth', linewidth=2)
        ax.plot(hours, pred_series, 'r--', label='GNN Prediction', linewidth=2)
        ax.fill_between(hours, truth_series, pred_series, alpha=0.3, color='red')

        rmse = np.sqrt(np.mean((np.array(pred_series) - np.array(truth_series))**2))
        ax.set_title(f'{name} | RMSE: {rmse:.4f}m ({rmse*100:.2f}cm)', fontsize=12)
        ax.set_ylabel('Water Level (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Forecast Hour')
    plt.tight_layout()
    plt.savefig(output_dir / 'station_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved station_timeseries.png")


def find_key_stations(mesh_data):
    """Find indices of key coastal stations based on location."""
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']

    # Define approximate locations of interest
    stations = {
        'NYC/Battery': (-74.0, 40.7),
        'Atlantic City': (-74.4, 39.4),
        'Chesapeake Bay': (-76.3, 37.0),
        'Norfolk': (-76.3, 36.9),
        'Outer Banks': (-75.5, 35.8),
    }

    indices = []
    names = []

    for name, (target_lon, target_lat) in stations.items():
        # Find nearest coastal node (depth < 10m)
        mask = depth < 10
        if not mask.any():
            mask = np.ones(len(lon), dtype=bool)

        dist = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
        dist[~mask] = np.inf
        idx = np.argmin(dist)

        if dist[idx] < 1.0:  # Within 1 degree
            indices.append(idx)
            names.append(f"{name} ({lon[idx]:.2f}, {lat[idx]:.2f})")

    return indices, names


def main():
    parser = argparse.ArgumentParser(description='Rollout visualization for 80k STOFS-GNN')
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_80k_h100/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                        default='/mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a',
                        help='Path to processed data directory')
    parser.add_argument('--date', type=str, default='20231015',
                        help='Date to run rollout on (YYYYMMDD)')
    parser.add_argument('--start-timestep', type=int, default=24,
                        help='Starting timestep for rollout')
    parser.add_argument('--num-steps', type=int, default=48,
                        help='Number of rollout steps (hours)')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/d/AI_4_STOFS/stofs_surrogate/plots/rollout_80k',
                        help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computing device')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print("80k Node STOFS-GNN Rollout Visualization")
    print(f"=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data dir: {args.data_dir}")
    print(f"Date: {args.date}")
    print(f"Rollout: {args.num_steps} steps from timestep {args.start_timestep}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 6)

    model = BatchedTemporalMemoryGNN(
        state_dim=1,
        temporal_dim=6,  # x_prev + dxdt (2) + tidal (4) -> but actually tidal is separate
        static_feature_dim=4,
        forcing_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Loaded epoch {ckpt.get('epoch', 'N/A')} with val_loss {ckpt.get('val_loss', 'N/A'):.6f}")

    # Load data
    print("Loading mesh and data...")
    mesh_path = Path(args.data_dir) / 'mesh.npz'
    mesh_data = load_mesh(mesh_path)
    print(f"  Mesh: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    date_data = load_date_data(args.data_dir, args.date)
    print(f"  Date {args.date}: {date_data['elevation'].shape[0]} timesteps")

    # Run rollout
    print(f"\nRunning {args.num_steps}-step rollout...")
    predictions, ground_truth = run_rollout(
        model, mesh_data, date_data,
        start_timestep=args.start_timestep,
        num_steps=args.num_steps,
        device=device
    )
    print(f"  Completed {len(predictions)} steps")

    # Calculate summary statistics
    print("\nRollout Statistics:")
    for t in [0, 5, 11, 23, 47]:
        if t < len(predictions):
            error = predictions[t] - ground_truth[t]
            rmse = np.sqrt((error**2).mean())
            print(f"  t+{t+1:2d}h: RMSE = {rmse:.4f}m ({rmse*100:.2f}cm)")

    # Generate plots
    print("\nGenerating visualizations...")

    # Spatial comparisons at key forecast hours
    plot_spatial_comparison(mesh_data, predictions, ground_truth,
                           timesteps=[0, 5, 11, 23], output_dir=output_dir)

    # Error over time
    rmse, mae = plot_error_over_time(predictions, ground_truth, output_dir)

    # Station time series
    station_indices, station_names = find_key_stations(mesh_data)
    if station_indices:
        plot_station_timeseries(mesh_data, predictions, ground_truth,
                               station_indices, station_names, output_dir)

    print(f"\nAll plots saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
