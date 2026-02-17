#!/usr/bin/env python3
"""
Generate rollout plots from trained multi-date CWL GNN model.
Standalone script that loads model and data directly.
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# Base directory
OUTPUT_DIR = str(Path(__file__).resolve().parent.parent)

# Constants
HIDDEN_DIM = 96
NUM_LAYERS = 6
WIND_SCALE = 20.0
ETA_SCALE = 2.0


class SWEInspiredGraphBlock(nn.Module):
    """Graph neural network block inspired by SWE physics."""

    def __init__(self, hidden_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing

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

    def _edge_update(self, edge_attr, h_src, h_dst, h_gradient):
        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        return edge_msg

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        h_gradient = h_dst - h_src

        edge_msg = self._edge_update(edge_attr, h_src, h_dst, h_gradient)

        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr


class PhysicsInformedCWLModel(nn.Module):
    """GNN for Coastal Water Level prediction with physics-informed design."""

    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int = 6,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing

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

        self.layers = nn.ModuleList([
            SWEInspiredGraphBlock(hidden_dim, use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        static_features: torch.Tensor,
        forcing_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        node_input = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_input)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        out = self.decoder(h)
        return out


def load_data_and_model(device):
    """Load preprocessed data and trained model."""

    # Load preprocessed file
    data_file = f'{OUTPUT_DIR}/data/processed/processed_20251130.npz'
    print(f"Loading data: {data_file}")
    npz = np.load(data_file)

    lon = npz['lon'].astype(np.float32)
    lat = npz['lat'].astype(np.float32)
    depth = npz['depth'].astype(np.float32)
    edge_index = npz['edge_index']
    elevation = npz['elevation'].astype(np.float32)
    u10 = npz['u10'].astype(np.float32)
    v10 = npz['v10'].astype(np.float32)
    pressure = npz['pressure'].astype(np.float32)

    print(f"  Nodes: {len(lon)}, Timesteps: {elevation.shape[0]}")

    # Find valid nodes (no NaN)
    valid_mask = np.all(~np.isnan(elevation), axis=0)
    valid_indices = np.where(valid_mask)[0]
    print(f"  Valid nodes: {len(valid_indices)} / {len(lon)}")

    # Filter to valid nodes
    lon = lon[valid_indices]
    lat = lat[valid_indices]
    depth = depth[valid_indices]
    elevation = elevation[:, valid_indices]
    u10 = u10[:, valid_indices]
    v10 = v10[:, valid_indices]
    pressure = pressure[:, valid_indices]

    # Rebuild edge index
    old_to_new = {old: new for new, old in enumerate(valid_indices)}
    new_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in old_to_new and dst in old_to_new:
            new_edges.append([old_to_new[src], old_to_new[dst]])
    edge_index = np.array(new_edges).T
    print(f"  Edges: {edge_index.shape[1]}")

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
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    # Load model
    model_path = f'{OUTPUT_DIR}/outputs/checkpoints/best_multidate_model.pt'
    print(f"\nLoading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = PhysicsInformedCWLModel(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_checkpointing=False
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'static_base': static_base,
        'x_cart': x_cart,
        'y_cart': y_cart,
    }, model


def run_rollout(data, model, start_t, num_steps, device):
    """Run autoregressive rollout prediction."""
    predictions = []
    ground_truth = []

    current_eta = data['elevation'][start_t].copy()

    # Prepare tensors that don't change
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long).to(device)
    edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        for step in range(num_steps):
            t = start_t + step
            if t >= data['elevation'].shape[0] - 1:
                break

            # Normalize elevation
            eta_normalized = current_eta / ETA_SCALE

            # Compute water level feature
            water_level = data['depth'] + current_eta
            water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

            static_features = np.concatenate([
                data['static_base'],
                water_level_norm[:, np.newaxis]
            ], axis=1)

            # Forcing
            u10 = data['u10'][t] / WIND_SCALE
            v10 = data['v10'][t] / WIND_SCALE
            pressure = data['pressure'][t]  # Already normalized

            forcing_features = np.stack([u10, v10, pressure], axis=1)

            # Convert to tensors
            x = torch.tensor(eta_normalized[:, np.newaxis], dtype=torch.float32).to(device)
            static_feat = torch.tensor(static_features, dtype=torch.float32).to(device)
            forcing_feat = torch.tensor(forcing_features, dtype=torch.float32).to(device)

            # Predict
            pred = model(x, static_feat, forcing_feat, edge_index, edge_attr)
            pred_eta = pred.squeeze(-1).cpu().numpy() * ETA_SCALE

            predictions.append(pred_eta)
            ground_truth.append(data['elevation'][t + 1])

            # Autoregressive: use prediction as next input
            current_eta = pred_eta

    return predictions, ground_truth


def plot_rollout_spatial(data, predictions, ground_truth, output_path):
    """Create spatial comparison plots at key forecast hours."""
    lon, lat = data['lon'], data['lat']
    timesteps = [0, 5, 11, 23, 47]

    fig, axes = plt.subplots(len(timesteps), 3, figsize=(16, 4*len(timesteps)))

    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        pred = predictions[t]
        truth = ground_truth[t]
        error = pred - truth

        vmin = min(pred.min(), truth.min())
        vmax = max(pred.max(), truth.max())

        # Prediction
        sc1 = axes[i, 0].scatter(lon, lat, c=pred, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f't+{t+1}h Predicted')
        axes[i, 0].set_xlabel('Longitude')
        axes[i, 0].set_ylabel('Latitude')
        plt.colorbar(sc1, ax=axes[i, 0], label='Elevation (m)')

        # Ground Truth
        sc2 = axes[i, 1].scatter(lon, lat, c=truth, s=1, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f't+{t+1}h Ground Truth')
        axes[i, 1].set_xlabel('Longitude')
        plt.colorbar(sc2, ax=axes[i, 1], label='Elevation (m)')

        # Error
        err_max = max(abs(error.min()), abs(error.max()), 0.1)
        sc3 = axes[i, 2].scatter(lon, lat, c=error, s=1, cmap='RdBu_r', vmin=-err_max, vmax=err_max)
        rmse = np.sqrt(np.mean(error**2))
        axes[i, 2].set_title(f't+{t+1}h Error (RMSE: {rmse:.4f}m)')
        axes[i, 2].set_xlabel('Longitude')
        plt.colorbar(sc3, ax=axes[i, 2], label='Error (m)')

    plt.suptitle('Multi-Date CWL GNN - 48h Rollout Predictions (Nov 30, 2025)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_rollout_timeseries(predictions, ground_truth, output_path):
    """Plot time series comparison."""
    times = np.arange(len(predictions)) + 1

    pred_mean = [p.mean() for p in predictions]
    truth_mean = [t.mean() for t in ground_truth]
    rmse = [np.sqrt(np.mean((p - t)**2)) for p, t in zip(predictions, ground_truth)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean elevation
    ax1 = axes[0, 0]
    ax1.plot(times, pred_mean, 'b-', label='Predicted', linewidth=2)
    ax1.plot(times, truth_mean, 'r--', label='Ground Truth', linewidth=2)
    ax1.set_xlabel('Forecast Hour')
    ax1.set_ylabel('Mean Elevation (m)')
    ax1.set_title('Domain-Average Water Surface Elevation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # RMSE over time
    ax2 = axes[0, 1]
    ax2.plot(times, rmse, 'g-', linewidth=2)
    ax2.set_xlabel('Forecast Hour')
    ax2.set_ylabel('RMSE (m)')
    ax2.set_title('RMSE vs Forecast Hour')
    ax2.grid(True, alpha=0.3)

    # Scatter t+6h
    if len(predictions) >= 6:
        ax3 = axes[1, 0]
        ax3.scatter(ground_truth[5], predictions[5], s=1, alpha=0.5)
        lims = [min(ground_truth[5].min(), predictions[5].min()),
                max(ground_truth[5].max(), predictions[5].max())]
        ax3.plot(lims, lims, 'r--', linewidth=2, label='1:1')
        ax3.set_xlabel('Ground Truth (m)')
        ax3.set_ylabel('Predicted (m)')
        ax3.set_title(f't+6h (RMSE: {rmse[5]:.4f}m, R={np.corrcoef(predictions[5], ground_truth[5])[0,1]:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Scatter t+24h
    if len(predictions) >= 24:
        ax4 = axes[1, 1]
        ax4.scatter(ground_truth[23], predictions[23], s=1, alpha=0.5)
        lims = [min(ground_truth[23].min(), predictions[23].min()),
                max(ground_truth[23].max(), predictions[23].max())]
        ax4.plot(lims, lims, 'r--', linewidth=2, label='1:1')
        ax4.set_xlabel('Ground Truth (m)')
        ax4.set_ylabel('Predicted (m)')
        ax4.set_title(f't+24h (RMSE: {rmse[23]:.4f}m, R={np.corrcoef(predictions[23], ground_truth[23])[0,1]:.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.suptitle('Multi-Date CWL GNN - Rollout Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("GENERATING ROLLOUT PLOTS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data and model
    data, model = load_data_and_model(device)

    # Run 48-hour rollout
    print("\nRunning 48-hour rollout...")
    start_t = 50  # Start from middle of sequence
    num_steps = 48

    predictions, ground_truth = run_rollout(data, model, start_t, num_steps, device)
    print(f"Generated {len(predictions)} predictions")

    # Create output directory
    fig_dir = f'{OUTPUT_DIR}/outputs/figures'
    os.makedirs(fig_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_rollout_spatial(data, predictions, ground_truth, f'{fig_dir}/multidate_rollout.png')
    plot_rollout_timeseries(predictions, ground_truth, f'{fig_dir}/multidate_rollout_timeseries.png')

    # Print RMSE table
    print("\n" + "=" * 50)
    print("ROLLOUT RMSE BY FORECAST HOUR")
    print("=" * 50)
    print(f"{'Hour':>8} | {'RMSE (m)':>10} | {'Correlation':>12} | {'Bias (m)':>10}")
    print("-" * 50)

    for t in [0, 5, 11, 17, 23, 35, 47]:
        if t < len(predictions):
            rmse = np.sqrt(np.mean((predictions[t] - ground_truth[t])**2))
            corr = np.corrcoef(predictions[t], ground_truth[t])[0, 1]
            bias = np.mean(predictions[t] - ground_truth[t])
            print(f"t+{t+1:>4}h | {rmse:>10.4f} | {corr:>12.4f} | {bias:>+10.4f}")

    print("-" * 50)
    print("\nDone!")


if __name__ == '__main__':
    main()
