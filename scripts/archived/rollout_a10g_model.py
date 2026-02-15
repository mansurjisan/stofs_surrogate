#!/usr/bin/env python3
"""
Generate rollout predictions from A10G trained model.
Uses the model architecture from train_cwl_gnn_a10g.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints/best_a10g_model.pt'
DATA_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_optimized'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures'

# Which date to use for rollout (validation date)
ROLLOUT_DATE = '20251130'

# Normalization constants
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Rollout settings
NUM_ROLLOUT_STEPS = 48  # 48 hours


# ============================================================
# Model Architecture (from train_cwl_gnn_a10g.py)
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

        # Edge update with gradient gating
        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        # Aggregate messages
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)

        # Node update
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
        hidden_dim: int = 96,
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
        # Encode inputs
        node_features = torch.cat([x, static_features, forcing], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        # Message passing
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        # Decode to residual
        delta = self.decoder(h)
        output = x + delta

        return output


def load_data():
    """Load mesh and validation data."""
    logger.info("Loading mesh and data...")

    # Load mesh
    mesh_path = Path(DATA_DIR) / 'mesh_optimized.npz'
    mesh_data = dict(np.load(mesh_path))

    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)

    logger.info(f"  Mesh: {len(lon)} nodes, {edge_index.shape[1]} edges")

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
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    )

    # Load validation data
    data_path = Path(DATA_DIR) / f'processed_{ROLLOUT_DATE}.npz'
    data = dict(np.load(data_path))

    elevation = data['elevation'].astype(np.float32)
    u10 = data['u10'].astype(np.float32)
    v10 = data['v10'].astype(np.float32)
    pressure = data['pressure'].astype(np.float32)

    logger.info(f"  Data: {elevation.shape[0]} timesteps")

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'static_base': static_base,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
    }


def run_rollout(model, data, device, num_steps):
    """Run autoregressive rollout."""
    model.eval()

    edge_index = data['edge_index'].to(device)
    edge_attr = data['edge_attr'].to(device)

    predictions = []
    ground_truth = []

    # Initial state (first timestep)
    elev = data['elevation']
    cwl_t = np.nan_to_num(elev[0], nan=0.0)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        for t in range(min(num_steps, len(elev) - 1)):
            # Static features with current water level
            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = data['depth'] + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([data['static_base'], wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).to(device)

            # Forcing
            u10 = data['u10'][t] / WIND_SCALE
            v10 = data['v10'][t] / WIND_SCALE
            pres = data['pressure'][t]  # Already normalized
            forcing = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing, dtype=torch.float32).to(device)

            # Predict
            pred = model(current_cwl, static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elev[t + 1], nan=0.0))

            # Use prediction as next input (autoregressive)
            current_cwl = pred

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    return predictions, ground_truth


def compute_metrics(predictions, ground_truth):
    """Compute RMSE at different lead times."""
    results = {}

    for lead_time in [1, 6, 12, 24, 48]:
        if lead_time <= len(predictions):
            rmse = np.sqrt(np.mean((predictions[lead_time-1] - ground_truth[lead_time-1])**2))
            results[f't+{lead_time}h'] = rmse

    # Overall RMSE
    results['overall'] = np.sqrt(np.mean((predictions - ground_truth)**2))

    return results


def plot_results(predictions, ground_truth, data, metrics, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lon = data['lon']
    lat = data['lat']

    # 1. Spatial comparison at different lead times
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    lead_times = [1, 6, 12, 24]

    for i, lt in enumerate(lead_times):
        if lt > len(predictions):
            continue

        pred = predictions[lt - 1]
        truth = ground_truth[lt - 1]
        error = pred - truth

        # Ground truth
        sc0 = axes[0, i].scatter(lon, lat, c=truth, s=1, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0, i].set_title(f'Ground Truth t+{lt}h')
        axes[0, i].set_xlabel('Longitude')
        axes[0, i].set_ylabel('Latitude')
        plt.colorbar(sc0, ax=axes[0, i], label='CWL (m)')

        # Prediction
        sc1 = axes[1, i].scatter(lon, lat, c=pred, s=1, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1, i].set_title(f'Prediction t+{lt}h')
        axes[1, i].set_xlabel('Longitude')
        plt.colorbar(sc1, ax=axes[1, i], label='CWL (m)')

        # Error
        sc2 = axes[2, i].scatter(lon, lat, c=error, s=1, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        axes[2, i].set_title(f'Error t+{lt}h (RMSE: {metrics[f"t+{lt}h"]:.4f}m)')
        axes[2, i].set_xlabel('Longitude')
        plt.colorbar(sc2, ax=axes[2, i], label='Error (m)')

    plt.tight_layout()
    plt.savefig(output_dir / 'a10g_rollout_spatial.png', dpi=150)
    plt.close()
    logger.info(f"  Saved: a10g_rollout_spatial.png")

    # 2. Time series at sample points
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pick 4 sample nodes (corners of bbox roughly)
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    sample_locs = [
        (lon_min + 0.2*(lon_max-lon_min), lat_min + 0.2*(lat_max-lat_min)),
        (lon_min + 0.8*(lon_max-lon_min), lat_min + 0.2*(lat_max-lat_min)),
        (lon_min + 0.2*(lon_max-lon_min), lat_min + 0.8*(lat_max-lat_min)),
        (lon_min + 0.8*(lon_max-lon_min), lat_min + 0.8*(lat_max-lat_min)),
    ]

    for idx, (target_lon, target_lat) in enumerate(sample_locs):
        ax = axes[idx // 2, idx % 2]

        # Find nearest node
        dist = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
        node_idx = np.argmin(dist)

        time_hours = np.arange(len(predictions))

        ax.plot(time_hours, ground_truth[:, node_idx], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(time_hours, predictions[:, node_idx], 'r--', label='Prediction', linewidth=2)

        ax.set_xlabel('Lead Time (hours)')
        ax.set_ylabel('CWL (m)')
        ax.set_title(f'Node at ({lon[node_idx]:.2f}, {lat[node_idx]:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'a10g_rollout_timeseries.png', dpi=150)
    plt.close()
    logger.info(f"  Saved: a10g_rollout_timeseries.png")

    # 3. RMSE vs lead time
    fig, ax = plt.subplots(figsize=(10, 6))

    lead_times = [int(k.split('+')[1].replace('h', '')) for k in metrics.keys() if 't+' in k]
    rmse_values = [metrics[f't+{lt}h'] for lt in lead_times]

    ax.bar(range(len(lead_times)), rmse_values, tick_label=[f't+{lt}h' for lt in lead_times], color='steelblue')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('RMSE (m)')
    ax.set_title(f'A10G Model Rollout Performance (Date: {ROLLOUT_DATE})')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(rmse_values):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'a10g_rollout_rmse.png', dpi=150)
    plt.close()
    logger.info(f"  Saved: a10g_rollout_rmse.png")


def main():
    logger.info("=" * 70)
    logger.info("A10G MODEL ROLLOUT EVALUATION")
    logger.info("=" * 70)

    # Load checkpoint
    logger.info(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    config = checkpoint['config']

    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Val loss: {checkpoint['val_loss']:.6f}")
    logger.info(f"  Config: {config}")

    # Load data
    data = load_data()

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice: {device}")

    model = PhysicsInformedCWLModel(
        state_dim=1,
        static_feature_dim=config['static_features'],
        forcing_feature_dim=config['forcing_features'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Run rollout
    logger.info(f"\nRunning {NUM_ROLLOUT_STEPS}-hour rollout on {ROLLOUT_DATE}...")
    predictions, ground_truth = run_rollout(model, data, device, NUM_ROLLOUT_STEPS)

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth)

    logger.info("\nRollout RMSE:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f} m")

    # Plot results
    logger.info("\nGenerating plots...")
    plot_results(predictions, ground_truth, data, metrics, OUTPUT_DIR)

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
