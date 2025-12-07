#!/usr/bin/env python3
"""
Validate the Mid-Atlantic CWL GNN model.

Includes:
- Single-step prediction accuracy
- Multi-step rollout evaluation
- Spatial error analysis
- Comparison against key tide gauge locations
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Model Architecture
# ============================================================

class MeshGraphNetBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        edge_input = torch.cat([edge_attr, h[row], h[col]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_attr_new)
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)
        return h_new, edge_attr_new


class MidAtlanticGNN(nn.Module):
    def __init__(self, state_dim=1, node_feature_dim=3, edge_feature_dim=3,
                 hidden_dim=128, num_layers=8):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim + node_feature_dim, hidden_dim),
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
            MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x, node_features, edge_index, edge_attr):
        h = self.node_encoder(torch.cat([x, node_features], dim=-1))
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        return self.decoder(h)


# ============================================================
# Mid-Atlantic Tide Gauges
# ============================================================

MIDATLANTIC_STATIONS = {
    'Sandy_Hook': {'lon': -74.0092, 'lat': 40.4669, 'id': '8531680'},
    'The_Battery': {'lon': -74.0142, 'lat': 40.7003, 'id': '8518750'},
    'Kings_Point': {'lon': -73.7650, 'lat': 40.8103, 'id': '8516945'},
    'Montauk': {'lon': -71.9600, 'lat': 41.0483, 'id': '8510560'},
    'Atlantic_City': {'lon': -74.4181, 'lat': 39.3550, 'id': '8534720'},
    'Cape_May': {'lon': -74.9600, 'lat': 38.9683, 'id': '8536110'},
    'Lewes': {'lon': -75.1194, 'lat': 38.7828, 'id': '8557380'},
    'Philadelphia': {'lon': -75.1417, 'lat': 39.9333, 'id': '8545240'},
}


# ============================================================
# Data Loading
# ============================================================

def load_model_and_data():
    """Load trained model and data."""
    base_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate'

    # Load mesh
    mesh_data = np.load(f'{base_dir}/data/processed/midatlantic_mesh.npz')
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index_np = mesh_data['edge_index']

    # Load elevation
    elev_data = np.load(f'{base_dir}/data/processed/midatlantic_elevation.npz')
    elevation = elev_data['elevation']

    logger.info(f"Loaded: {len(lon)} nodes, {elevation.shape[0]} timesteps")

    # Load model
    checkpoint = torch.load(
        f'{base_dir}/outputs/checkpoints/best_midatlantic_model.pt',
        map_location='cpu', weights_only=False
    )

    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_layers', 8)
    eta_scale = checkpoint.get('eta_scale', 2.0)

    model = MidAtlanticGNN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model from epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.6f}")

    # Build features
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1

    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    node_features = torch.tensor(
        np.stack([x_norm, y_norm, depth_norm], axis=1),
        dtype=torch.float32
    )

    src, dst = edge_index_np[0], edge_index_np[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    )

    return {
        'model': model,
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'elevation': elevation,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'node_features': node_features,
        'eta_scale': eta_scale,
    }


# ============================================================
# Validation Functions
# ============================================================

def validate_single_step(data, val_indices):
    """Validate single-step predictions."""
    model = data['model']
    elevation = data['elevation']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    node_features = data['node_features']
    eta_scale = data['eta_scale']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_features = node_features.to(device)

    all_errors = []

    with torch.no_grad():
        for idx in val_indices:
            x_in = torch.tensor(elevation[idx] / eta_scale, dtype=torch.float32).unsqueeze(1).to(device)
            pred = model(x_in, node_features, edge_index, edge_attr).squeeze()
            pred_np = pred.cpu().numpy() * eta_scale
            true_np = elevation[idx + 1]

            error = pred_np - true_np
            all_errors.append(error)

    all_errors = np.array(all_errors)

    rmse = np.sqrt(np.mean(all_errors**2))
    mae = np.mean(np.abs(all_errors))
    bias = np.mean(all_errors)

    # Per-node statistics
    node_mae = np.mean(np.abs(all_errors), axis=0)

    return {
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'node_mae': node_mae,
        'all_errors': all_errors,
    }


def validate_rollout(data, start_idx, num_steps):
    """Validate multi-step rollout."""
    model = data['model']
    elevation = data['elevation']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    node_features = data['node_features']
    eta_scale = data['eta_scale']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_features = node_features.to(device)

    current = torch.tensor(elevation[start_idx] / eta_scale, dtype=torch.float32).to(device)

    rmse_list = []
    mae_list = []
    corr_list = []

    with torch.no_grad():
        for step in range(num_steps):
            x = current.unsqueeze(1)
            next_state = model(x, node_features, edge_index, edge_attr).squeeze()
            current = next_state

            pred = current.cpu().numpy() * eta_scale

            if start_idx + step + 1 < len(elevation):
                true = elevation[start_idx + step + 1]
                rmse = np.sqrt(np.mean((pred - true)**2))
                mae = np.mean(np.abs(pred - true))
                corr = np.corrcoef(pred, true)[0, 1]

                rmse_list.append(rmse)
                mae_list.append(mae)
                corr_list.append(corr)

    return {
        'rmse': np.array(rmse_list),
        'mae': np.array(mae_list),
        'correlation': np.array(corr_list),
    }


def validate_stations(data, stations):
    """Validate at tide gauge locations."""
    lon = data['lon']
    lat = data['lat']

    # Find nearest nodes
    tree = cKDTree(np.column_stack([lon, lat]))

    station_results = {}

    for name, info in stations.items():
        dist, idx = tree.query([info['lon'], info['lat']])

        if dist < 0.2:  # Within 0.2 degrees
            station_results[name] = {
                'node_idx': idx,
                'distance': dist,
                'lon': lon[idx],
                'lat': lat[idx],
            }
        else:
            logger.warning(f"{name} too far from domain ({dist:.2f} deg)")

    return station_results


# ============================================================
# Visualization
# ============================================================

def plot_validation_summary(data, single_step, rollout, station_results, output_path):
    """Create comprehensive validation plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    lon = data['lon']
    lat = data['lat']

    # Panel 1: Spatial MAE
    ax = axes[0, 0]
    cf = ax.scatter(lon, lat, c=single_step['node_mae'], s=3, cmap='YlOrRd', vmin=0, vmax=0.15)
    ax.set_title(f"Single-Step MAE\nOverall: {single_step['mae']:.4f} m")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='MAE (m)')

    # Mark stations
    for name, info in station_results.items():
        ax.plot(info['lon'], info['lat'], 'k^', markersize=8)
        ax.annotate(name, (info['lon'], info['lat']), fontsize=7, ha='left')

    # Panel 2: Rollout error evolution
    ax = axes[0, 1]
    hours = np.arange(1, len(rollout['rmse']) + 1)
    ax.plot(hours, rollout['rmse'], 'r-', linewidth=2, label='RMSE')
    ax.plot(hours, rollout['mae'], 'b--', linewidth=2, label='MAE')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Error (m)')
    ax.set_title('Rollout Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Correlation evolution
    ax = axes[1, 0]
    ax.plot(hours, rollout['correlation'], 'g-', linewidth=2)
    ax.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='R=0.9')
    ax.axhline(y=0.8, color='k', linestyle=':', alpha=0.5, label='R=0.8')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Correlation')
    ax.set_title('Rollout Correlation')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Error histogram
    ax = axes[1, 1]
    errors = single_step['all_errors'].flatten()
    ax.hist(errors, bins=100, density=True, alpha=0.7, color='blue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution\nBias: {single_step["bias"]:.4f} m')
    ax.set_xlim(-0.5, 0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Mid-Atlantic CWL Model Validation', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_station_comparison(data, station_results, start_idx, num_steps, output_path):
    """Plot time series at station locations."""
    model = data['model']
    elevation = data['elevation']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    node_features = data['node_features']
    eta_scale = data['eta_scale']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_features = node_features.to(device)

    # Run rollout
    current = torch.tensor(elevation[start_idx] / eta_scale, dtype=torch.float32).to(device)
    predictions = [elevation[start_idx].copy()]

    with torch.no_grad():
        for step in range(num_steps):
            x = current.unsqueeze(1)
            next_state = model(x, node_features, edge_index, edge_attr).squeeze()
            current = next_state
            predictions.append(current.cpu().numpy() * eta_scale)

    predictions = np.array(predictions)

    # Get ground truth
    ground_truth = elevation[start_idx:start_idx + num_steps + 1]

    # Plot each station
    num_stations = len(station_results)
    fig, axes = plt.subplots(num_stations, 1, figsize=(12, 3*num_stations))

    if num_stations == 1:
        axes = [axes]

    times = np.arange(num_steps + 1)

    for i, (name, info) in enumerate(station_results.items()):
        ax = axes[i]
        idx = info['node_idx']

        ax.plot(times, ground_truth[:, idx], 'b-', linewidth=2, label='Ground Truth')
        ax.plot(times, predictions[:, idx], 'r--', linewidth=2, label='Prediction')

        # Compute station RMSE
        rmse = np.sqrt(np.mean((predictions[:, idx] - ground_truth[:, idx])**2))

        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('CWL (m)')
        ax.set_title(f'{name} (ID: {MIDATLANTIC_STATIONS[name]["id"]}) - RMSE: {rmse:.3f} m')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Mid-Atlantic Station Validation', fontsize=14)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    logger.info("="*60)
    logger.info("Mid-Atlantic CWL Model Validation")
    logger.info("="*60)

    # Load data
    logger.info("\nLoading model and data...")
    data = load_model_and_data()

    # Define validation set (last 20% of data)
    num_times = data['elevation'].shape[0]
    train_size = int(0.8 * num_times)
    val_indices = list(range(train_size, num_times - 1))

    logger.info(f"Validation samples: {len(val_indices)}")

    # Single-step validation
    logger.info("\nValidating single-step predictions...")
    single_step = validate_single_step(data, val_indices)

    logger.info(f"  RMSE: {single_step['rmse']:.4f} m")
    logger.info(f"  MAE:  {single_step['mae']:.4f} m")
    logger.info(f"  Bias: {single_step['bias']:.4f} m")

    # Rollout validation
    logger.info("\nValidating multi-step rollout...")
    rollout = validate_rollout(data, start_idx=train_size + 5, num_steps=48)

    for t in [1, 6, 12, 24, 48]:
        if t <= len(rollout['rmse']):
            logger.info(f"  t+{t:2d}h: RMSE={rollout['rmse'][t-1]:.4f}m, R={rollout['correlation'][t-1]:.4f}")

    # Station validation
    logger.info("\nValidating at tide gauge locations...")
    station_results = validate_stations(data, MIDATLANTIC_STATIONS)
    logger.info(f"  Found {len(station_results)} stations in domain")

    # Generate plots
    logger.info("\nGenerating validation plots...")
    output_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures'

    plot_validation_summary(
        data, single_step, rollout, station_results,
        f'{output_dir}/midatlantic_validation.png'
    )

    if len(station_results) > 0:
        plot_station_comparison(
            data, station_results, start_idx=train_size + 5, num_steps=48,
            f'{output_dir}/midatlantic_station_validation.png'
        )

    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Single-step RMSE: {single_step['rmse']:.4f} m")
    logger.info(f"Single-step MAE:  {single_step['mae']:.4f} m")
    logger.info(f"24h rollout RMSE: {rollout['rmse'][23]:.4f} m" if len(rollout['rmse']) >= 24 else "")
    logger.info(f"24h correlation:  {rollout['correlation'][23]:.4f}" if len(rollout['correlation']) >= 24 else "")

    logger.info("\nInterpretation:")
    logger.info("  RMSE < 0.10 m: Excellent")
    logger.info("  RMSE < 0.15 m: Good for operations")
    logger.info("  RMSE < 0.25 m: Acceptable")
    logger.info("  RMSE > 0.30 m: Needs improvement")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
