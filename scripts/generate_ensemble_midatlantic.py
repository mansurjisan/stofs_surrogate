#!/usr/bin/env python3
"""
Generate ensemble forecasts using the Mid-Atlantic CWL GNN model.

This version is optimized for the smaller Mid-Atlantic domain.
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Model Architecture (same as train_midatlantic.py)
# ============================================================

class MeshGraphNetBlock(nn.Module):
    """Message passing block with layer normalization."""

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
    """GNN for Mid-Atlantic CWL prediction."""

    def __init__(
        self,
        state_dim: int = 1,
        node_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 8,
    ):
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
# Data Loading
# ============================================================

def load_model_and_data():
    """Load trained Mid-Atlantic model and data."""
    base_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate'

    # Load mesh
    mesh_path = f'{base_dir}/data/processed/midatlantic_mesh.npz'
    mesh_data = np.load(mesh_path)
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index_np = mesh_data['edge_index']

    # Load elevation
    elev_path = f'{base_dir}/data/processed/midatlantic_elevation.npz'
    elev_data = np.load(elev_path)
    elevation = elev_data['elevation']

    logger.info(f"Loaded mesh: {len(lon)} nodes, {edge_index_np.shape[1]} edges")
    logger.info(f"Elevation shape: {elevation.shape}")

    # Load model
    checkpoint_path = f'{base_dir}/outputs/checkpoints/best_midatlantic_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    hidden_dim = checkpoint.get('hidden_dim', 128)
    num_layers = checkpoint.get('num_layers', 8)
    eta_scale = checkpoint.get('eta_scale', 2.0)

    logger.info(f"Model: hidden_dim={hidden_dim}, num_layers={num_layers}")
    logger.info(f"Eta scale: {eta_scale}")

    model = MidAtlanticGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")

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
# Ensemble Generation
# ============================================================

def generate_spatially_correlated_perturbation(lon, lat, num_members, noise_std=0.05,
                                                correlation_length=0.5):
    """Generate spatially correlated perturbations."""
    num_nodes = len(lon)

    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()

    n_coarse_lon = int(lon_range / correlation_length) + 1
    n_coarse_lat = int(lat_range / correlation_length) + 1
    n_coarse_lon = max(5, min(n_coarse_lon, 30))
    n_coarse_lat = max(5, min(n_coarse_lat, 30))

    coarse_lon = np.linspace(lon.min(), lon.max(), n_coarse_lon)
    coarse_lat = np.linspace(lat.min(), lat.max(), n_coarse_lat)

    perturbations = np.zeros((num_members, num_nodes))

    for m in range(num_members):
        coarse_field = np.random.randn(n_coarse_lat, n_coarse_lon) * noise_std

        interp = RegularGridInterpolator(
            (coarse_lat, coarse_lon),
            coarse_field,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        points = np.stack([lat, lon], axis=1)
        perturbations[m] = interp(points)

    return perturbations


def run_ensemble_forecast(data, start_idx, num_steps, num_members=20,
                          ic_noise_std=0.05, model_noise_std=0.0,
                          correlation_length=0.5):
    """Run ensemble forecast."""
    model = data['model']
    elevation = data['elevation']
    edge_index = data['edge_index']
    edge_attr = data['edge_attr']
    node_features = data['node_features']
    eta_scale = data['eta_scale']
    lon = data['lon']
    lat = data['lat']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_features = node_features.to(device)

    num_nodes = len(lon)
    initial_state = elevation[start_idx]

    # Generate perturbations
    logger.info(f"Generating {num_members} ensemble members...")
    perturbations = generate_spatially_correlated_perturbation(
        lon, lat, num_members, ic_noise_std, correlation_length
    )
    perturbed_ics = initial_state[np.newaxis, :] + perturbations

    # Add control member
    all_ics = np.vstack([initial_state[np.newaxis, :], perturbed_ics])
    num_members_total = num_members + 1

    # Storage
    ensemble_forecasts = np.zeros((num_members_total, num_steps + 1, num_nodes))
    ensemble_forecasts[:, 0, :] = all_ics

    # Run each member
    logger.info(f"Running {num_steps}-step forecast for {num_members_total} members...")

    with torch.no_grad():
        for m in range(num_members_total):
            current = torch.tensor(all_ics[m] / eta_scale, dtype=torch.float32).to(device)

            for step in range(num_steps):
                x = current.unsqueeze(1)
                next_state = model(x, node_features, edge_index, edge_attr).squeeze()

                if model_noise_std > 0 and m > 0:
                    noise = torch.randn_like(next_state) * (model_noise_std / eta_scale)
                    next_state = next_state + noise

                current = next_state
                ensemble_forecasts[m, step + 1, :] = current.cpu().numpy() * eta_scale

            if (m + 1) % 10 == 0:
                logger.info(f"  Completed member {m + 1}/{num_members_total}")

    # Compute statistics
    logger.info("Computing ensemble statistics...")

    control = ensemble_forecasts[0]
    perturbed = ensemble_forecasts[1:]

    ensemble_mean = np.mean(perturbed, axis=0)
    ensemble_std = np.std(perturbed, axis=0)
    ensemble_p10 = np.percentile(perturbed, 10, axis=0)
    ensemble_p25 = np.percentile(perturbed, 25, axis=0)
    ensemble_p50 = np.percentile(perturbed, 50, axis=0)
    ensemble_p75 = np.percentile(perturbed, 75, axis=0)
    ensemble_p90 = np.percentile(perturbed, 90, axis=0)

    # Ground truth
    ground_truth = np.zeros((num_steps + 1, num_nodes))
    for t in range(num_steps + 1):
        if start_idx + t < len(elevation):
            ground_truth[t] = elevation[start_idx + t]
        else:
            ground_truth[t] = np.nan

    return {
        'control': control,
        'ensemble_forecasts': perturbed,
        'ensemble_mean': ensemble_mean,
        'ensemble_std': ensemble_std,
        'ensemble_p10': ensemble_p10,
        'ensemble_p25': ensemble_p25,
        'ensemble_p50': ensemble_p50,
        'ensemble_p75': ensemble_p75,
        'ensemble_p90': ensemble_p90,
        'ground_truth': ground_truth,
        'num_members': num_members,
        'start_idx': start_idx,
        'num_steps': num_steps,
    }


# ============================================================
# Visualization
# ============================================================

def plot_ensemble_spatial(data, results, timestep=12, output_path=None):
    """Plot spatial ensemble statistics."""
    lon = data['lon']
    lat = data['lat']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    t = timestep
    s = 3

    # Row 1: Mean, Spread, Ground Truth
    ax = axes[0, 0]
    cf = ax.scatter(lon, lat, c=results['ensemble_mean'][t], s=s, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Ensemble Mean (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    ax = axes[0, 1]
    cf = ax.scatter(lon, lat, c=results['ensemble_std'][t], s=s, cmap='YlOrRd', vmin=0, vmax=0.15)
    ax.set_title(f'Ensemble Spread (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Std Dev (m)')

    ax = axes[0, 2]
    cf = ax.scatter(lon, lat, c=results['ground_truth'][t], s=s, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'Ground Truth (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    # Row 2: P10, P90, 80% CI Width
    ax = axes[1, 0]
    cf = ax.scatter(lon, lat, c=results['ensemble_p10'][t], s=s, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'10th Percentile (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    ax = axes[1, 1]
    cf = ax.scatter(lon, lat, c=results['ensemble_p90'][t], s=s, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_title(f'90th Percentile (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    ax = axes[1, 2]
    ci_width = results['ensemble_p90'][t] - results['ensemble_p10'][t]
    cf = ax.scatter(lon, lat, c=ci_width, s=s, cmap='YlOrRd', vmin=0, vmax=0.4)
    ax.set_title(f'80% CI Width (t+{t}h)')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='P90-P10 (m)')

    plt.suptitle(f'Mid-Atlantic Ensemble Forecast - {results["num_members"]} Members', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def plot_spread_skill(results, output_path=None):
    """Plot spread-skill analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rmse = np.sqrt(np.nanmean((results['ensemble_mean'] - results['ground_truth'])**2, axis=1))
    mean_spread = np.mean(results['ensemble_std'], axis=1)
    times = np.arange(len(rmse))

    # RMSE and Spread vs time
    ax = axes[0]
    ax.plot(times, rmse, 'r-', linewidth=2, label='RMSE')
    ax.plot(times, mean_spread, 'b-', linewidth=2, label='Spread')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Error / Spread (m)')
    ax.set_title('Spread-Skill Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter
    ax = axes[1]
    ax.scatter(mean_spread, rmse, c=times, cmap='viridis', s=50)
    max_val = max(rmse.max(), mean_spread.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', label='1:1 line')
    ax.set_xlabel('Spread (m)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Spread vs Skill')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Ratio
    ax = axes[2]
    ratio = mean_spread / (rmse + 1e-6)
    ax.plot(times, ratio, 'g-', linewidth=2)
    ax.axhline(y=1.0, color='k', linestyle='--')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Spread / RMSE')
    ax.set_title('Spread-Skill Ratio')
    ax.set_ylim(0, 3)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Mid-Atlantic Ensemble Spread-Skill Analysis', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Mid-Atlantic ensemble forecast')
    parser.add_argument('--members', type=int, default=30, help='Number of ensemble members')
    parser.add_argument('--steps', type=int, default=48, help='Forecast hours')
    parser.add_argument('--start', type=int, default=100, help='Starting timestep')
    parser.add_argument('--ic-noise', type=float, default=0.05, help='IC noise std (m)')
    parser.add_argument('--corr-length', type=float, default=0.5, help='Correlation length (deg)')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Mid-Atlantic Ensemble Forecast")
    logger.info("="*60)

    # Load model and data
    logger.info("\nLoading model and data...")
    data = load_model_and_data()

    # Run ensemble
    logger.info(f"\nGenerating {args.members}-member ensemble...")
    results = run_ensemble_forecast(
        data,
        start_idx=args.start,
        num_steps=args.steps,
        num_members=args.members,
        ic_noise_std=args.ic_noise,
        correlation_length=args.corr_length,
    )

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE SUMMARY")
    logger.info("="*60)

    for t in [0, 6, 12, 24, 48]:
        if t <= args.steps:
            rmse = np.sqrt(np.nanmean((results['ensemble_mean'][t] - results['ground_truth'][t])**2))
            spread = np.mean(results['ensemble_std'][t])
            logger.info(f"t+{t:2d}h: RMSE={rmse:.4f}m, Spread={spread:.4f}m, Ratio={spread/(rmse+1e-6):.2f}")

    # Generate plots
    logger.info("\nGenerating plots...")
    output_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures'

    plot_ensemble_spatial(data, results, timestep=12,
                         output_path=f'{output_dir}/midatlantic_ensemble_t12.png')

    plot_ensemble_spatial(data, results, timestep=24,
                         output_path=f'{output_dir}/midatlantic_ensemble_t24.png')

    plot_spread_skill(results,
                     output_path=f'{output_dir}/midatlantic_spread_skill.png')

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
