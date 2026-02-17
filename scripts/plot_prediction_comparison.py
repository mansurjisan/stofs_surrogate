#!/usr/bin/env python3
"""
Plot predicted vs ground truth water elevation snapshots.
Compares GNN predictions with actual STOFS CWL data.
"""

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Model Architecture (copied from train_cwl_bias_corrected.py)
# ============================================================

class MeshGraphNetBlock(nn.Module):
    """Message passing block."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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


class CWLGNN(nn.Module):
    """GNN for CWL (Coastal Water Level) prediction."""

    def __init__(
        self,
        state_dim: int = 1,
        node_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 6,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim + node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

def load_model_and_data():
    """Load trained model and data."""
    # Load mesh
    project_root = Path(__file__).resolve().parent.parent
    mesh_path = str(project_root / 'data/processed/us_east_coast_cwl_mesh.npz')
    mesh_data = np.load(mesh_path)
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index_np = mesh_data['edge_index']  # (2, num_edges)

    # Load elevation data
    elev_path = str(project_root / 'data/processed/us_east_coast_cwl_elevation.npz')
    elev_data = np.load(elev_path)
    elevation = elev_data['elevation']  # (time, nodes)

    logger.info(f"Loaded mesh: {len(lon)} nodes, {edge_index_np.shape[1]} edges")
    logger.info(f"Elevation shape: {elevation.shape}")

    # Load model
    checkpoint_path = str(project_root / 'outputs/checkpoints/best_cwl_model.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get eta_scale from checkpoint
    eta_scale = checkpoint.get('eta_scale', 2.0)
    logger.info(f"Eta scale from checkpoint: {eta_scale}")

    # Recreate model
    model = CWLGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=64,
        num_layers=6,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")

    # Build features exactly as in training script (CWLDataset)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # Compute Cartesian coordinates (same as training)
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    # Node features (same normalization as training)
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1

    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    node_features = torch.tensor(
        np.stack([x_norm, y_norm, depth_norm], axis=1),
        dtype=torch.float32
    )

    # Edge features (same normalization as training)
    src, dst = edge_index_np[0], edge_index_np[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    )

    logger.info(f"Node features shape: {node_features.shape}")
    logger.info(f"Edge attr shape: {edge_attr.shape}")

    return model, lon, lat, elevation, edge_index, edge_attr, node_features, eta_scale


def predict_sequence(model, elevation, edge_index, edge_attr, node_features, eta_scale, start_idx, num_steps):
    """Run autoregressive prediction for num_steps."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    node_features = node_features.to(device)

    # Start from ground truth (normalized)
    current = torch.tensor(elevation[start_idx] / eta_scale, dtype=torch.float32).to(device)

    predictions = [elevation[start_idx].copy()]  # Store unnormalized
    ground_truth = [elevation[start_idx]]

    with torch.no_grad():
        for step in range(num_steps):
            # Prepare input
            x = current.unsqueeze(1)  # (nodes, 1)

            # Model predicts the NEXT state directly (not delta!)
            # Model signature: forward(x, node_features, edge_index, edge_attr)
            next_state = model(x, node_features, edge_index, edge_attr).squeeze()

            # Update state for next iteration
            current = next_state

            # Store (unnormalized)
            predictions.append(current.cpu().numpy() * eta_scale)
            if start_idx + step + 1 < len(elevation):
                ground_truth.append(elevation[start_idx + step + 1])
            else:
                ground_truth.append(np.full_like(elevation[0], np.nan))

    return np.array(predictions), np.array(ground_truth)


def plot_comparison(lon, lat, predictions, ground_truth, timesteps=[0, 6, 12, 24]):
    """Plot side-by-side comparison of predicted vs ground truth using scatter."""

    # Determine color scale from all data
    vmin = min(np.nanmin(predictions), np.nanmin(ground_truth))
    vmax = max(np.nanmax(predictions), np.nanmax(ground_truth))
    # Symmetric around 0
    vabs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vabs, vabs

    # Limit for better visualization
    vmin, vmax = max(vmin, -2), min(vmax, 2)

    n_times = len(timesteps)
    fig, axes = plt.subplots(n_times, 3, figsize=(15, 4*n_times))

    if n_times == 1:
        axes = axes.reshape(1, -1)

    # Point size - smaller for dense scatter
    s = 1

    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        gt = ground_truth[t]
        pred = predictions[t]
        diff = pred - gt

        # Ground truth
        ax = axes[i, 0]
        cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'Ground Truth (t+{t}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Prediction
        ax = axes[i, 1]
        cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'GNN Prediction (t+{t}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Difference
        ax = axes[i, 2]
        diff_max = 0.5
        cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        rmse = np.sqrt(np.nanmean(diff**2))
        ax.set_title(f'Difference (RMSE={rmse:.3f}m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Error (m)')

    plt.suptitle('CWL GNN Model: Prediction vs Ground Truth', fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def plot_single_snapshot(lon, lat, predictions, ground_truth, timestep=12):
    """Plot a single detailed snapshot comparison using scatter."""

    gt = ground_truth[timestep]
    pred = predictions[timestep]
    diff = pred - gt

    # Color scale
    vabs = max(abs(np.nanmin(gt)), abs(np.nanmax(gt)),
               abs(np.nanmin(pred)), abs(np.nanmax(pred)))
    vabs = min(vabs, 2.0)  # Limit

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Point size
    s = 2

    # Ground truth
    ax = axes[0]
    cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax.set_title(f'Ground Truth\nt+{timestep} hours', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)', shrink=0.8)

    # Prediction
    ax = axes[1]
    cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=-vabs, vmax=vabs)
    ax.set_title(f'GNN Prediction\nt+{timestep} hours', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)', shrink=0.8)

    # Difference
    ax = axes[2]
    diff_max = 0.5
    cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
    rmse = np.sqrt(np.nanmean(diff**2))
    mae = np.nanmean(np.abs(diff))
    ax.set_title(f'Prediction Error\nRMSE={rmse:.3f}m, MAE={mae:.3f}m', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Error (m)', shrink=0.8)

    plt.suptitle('CWL Surrogate Model Validation', fontsize=14)
    plt.tight_layout()

    return fig


def main():
    logger.info("Loading model and data...")
    model, lon, lat, elevation, edge_index, edge_attr, node_features, eta_scale = load_model_and_data()

    # Start from middle of validation set
    start_idx = 120  # Start from timestep 120
    num_steps = 36   # Predict 36 hours ahead

    logger.info(f"Running {num_steps}-step prediction from timestep {start_idx}...")
    predictions, ground_truth = predict_sequence(
        model, elevation, edge_index, edge_attr, node_features, eta_scale, start_idx, num_steps
    )

    logger.info(f"Predictions shape: {predictions.shape}")
    logger.info(f"Ground truth shape: {ground_truth.shape}")

    # Plot single detailed snapshot at t+12h
    logger.info("Generating single snapshot comparison...")
    fig1 = plot_single_snapshot(lon, lat, predictions, ground_truth, timestep=12)
    project_root = Path(__file__).resolve().parent.parent
    fig1.savefig(str(project_root / 'outputs/figures/cwl_snapshot_comparison.png'),
                 dpi=150, bbox_inches='tight')
    logger.info("Saved: outputs/figures/cwl_snapshot_comparison.png")

    # Plot multi-timestep comparison
    logger.info("Generating multi-timestep comparison...")
    fig2 = plot_comparison(lon, lat, predictions, ground_truth,
                          timesteps=[0, 6, 12, 24])
    fig2.savefig(str(project_root / 'outputs/figures/cwl_multi_timestep.png'),
                 dpi=150, bbox_inches='tight')
    logger.info("Saved: outputs/figures/cwl_multi_timestep.png")

    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*50)
    for t in [0, 6, 12, 24, 36]:
        if t < len(predictions):
            diff = predictions[t] - ground_truth[t]
            rmse = np.sqrt(np.nanmean(diff**2))
            mae = np.nanmean(np.abs(diff))
            corr = np.corrcoef(predictions[t], ground_truth[t])[0, 1]
            logger.info(f"t+{t:2d}h: RMSE={rmse:.4f}m, MAE={mae:.4f}m, R={corr:.4f}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
