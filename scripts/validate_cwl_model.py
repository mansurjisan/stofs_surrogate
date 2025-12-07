#!/usr/bin/env python3
"""
Validate the CWL GNN Model

This script performs comprehensive validation:
1. Single-step prediction accuracy (RMSE, MAE, correlation)
2. Multi-step rollout accuracy (error accumulation over time)
3. Spatial error distribution (where does the model perform best/worst)
4. Comparison of predicted vs actual water levels

Usage:
    python scripts/validate_cwl_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.stats import pearsonr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/processed")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_cwl_model.pt")
OUTPUT_DIR = Path("outputs/figures")

# ============================================================
# Model Architecture (must match training script)
# ============================================================

class MeshGraphNetBlock(nn.Module):
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
    def __init__(self, state_dim=1, node_feature_dim=3, edge_feature_dim=3, hidden_dim=64, num_layers=6):
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
        self.layers = nn.ModuleList([MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)])
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


def load_data_and_model():
    """Load mesh, elevation data, and trained model."""
    logger.info("Loading data...")

    # Load mesh
    mesh = np.load(DATA_DIR / "us_east_coast_cwl_mesh.npz")
    lon = mesh['lon']
    lat = mesh['lat']
    depth = mesh['depth']
    edge_index = mesh['edge_index']

    # Load elevation
    elev_data = np.load(DATA_DIR / "us_east_coast_cwl_elevation.npz")
    elevation = elev_data['elevation']

    logger.info(f"Mesh: {len(lon)} nodes, {edge_index.shape[1]} edges")
    logger.info(f"Elevation: {elevation.shape[0]} timesteps")

    # Compute features (same as training)
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    node_features = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)

    # Edge features
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    # Load model
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    eta_scale = checkpoint.get('eta_scale', 2.0)

    model = CWLGNN(state_dim=1, node_feature_dim=3, edge_feature_dim=3, hidden_dim=64, num_layers=6).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Best val loss: {checkpoint.get('val_loss', 'unknown'):.6f}")

    return {
        'lon': lon, 'lat': lat, 'depth': depth,
        'elevation': elevation,
        'edge_index': edge_index,
        'node_features': node_features,
        'edge_attr': edge_attr,
        'eta_scale': eta_scale,
        'model': model,
        'device': device,
    }


def validate_single_step(data):
    """Validate single-step predictions on validation set."""
    logger.info("\n" + "="*60)
    logger.info("1. SINGLE-STEP VALIDATION")
    logger.info("="*60)

    model = data['model']
    device = data['device']
    elevation = data['elevation']
    eta_scale = data['eta_scale']

    # Prepare tensors
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
    edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float32, device=device)
    node_features = torch.tensor(data['node_features'], dtype=torch.float32, device=device)

    # Use last 20% as validation (same as training)
    num_samples = elevation.shape[0] - 1
    val_start = int(0.8 * num_samples)

    all_errors = []
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for t in range(val_start, num_samples):
            x_in = torch.tensor(elevation[t:t+1].T / eta_scale, dtype=torch.float32, device=device)
            y_true = elevation[t+1]

            pred = model(x_in, node_features, edge_index, edge_attr)
            y_pred = pred.cpu().numpy().flatten() * eta_scale

            errors = y_pred - y_true
            all_errors.append(errors)
            all_predictions.append(y_pred)
            all_targets.append(y_true)

    all_errors = np.array(all_errors)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Compute metrics
    rmse = np.sqrt(np.mean(all_errors**2))
    mae = np.mean(np.abs(all_errors))
    bias = np.mean(all_errors)

    # Correlation
    corr, _ = pearsonr(all_predictions.flatten(), all_targets.flatten())

    # Percentiles
    p95_error = np.percentile(np.abs(all_errors), 95)
    max_error = np.max(np.abs(all_errors))

    logger.info(f"  Validation samples: {len(all_errors)}")
    logger.info(f"  RMSE: {rmse:.4f} m")
    logger.info(f"  MAE: {mae:.4f} m")
    logger.info(f"  Bias: {bias:.4f} m")
    logger.info(f"  Correlation: {corr:.4f}")
    logger.info(f"  95th percentile error: {p95_error:.4f} m")
    logger.info(f"  Max error: {max_error:.4f} m")

    return {
        'rmse': rmse, 'mae': mae, 'bias': bias, 'corr': corr,
        'p95_error': p95_error, 'max_error': max_error,
        'errors': all_errors, 'predictions': all_predictions, 'targets': all_targets,
    }


def validate_rollout(data, rollout_steps=50):
    """Validate multi-step rollout predictions."""
    logger.info("\n" + "="*60)
    logger.info("2. ROLLOUT VALIDATION")
    logger.info("="*60)

    model = data['model']
    device = data['device']
    elevation = data['elevation']
    eta_scale = data['eta_scale']

    edge_index = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
    edge_attr = torch.tensor(data['edge_attr'], dtype=torch.float32, device=device)
    node_features = torch.tensor(data['node_features'], dtype=torch.float32, device=device)

    # Start from middle of dataset
    start_idx = elevation.shape[0] // 3

    if start_idx + rollout_steps >= elevation.shape[0]:
        rollout_steps = elevation.shape[0] - start_idx - 1
        logger.info(f"  Adjusted rollout steps to {rollout_steps}")

    # Initialize with ground truth
    current_state = torch.tensor(elevation[start_idx:start_idx+1].T / eta_scale,
                                  dtype=torch.float32, device=device)

    rollout_rmse = []
    rollout_mae = []
    rollout_corr = []

    with torch.no_grad():
        for step in range(rollout_steps):
            # Predict next step
            pred = model(current_state, node_features, edge_index, edge_attr)
            current_state = pred

            # Compare to ground truth
            y_pred = pred.cpu().numpy().flatten() * eta_scale
            y_true = elevation[start_idx + step + 1]

            errors = y_pred - y_true
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))
            corr, _ = pearsonr(y_pred, y_true)

            rollout_rmse.append(rmse)
            rollout_mae.append(mae)
            rollout_corr.append(corr)

    logger.info(f"  Rollout steps: {rollout_steps}")
    logger.info(f"  Step 1 RMSE: {rollout_rmse[0]:.4f} m")
    logger.info(f"  Step 10 RMSE: {rollout_rmse[min(9, len(rollout_rmse)-1)]:.4f} m")
    logger.info(f"  Step 25 RMSE: {rollout_rmse[min(24, len(rollout_rmse)-1)]:.4f} m")
    logger.info(f"  Step {rollout_steps} RMSE: {rollout_rmse[-1]:.4f} m")
    logger.info(f"  Final correlation: {rollout_corr[-1]:.4f}")

    return {
        'steps': list(range(1, rollout_steps + 1)),
        'rmse': rollout_rmse,
        'mae': rollout_mae,
        'corr': rollout_corr,
        'start_idx': start_idx,
    }


def validate_spatial(data, single_step_results):
    """Analyze spatial distribution of errors."""
    logger.info("\n" + "="*60)
    logger.info("3. SPATIAL ERROR ANALYSIS")
    logger.info("="*60)

    lon = data['lon']
    lat = data['lat']
    depth = data['depth']
    errors = single_step_results['errors']

    # Mean absolute error per node
    node_mae = np.mean(np.abs(errors), axis=0)

    # Find best and worst regions
    best_10pct = np.percentile(node_mae, 10)
    worst_10pct = np.percentile(node_mae, 90)

    best_nodes = node_mae <= best_10pct
    worst_nodes = node_mae >= worst_10pct

    logger.info(f"  Best 10% nodes - MAE <= {best_10pct:.4f} m")
    logger.info(f"    Mean depth: {np.mean(np.abs(depth[best_nodes])):.1f} m")
    logger.info(f"    Lat range: [{lat[best_nodes].min():.2f}, {lat[best_nodes].max():.2f}]")

    logger.info(f"  Worst 10% nodes - MAE >= {worst_10pct:.4f} m")
    logger.info(f"    Mean depth: {np.mean(np.abs(depth[worst_nodes])):.1f} m")
    logger.info(f"    Lat range: [{lat[worst_nodes].min():.2f}, {lat[worst_nodes].max():.2f}]")

    # Correlation with depth
    depth_corr, _ = pearsonr(np.abs(depth), node_mae)
    logger.info(f"  Error-depth correlation: {depth_corr:.4f}")

    return {'node_mae': node_mae, 'best_nodes': best_nodes, 'worst_nodes': worst_nodes}


def plot_validation_results(data, single_step, rollout, spatial):
    """Generate validation plots."""
    logger.info("\n" + "="*60)
    logger.info("4. GENERATING VALIDATION PLOTS")
    logger.info("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot: predicted vs actual
    ax = axes[0, 0]
    sample_idx = np.random.choice(len(single_step['predictions'].flatten()),
                                   min(50000, len(single_step['predictions'].flatten())),
                                   replace=False)
    pred_flat = single_step['predictions'].flatten()[sample_idx]
    targ_flat = single_step['targets'].flatten()[sample_idx]
    ax.scatter(targ_flat, pred_flat, alpha=0.1, s=1, c='blue')
    ax.plot([-4, 4], [-4, 4], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('Actual CWL (m)', fontsize=11)
    ax.set_ylabel('Predicted CWL (m)', fontsize=11)
    ax.set_title(f'Single-Step Prediction (R={single_step["corr"]:.3f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)

    # 2. Rollout error over time
    ax = axes[0, 1]
    ax.plot(rollout['steps'], rollout['rmse'], 'b-', linewidth=2, label='RMSE')
    ax.plot(rollout['steps'], rollout['mae'], 'g--', linewidth=2, label='MAE')
    ax.set_xlabel('Rollout Step (hours)', fontsize=11)
    ax.set_ylabel('Error (m)', fontsize=11)
    ax.set_title('Rollout Error Accumulation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Spatial error distribution
    ax = axes[1, 0]
    lon = data['lon']
    lat = data['lat']
    node_mae = spatial['node_mae']
    scatter = ax.scatter(lon, lat, c=node_mae, cmap='hot', s=1, vmin=0, vmax=np.percentile(node_mae, 95))
    plt.colorbar(scatter, ax=ax, label='MAE (m)')
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Spatial Error Distribution', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')

    # 4. Error histogram
    ax = axes[1, 1]
    errors_flat = single_step['errors'].flatten()
    ax.hist(errors_flat, bins=100, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Error (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Error Distribution (RMSE={single_step["rmse"]:.4f} m)', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.5, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('CWL GNN Model Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / "cwl_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved: {output_path}")

    return output_path


def main():
    print("="*70)
    print("CWL GNN Model Validation")
    print("="*70)

    # Load data and model
    data = load_data_and_model()

    # Run validations
    single_step = validate_single_step(data)
    rollout = validate_rollout(data, rollout_steps=50)
    spatial = validate_spatial(data, single_step)

    # Generate plots
    plot_path = plot_validation_results(data, single_step, rollout, spatial)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"""
Key Metrics:
  - Single-step RMSE: {single_step['rmse']:.4f} m
  - Single-step MAE: {single_step['mae']:.4f} m
  - Single-step Correlation: {single_step['corr']:.4f}
  - 50-step rollout final RMSE: {rollout['rmse'][-1]:.4f} m
  - 50-step rollout final Correlation: {rollout['corr'][-1]:.4f}

Interpretation:
  - RMSE < 0.1 m is excellent for storm surge prediction
  - RMSE < 0.2 m is good for operational use
  - RMSE > 0.3 m may need model improvement

Validation plot saved to: {plot_path}
""")


if __name__ == '__main__':
    main()
