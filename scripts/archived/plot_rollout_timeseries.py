#!/usr/bin/env -S /usr/bin/python3
"""
Plot rollout timeseries at tide gauge stations.

Usage:
    python scripts/plot_rollout_timeseries.py --checkpoint outputs/checkpoints/best_optimized_model.pt
    python scripts/plot_rollout_timeseries.py --checkpoint outputs/checkpoints/best_multidate_model.pt --date 20251130
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Model Definitions
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
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

    def forward(self, h, edge_index, edge_attr):
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
    """Model compatible with older checkpoints (uses 'layers')."""
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
            SWEInspiredGraphBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x, static_features, forcing_features, edge_index, edge_attr):
        node_features = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return self.decoder(h)


class PhysicsInformedCWLModelA10G(nn.Module):
    """Model compatible with A10G checkpoints (uses 'gnn_layers' and deeper decoder)."""
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

    def forward(self, x, static_features, forcing_features, edge_index, edge_attr):
        node_features = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        delta = self.decoder(h)
        return x + delta


# ============================================================
# Station Definitions
# ============================================================

STATIONS = {
    'Atlantic_City': {'lon': -74.4181, 'lat': 39.3550, 'coops_id': '8534720'},
    'Sandy_Hook': {'lon': -74.0092, 'lat': 40.4669, 'coops_id': '8531680'},
    'The_Battery': {'lon': -74.0142, 'lat': 40.6995, 'coops_id': '8518750'},
    'Lewes_DE': {'lon': -75.1194, 'lat': 38.7828, 'coops_id': '8557380'},
    'Cape_May': {'lon': -74.9600, 'lat': 38.9683, 'coops_id': '8536110'},
}

# ============================================================
# Helper Functions
# ============================================================

def find_valid_water_node(lon_mesh, lat_mesh, elev, station_lon, station_lat, search_radius=0.3):
    """Find a valid water node near the station (not on land)."""
    dist = np.sqrt((lon_mesh - station_lon)**2 + (lat_mesh - station_lat)**2)
    nearby_mask = dist < search_radius

    if not nearby_mask.any():
        return np.argmin(dist)

    nearby_indices = np.where(nearby_mask)[0]

    best_idx = None
    best_score = -1

    for nidx in nearby_indices:
        ts = elev[:, nidx]
        ts_valid = ts[~np.isnan(ts)]
        if len(ts_valid) > 10:
            ts_range = np.ptp(ts_valid)
            if ts_range > 0.01:
                proximity = 1.0 / (dist[nidx] + 0.01)
                score = ts_range * proximity
                if score > best_score:
                    best_score = score
                    best_idx = nidx

    return best_idx if best_idx is not None else np.argmin(dist)


def compute_static_features(lon, lat, depth):
    """Compute static features exactly as in training."""
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

    return static_base, x_cart, y_cart


def compute_edge_features(edge_index, x_cart, y_cart):
    """Compute edge features exactly as in training."""
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)
    return edge_attr


def run_rollout(model, initial_cwl, forcing_data, static_base, edge_index, edge_attr,
                depth, n_hours, device, eta_scale=2.0, wind_scale=15.0):
    """Run autoregressive rollout."""

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    predictions = [initial_cwl.copy()]

    with torch.no_grad():
        current_cwl = initial_cwl.copy()

        for t in range(n_hours):
            # Normalize current state
            cwl_norm = current_cwl / eta_scale

            # Water level feature
            water_level = depth + current_cwl
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

            # Static features with water level
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1).astype(np.float32)

            # Forcing (pressure is RAW)
            u = np.nan_to_num(forcing_data['u10'][t].astype(np.float32), nan=0.0) / wind_scale
            v = np.nan_to_num(forcing_data['v10'][t].astype(np.float32), nan=0.0) / wind_scale
            p = np.nan_to_num(forcing_data['pressure'][t].astype(np.float32), nan=0.0)
            forcing = np.stack([u, v, p], axis=1).astype(np.float32)

            # To tensors
            x = torch.tensor(cwl_norm[:, np.newaxis], dtype=torch.float32, device=device)
            static_tensor = torch.tensor(static, dtype=torch.float32, device=device)
            forcing_tensor = torch.tensor(forcing, dtype=torch.float32, device=device)

            # Predict
            pred_norm = model(x, static_tensor, forcing_tensor, edge_index_tensor, edge_attr_tensor)
            pred_physical = pred_norm.squeeze().cpu().numpy() * eta_scale

            predictions.append(pred_physical)
            current_cwl = pred_physical

            if (t + 1) % 12 == 0:
                logger.info(f"  t+{t+1}h complete, pred range: [{pred_physical.min():.2f}, {pred_physical.max():.2f}]")

    return np.array(predictions)


def plot_station_timeseries(predictions, ground_truth, lon, lat, elevation,
                            start_date, output_path, use_dots=True, checkpoint_name=""):
    """Plot timeseries at all stations."""

    n_hours = len(predictions) - 1
    hours = np.arange(n_hours + 1)

    # Parse start date
    year = int(start_date[:4])
    month = int(start_date[4:6])
    day = int(start_date[6:8])
    start_time = datetime(year, month, day, 0, 0)
    times = [start_time + timedelta(hours=int(h)) for h in hours]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    stats = {}

    for i, (station_name, coords) in enumerate(STATIONS.items()):
        if i >= 5:
            break

        ax = axes[i]

        node_idx = find_valid_water_node(lon, lat, elevation, coords['lon'], coords['lat'])

        gt = ground_truth[:n_hours+1, node_idx]
        pred = predictions[:, node_idx]

        if use_dots:
            ax.scatter(times, gt, c='black', s=20, marker='o', label='STOFS Ground Truth', zorder=5)
            ax.scatter(times, pred, c='blue', s=15, marker='s', label='GNN Prediction', zorder=4, alpha=0.7)
        else:
            ax.plot(times, gt, 'k-', linewidth=2, label='STOFS Ground Truth')
            ax.plot(times, pred, 'b-', linewidth=1.5, label='GNN Prediction')

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

        for thresh in [0.5, 1.0, -0.5, -1.0]:
            ax.axhline(y=thresh, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)

        valid = ~np.isnan(gt)
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((pred[valid] - gt[valid])**2))
            corr = np.corrcoef(pred[valid], gt[valid])[0, 1] if valid.sum() > 5 else 0
        else:
            rmse, corr = np.nan, np.nan

        stats[station_name] = {'rmse': rmse, 'corr': corr}

        ax.set_title(f'{station_name}\nRMSE: {rmse:.3f}m, R: {corr:.3f}')
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Water Level (m, MSL)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    axes[5].set_visible(False)

    title = f'GNN Rollout vs STOFS Ground Truth ({n_hours}h forecast from {start_date})'
    if checkpoint_name:
        title += f'\nModel: {checkpoint_name}'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Plot rollout timeseries at tide gauge stations')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--date', type=str, default='20251130', help='Forecast date (YYYYMMDD)')
    parser.add_argument('--hours', type=int, default=48, help='Forecast hours')
    parser.add_argument('--mesh', type=str, default=None, help='Path to mesh file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to preprocessed data directory')
    parser.add_argument('--output', type=str, default=None, help='Output plot path')
    parser.add_argument('--lines', action='store_true', help='Use lines instead of dots')
    args = parser.parse_args()

    # Find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Default paths
    if args.mesh is None:
        args.mesh = os.path.join(project_root, 'data/processed_optimized/mesh_optimized.npz')
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data/processed_optimized')
    if args.output is None:
        checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(project_root, f'outputs/figures/rollout_{checkpoint_name}_{args.date}.png')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info("=" * 60)
    logger.info("ROLLOUT TIMESERIES PLOT")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Date: {args.date}")
    logger.info(f"Hours: {args.hours}")

    # Load mesh
    logger.info(f"Loading mesh from {args.mesh}")
    mesh = np.load(args.mesh)
    lon = mesh['lon'].astype(np.float32)
    lat = mesh['lat'].astype(np.float32)
    depth = mesh['depth'].astype(np.float32)
    edge_index = mesh['edge_index']

    logger.info(f"Mesh: {len(lon)} nodes, {edge_index.shape[1]} edges")

    # Load data
    data_path = os.path.join(args.data_dir, f'processed_{args.date}.npz')
    logger.info(f"Loading data from {data_path}")
    data = np.load(data_path)
    elevation = data['elevation']
    forcing_data = {
        'u10': data['u10'],
        'v10': data['v10'],
        'pressure': data['pressure'],
    }

    logger.info(f"Elevation shape: {elevation.shape}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_data = torch.load(args.checkpoint, map_location=device)

    config = checkpoint_data['config']
    logger.info(f"Model config: {config}")

    # Handle different config formats
    hidden_dim = config.get('hidden_dim', config.get('HIDDEN_DIM', 96))
    num_layers = config.get('num_layers', config.get('NUM_LAYERS', 6))
    static_features = config.get('static_features', 4)
    forcing_features = config.get('forcing_features', 3)
    eta_scale = config.get('eta_scale', config.get('ETA_SCALE', 2.0))

    # Detect model architecture from state_dict keys
    state_dict_keys = list(checkpoint_data['model_state_dict'].keys())
    use_a10g_model = any('gnn_layers' in k for k in state_dict_keys)

    if use_a10g_model:
        logger.info("Detected A10G model architecture")
        model = PhysicsInformedCWLModelA10G(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            static_feature_dim=static_features,
            forcing_feature_dim=forcing_features,
        )
    else:
        logger.info("Detected standard model architecture")
        model = PhysicsInformedCWLModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            static_feature_dim=static_features,
            forcing_feature_dim=forcing_features,
        )
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: hidden={hidden_dim}, layers={num_layers}")

    # Compute features
    static_base, x_cart, y_cart = compute_static_features(lon, lat, depth)
    edge_attr = compute_edge_features(edge_index, x_cart, y_cart)

    # Initial condition
    initial_cwl = elevation[0].astype(np.float32)
    initial_cwl = np.nan_to_num(initial_cwl, nan=0.0)

    # Run rollout
    n_hours = min(args.hours, elevation.shape[0] - 1)
    logger.info(f"Running {n_hours}h rollout...")

    predictions = run_rollout(
        model=model,
        initial_cwl=initial_cwl,
        forcing_data=forcing_data,
        static_base=static_base,
        edge_index=edge_index,
        edge_attr=edge_attr,
        depth=depth,
        n_hours=n_hours,
        device=device,
        eta_scale=eta_scale,
    )

    logger.info("Rollout complete!")

    # Plot
    checkpoint_name = os.path.basename(args.checkpoint)
    stats = plot_station_timeseries(
        predictions=predictions,
        ground_truth=elevation,
        lon=lon,
        lat=lat,
        elevation=elevation,
        start_date=args.date,
        output_path=args.output,
        use_dots=not args.lines,
        checkpoint_name=checkpoint_name,
    )

    # Print statistics
    logger.info("\nStation Statistics:")
    for station, s in stats.items():
        logger.info(f"  {station}: RMSE={s['rmse']:.3f}m, R={s['corr']:.3f}")


if __name__ == '__main__':
    main()
