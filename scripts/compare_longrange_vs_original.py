#!/usr/bin/env python3
"""
Compare Long-Range Enhanced Model vs Original 25k V2

Evaluates whether long-range edges improve:
1. RMSE at different forecast horizons
2. Regional performance (bays vs coastal)
3. Amplitude preservation
4. Station-level predictions
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Data directories
DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
MESH_ORIGINAL = DATA_DIR / 'mesh.npz'
MESH_LONGRANGE = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2_longrange/mesh.npz')

# Checkpoints to compare
CHECKPOINT_ORIGINAL = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2/checkpoint_epoch_55.pt')
CHECKPOINT_LONGRANGE = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_longrange/checkpoint_longrange_epoch_5.pt')

OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/plots/longrange_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model config
HIDDEN_DIM = 128
NUM_LAYERS = 6
ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Evaluation config
MAX_ROLLOUT = 24  # Compare up to 24-hour forecasts
VAL_DATE = '20250101'  # Validation date to use

# Tidal periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}

# Region definitions for regional analysis
REGIONS = {
    'chesapeake_inner': {'lon': (-76.8, -76.0), 'lat': (38.5, 39.5)},
    'chesapeake_mouth': {'lon': (-76.5, -75.8), 'lat': (36.8, 37.5)},
    'delaware_bay': {'lon': (-75.6, -74.8), 'lat': (38.7, 40.0)},
    'nj_coast': {'lon': (-74.5, -73.8), 'lat': (39.0, 40.5)},
    'open_ocean': {'lon': (-74.0, -72.0), 'lat': (38.0, 41.0)},
}

# Stations for detailed comparison
STATIONS = {
    'Baltimore': (-76.578, 39.267),
    'Annapolis': (-76.480, 38.983),
    'Lewes_DE': (-75.131, 38.797),
    'Atlantic_City': (-74.406, 39.349),
    'The_Battery': (-74.003, 40.704),
    'Cape_May': (-74.972, 38.974),
}


# ============================================================
# Model Architecture (same as training)
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
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
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]

        h_src = h[:, row, :]
        h_dst = h[:, col, :]
        h_gradient = h_dst - h_src

        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1)

        edge_input_flat = edge_input.reshape(B * E, -1)
        edge_msg_flat = self.edge_mlp(edge_input_flat)
        edge_msg = edge_msg_flat.reshape(B, E, F)

        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, E, F)
        aggr.scatter_add_(1, row_expanded, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        node_input_flat = node_input.reshape(B * N, -1)
        node_out_flat = self.node_mlp(node_input_flat)
        node_out = node_out_flat.reshape(B, N, F)

        return h + node_out, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim

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
# Helper Functions
# ============================================================

def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    model = BatchedTemporalMemoryGNN(
        state_dim=1,
        temporal_dim=12,
        static_feature_dim=4,
        forcing_feature_dim=8,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def prepare_mesh_tensors(mesh_path, device):
    """Load and prepare mesh tensors."""
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))

    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = mesh_data['edge_index']

    # Static features
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

    # Edge attributes
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth,
        'static_base': torch.tensor(static_base).to(device),
        'edge_index': torch.tensor(edge_index, dtype=torch.long).to(device),
        'edge_attr': torch.tensor(edge_attr).to(device),
    }


def compute_tidal_harmonics(global_hour, num_nodes, device):
    """Compute tidal harmonics for given hour."""
    harmonics = []
    for name, period in TIDAL_PERIODS.items():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    tidal = np.array(harmonics, dtype=np.float32)
    tidal_expanded = np.tile(tidal, (num_nodes, 1))
    return torch.tensor(tidal_expanded).unsqueeze(0).to(device)


def rollout_prediction(model, mesh_tensors, elevation, forcing, start_t, num_steps, device):
    """Run autoregressive rollout and return predictions."""
    num_nodes = len(mesh_tensors['lon'])
    depth = mesh_tensors['depth']

    # Initial state
    cwl_t = np.nan_to_num(elevation[start_t], nan=0.0).astype(np.float32) / ETA_SCALE
    cwl_prev = np.nan_to_num(elevation[start_t - 1], nan=0.0).astype(np.float32) / ETA_SCALE

    current = torch.tensor(cwl_t).unsqueeze(0).unsqueeze(-1).to(device)
    prev = torch.tensor(cwl_prev).unsqueeze(0).unsqueeze(-1).to(device)

    # Global time
    date_dt = datetime.strptime(VAL_DATE, '%Y%m%d')
    global_hour = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0 + start_t * DT_HOURS

    predictions = []

    with torch.no_grad():
        for step in range(num_steps):
            # Tidal harmonics
            tidal = compute_tidal_harmonics(global_hour + step, num_nodes, device)

            # Static features with water level
            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = torch.cat([
                mesh_tensors['static_base'].unsqueeze(0),
                torch.tensor(wl_norm[:, np.newaxis]).unsqueeze(0).to(device)
            ], dim=-1)

            # Forcing
            forcing_t = np.stack([
                forcing['u10'][start_t + step],
                forcing['v10'][start_t + step],
                forcing['wind_speed'][start_t + step],
                forcing['wind_speed_sq'][start_t + step],
                forcing['wind_dir'][start_t + step],
                forcing['pressure'][start_t + step],
                forcing['dP_dx'][start_t + step],
                forcing['dP_dy'][start_t + step],
            ], axis=1).astype(np.float32)
            forcing_tensor = torch.tensor(forcing_t).unsqueeze(0).to(device)

            # Rate of change
            dxdt = (current - prev) / DT_HOURS

            # Forward pass
            pred = model(current, prev, dxdt, tidal, static, forcing_tensor,
                        mesh_tensors['edge_index'], mesh_tensors['edge_attr'])

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)

            # Update for next step
            prev = current
            current = pred

    return np.array(predictions)


def get_region_mask(lon, lat, region):
    """Get node indices within a region."""
    lon_min, lon_max = region['lon']
    lat_min, lat_max = region['lat']
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


def find_nearest_node(lon, lat, target_lon, target_lat):
    """Find nearest mesh node to target coordinates."""
    dist = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
    return np.argmin(dist)


# ============================================================
# Main Comparison
# ============================================================

def main():
    logger.info("="*70)
    logger.info("COMPARING LONG-RANGE VS ORIGINAL 25K V2")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Check if checkpoints exist
    if not CHECKPOINT_ORIGINAL.exists():
        logger.error(f"Original checkpoint not found: {CHECKPOINT_ORIGINAL}")
        return

    if not CHECKPOINT_LONGRANGE.exists():
        logger.error(f"Long-range checkpoint not found: {CHECKPOINT_LONGRANGE}")
        logger.info("Run this script after long-range training produces checkpoints.")
        return

    # Load models
    logger.info("\nLoading models...")
    model_original = load_model(CHECKPOINT_ORIGINAL, device)
    model_longrange = load_model(CHECKPOINT_LONGRANGE, device)
    logger.info("  Original model loaded")
    logger.info("  Long-range model loaded")

    # Load meshes
    logger.info("\nLoading meshes...")
    mesh_original = prepare_mesh_tensors(MESH_ORIGINAL, device)
    mesh_longrange = prepare_mesh_tensors(MESH_LONGRANGE, device)
    logger.info(f"  Original: {len(mesh_original['lon']):,} nodes, {mesh_original['edge_index'].shape[1]:,} edges")
    logger.info(f"  Long-range: {len(mesh_longrange['lon']):,} nodes, {mesh_longrange['edge_index'].shape[1]:,} edges")

    # Load validation data
    logger.info(f"\nLoading validation data: {VAL_DATE}")
    val_file = DATA_DIR / f'processed_{VAL_DATE}.npz'
    if not val_file.exists():
        logger.error(f"Validation file not found: {val_file}")
        return

    val_data = np.load(val_file)
    elevation = val_data['elevation']
    forcing = {k: val_data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                         'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}

    # Run rollouts from multiple start times
    logger.info(f"\nRunning {MAX_ROLLOUT}-hour rollouts...")
    start_times = list(range(6, 48, 6))  # Start at hours 6, 12, 18, 24, 30, 36, 42

    all_rmse_original = []
    all_rmse_longrange = []
    regional_rmse = {region: {'original': [], 'longrange': []} for region in REGIONS}

    for start_t in start_times:
        logger.info(f"  Start time: t={start_t}")

        # Get predictions
        pred_original = rollout_prediction(model_original, mesh_original, elevation, forcing,
                                           start_t, MAX_ROLLOUT, device)
        pred_longrange = rollout_prediction(model_longrange, mesh_longrange, elevation, forcing,
                                            start_t, MAX_ROLLOUT, device)

        # Ground truth
        truth = np.array([elevation[start_t + i + 1] for i in range(MAX_ROLLOUT)])

        # Compute RMSE per timestep
        rmse_original = np.sqrt(np.nanmean((pred_original - truth)**2, axis=1)) * 100  # cm
        rmse_longrange = np.sqrt(np.nanmean((pred_longrange - truth)**2, axis=1)) * 100

        all_rmse_original.append(rmse_original)
        all_rmse_longrange.append(rmse_longrange)

        # Regional RMSE
        lon, lat = mesh_original['lon'], mesh_original['lat']
        for region_name, region_bounds in REGIONS.items():
            mask = get_region_mask(lon, lat, region_bounds)
            if mask.sum() > 0:
                rmse_orig_region = np.sqrt(np.nanmean((pred_original[:, mask] - truth[:, mask])**2, axis=1)) * 100
                rmse_lr_region = np.sqrt(np.nanmean((pred_longrange[:, mask] - truth[:, mask])**2, axis=1)) * 100
                regional_rmse[region_name]['original'].append(rmse_orig_region)
                regional_rmse[region_name]['longrange'].append(rmse_lr_region)

    # Average RMSE across start times
    mean_rmse_original = np.mean(all_rmse_original, axis=0)
    mean_rmse_longrange = np.mean(all_rmse_longrange, axis=0)

    # ========================================
    # Plot 1: RMSE vs Lead Time
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    hours = np.arange(1, MAX_ROLLOUT + 1)
    ax.plot(hours, mean_rmse_original, 'b-o', label='Original 25k V2', linewidth=2, markersize=6)
    ax.plot(hours, mean_rmse_longrange, 'r-s', label='Long-Range Enhanced', linewidth=2, markersize=6)

    ax.set_xlabel('Forecast Lead Time (hours)', fontsize=12)
    ax.set_ylabel('RMSE (cm)', fontsize=12)
    ax.set_title('RMSE Comparison: Long-Range vs Original', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, MAX_ROLLOUT + 1)

    # Add improvement annotation
    improvement = (mean_rmse_original - mean_rmse_longrange) / mean_rmse_original * 100
    for i in [5, 11, 17, 23]:  # t+6, t+12, t+18, t+24
        if improvement[i] > 0:
            ax.annotate(f'{improvement[i]:.1f}%', xy=(i+1, mean_rmse_longrange[i]),
                       xytext=(5, -15), textcoords='offset points', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rmse_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {OUTPUT_DIR / 'rmse_comparison.png'}")
    plt.close()

    # ========================================
    # Plot 2: Regional RMSE Comparison
    # ========================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (region_name, data) in zip(axes.flat, regional_rmse.items()):
        if len(data['original']) > 0:
            mean_orig = np.mean(data['original'], axis=0)
            mean_lr = np.mean(data['longrange'], axis=0)

            ax.plot(hours, mean_orig, 'b-o', label='Original', linewidth=2, markersize=4)
            ax.plot(hours, mean_lr, 'r-s', label='Long-Range', linewidth=2, markersize=4)
            ax.set_title(region_name.replace('_', ' ').title())
            ax.set_xlabel('Lead Time (hours)')
            ax.set_ylabel('RMSE (cm)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    # Remove empty subplot if odd number of regions
    if len(regional_rmse) < 6:
        axes.flat[-1].axis('off')

    plt.suptitle('Regional RMSE Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regional_rmse_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {OUTPUT_DIR / 'regional_rmse_comparison.png'}")
    plt.close()

    # ========================================
    # Plot 3: Station Time Series
    # ========================================
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    lon, lat = mesh_original['lon'], mesh_original['lat']
    start_t = 12  # Use hour 12 for time series

    pred_original = rollout_prediction(model_original, mesh_original, elevation, forcing,
                                       start_t, MAX_ROLLOUT, device)
    pred_longrange = rollout_prediction(model_longrange, mesh_longrange, elevation, forcing,
                                        start_t, MAX_ROLLOUT, device)
    truth = np.array([elevation[start_t + i + 1] for i in range(MAX_ROLLOUT)])

    for ax, (station_name, (stn_lon, stn_lat)) in zip(axes.flat, STATIONS.items()):
        node_idx = find_nearest_node(lon, lat, stn_lon, stn_lat)

        ax.plot(hours, truth[:, node_idx] * 100, 'k-', label='Truth', linewidth=2)
        ax.plot(hours, pred_original[:, node_idx] * 100, 'b--', label='Original', linewidth=1.5, alpha=0.8)
        ax.plot(hours, pred_longrange[:, node_idx] * 100, 'r--', label='Long-Range', linewidth=1.5, alpha=0.8)

        ax.set_title(station_name.replace('_', ' '))
        ax.set_xlabel('Lead Time (hours)')
        ax.set_ylabel('Water Level (cm)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Station Time Series Comparison (start t={start_t})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'station_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {OUTPUT_DIR / 'station_comparison.png'}")
    plt.close()

    # ========================================
    # Summary Statistics
    # ========================================
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nRMSE at key lead times (cm):")
    print(f"{'Lead Time':<12} {'Original':<12} {'Long-Range':<12} {'Improvement':<12}")
    print("-" * 48)
    for t in [1, 6, 12, 18, 24]:
        idx = t - 1
        orig = mean_rmse_original[idx]
        lr = mean_rmse_longrange[idx]
        imp = (orig - lr) / orig * 100
        print(f"t+{t:<10} {orig:<12.2f} {lr:<12.2f} {imp:+.1f}%")

    print(f"\nMean RMSE (1-24h):")
    print(f"  Original:   {mean_rmse_original.mean():.2f} cm")
    print(f"  Long-Range: {mean_rmse_longrange.mean():.2f} cm")
    print(f"  Improvement: {(mean_rmse_original.mean() - mean_rmse_longrange.mean()) / mean_rmse_original.mean() * 100:+.1f}%")
    print("="*70)


if __name__ == '__main__':
    main()
