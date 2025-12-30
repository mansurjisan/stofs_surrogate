#!/usr/bin/env python3
"""
Rollout script for 80k node STOFS-GNN model.

Domain: Long Island Sound to Southern Maine (40-44N, 74-69W)
Resolution: ~1.5 km (80,000 nodes)

Usage:
    python rollout_80k_model.py --date 20250115 --hours 48
    python rollout_80k_model.py --date 20250115 --hours 48 --obs
    python rollout_80k_model.py --date 20250115 --hours 48 --save-ts
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import requests

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_80k_option_a'))
MODEL_PATH = Path(os.environ.get('STOFS_MODEL_PATH', '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_80k_optimized/best_model.pt'))
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures_80k')
TS_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/timeseries_80k')

ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0

# Tidal harmonic periods (hours)
M2_PERIOD = 12.42
S2_PERIOD = 12.00

# Global epoch (must match training)
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Stations in Long Island Sound to Southern Maine domain
STATIONS = {
    'Boston': {'lat': 42.355, 'lon': -71.050, 'coops_id': '8443970'},
    'Portland_ME': {'lat': 43.657, 'lon': -70.246, 'coops_id': '8418150'},
    'New_London': {'lat': 41.361, 'lon': -72.090, 'coops_id': '8461490'},
    'Bridgeport': {'lat': 41.173, 'lon': -73.182, 'coops_id': '8467150'},
    'Kings_Point': {'lat': 40.810, 'lon': -73.765, 'coops_id': '8516945'},
    'The_Battery': {'lat': 40.700, 'lon': -74.014, 'coops_id': '8518750'},
    'Montauk': {'lat': 41.048, 'lon': -71.960, 'coops_id': '8510560'},
    'Providence': {'lat': 41.807, 'lon': -71.401, 'coops_id': '8454000'},
    'Newport': {'lat': 41.505, 'lon': -71.327, 'coops_id': '8452660'},
}


# ============================================================
# Model Architecture (must match training)
# ============================================================

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


class TemporalMemoryGNN(nn.Module):
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
            SWEInspiredGraphBlock(hidden_dim) for _ in range(num_layers)
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
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)
        delta = self.decoder(h)
        return x + delta


def compute_tidal_harmonics(global_hour: float, num_nodes: int) -> np.ndarray:
    phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
    phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
    tidal = np.array([np.sin(phase_m2), np.cos(phase_m2),
                      np.sin(phase_s2), np.cos(phase_s2)], dtype=np.float32)
    return np.tile(tidal, (num_nodes, 1))


def fetch_coops_observations(station_id, start_date, end_date):
    """Fetch observations from CO-OPS API."""
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'begin_date': start_date.strftime('%Y%m%d %H:%M'),
        'end_date': end_date.strftime('%Y%m%d %H:%M'),
        'station': station_id,
        'product': 'water_level',
        'datum': 'MSL',
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'json',
        'application': 'stofs_gnn'
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        if 'data' in data:
            times, values = [], []
            for record in data['data']:
                try:
                    t = datetime.strptime(record['t'], '%Y-%m-%d %H:%M')
                    v = float(record['v'])
                    times.append(t)
                    values.append(v)
                except (ValueError, KeyError):
                    continue
            return times, values
    except Exception as e:
        print(f"    Warning: Could not fetch obs for {station_id}: {e}")
    return None, None


def find_nearest_node(lon_target, lat_target, mesh_lon, mesh_lat):
    dist = np.sqrt((mesh_lon - lon_target)**2 + (mesh_lat - lat_target)**2)
    return np.argmin(dist)


def load_model_and_data(date_str, device):
    """Load model and data for rollout."""
    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    if not mesh_path.exists():
        print(f"ERROR: Mesh not found at {mesh_path}")
        sys.exit(1)

    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)

    print(f"Mesh: {len(lon):,} nodes, {edge_index.shape[1]:,} edges")

    # Compute edge features
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    src, dst = mesh_data['edge_index']
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32
    ).to(device)

    # Static features
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
    static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)

    # Load date data
    data_path = DATA_DIR / f'processed_{date_str}.npz'
    if not data_path.exists():
        print(f"ERROR: Data not found for {date_str}")
        print(f"  Expected: {data_path}")
        sys.exit(1)

    data = dict(np.load(data_path))
    elevation = data['elevation']
    forcing = {
        'u10': data['u10'],
        'v10': data['v10'],
        'pressure': data['pressure'],
    }
    print(f"Loaded {date_str}: {elevation.shape[0]} timesteps")

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = TemporalMemoryGNN(
        state_dim=1,
        temporal_dim=6,
        static_feature_dim=4,
        forcing_feature_dim=3,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"  Nodes: {config.get('num_nodes', 'N/A'):,}")

    return model, lon, lat, depth, elevation, forcing, edge_index, edge_attr, static_base, x_cart, y_cart


def run_rollout(model, elevation, forcing, depth, static_base, x_cart, y_cart,
                edge_index, edge_attr, device, num_steps=48, date_str='20250115'):
    """Run autoregressive rollout."""
    predictions = []
    ground_truth = []

    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_offset = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
    num_nodes = len(depth)

    # Initialize with first two timesteps
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Running {num_steps}h rollout...")

    with torch.no_grad():
        for t in range(1, min(num_steps + 1, len(elevation) - 1)):
            # Temporal features
            dxdt = (current_cwl - current_prev) / DT_HOURS

            # Tidal harmonics
            global_hour_t = global_hours_offset + t * DT_HOURS
            tidal_harmonics = compute_tidal_harmonics(global_hour_t, num_nodes)
            tidal_tensor = torch.tensor(tidal_harmonics, dtype=torch.float32).to(device)

            # Static features with water level
            cwl_np = current_cwl.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).to(device)

            # Forcing
            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pres = forcing['pressure'][t].astype(np.float32)
            forcing_arr = np.stack([u10, v10, pres], axis=1)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).to(device)

            with autocast('cuda'):
                pred = model(current_cwl, current_prev, dxdt, tidal_tensor,
                           static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            # Update state
            current_prev = current_cwl
            current_cwl = pred

            if (t + 1) % 12 == 0:
                print(f"  Step {t+1}/{num_steps}")

    return np.array(predictions), np.array(ground_truth)


def plot_spatial(predictions, ground_truth, lon, lat, output_path, date_str, hour=24):
    """Plot spatial comparison at a given hour."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    idx = min(hour - 1, len(predictions) - 1)
    pred = predictions[idx]
    truth = ground_truth[idx]
    error = pred - truth

    vmin, vmax = -1.0, 1.0
    emin, emax = -0.3, 0.3

    sc1 = axes[0].scatter(lon, lat, c=truth, s=0.5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'STOFS Ground Truth (Hour {hour})')
    plt.colorbar(sc1, ax=axes[0], label='Water Level (m)')

    sc2 = axes[1].scatter(lon, lat, c=pred, s=0.5, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'GNN Prediction (Hour {hour})')
    plt.colorbar(sc2, ax=axes[1], label='Water Level (m)')

    sc3 = axes[2].scatter(lon, lat, c=error, s=0.5, cmap='RdBu_r', vmin=emin, vmax=emax)
    axes[2].set_title(f'Error (Pred - Truth)\nRMSE: {np.sqrt(np.mean(error**2)):.3f}m')
    plt.colorbar(sc3, ax=axes[2], label='Error (m)')

    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.suptitle(f'80k Node GNN Rollout - {date_str}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='80k Node GNN Rollout')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYYMMDD)')
    parser.add_argument('--hours', type=int, default=48, help='Rollout hours')
    parser.add_argument('--obs', action='store_true', help='Fetch CO-OPS observations')
    parser.add_argument('--save-ts', action='store_true', help='Save timeseries files')
    parser.add_argument('--spatial', action='store_true', help='Create spatial plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Date: {args.date}")
    print(f"Rollout hours: {args.hours}")

    # Load model and data
    model, lon, lat, depth, elevation, forcing, edge_index, edge_attr, static_base, x_cart, y_cart = \
        load_model_and_data(args.date, device)

    # Find station indices
    print("\nStation locations:")
    station_indices = {}
    for name, info in STATIONS.items():
        # Check if station is within domain
        if info['lon'] < lon.min() or info['lon'] > lon.max():
            continue
        if info['lat'] < lat.min() or info['lat'] > lat.max():
            continue

        idx = find_nearest_node(info['lon'], info['lat'], lon, lat)
        dist_km = np.sqrt((lon[idx] - info['lon'])**2 + (lat[idx] - info['lat'])**2) * 111
        station_indices[name] = idx
        print(f"  {name}: node {idx:,} ({dist_km:.1f} km from actual)")

    # Run rollout
    predictions, ground_truth = run_rollout(
        model, elevation, forcing, depth, static_base, x_cart, y_cart,
        edge_index, edge_attr, device, args.hours, args.date
    )

    # Fetch observations if requested
    obs_data = {}
    if args.obs:
        print("\nFetching CO-OPS observations...")
        start_dt = datetime.strptime(args.date, '%Y%m%d')
        end_dt = start_dt + timedelta(hours=args.hours)

        for name, info in STATIONS.items():
            if name not in station_indices:
                continue
            times, values = fetch_coops_observations(info['coops_id'], start_dt, end_dt)
            if times:
                obs_data[name] = {'times': times, 'values': values}
                print(f"  {name}: {len(values)} observations")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_ts:
        ts_dir = TS_DIR / args.date
        ts_dir.mkdir(parents=True, exist_ok=True)

    # Plot timeseries
    n_stations = len(station_indices)
    n_cols = min(3, n_stations)
    n_rows = (n_stations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_stations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    start_dt = datetime.strptime(args.date, '%Y%m%d')
    times = np.arange(len(predictions)) * DT_HOURS

    for i, (name, idx) in enumerate(station_indices.items()):
        ax = axes[i]

        stofs_wl = ground_truth[:, idx]
        gnn_wl = predictions[:, idx]

        # Metrics
        valid = ~(np.isnan(stofs_wl) | np.isnan(gnn_wl))
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((stofs_wl[valid] - gnn_wl[valid])**2))
            corr = np.corrcoef(stofs_wl[valid], gnn_wl[valid])[0, 1] if np.std(stofs_wl[valid]) > 0 else np.nan
        else:
            rmse, corr = np.nan, np.nan

        ax.plot(times, stofs_wl, 'k-', label='STOFS', linewidth=1.5)
        ax.plot(times, gnn_wl, 'b--', label='GNN', linewidth=1.5)

        if name in obs_data:
            obs_times = [(t - start_dt).total_seconds() / 3600 for t in obs_data[name]['times']]
            ax.plot(obs_times, obs_data[name]['values'], 'g.', alpha=0.5, markersize=3, label='Obs')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Water Level (m)')
        ax.set_title(f'{name}\nRMSE: {rmse:.3f}m, R: {corr:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Save timeseries
        if args.save_ts:
            ts_file = ts_dir / f'{name}_rollout.txt'
            with open(ts_file, 'w') as f:
                f.write(f'# Station: {name}\n')
                f.write(f'# datetime, hours, stofs_wl(m), gnn_wl(m)\n')
                for t_idx in range(len(predictions)):
                    dt = start_dt + timedelta(hours=t_idx)
                    f.write(f'{dt.strftime("%Y-%m-%d %H:%M")}, {t_idx}, {stofs_wl[t_idx]:.4f}, {gnn_wl[t_idx]:.4f}\n')

    # Hide unused subplots
    for i in range(len(station_indices), len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'80k Node GNN - {args.hours}h Rollout ({args.date})', fontsize=12)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f'rollout_80k_{args.date}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()

    # Spatial plots
    if args.spatial:
        spatial_path = OUTPUT_DIR / f'spatial_80k_{args.date}.png'
        plot_spatial(predictions, ground_truth, lon, lat, spatial_path, args.date, hour=24)

    # Summary statistics
    print("\n" + "="*60)
    print("ROLLOUT SUMMARY")
    print("="*60)
    all_rmse = np.sqrt(np.mean((predictions - ground_truth)**2))
    print(f"Overall RMSE: {all_rmse:.4f} m")
    print(f"Max error: {np.abs(predictions - ground_truth).max():.4f} m")

    for name, idx in station_indices.items():
        rmse = np.sqrt(np.mean((predictions[:, idx] - ground_truth[:, idx])**2))
        print(f"  {name}: RMSE = {rmse:.4f} m")


if __name__ == '__main__':
    main()
