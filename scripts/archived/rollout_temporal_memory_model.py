#!/usr/bin/env python3
"""
Rollout script for Temporal Memory GNN model.

This model uses η(t-1) and dη/dt as additional inputs to resolve
phase ambiguity and reduce the ~3.5 hour lag seen in the baseline model.

Usage:
    python rollout_temporal_memory_model.py --date 20251128 --hours 48
    python rollout_temporal_memory_model.py --date 20251128 --hours 48 --obs
    python rollout_temporal_memory_model.py --date 20251128 --hours 48 --save-ts
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import requests

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_25k')
MODEL_PATH = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints/best_temporal_memory_model.pt')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures')
TS_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/timeseries')

ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0  # 1-hour timesteps

# Tidal harmonic periods (hours) - must match training
M2_PERIOD = 12.42  # Principal lunar semi-diurnal
S2_PERIOD = 12.00  # Principal solar semi-diurnal

# Global epoch for continuous time (must match training)
EPOCH_DATETIME = datetime(2025, 1, 1, 0, 0, 0)

# Mid-Atlantic stations
STATIONS = {
    'Atlantic_City': {'lat': 39.355, 'lon': -74.418, 'coops_id': '8534720'},
    'Sandy_Hook': {'lat': 40.467, 'lon': -74.009, 'coops_id': '8531680'},
    'The_Battery': {'lat': 40.700, 'lon': -74.014, 'coops_id': '8518750'},
    'Lewes_DE': {'lat': 38.783, 'lon': -75.119, 'coops_id': '8557380'},
    'Cape_May': {'lat': 38.968, 'lon': -74.960, 'coops_id': '8536110'},
}


# ============================================================
# Model Architecture (must match training)
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
    """GNN with temporal memory and tidal harmonics for resolving phase ambiguity."""

    def __init__(
        self,
        state_dim: int = 1,
        temporal_dim: int = 6,  # η(t-1), dη/dt, + 4 tidal harmonics
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
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

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        delta = self.decoder(h)
        output = x + delta

        return output


def compute_tidal_harmonics(global_hour: float, num_nodes: int) -> np.ndarray:
    """Compute tidal harmonic features for a given global hour."""
    # M2 tidal constituent (12.42 hour period)
    phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
    sin_m2 = np.sin(phase_m2)
    cos_m2 = np.cos(phase_m2)

    # S2 tidal constituent (12.00 hour period)
    phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
    sin_s2 = np.sin(phase_s2)
    cos_s2 = np.cos(phase_s2)

    # Broadcast to all nodes [4] -> [num_nodes, 4]
    tidal_harmonics = np.array([sin_m2, cos_m2, sin_s2, cos_s2], dtype=np.float32)
    return np.tile(tidal_harmonics, (num_nodes, 1))


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
            times = []
            values = []
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


def find_nearest_node(lon, lat, mesh_lon, mesh_lat):
    """Find nearest mesh node to given coordinates."""
    dist = np.sqrt((mesh_lon - lon)**2 + (mesh_lat - lat)**2)
    return np.argmin(dist)


def load_model_and_data(date_str, device):
    """Load model and data for rollout."""
    # Load mesh
    mesh_data = dict(np.load(DATA_DIR / 'mesh_25k.npz'))
    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)

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
    data = dict(np.load(DATA_DIR / f'processed_{date_str}.npz'))
    elevation = data['elevation']
    forcing = {
        'u10': data['u10'],
        'v10': data['v10'],
        'pressure': data['pressure'],
    }

    # Load model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = TemporalMemoryGNN(
        state_dim=1,
        temporal_dim=config.get('temporal_features', 2),
        static_feature_dim=config.get('static_features', 4),
        forcing_feature_dim=config.get('forcing_features', 3),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"  Config: {config}")

    return model, lon, lat, depth, elevation, forcing, edge_index, edge_attr, static_base, x_cart, y_cart


def run_rollout(model, elevation, forcing, depth, static_base, x_cart, y_cart,
                edge_index, edge_attr, device, num_steps=96, date_str='20251128'):
    """Run autoregressive rollout with temporal memory and tidal harmonics."""
    predictions = []
    ground_truth = []

    # Compute global hours offset from epoch
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_offset = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0
    num_nodes = len(depth)

    # Initialize with first two timesteps for temporal memory
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        for t in range(1, min(num_steps + 1, len(elevation) - 1)):
            # Compute temporal features
            dxdt = (current_cwl - current_prev) / DT_HOURS

            # Compute tidal harmonics with continuous global time
            global_hour_t = global_hours_offset + t * DT_HOURS
            tidal_harmonics = compute_tidal_harmonics(global_hour_t, num_nodes)
            tidal_harmonics_tensor = torch.tensor(tidal_harmonics, dtype=torch.float32).to(device)

            # Static features
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
                pred = model(current_cwl, current_prev, dxdt, tidal_harmonics_tensor, static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            # Update temporal state
            current_prev = current_cwl
            current_cwl = pred

    return np.array(predictions), np.array(ground_truth)


def main():
    parser = argparse.ArgumentParser(description='Temporal Memory GNN Rollout')
    parser.add_argument('--date', type=str, default='20251128', help='Date (YYYYMMDD)')
    parser.add_argument('--hours', type=int, default=48, help='Rollout hours')
    parser.add_argument('--obs', action='store_true', help='Fetch and plot CO-OPS observations')
    parser.add_argument('--save-ts', action='store_true', help='Save timeseries to text files')
    parser.add_argument('--ts-dir', type=str, default=None, help='Directory for timeseries files')
    args = parser.parse_args()

    date_str = args.date
    num_steps = args.hours  # 1-hour timesteps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Please train the temporal memory model first:")
        print("  python train_25k_temporal_memory.py")
        sys.exit(1)

    # Load data
    print(f"\nLoading data for {date_str}...")
    model, lon, lat, depth, elevation, forcing, edge_index, edge_attr, static_base, x_cart, y_cart = \
        load_model_and_data(date_str, device)

    # Find station indices
    station_indices = {}
    for name, info in STATIONS.items():
        idx = find_nearest_node(info['lon'], info['lat'], lon, lat)
        dist_km = np.sqrt((lon[idx] - info['lon'])**2 + (lat[idx] - info['lat'])**2) * 111
        station_indices[name] = idx
        print(f"  {name}: node {idx} ({dist_km:.1f} km from actual)")

    # Run rollout
    print(f"\nRunning {args.hours}h rollout ({num_steps} steps)...")
    predictions, ground_truth = run_rollout(
        model, elevation, forcing, depth, static_base, x_cart, y_cart,
        edge_index, edge_attr, device, num_steps, date_str
    )

    # Fetch observations if requested
    obs_data = {}
    if args.obs:
        print("\nFetching CO-OPS observations...")
        start_dt = datetime.strptime(date_str, '%Y%m%d')
        end_dt = start_dt + timedelta(hours=args.hours)

        for name, info in STATIONS.items():
            times, values = fetch_coops_observations(info['coops_id'], start_dt, end_dt)
            if times:
                obs_data[name] = {'times': times, 'values': values}
                print(f"  {name}: {len(values)} observations")

    # Setup timeseries output
    ts_output_dir = Path(args.ts_dir) if args.ts_dir else TS_DIR / date_str
    if args.save_ts:
        ts_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    start_dt = datetime.strptime(date_str, '%Y%m%d')
    times = np.arange(len(predictions)) * DT_HOURS  # hours

    for i, (name, idx) in enumerate(station_indices.items()):
        if i >= 5:
            break

        ax = axes[i]

        stofs_wl = ground_truth[:, idx]
        gnn_wl = predictions[:, idx]

        # Calculate RMSE and correlation
        valid = ~(np.isnan(stofs_wl) | np.isnan(gnn_wl))
        if valid.sum() > 0:
            rmse = np.sqrt(np.mean((stofs_wl[valid] - gnn_wl[valid])**2))
            if np.std(stofs_wl[valid]) > 0 and np.std(gnn_wl[valid]) > 0:
                corr = np.corrcoef(stofs_wl[valid], gnn_wl[valid])[0, 1]
            else:
                corr = np.nan
        else:
            rmse = np.nan
            corr = np.nan

        ax.plot(times, stofs_wl, 'k-', label='STOFS Ground Truth', linewidth=1.5)
        ax.plot(times, gnn_wl, 'b--', label='GNN Prediction', linewidth=1.5)

        # Plot observations if available
        if name in obs_data:
            obs_times = [(t - start_dt).total_seconds() / 3600 for t in obs_data[name]['times']]
            ax.plot(obs_times, obs_data[name]['values'], 'g.', alpha=0.5, markersize=3, label='CO-OPS Obs')

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Water Level (m MSL)')
        ax.set_title(f'{name}\nRMSE vs STOFS: {rmse:.3f}m, R: {corr:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Save timeseries
        if args.save_ts:
            ts_file = ts_output_dir / f'{name}_temporal_memory_rollout.txt'

            # Build observation lookup dict if available
            obs_lookup = {}
            if name in obs_data:
                for obs_t, obs_v in zip(obs_data[name]['times'], obs_data[name]['values']):
                    obs_lookup[obs_t.strftime("%Y-%m-%d %H:%M")] = obs_v

            with open(ts_file, 'w') as f:
                f.write(f'# Station: {name}\n')
                f.write(f'# CO-OPS ID: {STATIONS[name]["coops_id"]}\n')
                f.write(f'# Lat: {STATIONS[name]["lat"]}, Lon: {STATIONS[name]["lon"]}\n')
                f.write(f'# Model: best_temporal_memory_model.pt\n')
                f.write(f'# Start: {start_dt.strftime("%Y-%m-%d %H:%M:%S")} UTC\n')
                f.write(f'# Columns: datetime, hours_from_start, stofs_wl(m), gnn_wl(m), obs_wl(m)\n')
                f.write(f'# Note: obs_wl is NaN if no observation available at that time\n')
                f.write('#\n')

                for t_idx in range(len(predictions)):
                    dt = start_dt + timedelta(hours=t_idx * DT_HOURS)
                    dt_key = dt.strftime("%Y-%m-%d %H:%M")
                    obs_val = obs_lookup.get(dt_key, np.nan)
                    f.write(f'{dt.strftime("%Y-%m-%d %H:%M:%S")}  {t_idx*DT_HOURS:6.1f}  {stofs_wl[t_idx]:8.4f}  {gnn_wl[t_idx]:8.4f}  {obs_val:8.4f}\n')
            print(f"  Saved: {ts_file}")

    # Remove unused subplot
    axes[5].axis('off')

    plt.suptitle(f'Temporal Memory GNN - {args.hours}h Rollout ({date_str})\nModel: best_temporal_memory_model.pt',
                 fontsize=12)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'rollout_temporal_memory_{date_str}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
