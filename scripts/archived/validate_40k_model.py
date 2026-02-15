#!/usr/bin/env python3
"""
Validation Script for 40k Mid-Atlantic STOFS Surrogate Model

Performs:
1. Autoregressive rollout on validation dates
2. RMSE by lead time (t+1h, t+6h, t+12h, t+24h, t+48h)
3. Spatial error maps
4. Station comparison with NOAA tide gauge observations
5. Scatter plots (pred vs truth)

Usage:
    python scripts/validate_40k_model.py
    python scripts/validate_40k_model.py --date 20230204
    python scripts/validate_40k_model.py --hours 96 --save-timeseries
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import nullcontext
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Paths - update these for your environment
DATA_DIR = Path(os.environ.get('STOFS_DATA_DIR', '/mnt/f/STOFS_TRAINING_DATA/processed'))
MODEL_PATH = Path(os.environ.get('MODEL_PATH', '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints/best_model_40k.pt'))
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/validation_40k')

# Normalization constants (must match training)
ETA_SCALE = 2.0
WIND_SCALE = 15.0
DT_HOURS = 1.0

# Tidal harmonic periods (hours) - must match training
M2_PERIOD = 12.42  # Principal lunar semi-diurnal
S2_PERIOD = 12.00  # Principal solar semi-diurnal

# Reference epoch for tidal harmonics - must match training
# The 40k model was trained with 2023 data, using 2023-01-01 as epoch
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Mid-Atlantic + New England stations (40k domain: 37-45N, 77-66W)
STATIONS = {
    'The_Battery_NYC': {'lat': 40.700, 'lon': -74.014, 'coops_id': '8518750'},
    'Sandy_Hook_NJ': {'lat': 40.467, 'lon': -74.009, 'coops_id': '8531680'},
    'Atlantic_City_NJ': {'lat': 39.355, 'lon': -74.418, 'coops_id': '8534720'},
    'Lewes_DE': {'lat': 38.783, 'lon': -75.119, 'coops_id': '8557380'},
    'Boston_MA': {'lat': 42.355, 'lon': -71.052, 'coops_id': '8443970'},
    'Providence_RI': {'lat': 41.807, 'lon': -71.401, 'coops_id': '8454000'},
    'New_London_CT': {'lat': 41.361, 'lon': -72.090, 'coops_id': '8461490'},
    'Portland_ME': {'lat': 43.657, 'lon': -70.246, 'coops_id': '8418150'},
    'Norfolk_VA': {'lat': 36.946, 'lon': -76.330, 'coops_id': '8638610'},
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
    def __init__(
        self,
        state_dim: int = 1,
        temporal_dim: int = 6,
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


# ============================================================
# Helper Functions
# ============================================================

def compute_tidal_harmonics(global_hour: float) -> np.ndarray:
    phase_m2 = 2.0 * np.pi * global_hour / M2_PERIOD
    phase_s2 = 2.0 * np.pi * global_hour / S2_PERIOD
    return np.array([np.sin(phase_m2), np.cos(phase_m2),
                     np.sin(phase_s2), np.cos(phase_s2)], dtype=np.float32)


def compute_static_features(lon, lat, depth):
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1

    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    return np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32), x_cart, y_cart


def compute_edge_features(edge_index, x_cart, y_cart):
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    return np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)


def find_nearest_node(lon, lat, target_lon, target_lat):
    """Find index of nearest mesh node to target location."""
    dist = np.sqrt((lon - target_lon)**2 + (lat - target_lat)**2)
    return np.argmin(dist)


def fetch_noaa_observations(station_id, start_date, end_date):
    """Fetch water level observations from NOAA CO-OPS API.

    Args:
        station_id: NOAA CO-OPS station ID
        start_date: Start datetime
        end_date: End datetime

    Returns:
        times: List of datetime objects
        values: numpy array of water level values (meters, MSL)
    """
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
        'application': 'stofs_validation'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                times = []
                values = []
                for entry in data['data']:
                    times.append(datetime.strptime(entry['t'], '%Y-%m-%d %H:%M'))
                    val = entry.get('v')
                    values.append(float(val) if val and val != '' else np.nan)
                # Return times as list (matplotlib handles this better than np.array of datetime)
                return times, np.array(values)
    except Exception as e:
        logger.warning(f"Failed to fetch NOAA data for station {station_id}: {e}")

    return None, None


# ============================================================
# Validation Functions
# ============================================================

def run_rollout(model, mesh_data, data, device, num_hours=48, date_str='20230101'):
    """Run autoregressive rollout prediction."""
    model.eval()

    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)

    static_base, x_cart, y_cart = compute_static_features(lon, lat, depth)
    edge_attr = torch.tensor(compute_edge_features(mesh_data['edge_index'], x_cart, y_cart),
                             dtype=torch.float32).to(device)

    elevation = data['elevation']
    forcing = {'u10': data['u10'], 'v10': data['v10'], 'pressure': data['pressure']}

    # Compute global hours offset
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_offset = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    predictions = []
    ground_truth = []

    # Initialize with first two timesteps
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)

    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)
    current_cwl = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(1).to(device)

    num_nodes = len(lon)

    with torch.no_grad():
        for t in range(1, min(num_hours + 1, len(elevation) - 1)):
            # Compute temporal features
            dxdt = (current_cwl - current_prev) / DT_HOURS

            # Tidal harmonics
            global_hour_t = global_hours_offset + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour_t)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).to(device)

            # Update static features with water level
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

            # Predict
            with autocast('cuda') if device.type == 'cuda' else nullcontext():
                pred = model(current_cwl, current_prev, dxdt, tidal_tensor,
                           static_tensor, forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            # Update state
            current_prev = current_cwl
            current_cwl = pred

    return np.array(predictions), np.array(ground_truth)


def compute_rmse_by_lead_time(predictions, ground_truth, lead_times=[1, 6, 12, 24, 48]):
    """Compute RMSE at specific lead times."""
    results = {}
    for lead in lead_times:
        if lead - 1 < len(predictions):
            rmse = np.sqrt(np.mean((predictions[lead-1] - ground_truth[lead-1])**2))
            results[f't+{lead}h'] = rmse
    return results


def plot_spatial_error(predictions, ground_truth, lon, lat, lead_hours, output_path):
    """Plot spatial error map at given lead time."""
    idx = lead_hours - 1
    if idx >= len(predictions):
        return

    error = predictions[idx] - ground_truth[idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Ground truth
    sc1 = axes[0].scatter(lon, lat, c=ground_truth[idx], s=1, cmap='coolwarm',
                          vmin=-0.5, vmax=0.5)
    axes[0].set_title(f'Ground Truth (t+{lead_hours}h)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(sc1, ax=axes[0], label='CWL (m)')

    # Prediction
    sc2 = axes[1].scatter(lon, lat, c=predictions[idx], s=1, cmap='coolwarm',
                          vmin=-0.5, vmax=0.5)
    axes[1].set_title(f'Prediction (t+{lead_hours}h)')
    axes[1].set_xlabel('Longitude')
    plt.colorbar(sc2, ax=axes[1], label='CWL (m)')

    # Error
    sc3 = axes[2].scatter(lon, lat, c=error, s=1, cmap='RdBu_r',
                          vmin=-0.1, vmax=0.1)
    rmse = np.sqrt(np.mean(error**2))
    axes[2].set_title(f'Error (t+{lead_hours}h)\nRMSE: {rmse:.4f} m')
    axes[2].set_xlabel('Longitude')
    plt.colorbar(sc3, ax=axes[2], label='Error (m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter(predictions, ground_truth, lead_hours, output_path):
    """Plot scatter plot of predictions vs ground truth."""
    idx = lead_hours - 1
    if idx >= len(predictions):
        return

    pred_flat = predictions[idx].flatten()
    truth_flat = ground_truth[idx].flatten()

    # Subsample for plotting
    if len(pred_flat) > 5000:
        idx_sample = np.random.choice(len(pred_flat), 5000, replace=False)
        pred_flat = pred_flat[idx_sample]
        truth_flat = truth_flat[idx_sample]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(truth_flat, pred_flat, s=1, alpha=0.5)

    # Perfect prediction line
    lims = [min(truth_flat.min(), pred_flat.min()), max(truth_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, 'r--', label='Perfect prediction')

    # Statistics
    rmse = np.sqrt(np.mean((pred_flat - truth_flat)**2))
    correlation = np.corrcoef(truth_flat, pred_flat)[0, 1]
    bias = np.mean(pred_flat - truth_flat)

    ax.set_xlabel('Ground Truth CWL (m)')
    ax.set_ylabel('Predicted CWL (m)')
    ax.set_title(f't+{lead_hours}h Prediction\nRMSE: {rmse:.4f} m, R: {correlation:.4f}, Bias: {bias:.4f} m')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_station_timeseries(predictions, ground_truth, lon, lat, date_str,
                            stations, output_path, fetch_obs=True, stofs_start_hour=0):
    """Plot time series at tide gauge stations.

    Args:
        predictions: Model predictions [T, N]
        ground_truth: STOFS ground truth [T, N]
        lon, lat: Mesh coordinates
        date_str: Date string YYYYMMDD
        stations: Dictionary of station info
        output_path: Output file path
        fetch_obs: Whether to fetch NOAA observations
        stofs_start_hour: Hour offset for STOFS data (default: 7 for nowcast)
    """
    import matplotlib.dates as mdates

    num_stations = len(stations)
    fig, axes = plt.subplots(num_stations, 1, figsize=(14, 3 * num_stations))
    if num_stations == 1:
        axes = [axes]

    # Create proper datetime axis accounting for STOFS start hour offset
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    start_time = date_dt + timedelta(hours=stofs_start_hour)

    # Create datetime array for x-axis
    num_hours = len(predictions)
    times = [start_time + timedelta(hours=h) for h in range(num_hours)]

    for idx, (name, info) in enumerate(stations.items()):
        ax = axes[idx]

        # Find nearest node
        node_idx = find_nearest_node(lon, lat, info['lon'], info['lat'])

        pred_ts = predictions[:, node_idx]
        truth_ts = ground_truth[:, node_idx]

        # Plot with datetime x-axis
        ax.plot(times, truth_ts, 'b-', label='STOFS (Ground Truth)', linewidth=1.5)
        ax.plot(times, pred_ts, 'r--', label='GNN Prediction', linewidth=1.5)

        # Fetch observations if requested
        if fetch_obs and 'coops_id' in info:
            end_time = start_time + timedelta(hours=num_hours)
            obs_times, obs_values = fetch_noaa_observations(info['coops_id'], start_time, end_time)
            if obs_times is not None and len(obs_times) > 0:
                # Plot observations directly with their datetime values
                ax.plot(obs_times, obs_values, 'g.', label='NOAA Observations', markersize=3, alpha=0.7)

        rmse = np.sqrt(np.mean((pred_ts - truth_ts)**2))
        ax.set_title(f'{name.replace("_", " ")} (RMSE: {rmse:.4f} m)')
        ax.set_xlabel('Date/Time (UTC)')
        ax.set_ylabel('Water Level (m, MSL)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Validate 40k STOFS Surrogate Model')
    parser.add_argument('--date', type=str, default='20230204', help='Validation date (YYYYMMDD)')
    parser.add_argument('--hours', type=int, default=48, help='Rollout hours')
    parser.add_argument('--no-obs', action='store_true', help='Skip fetching NOAA observations')
    parser.add_argument('--model-path', type=str, default=str(MODEL_PATH), help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default=str(DATA_DIR), help='Data directory')
    parser.add_argument('--start-hour', type=int, default=4,
                        help='Hour offset for STOFS data start time (default: 4 for 40k preprocessed data)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("40k MODEL VALIDATION")
    logger.info("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load checkpoint
    logger.info(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    config = checkpoint.get('config', {})
    logger.info(f"Model config: {config}")

    # Create model
    model = TemporalMemoryGNN(
        state_dim=config.get('state_dim', 1),
        temporal_dim=config.get('temporal_features', 6),
        static_feature_dim=config.get('static_features', 4),
        forcing_feature_dim=config.get('forcing_features', 3),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load mesh
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / 'mesh.npz'
    logger.info(f"\nLoading mesh from: {mesh_path}")
    mesh_data = dict(np.load(mesh_path))
    logger.info(f"Mesh: {len(mesh_data['lon']):,} nodes, {mesh_data['edge_index'].shape[1]:,} edges")

    # Load validation data
    data_path = data_dir / f'processed_{args.date}.npz'
    logger.info(f"\nLoading data from: {data_path}")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    data = dict(np.load(data_path))
    logger.info(f"Data shape: {data['elevation'].shape}")

    # Run rollout
    logger.info(f"\nRunning {args.hours}-hour rollout...")
    predictions, ground_truth = run_rollout(model, mesh_data, data, device,
                                            num_hours=args.hours, date_str=args.date)

    # Compute RMSE by lead time
    logger.info("\n" + "=" * 40)
    logger.info("RMSE by Lead Time")
    logger.info("=" * 40)
    rmse_results = compute_rmse_by_lead_time(predictions, ground_truth, [1, 6, 12, 24, 48])
    for lead, rmse in rmse_results.items():
        logger.info(f"  {lead}: {rmse:.4f} m")

    # Generate plots
    logger.info("\nGenerating validation plots...")

    # Spatial error maps
    for lead in [6, 12, 24, 48]:
        if lead <= args.hours:
            plot_spatial_error(predictions, ground_truth, mesh_data['lon'], mesh_data['lat'],
                             lead, OUTPUT_DIR / f'spatial_error_t{lead}h_{args.date}.png')
            logger.info(f"  Saved spatial error map for t+{lead}h")

    # Scatter plots
    for lead in [6, 24, 48]:
        if lead <= args.hours:
            plot_scatter(predictions, ground_truth, lead,
                        OUTPUT_DIR / f'scatter_t{lead}h_{args.date}.png')
            logger.info(f"  Saved scatter plot for t+{lead}h")

    # Station time series
    plot_station_timeseries(predictions, ground_truth, mesh_data['lon'], mesh_data['lat'],
                           args.date, STATIONS, OUTPUT_DIR / f'stations_{args.date}.png',
                           fetch_obs=not args.no_obs, stofs_start_hour=args.start_hour)
    logger.info("  Saved station time series")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE vs lead time
    leads = [int(k.split('+')[1].replace('h', '')) for k in rmse_results.keys()]
    rmses = list(rmse_results.values())
    axes[0].plot(leads, rmses, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Lead Time (hours)')
    axes[0].set_ylabel('RMSE (m)')
    axes[0].set_title(f'RMSE vs Lead Time\n40k Model, Date: {args.date}')
    axes[0].grid(True, alpha=0.3)

    # Domain-averaged time series
    pred_mean = predictions.mean(axis=1)
    truth_mean = ground_truth.mean(axis=1)
    hours = np.arange(1, len(pred_mean) + 1)
    axes[1].plot(hours, truth_mean, 'b-', label='Ground Truth', linewidth=1.5)
    axes[1].plot(hours, pred_mean, 'r--', label='Prediction', linewidth=1.5)
    axes[1].set_xlabel('Forecast Hour')
    axes[1].set_ylabel('Domain-Averaged CWL (m)')
    axes[1].set_title('Domain-Averaged Water Level')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'validation_summary_{args.date}.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"\nValidation complete! Results saved to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
