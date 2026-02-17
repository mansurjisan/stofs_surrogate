#!/usr/bin/env python3
"""
Ensemble Inference for STOFS-GNN V2 (25K mesh, 8 forcing features)

Generates ensemble forecasts by perturbing meteorological forcing
(wind speed, direction, pressure) and initial conditions.

Usage:
    python scripts/ensemble_v2.py --date 20250120 --checkpoint checkpoint_epoch_95.pt \
        --n_members 20 --forecast_hours 48
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/e/STOFS_TRAINING_DATA/processed_25k_v2')
CHECKPOINT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/ensemble_v2')

HIDDEN_DIM = 128
NUM_LAYERS = 6
STATE_DIM = 1
TEMPORAL_FEATURES = 12
STATIC_NODE_FEATURES = 4
FORCING_FEATURES = 8

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}

STATIONS = {
    'The_Battery': (-74.003, 40.704),
    'Sandy_Hook': (-74.025, 40.474),
    'Atlantic_City': (-74.406, 39.349),
    'Philadelphia_PA': (-75.225, 39.856),
    'Cape_May': (-74.972, 38.974),
    'Lewes_DE': (-75.131, 38.797),
    'Baltimore': (-76.578, 39.267),
    'Annapolis': (-76.480, 38.983),
}

# Perturbation config
PERTURBATION = {
    'wind_speed_std': 0.05,       # 5% multiplicative std
    'wind_direction_std': 5.0,    # degrees
    'pressure_std': 100.0,        # Pa (1 hPa)
    'initial_cwl_std': 0.02,      # meters (2 cm)
    'spatial_smooth_sigma': 3.0,  # temporal smoothing for coherent perturbations
}


# ============================================================
# Model Architecture (same as training V2)
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.gradient_scale = nn.Parameter(torch.ones(1))

    def forward(self, h, edge_index, edge_attr):
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]
        h_src, h_dst = h[:, row, :], h[:, col, :]
        h_gradient = h_dst - h_src
        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input.reshape(B * E, -1)).reshape(B, E, F)
        edge_msg = edge_msg * (1.0 + torch.tanh(self.gradient_scale * h_gradient))
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        aggr.scatter_add_(1, row.unsqueeze(0).unsqueeze(-1).expand(B, E, F), edge_msg)
        node_out = self.node_mlp(torch.cat([h, aggr], dim=-1).reshape(B * N, -1)).reshape(B, N, F)
        return h + node_out, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = HIDDEN_DIM
        node_input_dim = 3 * STATE_DIM + TEMPORAL_FEATURES + STATIC_NODE_FEATURES + FORCING_FEATURES
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM))
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM))
        self.gnn_layers = nn.ModuleList([BatchedSWEGraphBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)])
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, STATE_DIM))

    def forward(self, x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, x_prev, dxdt, tidal, static, forcing], dim=-1)
        B, N, F_in = node_features.shape
        h = self.node_encoder(node_features.reshape(B * N, F_in)).reshape(B, N, self.hidden_dim)
        e = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)
        delta = self.decoder(h.reshape(B * N, self.hidden_dim)).reshape(B, N, -1)
        return x + delta


# ============================================================
# Helpers
# ============================================================

def compute_tidal_harmonics(global_hour):
    harmonics = []
    for period in TIDAL_PERIODS.values():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


def get_forcing(forcing_dict, t, num_nodes):
    return np.stack([
        forcing_dict['u10'][t], forcing_dict['v10'][t],
        forcing_dict['wind_speed'][t], forcing_dict['wind_speed_sq'][t],
        forcing_dict['wind_dir'][t], forcing_dict['pressure'][t],
        forcing_dict['dP_dx'][t], forcing_dict['dP_dy'][t],
    ], axis=1).astype(np.float32)


def perturb_forcing(forcing_dict, rng, num_timesteps):
    """Create perturbed copy of forcing with temporally correlated noise."""
    p = PERTURBATION
    perturbed = {}

    # Wind speed scaling (temporally correlated)
    wind_scale = 1.0 + rng.normal(0, p['wind_speed_std'], num_timesteps)
    wind_scale = gaussian_filter1d(wind_scale, p['spatial_smooth_sigma'])
    wind_scale = np.clip(wind_scale, 0.3, 2.0)

    # Wind direction rotation
    wind_rot_deg = rng.normal(0, p['wind_direction_std'], num_timesteps)
    wind_rot_deg = gaussian_filter1d(wind_rot_deg, p['spatial_smooth_sigma'])
    wind_rot_rad = np.radians(wind_rot_deg)

    # Pressure offset
    pressure_offset = rng.normal(0, p['pressure_std'], num_timesteps)
    pressure_offset = gaussian_filter1d(pressure_offset, p['spatial_smooth_sigma'])

    for t in range(num_timesteps):
        u10 = forcing_dict['u10'][t].copy()
        v10 = forcing_dict['v10'][t].copy()

        # Rotate wind
        cos_r, sin_r = np.cos(wind_rot_rad[t]), np.sin(wind_rot_rad[t])
        u_rot = u10 * cos_r - v10 * sin_r
        v_rot = u10 * sin_r + v10 * cos_r

        # Scale wind
        u_rot *= wind_scale[t]
        v_rot *= wind_scale[t]

        for key in forcing_dict:
            if key not in perturbed:
                perturbed[key] = forcing_dict[key].copy()

        perturbed['u10'][t] = u_rot
        perturbed['v10'][t] = v_rot
        speed = np.sqrt(u_rot**2 + v_rot**2)
        perturbed['wind_speed'][t] = speed
        perturbed['wind_speed_sq'][t] = speed**2
        perturbed['wind_dir'][t] = np.arctan2(v_rot, u_rot)
        perturbed['pressure'][t] = forcing_dict['pressure'][t] + pressure_offset[t]

    return perturbed


def load_model(checkpoint_path, device):
    model = BatchedTemporalMemoryGNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def run_single_rollout(model, mesh_data, elevation, forcing_dict, date_str, device, max_hours):
    """Run a single deterministic rollout. Returns predictions array [T, N]."""
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_base = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    lon = mesh_data['lon'].astype(np.float32)
    lat = mesh_data['lat'].astype(np.float32)
    depth = mesh_data['depth'].astype(np.float32)
    edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long).to(device)
    num_nodes = len(lon)

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

    # Edge features
    src, dst = mesh_data['edge_index'][0], mesh_data['edge_index'][1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = torch.tensor(
        np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
        dtype=torch.float32).to(device)

    # Initialize
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = []
    num_steps = min(max_hours, elevation.shape[0] - 2)

    with torch.no_grad():
        for t in range(1, num_steps + 1):
            dxdt = (current - current_prev) / DT_HOURS
            global_hour = global_hours_base + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)

            forcing_arr = get_forcing(forcing_dict, t, num_nodes)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            current_prev = current
            current = pred

    return np.array(predictions)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Ensemble inference for STOFS-GNN V2')
    parser.add_argument('--date', type=str, default='20250120')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_95.pt')
    parser.add_argument('--n_members', type=int, default=20)
    parser.add_argument('--forecast_hours', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    rng = np.random.default_rng(args.seed)

    # Load mesh and data
    mesh_data = dict(np.load(DATA_DIR / 'mesh.npz', allow_pickle=True))
    lon, lat = mesh_data['lon'], mesh_data['lat']
    num_nodes = len(lon)
    logger.info(f"Mesh: {num_nodes:,} nodes")

    val_file = DATA_DIR / f'processed_{args.date}.npz'
    val_data_raw = np.load(val_file)
    elevation = val_data_raw['elevation']
    base_forcing = {k: val_data_raw[k] for k in
                    ['u10', 'v10', 'wind_speed', 'wind_speed_sq', 'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}

    # Load model
    ckpt_path = CHECKPOINT_DIR / args.checkpoint
    logger.info(f"Loading model: {ckpt_path}")
    model = load_model(ckpt_path, device)

    # Run ensemble
    n = args.n_members
    fh = args.forecast_hours
    num_timesteps = min(fh, elevation.shape[0] - 2)
    logger.info(f"\nRunning {n} ensemble members for {fh}h forecast...")

    all_predictions = np.zeros((n, num_timesteps, num_nodes), dtype=np.float32)
    t_start = time.time()

    for i in range(n):
        if i == 0:
            # Member 0 = control (unperturbed)
            forcing_i = base_forcing
        else:
            forcing_i = perturb_forcing(base_forcing, rng, elevation.shape[0])

        # Perturb initial conditions for non-control members
        elev_i = elevation.copy()
        if i > 0:
            ic_noise = rng.normal(0, PERTURBATION['initial_cwl_std'], num_nodes).astype(np.float32)
            elev_i[0] += ic_noise
            elev_i[1] += ic_noise

        preds = run_single_rollout(model, mesh_data, elev_i, forcing_i, args.date, device, fh)
        all_predictions[i] = preds

        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        remaining = (n - i - 1) / rate if rate > 0 else 0
        logger.info(f"  Member {i+1}/{n} done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    elapsed_total = time.time() - t_start
    logger.info(f"\nEnsemble complete: {elapsed_total:.1f}s total, {elapsed_total/n:.1f}s per member")

    # Compute statistics
    ens_mean = all_predictions.mean(axis=0)
    ens_std = all_predictions.std(axis=0)
    ground_truth = np.array([np.nan_to_num(elevation[t+2].astype(np.float32), nan=0.0)
                             for t in range(num_timesteps)])

    # Save results
    run_dir = OUTPUT_DIR / f'run_{args.date}_{args.checkpoint.replace(".pt","")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / 'ensemble_results.npz',
                        predictions=all_predictions, mean=ens_mean, std=ens_std,
                        ground_truth=ground_truth, lon=lon, lat=lat)

    # ========== PLOTTING ==========

    # Find station indices
    station_indices = {}
    for name, (slon, slat) in STATIONS.items():
        dist = np.sqrt((lon - slon)**2 + (lat - slat)**2)
        station_indices[name] = np.argmin(dist)

    hours = np.arange(1, num_timesteps + 1)

    # --- Figure 1: Ensemble Dashboard ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Top row: spatial maps at t+6h
    t_map = min(5, num_timesteps - 1)  # t+6h

    # Mean prediction
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(lon, lat, c=ens_mean[t_map], cmap='RdBu_r', s=0.5, vmin=-1, vmax=1)
    ax1.set_title(f'Ensemble Mean at t+{t_map+1}h')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(sc, ax=ax1, label='Water Level (m)')

    # Uncertainty (std)
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(lon, lat, c=ens_std[t_map], cmap='YlOrRd', s=0.5, vmin=0)
    ax2.set_title(f'Ensemble Spread (Std) at t+{t_map+1}h')
    ax2.set_xlabel('Longitude')
    plt.colorbar(sc, ax=ax2, label='Std (m)')

    # Exceedance probability P(η > 0.5m)
    ax3 = fig.add_subplot(gs[0, 2])
    exceed_prob = (all_predictions[:, t_map, :] > 0.5).mean(axis=0)
    sc = ax3.scatter(lon, lat, c=exceed_prob, cmap='YlOrRd', s=0.5, vmin=0, vmax=1)
    ax3.set_title(f'P(η > 0.5m) at t+{t_map+1}h')
    ax3.set_xlabel('Longitude')
    plt.colorbar(sc, ax=ax3, label='Probability')

    # Middle row: station spaghetti plots (3 stations)
    key_stations = ['The_Battery', 'Baltimore', 'Atlantic_City']
    for i, name in enumerate(key_stations):
        ax = fig.add_subplot(gs[1, i])
        idx = station_indices[name]

        # Spaghetti (all members)
        for m in range(min(n, 30)):
            ax.plot(hours, all_predictions[m, :, idx], color='steelblue', alpha=0.15, linewidth=0.5)

        # Mean + CI
        mean_ts = ens_mean[:, idx]
        std_ts = ens_std[:, idx]
        ax.fill_between(hours, mean_ts - 2*std_ts, mean_ts + 2*std_ts,
                        alpha=0.2, color='blue', label='95% CI')
        ax.fill_between(hours, mean_ts - std_ts, mean_ts + std_ts,
                        alpha=0.3, color='blue', label='68% CI')
        ax.plot(hours, mean_ts, 'b-', linewidth=2, label='Ensemble Mean')
        ax.plot(hours, ground_truth[:, idx], 'g-', linewidth=2, label='STOFS Truth')

        rmse = np.sqrt(np.mean((mean_ts - ground_truth[:, idx])**2))
        ax.set_title(f'{name}\nRMSE: {rmse*100:.1f} cm')
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('Water Level (m)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Bottom row: domain-avg response, peak surge distribution, summary
    ax4 = fig.add_subplot(gs[2, 0])
    domain_mean = ens_mean.mean(axis=1)
    domain_std = ens_std.mean(axis=1)
    ax4.fill_between(hours, domain_mean - 2*domain_std, domain_mean + 2*domain_std,
                     alpha=0.2, color='blue')
    ax4.plot(hours, domain_mean, 'b-', linewidth=2)
    ax4.plot(hours, ground_truth.mean(axis=1), 'g-', linewidth=2)
    ax4.set_title('Domain-Averaged Response')
    ax4.set_xlabel('Forecast Hour')
    ax4.set_ylabel('Mean Water Level (m)')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 1])
    peak_surges = all_predictions.max(axis=(1, 2))
    ax5.hist(peak_surges, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.axvline(peak_surges.mean(), color='red', linestyle='--', label=f'Mean: {peak_surges.mean():.2f}m')
    ax5.set_title('Distribution of Peak Surge')
    ax5.set_xlabel('Maximum Water Level (m)')
    ax5.set_ylabel('Count')
    ax5.legend()

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    summary_text = (
        f"Ensemble Summary\n"
        f"{'─'*30}\n"
        f"Members: {n}\n"
        f"Forecast: {num_timesteps}h\n"
        f"Runtime: {elapsed_total:.1f}s\n"
        f"Per member: {elapsed_total/n:.1f}s\n"
        f"{'─'*30}\n"
        f"Peak ensemble mean: {ens_mean.max():.2f}m\n"
        f"Peak ensemble max: {all_predictions.max():.2f}m\n"
        f"Max uncertainty: {ens_std.max():.2f}m\n"
        f"{'─'*30}\n"
        f"Checkpoint: {args.checkpoint}\n"
        f"Date: {args.date}\n"
    )
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'Storm Surge Ensemble Forecast — {args.date}', fontsize=16, y=1.01)
    plt.savefig(run_dir / 'ensemble_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {run_dir / 'ensemble_dashboard.png'}")

    # --- Figure 2: All-station spaghetti ---
    n_stations = len(STATIONS)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (name, idx) in enumerate(station_indices.items()):
        if i >= 8:
            break
        ax = axes[i]
        for m in range(min(n, 30)):
            ax.plot(hours, all_predictions[m, :, idx], color='steelblue', alpha=0.15, linewidth=0.5)
        mean_ts = ens_mean[:, idx]
        std_ts = ens_std[:, idx]
        ax.fill_between(hours, mean_ts - 2*std_ts, mean_ts + 2*std_ts, alpha=0.15, color='blue')
        ax.plot(hours, mean_ts, 'b-', linewidth=2, label='Mean')
        ax.plot(hours, ground_truth[:, idx], 'g-', linewidth=2, label='Truth')
        rmse = np.sqrt(np.mean((mean_ts - ground_truth[:, idx])**2))
        ax.set_title(f'{name} (RMSE: {rmse*100:.1f}cm)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Hour')
        ax.set_ylabel('WL (m)')

    plt.suptitle(f'Ensemble Station Forecasts ({n} members) — {args.date}', fontsize=14)
    plt.tight_layout()
    plt.savefig(run_dir / 'ensemble_stations.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {run_dir / 'ensemble_stations.png'}")

    logger.info(f"\nAll outputs saved to: {run_dir}")


if __name__ == '__main__':
    main()
