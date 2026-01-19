#!/usr/bin/env python3
"""
Multi-date Spatial Error Visualization for 25K V2 Model

Generates spatial error maps across multiple dates and forecast hours.
Shows error patterns for hours 6, 12, 18, 24, 30, 36, 42, 48.

Usage:
    python scripts/visualize_spatial_multidate.py --checkpoint checkpoint_epoch_60.pt
    python scripts/visualize_spatial_multidate.py --checkpoint checkpoint_epoch_60.pt --dates 20250101 20250115
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
CHECKPOINT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures_25k_v2')

# Model config
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

# Default validation dates (spread across seasons)
DEFAULT_DATES = ['20230115', '20230415', '20230715', '20231015', '20250101']


# ============================================================
# Model Architecture
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.gradient_scale = nn.Parameter(torch.ones(1))

    def forward(self, h, edge_index, edge_attr):
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]
        h_src, h_dst = h[:, row, :], h[:, col, :]
        h_gradient = h_dst - h_src
        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1).reshape(B * E, -1)
        edge_msg = self.edge_mlp(edge_input).reshape(B, E, F)
        edge_msg = edge_msg * (1.0 + torch.tanh(self.gradient_scale * h_gradient))
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        aggr = torch.zeros(B, N, F, device=h.device).scatter_add_(
            1, row.unsqueeze(0).unsqueeze(-1).expand(B, E, F), edge_msg
        )
        return h + self.node_mlp(torch.cat([h, aggr], dim=-1).reshape(B * N, -1)).reshape(B, N, F), edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.gnn_layers = nn.ModuleList([BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim)
        )

    def forward(self, x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr):
        B, N, _ = x.shape
        node_features = torch.cat([x, x_prev, dxdt, tidal, static, forcing], dim=-1).reshape(B * N, -1)
        h = self.node_encoder(node_features).reshape(B, N, self.hidden_dim)
        e = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)
        return x + self.decoder(h.reshape(B * N, self.hidden_dim)).reshape(B, N, -1)


# ============================================================
# Helper Functions
# ============================================================

def compute_tidal_harmonics(global_hour):
    harmonics = []
    for period in TIDAL_PERIODS.values():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


def load_model(checkpoint_path, device):
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM, temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES, forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, ckpt


def run_rollout(model, mesh_data, val_data, device, max_hours, date_str):
    """Run autoregressive rollout."""
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
    depth_log = np.log10(np.maximum(np.abs(depth), 0.1))
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
        dtype=torch.float32
    ).to(device)

    # Initialize
    elevation = val_data['elevation']
    date_dt = datetime.strptime(date_str, '%Y%m%d')
    global_hours_base = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0

    cwl_prev = np.nan_to_num(elevation[0], nan=0.0)
    cwl_t = np.nan_to_num(elevation[1], nan=0.0)
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = [cwl_prev, cwl_t]
    ground_truth = [elevation[0], elevation[1]]

    with torch.no_grad():
        for t in range(1, max_hours + 1):
            dxdt = (current - current_prev) / DT_HOURS
            global_hour = global_hours_base + t * DT_HOURS
            tidal = compute_tidal_harmonics(global_hour)
            tidal_tensor = torch.tensor(np.tile(tidal, (num_nodes, 1)), dtype=torch.float32).unsqueeze(0).to(device)

            cwl_np = current.squeeze().cpu().numpy() * ETA_SCALE
            water_level = depth + cwl_np
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            static = np.concatenate([static_base, wl_norm[:, np.newaxis]], axis=1)
            static_tensor = torch.tensor(static, dtype=torch.float32).unsqueeze(0).to(device)

            forcing_arr = np.stack([
                val_data['u10'][t], val_data['v10'][t], val_data['wind_speed'][t],
                val_data['wind_speed_sq'][t], val_data['wind_dir'][t], val_data['pressure'][t],
                val_data['dP_dx'][t], val_data['dP_dy'][t]
            ], axis=1).astype(np.float32)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1], nan=0.0))

            current_prev = current
            current = pred

    coords = np.stack([lon, lat], axis=1)
    return np.array(predictions), np.array(ground_truth), coords


def plot_spatial_error_multidate(all_results, hours, output_path, epoch_num):
    """Generate spatial error maps across multiple dates and hours."""
    num_dates = len(all_results)
    num_hours = len(hours)

    fig, axes = plt.subplots(num_dates, num_hours, figsize=(3 * num_hours, 3 * num_dates))

    if num_dates == 1:
        axes = axes.reshape(1, -1)

    # Compute global error limits across all dates/hours
    all_errors = []
    for date_str, (predictions, ground_truth, coords) in all_results.items():
        for h in hours:
            timestep = h + 1
            if timestep < len(predictions):
                error = predictions[timestep] - ground_truth[timestep]
                valid = ~np.isnan(error)
                all_errors.append(error[valid])

    if all_errors:
        all_errors = np.concatenate(all_errors)
        vmax_err = np.percentile(np.abs(all_errors), 95)
    else:
        vmax_err = 0.5

    # Track RMSE for summary
    rmse_table = {}

    for row, (date_str, (predictions, ground_truth, coords)) in enumerate(all_results.items()):
        lon = coords[:, 0]
        lat = coords[:, 1]
        rmse_table[date_str] = {}

        for col, h in enumerate(hours):
            ax = axes[row, col]
            timestep = h + 1

            if timestep >= len(predictions):
                ax.axis('off')
                continue

            error = predictions[timestep] - ground_truth[timestep]
            valid = ~np.isnan(error)
            rmse = np.sqrt(np.mean(error[valid]**2))
            rmse_table[date_str][h] = rmse

            sc = ax.scatter(lon, lat, c=error, s=0.3, cmap='RdBu_r', vmin=-vmax_err, vmax=vmax_err)

            # Add RMSE text
            ax.text(0.02, 0.98, f'RMSE: {rmse*100:.1f}cm',
                   transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Row labels (dates)
            if col == 0:
                ax.set_ylabel(f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}', fontsize=10)

            # Column labels (hours)
            if row == 0:
                ax.set_title(f't+{h}h', fontsize=11, fontweight='bold')

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Error (m)', fontsize=10)

    plt.suptitle(f'25K V2 Model (Epoch {epoch_num}) - Spatial Error Maps\nMulti-date Validation',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return rmse_table


def plot_rmse_by_lead_time(rmse_table, output_path, epoch_num):
    """Plot RMSE vs lead time for all dates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(rmse_table)))

    all_hours = sorted(list(list(rmse_table.values())[0].keys()))

    # Plot individual dates
    for i, (date_str, rmse_by_hour) in enumerate(rmse_table.items()):
        hours = sorted(rmse_by_hour.keys())
        rmses = [rmse_by_hour[h] * 100 for h in hours]  # Convert to cm
        label = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
        ax.plot(hours, rmses, 'o-', color=colors[i], label=label, alpha=0.7, markersize=6)

    # Plot mean across dates
    mean_rmse = []
    for h in all_hours:
        rmses = [rmse_table[d][h] * 100 for d in rmse_table if h in rmse_table[d]]
        mean_rmse.append(np.mean(rmses))

    ax.plot(all_hours, mean_rmse, 's-', color='black', linewidth=2.5,
           markersize=8, label='Mean', zorder=10)

    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('RMSE (cm)', fontsize=12)
    ax.set_title(f'25K V2 Model (Epoch {epoch_num}) - RMSE vs Lead Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_hours)

    # Add text annotations for mean
    for h, rmse in zip(all_hours, mean_rmse):
        ax.annotate(f'{rmse:.1f}', (h, rmse), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Generate multi-date spatial error visualizations')
    parser.add_argument('--dates', type=str, nargs='+', default=None,
                        help='Validation dates (YYYYMMDD). If not specified, uses default dates.')
    parser.add_argument('--hours', type=int, nargs='+', default=[6, 12, 18, 24, 30, 36, 42, 48],
                        help='Forecast hours to visualize')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_60.pt',
                        help='Checkpoint filename')

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    print(f"Loading model: {checkpoint_path}")
    model, ckpt = load_model(checkpoint_path, device)
    epoch_num = ckpt.get('epoch', 'unknown')
    print(f"  Epoch: {epoch_num}")

    # Load mesh
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))
    print(f"Mesh: {len(mesh_data['lon']):,} nodes")

    # Determine validation dates
    if args.dates:
        dates = args.dates
    else:
        # Use default dates, but filter to those that exist
        dates = []
        for d in DEFAULT_DATES:
            if (DATA_DIR / f'processed_{d}.npz').exists():
                dates.append(d)

        # Also add some from 2025 if available
        for d in ['20250101', '20250115']:
            if (DATA_DIR / f'processed_{d}.npz').exists() and d not in dates:
                dates.append(d)

    print(f"\nValidation dates: {dates}")
    print(f"Forecast hours: {args.hours}")

    # Run rollouts for all dates
    max_hour = max(args.hours)
    all_results = {}

    for date_str in dates:
        data_path = DATA_DIR / f'processed_{date_str}.npz'
        if not data_path.exists():
            print(f"  Skipping {date_str}: data file not found")
            continue

        print(f"\nProcessing {date_str}...")
        val_data = dict(np.load(data_path))

        # Check if data has enough timesteps
        if len(val_data['elevation']) < max_hour + 2:
            print(f"  Skipping {date_str}: only {len(val_data['elevation'])} timesteps")
            continue

        predictions, ground_truth, coords = run_rollout(
            model, mesh_data, val_data, device, max_hour, date_str
        )
        all_results[date_str] = (predictions, ground_truth, coords)
        print(f"  Rollout complete: {len(predictions)} timesteps")

    if not all_results:
        print("Error: No valid dates to process")
        return

    # Generate multi-date spatial error plot
    dates_str = '_'.join(list(all_results.keys())[:3])  # Use first 3 for filename
    if len(all_results) > 3:
        dates_str += f'_plus{len(all_results)-3}'

    output_path = OUTPUT_DIR / f'spatial_error_multidate_{dates_str}_ep{epoch_num}.png'
    rmse_table = plot_spatial_error_multidate(all_results, args.hours, output_path, epoch_num)

    # Generate RMSE vs lead time plot
    rmse_path = OUTPUT_DIR / f'rmse_vs_leadtime_multidate_{dates_str}_ep{epoch_num}.png'
    plot_rmse_by_lead_time(rmse_table, rmse_path, epoch_num)

    # Print summary table
    print("\n" + "="*80)
    print(f"RMSE Summary (cm) - Epoch {epoch_num}")
    print("="*80)

    header = "Date        | " + " | ".join([f't+{h}h' for h in args.hours])
    print(header)
    print("-" * len(header))

    for date_str, rmse_by_hour in rmse_table.items():
        row = f"{date_str}  | "
        row += " | ".join([f'{rmse_by_hour.get(h, 0)*100:5.1f}' for h in args.hours])
        print(row)

    # Mean row
    print("-" * len(header))
    mean_row = "Mean        | "
    mean_rmses = []
    for h in args.hours:
        rmses = [rmse_table[d][h] * 100 for d in rmse_table if h in rmse_table[d]]
        mean_rmses.append(np.mean(rmses))
    mean_row += " | ".join([f'{r:5.1f}' for r in mean_rmses])
    print(mean_row)
    print("="*80)

    print("\nDone!")


if __name__ == '__main__':
    main()
