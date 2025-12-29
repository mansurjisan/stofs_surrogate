"""
Rollout script for 25k SWE-inspired GNN model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import argparse
import requests
from datetime import datetime, timedelta

# Constants
WIND_SCALE = 30.0
ETA_SCALE = 2.0


# Model architecture (from train_25k_15day.py)
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

        return h_new, edge_attr  # edge_attr unchanged


class PhysicsInformedCWLModel(nn.Module):
    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
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

    def forward(self, x, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, static_features, forcing], dim=-1)
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_attr)  # Encode edges to hidden_dim

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        delta = self.decoder(h)
        output = x + delta

        return output


# Station definitions
STATIONS = {
    'Atlantic_City': {'lon': -74.418, 'lat': 39.355, 'id': '8534720'},
    'Sandy_Hook': {'lon': -74.009, 'lat': 40.467, 'id': '8531680'},
    'The_Battery': {'lon': -74.014, 'lat': 40.700, 'id': '8518750'},
    'Lewes_DE': {'lon': -75.119, 'lat': 38.782, 'id': '8557380'},
    'Cape_May': {'lon': -74.960, 'lat': 38.968, 'id': '8536110'},
}


def find_nearest_node(lon, lat, mesh_lon, mesh_lat):
    """Find nearest mesh node to station location."""
    dist = np.sqrt((mesh_lon - lon)**2 + (mesh_lat - lat)**2)
    return np.argmin(dist)


def fetch_coops_observations(station_id, start_date, hours):
    """
    Fetch CO-OPS water level observations from NOAA API.

    Args:
        station_id: CO-OPS station ID (e.g., '8534720')
        start_date: Start date string 'YYYYMMDD'
        hours: Number of hours of data to fetch

    Returns:
        times: array of hours from start
        values: array of water levels in meters (MSL)
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = start_dt + timedelta(hours=hours)

        base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            'begin_date': start_dt.strftime('%Y%m%d %H:%M'),
            'end_date': end_dt.strftime('%Y%m%d %H:%M'),
            'station': station_id,
            'product': 'water_level',
            'datum': 'MSL',
            'units': 'metric',
            'time_zone': 'gmt',
            'format': 'json',
            'application': 'stofs_surrogate'
        }

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data:
            print(f"  No data for station {station_id}")
            return None, None

        times = []
        values = []

        for record in data['data']:
            try:
                t = datetime.strptime(record['t'], '%Y-%m-%d %H:%M')
                v = float(record['v'])
                hours_from_start = (t - start_dt).total_seconds() / 3600.0
                times.append(hours_from_start)
                values.append(v)
            except (ValueError, KeyError):
                continue

        return np.array(times), np.array(values)

    except Exception as e:
        print(f"  Error fetching observations for {station_id}: {e}")
        return None, None


def compute_static_features(lon, lat, depth):
    """Compute normalized static features matching training code."""
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


def compute_edge_features(x_cart, y_cart, edge_index):
    """Compute normalized edge features matching training code."""
    src = edge_index[0]
    dst = edge_index[1]

    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8

    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1)
    return torch.tensor(edge_attr, dtype=torch.float32)


def rollout(model, data, mesh, device, num_steps, static_base, depth, eta_scale=ETA_SCALE):
    """Autoregressive rollout matching training approach."""
    model.eval()

    predictions = []

    # Get data
    elevation = data['elevation']  # (T, N)
    u10 = data['u10']
    v10 = data['v10']
    pressure = data['pressure']

    edge_index = torch.tensor(mesh['edge_index'], dtype=torch.long).to(device)
    edge_attr = data['edge_attr'].to(device)

    # Initial normalized cwl
    cwl_t = elevation[0].astype(np.float32)
    cwl_t = np.nan_to_num(cwl_t, nan=0.0)
    x = torch.tensor(cwl_t / eta_scale, dtype=torch.float32).unsqueeze(-1).to(device)

    print(f"Initial cwl range: [{cwl_t.min():.4f}, {cwl_t.max():.4f}]")
    print(f"Initial x (normalized) range: [{x.min().item():.4f}, {x.max().item():.4f}]")

    static_base_t = torch.tensor(static_base, dtype=torch.float32).to(device)
    depth_t = torch.tensor(depth, dtype=torch.float32).to(device)

    with torch.no_grad():
        for t in range(min(num_steps, len(elevation) - 1)):
            # Update water level feature (4th static feature)
            current_cwl = x.squeeze(-1) * eta_scale  # Unnormalize
            water_level = depth_t + current_cwl
            wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

            # Full static features [x_norm, y_norm, depth_norm, wl_norm]
            static_full = torch.cat([static_base_t, wl_norm.unsqueeze(-1)], dim=-1)

            # Get forcing for current timestep
            u10_t = u10[t].astype(np.float32) / WIND_SCALE
            v10_t = v10[t].astype(np.float32) / WIND_SCALE
            pres_t = pressure[t].astype(np.float32)  # Already normalized during preprocessing

            forcing_t = torch.tensor(
                np.stack([u10_t, v10_t, pres_t], axis=1), dtype=torch.float32
            ).to(device)

            # Model forward pass
            x_next = model(x, static_full, forcing_t, edge_index, edge_attr)

            # Store prediction (denormalized)
            pred_cwl = x_next.squeeze(-1).cpu().numpy() * eta_scale
            predictions.append(pred_cwl)

            if t == 0:
                print(f"Step 0 prediction range: [{pred_cwl.min():.4f}, {pred_cwl.max():.4f}]")
                print(f"Step 0 delta: {(x_next - x).abs().mean().item():.6f}")

            # Update state for next step
            x = x_next

    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description='25k Model Rollout')
    parser.add_argument('--checkpoint', type=str, default='best_25k_15day_model.pt',
                        help='Checkpoint filename or path')
    parser.add_argument('--date', type=str, default='20251128',
                        help='Date to run rollout on (YYYYMMDD)')
    parser.add_argument('--hours', type=int, default=48,
                        help='Number of hours to rollout')
    parser.add_argument('--output', type=str, default=None,
                        help='Output figure path')
    parser.add_argument('--obs', action='store_true',
                        help='Fetch and plot CO-OPS observations')
    parser.add_argument('--save-ts', action='store_true',
                        help='Save timeseries to text files')
    parser.add_argument('--ts-dir', type=str, default=None,
                        help='Directory for timeseries output (default: outputs/timeseries)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    if os.path.isabs(args.checkpoint) or os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = f'outputs/checkpoints/{args.checkpoint}'

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    print(f"Model config: {config}")
    print(f"Epoch: {checkpoint['epoch']}, Val loss: {checkpoint['val_loss']:.6f}")

    # Create model
    model = PhysicsInformedCWLModel(
        state_dim=1,
        static_feature_dim=config['static_features'],
        forcing_feature_dim=config['forcing_features'],
        edge_feature_dim=3,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load mesh
    mesh = np.load('data/processed_25k/mesh_25k.npz')
    lon = mesh['lon']
    lat = mesh['lat']
    depth = mesh['depth']
    edge_index = mesh['edge_index']

    print(f"\nMesh: {len(lon)} nodes, {edge_index.shape[1]} edges")

    # Load date data
    data_path = f'data/processed_25k/processed_{args.date}.npz'
    data = np.load(data_path)
    elevation = data['elevation']
    u10 = data['u10']
    v10 = data['v10']
    pressure = data['pressure']

    print(f"Elevation shape: {elevation.shape}")
    print(f"Elevation range: [{np.nanmin(elevation):.4f}, {np.nanmax(elevation):.4f}]")

    # Compute features
    static_base, x_cart, y_cart = compute_static_features(lon, lat, depth)
    edge_attr = compute_edge_features(x_cart, y_cart, edge_index)

    print(f"Static base shape: {static_base.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")

    # Pack data for rollout
    data_dict = {
        'elevation': elevation,
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
        'edge_attr': edge_attr,
    }

    # Run rollout (2 steps per hour)
    num_steps = args.hours * 2
    print(f"\nRunning {num_steps}-step ({args.hours}h) rollout...")
    predictions = rollout(model, data_dict, mesh, device, num_steps, static_base, depth)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{np.nanmin(predictions):.4f}, {np.nanmax(predictions):.4f}]")

    # Find station indices
    station_indices = {}
    for name, info in STATIONS.items():
        idx = find_nearest_node(info['lon'], info['lat'], lon, lat)
        dist = np.sqrt((lon[idx] - info['lon'])**2 + (lat[idx] - info['lat'])**2)
        station_indices[name] = idx
        print(f"Station {name}: node {idx}, dist {dist:.4f} deg")

    # Fetch CO-OPS observations (optional)
    observations = {}
    if args.obs:
        print(f"\nFetching CO-OPS observations for {args.date}...")
        for name, info in STATIONS.items():
            obs_times, obs_values = fetch_coops_observations(info['id'], args.date, args.hours + 6)
            if obs_times is not None and len(obs_times) > 0:
                observations[name] = {'times': obs_times, 'values': obs_values}
                print(f"  {name}: {len(obs_times)} observations")
            else:
                observations[name] = None

    # Ground truth
    ground_truth = elevation[1:num_steps+1]  # Shift by 1 since we predict t+1

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    time_hours = np.arange(num_steps) * 0.5

    for i, (name, idx) in enumerate(station_indices.items()):
        if i >= 5:
            break
        ax = axes[i]

        gt = ground_truth[:, idx]
        pred = predictions[:, idx]

        # Plot CO-OPS observations first (if available)
        obs = observations.get(name)
        if obs is not None:
            ax.plot(obs['times'], obs['values'], 'k.', markersize=3,
                    label='CO-OPS Obs', alpha=0.7, zorder=1)

        # Compute metrics vs observations if available
        if obs is not None and len(obs['times']) > 10:
            # Interpolate predictions to observation times
            pred_interp = np.interp(obs['times'], time_hours, pred)
            valid_obs = ~np.isnan(obs['values']) & ~np.isnan(pred_interp)
            valid_obs &= (obs['times'] >= 0) & (obs['times'] <= args.hours)
            if valid_obs.sum() > 0:
                rmse_obs = np.sqrt(np.mean((obs['values'][valid_obs] - pred_interp[valid_obs])**2))
                corr_obs = np.corrcoef(obs['values'][valid_obs], pred_interp[valid_obs])[0, 1]
            else:
                rmse_obs, corr_obs = np.nan, np.nan
        else:
            rmse_obs, corr_obs = np.nan, np.nan

        # Compute metrics vs STOFS ground truth
        valid = ~np.isnan(gt) & ~np.isnan(pred)
        if valid.sum() > 0:
            rmse_stofs = np.sqrt(np.mean((gt[valid] - pred[valid])**2))
            corr_stofs = np.corrcoef(gt[valid], pred[valid])[0, 1] if len(gt[valid]) > 1 else np.nan
        else:
            rmse_stofs, corr_stofs = np.nan, np.nan

        # Plot STOFS and GNN prediction
        ax.plot(time_hours, gt, 'g-', label='STOFS', linewidth=1.5, alpha=0.8, zorder=2)
        ax.plot(time_hours, pred, 'b--', label='GNN Prediction', linewidth=1.5, alpha=0.8, zorder=3)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Water Level (m, MSL)')

        # Title with metrics
        if not np.isnan(rmse_obs):
            ax.set_title(f'{name}\nRMSE vs Obs: {rmse_obs:.3f}m, R: {corr_obs:.2f}')
        else:
            ax.set_title(f'{name}\nRMSE vs STOFS: {rmse_stofs:.3f}m, R: {corr_stofs:.2f}')

        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide last subplot
    axes[-1].axis('off')

    plt.suptitle(f'25K Node Model - {args.hours}h Rollout ({args.date})\nModel: {os.path.basename(checkpoint_path)}', fontsize=12)
    plt.tight_layout()

    # Save timeseries to text files (optional)
    if args.save_ts:
        ts_dir = args.ts_dir if args.ts_dir else f'outputs/timeseries/{args.date}'
        os.makedirs(ts_dir, exist_ok=True)

        start_dt = datetime.strptime(args.date, '%Y%m%d')

        print(f"\nSaving timeseries to {ts_dir}/...")
        for name, idx in station_indices.items():
            ts_file = os.path.join(ts_dir, f'{name}_rollout.txt')

            gt = ground_truth[:, idx]
            pred = predictions[:, idx]

            with open(ts_file, 'w') as f:
                f.write(f"# Station: {name}\n")
                f.write(f"# CO-OPS ID: {STATIONS[name]['id']}\n")
                f.write(f"# Lat: {STATIONS[name]['lat']}, Lon: {STATIONS[name]['lon']}\n")
                f.write(f"# Model: {os.path.basename(checkpoint_path)}\n")
                f.write(f"# Start: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
                f.write(f"# Columns: datetime, hours_from_start, stofs_wl(m), gnn_wl(m)\n")
                f.write("#\n")

                for t_idx in range(len(time_hours)):
                    dt = start_dt + timedelta(hours=time_hours[t_idx])
                    dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    stofs_val = gt[t_idx] if not np.isnan(gt[t_idx]) else -999.0
                    gnn_val = pred[t_idx] if not np.isnan(pred[t_idx]) else -999.0
                    f.write(f"{dt_str}  {time_hours[t_idx]:6.1f}  {stofs_val:8.4f}  {gnn_val:8.4f}\n")

            print(f"  Saved: {ts_file}")

    if args.output:
        output_path = args.output
    else:
        output_path = f'outputs/figures/rollout_25k_{args.date}.png'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
