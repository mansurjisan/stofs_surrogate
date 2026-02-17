#!/usr/bin/env python3
"""
Enhanced STOFS-GNN Visualization Script
Creates publication-quality tricontourf visualizations with:
- Custom Blue→White→Yellow/Orange/Red colormap
- 300 DPI output
- GSHHS coastline overlay
- Light blue ocean background
- Smooth filled contours
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.spatial import Delaunay
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import warnings
import argparse
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import geopandas for coastlines
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("geopandas not available. Coastlines will not be drawn.")

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/e/STOFS_TRAINING_DATA/processed_25k_v2')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(os.environ.get('STOFS_CHECKPOINT_DIR', PROJECT_ROOT / 'outputs/checkpoints_25k_v2'))
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', PROJECT_ROOT / 'outputs/figures_25k_v2/enhanced_snapshots'))
GSHHS_PATH = os.environ.get('GSHHS_PATH', None)  # Optional: path to GSHHS coastline shapefile

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

# Region bounds for Mid-Atlantic
LON_MIN, LON_MAX = -77.5, -71.5
LAT_MIN, LAT_MAX = 36.5, 42.5


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
        h_new = h + node_out
        return h_new, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.gnn_layers = nn.ModuleList([BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        B = x.shape[0]
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


def compute_tidal_harmonics(global_hour):
    harmonics = []
    for name, period in TIDAL_PERIODS.items():
        phase = 2.0 * np.pi * global_hour / period
        harmonics.extend([np.sin(phase), np.cos(phase)])
    return np.array(harmonics, dtype=np.float32)


def get_forcing(forcing_dict, t, num_nodes):
    return np.stack([
        forcing_dict['u10'][t], forcing_dict['v10'][t],
        forcing_dict['wind_speed'][t], forcing_dict['wind_speed_sq'][t],
        forcing_dict['wind_dir'][t], forcing_dict['pressure'][t],
        forcing_dict['dP_dx'][t], forcing_dict['dP_dy'][t],
    ], axis=-1).astype(np.float32)


def load_model(checkpoint_path, device):
    model = BatchedTemporalMemoryGNN(
        state_dim=STATE_DIM, temporal_dim=TEMPORAL_FEATURES,
        static_feature_dim=STATIC_NODE_FEATURES, forcing_feature_dim=FORCING_FEATURES,
        hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def run_rollout(model, mesh_data, data, device, max_hours=48):
    """Run autoregressive rollout."""
    elevation = data['elevation']
    forcing = data['forcing']
    date_str = data['date']

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
        dtype=torch.float32
    ).to(device)

    # Initialize
    cwl_prev = np.nan_to_num(elevation[0].astype(np.float32), nan=0.0)
    cwl_t = np.nan_to_num(elevation[1].astype(np.float32), nan=0.0)
    current_prev = torch.tensor(cwl_prev / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    current = torch.tensor(cwl_t / ETA_SCALE, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    predictions = []
    ground_truth = []
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

            forcing_arr = get_forcing(forcing, t, num_nodes)
            forcing_tensor = torch.tensor(forcing_arr, dtype=torch.float32).unsqueeze(0).to(device)

            pred = model(current, current_prev, dxdt, tidal_tensor, static_tensor,
                        forcing_tensor, edge_index, edge_attr)

            predictions.append(pred.squeeze().cpu().numpy() * ETA_SCALE)
            ground_truth.append(np.nan_to_num(elevation[t + 1].astype(np.float32), nan=0.0))

            current_prev = current
            current = pred

    return np.array(predictions), np.array(ground_truth)


def create_enhanced_plot(lon, lat, stofs_data, gnn_data, triangles, output_file,
                         date_str, hour, rmse, corr, vmin=-1.5, vmax=1.5):
    """Create enhanced side-by-side plot using grid interpolation for smooth rendering."""
    from scipy.interpolate import griddata

    # Create fine regular grid for smooth interpolation
    n_grid = 800  # High resolution grid
    lon_grid = np.linspace(LON_MIN, LON_MAX, n_grid)
    lat_grid = np.linspace(LAT_MIN, LAT_MAX, n_grid)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate data onto regular grid (cubic for smoothness)
    points = np.column_stack([lon, lat])

    # Clean data before interpolation
    mask_stofs = np.isnan(stofs_data) | (np.abs(stofs_data) > 10)
    mask_gnn = np.isnan(gnn_data) | (np.abs(gnn_data) > 10)

    stofs_clean = np.where(mask_stofs, 0, stofs_data)
    gnn_clean = np.where(mask_gnn, 0, gnn_data)

    # Use cubic interpolation for smooth results
    from scipy.ndimage import gaussian_filter

    stofs_grid = griddata(points, stofs_clean, (lon_mesh, lat_mesh), method='cubic')
    gnn_grid = griddata(points, gnn_clean, (lon_mesh, lat_mesh), method='cubic')

    # Apply Gaussian smoothing to remove mesh artifacts
    sigma = 3  # Smoothing strength
    stofs_grid = gaussian_filter(stofs_grid, sigma=sigma)
    gnn_grid = gaussian_filter(gnn_grid, sigma=sigma)

    # Ocean-style colormap for water elevation: deep blue -> light blue -> white -> yellow -> orange -> red
    # This represents: very low water -> low water -> mean -> high water -> very high water
    colors_list = [
        (0.0, '#08306b'),   # Deep blue (very low)
        (0.2, '#2171b5'),   # Medium blue
        (0.4, '#6baed6'),   # Light blue
        (0.5, '#f7f7f7'),   # White/neutral (mean sea level)
        (0.6, '#fee090'),   # Light yellow
        (0.8, '#fc8d59'),   # Orange
        (1.0, '#b30000'),   # Deep red (very high)
    ]
    cmap = LinearSegmentedColormap.from_list('ocean_elevation',
                                              [(pos, color) for pos, color in colors_list])

    # Create figure with space for horizontal colorbar at bottom
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=300)

    # Adjust subplot positions to leave room for colorbar
    plt.subplots_adjust(bottom=0.15, top=0.90, left=0.05, right=0.95, wspace=0.1)

    for ax in axes:
        ax.set_facecolor('white')  # White background for areas without data

    # Use pcolormesh with gouraud shading for truly smooth rendering (no contour banding)
    im1 = axes[0].pcolormesh(lon_mesh, lat_mesh, stofs_grid, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='gouraud', rasterized=True)
    axes[0].set_title(f'STOFS Ground Truth - t+{hour}h', fontsize=11, fontweight='bold', pad=6)

    im2 = axes[1].pcolormesh(lon_mesh, lat_mesh, gnn_grid, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='gouraud', rasterized=True)
    axes[1].set_title(f'GNN Prediction - t+{hour}h | RMSE: {rmse*100:.1f}cm, R: {corr:.3f}',
                      fontsize=11, fontweight='bold', pad=6)

    # Add coastlines
    if GEOPANDAS_AVAILABLE:
        try:
            coastline = gpd.read_file(GSHHS_PATH, bbox=(LON_MIN-0.5, LAT_MIN-0.5, LON_MAX+0.5, LAT_MAX+0.5))
            for ax in axes:
                coastline.plot(ax=ax, facecolor='#D4D4D4', edgecolor='#404040', linewidth=0.8, zorder=5)
        except Exception as e:
            logger.warning(f"Coastline error: {e}")

    # Set limits and labels
    for ax in axes:
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.set_xlabel('Longitude (degrees)', fontsize=9)
        ax.set_ylabel('Latitude (degrees)', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_aspect('equal')

        # Forecast time label
        ax.text(0.02, 0.98, f'Forecast: {date_str} +{hour}h',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
                zorder=20)

    # Horizontal colorbar at bottom spanning both panels
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label('Water Level (m MSL)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Set nice tick values
    tick_step = 0.5 if vmax <= 2 else 1.0
    cbar.set_ticks(np.arange(vmin, vmax + tick_step, tick_step))

    plt.suptitle(f'Water Elevation Comparison: STOFS vs GNN - {date_str}',
                 fontsize=12, fontweight='bold', y=0.95)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced STOFS-GNN comparison plots')
    parser.add_argument('--date', type=str, default='20250101', help='Validation date (YYYYMMDD)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch_95.pt', help='Checkpoint file')
    parser.add_argument('--interval', type=int, default=6, help='Hour interval for snapshots')
    parser.add_argument('--max-hours', type=int, default=48, help='Maximum forecast hours')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # Load mesh
    mesh_data = dict(np.load(DATA_DIR / 'mesh.npz', allow_pickle=True))
    lon = mesh_data['lon']
    lat = mesh_data['lat']
    logger.info(f"Mesh: {len(lon):,} nodes")

    # Create Delaunay triangulation
    logger.info("Creating Delaunay triangulation...")
    points = np.column_stack([lon, lat])
    delaunay = Delaunay(points)
    triangles = delaunay.simplices
    logger.info(f"  Triangles: {len(triangles):,}")

    # Load data
    val_file = DATA_DIR / f'processed_{args.date}.npz'
    val_data_raw = np.load(val_file)
    val_data = {
        'date': args.date,
        'elevation': val_data_raw['elevation'],
        'forcing': {
            'u10': val_data_raw['u10'], 'v10': val_data_raw['v10'],
            'wind_speed': val_data_raw['wind_speed'], 'wind_speed_sq': val_data_raw['wind_speed_sq'],
            'wind_dir': val_data_raw['wind_dir'], 'pressure': val_data_raw['pressure'],
            'dP_dx': val_data_raw['dP_dx'], 'dP_dy': val_data_raw['dP_dy'],
        }
    }
    logger.info(f"Loaded {val_data['elevation'].shape[0]} timesteps for {args.date}")

    # Load model
    ckpt_path = CHECKPOINT_DIR / args.checkpoint
    model = load_model(ckpt_path, device)
    logger.info(f"Loaded: {args.checkpoint}")

    # Run rollout
    logger.info("\nRunning rollout...")
    predictions, ground_truth = run_rollout(model, mesh_data, val_data, device, args.max_hours)
    logger.info(f"Predictions: {predictions.shape}")

    # Generate enhanced plots
    logger.info("\nGenerating enhanced plots...")

    for h in range(args.interval, args.max_hours + 1, args.interval):
        if h > len(predictions):
            continue

        pred = predictions[h - 1]
        truth = ground_truth[h - 1]
        rmse = np.sqrt(np.mean((pred - truth)**2))
        corr = np.corrcoef(pred, truth)[0, 1]

        # Determine color scale based on data
        all_vals = np.concatenate([pred, truth])
        vmax = np.percentile(np.abs(all_vals[~np.isnan(all_vals)]), 98)
        vmax = max(1.0, np.ceil(vmax * 2) / 2)  # Round up to nearest 0.5
        vmin = -vmax

        out_path = OUTPUT_DIR / f'enhanced_{args.date}_h{h:02d}.png'

        success = create_enhanced_plot(
            lon, lat, truth, pred, triangles, out_path,
            args.date, h, rmse, corr, vmin, vmax
        )

        if success:
            logger.info(f"  Saved: {out_path.name} (RMSE: {rmse*100:.1f}cm, R: {corr:.3f})")

    logger.info(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
