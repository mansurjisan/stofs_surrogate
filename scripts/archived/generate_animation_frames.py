#!/usr/bin/env python3
"""
Generate individual frames for animation of STOFS GNN predictions.

Uses GSHHS shapefile for high-quality coastlines with land fill.
Plotting style matches STOFS-3D visualization (jet colormap, 0-3m range).
"""

import matplotlib
matplotlib.use('Agg')

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import warnings
import logging
import os

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

# Paths
MESH_PATH = Path("data/processed/us_east_coast_mesh.npz")
ELEVATION_PATH = Path("data/processed/us_east_coast_elevation.npz")
MODEL_PATH = Path("outputs/checkpoints/best_real_stofs.pt")
OUTPUT_DIR = Path("outputs/figures/animation_frames")

# GSHHS Coastline shapefile path
COASTLINE_PATH = Path("/mnt/d/STOFS2D-Analysis/My_Scripts/2D-Global-Points-CWL/GSHHS_shp/f/GSHHS_f_L1.shp")

# Colorbar range (matching STOFS-3D style)
VMIN = 0.0
VMAX = 3.0


# ============================================================
# Model classes (copied from train_us_east_coast_real.py)
# ============================================================

class MeshGraphNetBlock(nn.Module):
    """Message passing block."""

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


class RealSTOFSGNN(nn.Module):
    """GNN for real STOFS elevation prediction."""

    def __init__(
        self,
        state_dim: int = 1,
        node_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 6,
    ):
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

        self.layers = nn.ModuleList([
            MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)
        ])

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


# ============================================================
# Data loading
# ============================================================

def load_model_and_data():
    """Load trained model and data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load mesh
    mesh = np.load(MESH_PATH)
    lon = mesh['lon']
    lat = mesh['lat']
    depth = mesh['depth']
    edge_index = mesh['edge_index']

    coords = np.stack([lon, lat], axis=1)

    # Load elevation
    elev_data = np.load(ELEVATION_PATH)
    elevation = elev_data['elevation']  # [time, nodes]
    times = elev_data['times']

    # Compute node features (same as training)
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

    # Compute edge features
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    # Load model
    model = RealSTOFSGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=64,
        num_layers=6
    )
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    eta_scale = checkpoint.get('eta_scale', 2.0)

    return (model, coords, edge_index, edge_attr, node_features,
            elevation, eta_scale, times, device)


def load_coastline(lon_min, lon_max, lat_min, lat_max):
    """Load GSHHS coastline for the region."""
    if not GEOPANDAS_AVAILABLE:
        return None

    if not COASTLINE_PATH.exists():
        logger.warning(f"Coastline file not found: {COASTLINE_PATH}")
        return None

    try:
        coastline_gdf = gpd.read_file(
            COASTLINE_PATH,
            bbox=(lon_min - 0.5, lat_min - 0.5, lon_max + 0.5, lat_max + 0.5)
        )
        logger.info(f"Loaded coastline with {len(coastline_gdf)} features")
        return coastline_gdf
    except Exception as e:
        logger.warning(f"Failed to load coastline: {e}")
        return None


# ============================================================
# Frame generation with STOFS-3D style (jet colormap, 0-3m)
# ============================================================

def create_frame(coords, elevation, step, time_label, output_path,
                coastline_gdf=None, frame_type="prediction"):
    """Create a single frame with STOFS-3D style visualization."""
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)

    # Dark blue ocean background (for values below 0)
    ax.set_facecolor('#000080')

    # Create triangulation
    triang = mtri.Triangulation(coords[:, 0], coords[:, 1])

    # Use jet colormap (STOFS-3D style) with 0-3m range
    cmap = plt.cm.jet
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    levels = np.linspace(VMIN, VMAX, 61)

    # Clip elevation to range and handle NaN
    elevation_clipped = np.clip(elevation, VMIN, VMAX)
    mask_nan = np.isnan(elevation)

    if mask_nan.any():
        tri_has_bad = mask_nan[triang.triangles].any(axis=1)
        triang.set_mask(tri_has_bad)
        elevation_clean = np.where(mask_nan, 0, elevation_clipped)
    else:
        elevation_clean = elevation_clipped

    # Plot water elevation using tricontourf
    im = ax.tricontourf(triang, elevation_clean, levels=levels, cmap=cmap, norm=norm, extend='both')

    # Add coastline with land fill
    if coastline_gdf is not None:
        coastline_gdf.plot(
            ax=ax,
            facecolor='#C0C0C0',  # Light gray land
            edgecolor='#404040',   # Dark gray coastline
            linewidth=0.5,
            zorder=5
        )

    # Set axis limits
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    ax.set_xlim(lon_min - 0.2, lon_max + 0.2)
    ax.set_ylim(lat_min - 0.2, lat_max + 0.2)
    ax.set_aspect('equal')

    # Title based on frame type
    if frame_type == "prediction":
        title_prefix = "GNN Surrogate Prediction"
    elif frame_type == "ground_truth":
        title_prefix = "STOFS Ground Truth"
    else:
        title_prefix = "Water Surface Elevation"

    ax.set_title(f'{title_prefix} - US East Coast\n{time_label}',
                fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude (degrees)', fontsize=12)
    ax.set_ylabel('Latitude (degrees)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

    # Step indicator box
    if frame_type == "prediction":
        box_color = "#FF6347"
    else:
        box_color = "#1E90FF"

    ax.text(0.02, 0.98, f'Step {step}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, edgecolor='black', alpha=0.9),
            color='white', zorder=20)

    # Horizontal colorbar at bottom (STOFS-3D style)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.08, aspect=40)
    cbar.set_label('Total Water Level (m above xGEOID20B)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)

    # Set colorbar ticks: 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def create_comparison_frame(coords, gt_elev, pred_elev, step, time_label, output_path,
                           coastline_gdf=None):
    """Create side-by-side comparison frame."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), dpi=200)

    cmap = plt.cm.jet
    norm = Normalize(vmin=VMIN, vmax=VMAX)
    levels = np.linspace(VMIN, VMAX, 61)

    triang = mtri.Triangulation(coords[:, 0], coords[:, 1])

    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

    titles = ['STOFS Ground Truth', 'GNN Prediction', 'Difference']
    data_list = [gt_elev, pred_elev, pred_elev - gt_elev]

    for idx, (ax, title, data) in enumerate(zip(axes, titles, data_list)):
        ax.set_facecolor('#000080')

        if idx == 2:  # Difference plot
            diff = data
            diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)), 0.5)
            diff_norm = Normalize(vmin=-diff_max, vcenter=0, vmax=diff_max)
            from matplotlib.colors import TwoSlopeNorm
            diff_norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
            diff_levels = np.linspace(-diff_max, diff_max, 61)

            # Use RdBu for difference
            im = ax.tricontourf(triang, np.nan_to_num(diff, 0), levels=diff_levels,
                               cmap='RdBu_r', norm=diff_norm, extend='both')
            rmse = np.sqrt(np.nanmean(diff**2))
            title = f'Difference (RMSE: {rmse:.3f} m)'
        else:
            elev_clipped = np.clip(data, VMIN, VMAX)
            elev_clean = np.nan_to_num(elev_clipped, 0)
            im = ax.tricontourf(triang, elev_clean, levels=levels, cmap=cmap, norm=norm, extend='both')

        # Add coastline
        if coastline_gdf is not None:
            coastline_gdf.plot(ax=ax, facecolor='#C0C0C0', edgecolor='#404040', linewidth=0.5, zorder=5)

        ax.set_xlim(lon_min - 0.2, lon_max + 0.2)
        ax.set_ylim(lat_min - 0.2, lat_max + 0.2)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)

        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.9, pad=0.08)
        if idx == 2:
            cbar.set_label('Difference (m)', fontsize=11)
        else:
            cbar.set_label('Water Level (m)', fontsize=11)
            cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    fig.suptitle(f'Step {step} - {time_label}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ============================================================
# Main frame generation functions
# ============================================================

def generate_rollout_frames(model, coords, edge_index, edge_attr, node_features,
                           elevation, eta_scale, times, device, coastline_gdf, num_steps=126):
    """Generate frames from model rollout."""
    pred_dir = OUTPUT_DIR / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)

    current_state = torch.tensor(elevation[0:1].T / eta_scale, dtype=torch.float32, device=device)

    logger.info(f"Generating {num_steps} prediction frames...")

    with torch.no_grad():
        for step in range(num_steps):
            elev_actual = current_state.cpu().numpy().flatten() * eta_scale

            if step < len(times):
                time_str = str(times[step])[:19]
            else:
                time_str = f"T+{step}h (forecast)"

            frame_path = pred_dir / f"frame_{step:04d}.png"
            create_frame(coords, elev_actual, step, time_str, frame_path,
                        coastline_gdf=coastline_gdf, frame_type="prediction")

            if (step + 1) % 10 == 0:
                logger.info(f"  Prediction frame {step + 1}/{num_steps}")

            pred = model(current_state, node_features_tensor, edge_index_tensor, edge_attr_tensor)
            current_state = pred


def generate_ground_truth_frames(coords, elevation, times, coastline_gdf, num_steps=126):
    """Generate ground truth frames."""
    gt_dir = OUTPUT_DIR / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    num_frames = min(num_steps, len(times))
    logger.info(f"Generating {num_frames} ground truth frames...")

    for step in range(num_frames):
        elev_actual = elevation[step]
        time_str = str(times[step])[:19]

        frame_path = gt_dir / f"frame_{step:04d}.png"
        create_frame(coords, elev_actual, step, time_str, frame_path,
                    coastline_gdf=coastline_gdf, frame_type="ground_truth")

        if (step + 1) % 10 == 0:
            logger.info(f"  Ground truth frame {step + 1}/{num_frames}")


def generate_comparison_frames(model, coords, edge_index, edge_attr, node_features,
                              elevation, eta_scale, times, device, coastline_gdf, num_steps=126):
    """Generate side-by-side comparison frames."""
    comp_dir = OUTPUT_DIR / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)

    current_state = torch.tensor(elevation[0:1].T / eta_scale, dtype=torch.float32, device=device)

    num_frames = min(num_steps, len(times) - 1)
    logger.info(f"Generating {num_frames} comparison frames...")

    with torch.no_grad():
        for step in range(num_frames):
            pred_elev = current_state.cpu().numpy().flatten() * eta_scale
            gt_elev = elevation[step]
            time_str = str(times[step])[:19]

            frame_path = comp_dir / f"frame_{step:04d}.png"
            create_comparison_frame(coords, gt_elev, pred_elev, step, time_str, frame_path,
                                   coastline_gdf=coastline_gdf)

            if (step + 1) % 10 == 0:
                logger.info(f"  Comparison frame {step + 1}/{num_frames}")

            pred = model(current_state, node_features_tensor, edge_index_tensor, edge_attr_tensor)
            current_state = pred


def create_gif(frame_dir, output_path, duration=150):
    """Create GIF from frames."""
    try:
        from PIL import Image

        frames = sorted(frame_dir.glob("frame_*.png"))
        if not frames:
            logger.warning(f"No frames found in {frame_dir}")
            return

        images = [Image.open(f) for f in frames]
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        logger.info(f"Created GIF: {output_path}")

    except ImportError:
        logger.warning("PIL not installed. Skipping GIF creation.")


def main():
    print("=" * 70)
    print("Generate Animation Frames for STOFS GNN Predictions")
    print("Using STOFS-3D style (jet colormap, 0-3m range)")
    print("=" * 70)

    # Load model and data
    print("\n1. Loading model and data...")
    (model, coords, edge_index, edge_attr, node_features,
     elevation, eta_scale, times, device) = load_model_and_data()

    logger.info(f"Device: {device}")
    logger.info(f"Nodes: {len(coords):,}")
    logger.info(f"Timesteps: {len(times)}")
    logger.info(f"Eta scale: {eta_scale}")
    logger.info(f"Elevation range: [{elevation.min():.2f}, {elevation.max():.2f}] m")

    # Load coastline
    print("\n2. Loading GSHHS coastline...")
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()
    coastline_gdf = load_coastline(lon_min, lon_max, lat_min, lat_max)

    num_steps = min(126, len(times))

    # Generate frames
    print("\n3. Generating prediction frames...")
    generate_rollout_frames(
        model, coords, edge_index, edge_attr, node_features,
        elevation, eta_scale, times, device, coastline_gdf, num_steps=num_steps
    )

    print("\n4. Generating ground truth frames...")
    generate_ground_truth_frames(coords, elevation, times, coastline_gdf, num_steps=num_steps)

    print("\n5. Generating comparison frames...")
    generate_comparison_frames(
        model, coords, edge_index, edge_attr, node_features,
        elevation, eta_scale, times, device, coastline_gdf, num_steps=num_steps
    )

    # Create GIFs
    print("\n6. Creating GIF animations...")
    create_gif(OUTPUT_DIR / "prediction", OUTPUT_DIR.parent / "prediction_rollout.gif", duration=150)
    create_gif(OUTPUT_DIR / "ground_truth", OUTPUT_DIR.parent / "ground_truth.gif", duration=150)
    create_gif(OUTPUT_DIR / "comparison", OUTPUT_DIR.parent / "comparison.gif", duration=200)

    # Summary
    print("\n" + "=" * 70)
    print("Animation Frames Generated!")
    print("=" * 70)
    print(f"""
Output directories:
  - Prediction frames: {OUTPUT_DIR / 'prediction'}
  - Ground truth frames: {OUTPUT_DIR / 'ground_truth'}
  - Comparison frames: {OUTPUT_DIR / 'comparison'}

GIF animations:
  - {OUTPUT_DIR.parent / 'prediction_rollout.gif'}
  - {OUTPUT_DIR.parent / 'ground_truth.gif'}
  - {OUTPUT_DIR.parent / 'comparison.gif'}

To create a video from frames:
  ffmpeg -framerate 10 -i {OUTPUT_DIR}/prediction/frame_%04d.png -c:v libx264 -pix_fmt yuv420p prediction.mp4
""")


if __name__ == '__main__':
    main()
