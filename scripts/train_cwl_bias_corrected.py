#!/usr/bin/env python3
"""
Train GNN surrogate on bias-corrected CWL (Coastal Water Level) data.

This script:
1. Extracts US East Coast subset from bias-corrected CWL file
2. Builds graph from mesh
3. Trains GNN to predict water level evolution
4. Saves model and generates visualizations

Usage:
    python scripts/train_cwl_bias_corrected.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from netCDF4 import Dataset as NCDataset, num2date
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CWL_FILE = Path("/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.nc")
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data/processed")

# US East Coast bounding box
BBOX = {
    'lon_min': -82.0,
    'lon_max': -65.0,
    'lat_min': 24.0,
    'lat_max': 46.0
}

# Training parameters
TARGET_NODES = 50000
HIDDEN_DIM = 64
NUM_LAYERS = 6
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 5e-4


# ============================================================
# Model Architecture
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


class CWLGNN(nn.Module):
    """GNN for CWL (Coastal Water Level) prediction."""

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

        logger.info(f"CWLGNN: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x, node_features, edge_index, edge_attr):
        h = self.node_encoder(torch.cat([x, node_features], dim=-1))
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return self.decoder(h)


# ============================================================
# Data Processing
# ============================================================

def extract_us_east_coast(nc_file, bbox, target_nodes=50000):
    """Extract US East Coast subset from CWL file."""
    logger.info(f"Opening {nc_file}")
    nc = NCDataset(nc_file, 'r')

    # Get coordinates
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    depth = nc.variables['depth'][:]
    elements = nc.variables['element'][:] - 1  # Convert to 0-indexed

    logger.info(f"Full mesh: {len(x):,} nodes, {len(elements):,} elements")

    # Filter to bounding box
    mask = ((x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
            (y >= bbox['lat_min']) & (y <= bbox['lat_max']))

    regional_indices = np.where(mask)[0]
    logger.info(f"Nodes in bbox: {len(regional_indices):,}")

    # Subsample if needed
    if len(regional_indices) > target_nodes:
        logger.info(f"Subsampling to {target_nodes:,} nodes...")
        np.random.seed(42)

        # Stratified sampling based on grid cells
        x_reg = x[regional_indices]
        y_reg = y[regional_indices]

        n_cells = int(np.sqrt(target_nodes / 10))
        x_bins = np.linspace(x_reg.min(), x_reg.max(), n_cells + 1)
        y_bins = np.linspace(y_reg.min(), y_reg.max(), n_cells + 1)

        x_idx = np.digitize(x_reg, x_bins) - 1
        y_idx = np.digitize(y_reg, y_bins) - 1
        cell_ids = x_idx * n_cells + y_idx

        selected = []
        unique_cells = np.unique(cell_ids)
        samples_per_cell = max(1, target_nodes // len(unique_cells))

        for cell in unique_cells:
            cell_mask = cell_ids == cell
            cell_indices = np.where(cell_mask)[0]
            n_sample = min(len(cell_indices), samples_per_cell)
            sampled = np.random.choice(cell_indices, n_sample, replace=False)
            selected.extend(sampled)

        # Add more if needed
        if len(selected) < target_nodes:
            remaining = list(set(range(len(regional_indices))) - set(selected))
            extra = np.random.choice(remaining, target_nodes - len(selected), replace=False)
            selected.extend(extra)

        selected = np.array(selected[:target_nodes])
        subset_indices = regional_indices[selected]
    else:
        subset_indices = regional_indices

    logger.info(f"Final subset: {len(subset_indices):,} nodes")

    # Extract coordinates for subset
    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_subset = depth[subset_indices]

    # Build edge connectivity using KDTree
    logger.info("Building edge connectivity...")
    coords = np.stack([lon, lat], axis=1)
    tree = cKDTree(coords)

    # Find k nearest neighbors
    k = 8
    distances, neighbors = tree.query(coords, k=k+1)

    edges_src = []
    edges_dst = []

    for i in range(len(coords)):
        for j in range(1, k+1):
            if neighbors[i, j] < len(coords):
                edges_src.append(i)
                edges_dst.append(neighbors[i, j])

    edge_index = np.array([edges_src, edges_dst])
    logger.info(f"Edges: {edge_index.shape[1]:,}")

    # Extract elevation time series
    logger.info("Extracting elevation time series...")
    zeta = nc.variables['zeta']
    num_times = zeta.shape[0]

    # Find contiguous regions in subset_indices for faster reading
    # Sort indices for efficient reading
    sorted_order = np.argsort(subset_indices)
    sorted_indices = subset_indices[sorted_order]

    # Create mask for faster extraction
    # Instead of advanced indexing, read blocks of data
    logger.info(f"Extracting {num_times} timesteps for {len(subset_indices)} nodes...")

    # Pre-allocate output
    elevation = np.zeros((num_times, len(subset_indices)), dtype=np.float32)

    # Read timestep by timestep with progress reporting
    for t in range(num_times):
        if t % 20 == 0 or t == num_times - 1:
            logger.info(f"  Timestep {t+1}/{num_times}")
        # Read single timestep, subset in memory
        all_nodes = zeta[t, :]
        elevation[t, :] = all_nodes[subset_indices]

    # Handle masked/fill values (ADCIRC uses -99999 for dry nodes)
    if hasattr(elevation, 'mask'):
        elevation = np.where(elevation.mask, np.nan, elevation.data)
    # Replace ADCIRC dry node fill values with NaN
    elevation = np.where(elevation < -9000, np.nan, elevation)

    # For each node, check how many valid timesteps it has
    valid_counts = np.sum(~np.isnan(elevation), axis=0)
    min_valid_ratio = 0.8  # Require at least 80% valid timesteps
    good_nodes = valid_counts >= (min_valid_ratio * num_times)

    if not np.all(good_nodes):
        logger.info(f"Filtering out {np.sum(~good_nodes)} nodes with too many dry timesteps")
        good_indices = np.where(good_nodes)[0]
        elevation = elevation[:, good_indices]
        lon = lon[good_indices]
        lat = lat[good_indices]
        depth_subset = depth_subset[good_indices]
        subset_indices = subset_indices[good_indices]

        # Rebuild edge connectivity for filtered nodes
        logger.info(f"Rebuilding edge connectivity for {len(good_indices)} nodes...")
        coords = np.stack([lon, lat], axis=1)
        tree = cKDTree(coords)

        k = 8
        distances, neighbors = tree.query(coords, k=k+1)

        edges_src = []
        edges_dst = []

        for i in range(len(coords)):
            for j in range(1, k+1):
                if neighbors[i, j] < len(coords):
                    edges_src.append(i)
                    edges_dst.append(neighbors[i, j])

        edge_index = np.array([edges_src, edges_dst])
        logger.info(f"New edges: {edge_index.shape[1]:,}")

    # Replace remaining NaN with 0 (shouldn't be many after filtering)
    elevation = np.nan_to_num(elevation, nan=0.0)

    # Get time info
    time_var = nc.variables['time']
    try:
        times = num2date(time_var[:], time_var.units)
        times = np.array([t.strftime('%Y-%m-%d %H:%M') for t in times])
    except:
        times = np.arange(num_times)

    nc.close()

    logger.info(f"Elevation shape: {elevation.shape}")
    logger.info(f"Elevation range: [{elevation.min():.2f}, {elevation.max():.2f}] m")

    return {
        'lon': lon.astype(np.float32),
        'lat': lat.astype(np.float32),
        'depth': depth_subset.astype(np.float32),
        'edge_index': edge_index,
        'elevation': elevation.astype(np.float32),
        'times': times,
        'original_indices': subset_indices,
    }


class CWLDataset(Dataset):
    """Dataset for CWL time series."""

    def __init__(self, data_dict, eta_scale=2.0):
        self.lon = data_dict['lon']
        self.lat = data_dict['lat']
        self.depth = data_dict['depth']
        self.edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        self.elevation = data_dict['elevation']
        self.eta_scale = eta_scale

        self.num_nodes = len(self.lon)
        self.num_times = len(self.elevation)
        self.num_samples = self.num_times - 1

        # Compute Cartesian coordinates
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = 6371000.0
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)

        # Node features
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1

        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

        self.node_features = torch.tensor(
            np.stack([x_norm, y_norm, depth_norm], axis=1),
            dtype=torch.float32
        )

        # Edge features
        src, dst = self.edge_index[0].numpy(), self.edge_index[1].numpy()
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)
        char_length = np.median(dist) + 1e-8

        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        )

        logger.info(f"Dataset: {self.num_samples} samples, {self.num_nodes} nodes")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        eta_in = self.elevation[idx] / self.eta_scale
        eta_out = self.elevation[idx + 1] / self.eta_scale

        return Data(
            x=torch.tensor(eta_in[:, np.newaxis], dtype=torch.float32),
            y=torch.tensor(eta_out[:, np.newaxis], dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_features=self.node_features,
        )


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.x, batch.node_features, batch.edge_index, batch.edge_attr)
        loss = criterion(pred, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.node_features, batch.edge_index, batch.edge_attr)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("=" * 70)
    print("Train GNN on Bias-Corrected CWL Data")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Check file exists
    if not CWL_FILE.exists():
        logger.error(f"CWL file not found: {CWL_FILE}")
        return

    # Extract data
    print("\n1. Extracting US East Coast subset...")
    data = extract_us_east_coast(CWL_FILE, BBOX, TARGET_NODES)

    # Save extracted data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mesh_file = DATA_DIR / "us_east_coast_cwl_mesh.npz"
    np.savez_compressed(
        mesh_file,
        lon=data['lon'],
        lat=data['lat'],
        depth=data['depth'],
        edge_index=data['edge_index'],
        original_indices=data['original_indices'],
    )
    logger.info(f"Saved mesh: {mesh_file}")

    elev_file = DATA_DIR / "us_east_coast_cwl_elevation.npz"
    np.savez_compressed(
        elev_file,
        elevation=data['elevation'],
        times=data['times'],
    )
    logger.info(f"Saved elevation: {elev_file}")

    # Create dataset
    print("\n2. Creating dataset...")
    eta_scale = 2.0
    dataset = CWLDataset(data, eta_scale=eta_scale)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print("\n3. Creating model...")
    model = CWLGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n4. Training for {EPOCHS} epochs...")
    print("-" * 50)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'eta_scale': eta_scale,
            }, OUTPUT_DIR / "checkpoints" / "best_cwl_model.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    print("-" * 50)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Plot training curves
    print("\n5. Generating visualizations...")
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)
    ax.semilogy(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.semilogy(epochs_range[4::5], val_losses[4::5], 'ro-', label='Val Loss', markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('CWL Bias-Corrected Data Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "cwl_training.png", dpi=150)
    plt.close()

    # Rollout visualization
    print("\n6. Testing rollout...")
    model.eval()

    coords = np.stack([data['lon'], data['lat']], axis=1)
    edge_index_tensor = torch.tensor(data['edge_index'], dtype=torch.long, device=device)
    edge_attr_tensor = dataset.edge_attr.to(device)
    node_features_tensor = dataset.node_features.to(device)

    current_state = torch.tensor(data['elevation'][0:1].T / eta_scale, dtype=torch.float32, device=device)

    rollout_steps = [0, 10, 20, 30, 40, 50]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    cmap = plt.cm.jet
    vmin, vmax = 0, 3

    with torch.no_grad():
        # Reset state for each rollout
        for i, step in enumerate(rollout_steps):
            # Reset to initial state and roll forward to this step
            current_state = torch.tensor(data['elevation'][0:1].T / eta_scale, dtype=torch.float32, device=device)

            for s in range(step):
                pred = model(current_state, node_features_tensor, edge_index_tensor, edge_attr_tensor)
                current_state = pred

            elev = current_state.cpu().numpy().flatten() * eta_scale

            ax = axes[i]
            triang = mtri.Triangulation(coords[:, 0], coords[:, 1])
            tcf = ax.tricontourf(triang, np.clip(elev, vmin, vmax),
                                levels=np.linspace(vmin, vmax, 31), cmap=cmap, extend='both')
            ax.set_title(f'Step {step} (T+{step}h)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal')
            plt.colorbar(tcf, ax=ax, label='CWL (m)')

    plt.suptitle('CWL GNN Surrogate - Rollout Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "cwl_rollout.png", dpi=150)
    plt.close()

    # Summary
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"""
Results:
  - Final train loss: {train_losses[-1]:.6f}
  - Best val loss: {best_val_loss:.6f}
  - Model saved: {OUTPUT_DIR / "checkpoints" / "best_cwl_model.pt"}
  - Mesh saved: {mesh_file}
  - Elevation saved: {elev_file}

This model was trained on BIAS-CORRECTED CWL data!
""")


if __name__ == '__main__':
    main()
