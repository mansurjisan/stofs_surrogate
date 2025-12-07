#!/usr/bin/env python3
"""
Train CWL GNN model on Mid-Atlantic region.

Domain: [-76, -73] × [38, 41]
Covers: New York, Atlantic City, Delaware Bay, Philadelphia area

Benefits of smaller domain:
- Better resolution in target area
- Faster training, more epochs
- Can use larger model (hidden_dim, layers)
- Easier validation against tide gauges
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

# Mid-Atlantic bounding box
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Data paths
CWL_FILE = '/mnt/d/STOFS2D-Analysis/MAXELE_PLOTS/FIELD2DPLOTS/20251122/stofs_2d_glo.t00z.fields.cwl.nc'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'

# Model parameters - can use larger model with smaller domain
HIDDEN_DIM = 128        # Increased from 64
NUM_LAYERS = 8          # Increased from 6
TARGET_NODES = 15000    # Keep all nodes in bbox, no subsampling

# Training parameters
EPOCHS = 200
BATCH_SIZE = 4          # Can increase with smaller domain
LEARNING_RATE = 1e-4    # Lower LR for longer training
WEIGHT_DECAY = 1e-5
ETA_SCALE = 2.0


# ============================================================
# Model Architecture
# ============================================================

class MeshGraphNetBlock(nn.Module):
    """Message passing block with layer normalization."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index

        # Edge update
        edge_input = torch.cat([edge_attr, h[row], h[col]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)

        # Aggregate messages
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_attr_new)

        # Node update with residual
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr_new


class MidAtlanticGNN(nn.Module):
    """GNN for Mid-Atlantic CWL prediction."""

    def __init__(
        self,
        state_dim: int = 1,
        node_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 8,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(state_dim + node_feature_dim, hidden_dim),
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

        self.layers = nn.ModuleList([
            MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MidAtlanticGNN: {total_params:,} parameters")

    def forward(self, x, node_features, edge_index, edge_attr):
        # Encode
        h = self.node_encoder(torch.cat([x, node_features], dim=-1))
        e = self.edge_encoder(edge_attr)

        # Process
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        # Decode
        return self.decoder(h)


# ============================================================
# Data Processing
# ============================================================

def extract_midatlantic(nc_file, bbox):
    """Extract Mid-Atlantic region from CWL file."""
    logger.info(f"Opening {nc_file}")
    nc = NCDataset(nc_file, 'r')

    # Get coordinates
    x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    depth = nc.variables['depth'][:]

    logger.info(f"Global mesh: {len(x):,} nodes")

    # Filter to bounding box
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    subset_indices = np.where(mask)[0]
    logger.info(f"Nodes in Mid-Atlantic bbox: {len(subset_indices):,}")

    # Extract subset
    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_sub = depth[subset_indices]

    # Build connectivity using Delaunay triangulation
    logger.info("Building mesh connectivity...")
    points = np.column_stack([lon, lat])
    tri = Delaunay(points)

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                e = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(e)

    edges = np.array(list(edges))
    # Make bidirectional
    edge_index = np.vstack([edges, edges[:, ::-1]]).T

    logger.info(f"Created {len(edges):,} edges ({edge_index.shape[1]:,} directed)")

    # Extract elevation time series
    logger.info("Extracting elevation time series...")
    zeta = nc.variables['zeta']
    num_times = zeta.shape[0]
    logger.info(f"Time steps: {num_times}")

    elevation = np.zeros((num_times, len(subset_indices)), dtype=np.float32)

    for t in range(num_times):
        if t % 20 == 0:
            logger.info(f"  Timestep {t+1}/{num_times}")
        all_nodes = zeta[t, :]
        elevation[t, :] = all_nodes[subset_indices]

    # Handle dry nodes (ADCIRC fill value)
    elevation = np.where(elevation < -9000, np.nan, elevation)

    # Filter nodes with too many missing values
    valid_counts = np.sum(~np.isnan(elevation), axis=0)
    min_valid_ratio = 0.8
    good_nodes = valid_counts >= (min_valid_ratio * num_times)

    logger.info(f"Nodes with >=80% valid data: {np.sum(good_nodes):,} / {len(good_nodes):,}")

    # Apply filter
    good_indices = np.where(good_nodes)[0]
    lon = lon[good_indices]
    lat = lat[good_indices]
    depth_sub = depth_sub[good_indices]
    elevation = elevation[:, good_indices]

    # Rebuild node mapping for edges
    old_to_new = {old: new for new, old in enumerate(good_indices)}
    new_edges = []
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src in old_to_new and dst in old_to_new:
            new_edges.append([old_to_new[src], old_to_new[dst]])
    edge_index = np.array(new_edges).T

    logger.info(f"Final mesh: {len(lon):,} nodes, {edge_index.shape[1]:,} edges")

    # Fill remaining NaNs with interpolation
    for i in range(len(lon)):
        col = elevation[:, i]
        if np.any(np.isnan(col)):
            mask = ~np.isnan(col)
            if np.sum(mask) > 0:
                col[~mask] = np.interp(
                    np.where(~mask)[0],
                    np.where(mask)[0],
                    col[mask]
                )
            elevation[:, i] = col

    # Get times
    times = nc.variables['time'][:]

    nc.close()

    return {
        'lon': lon,
        'lat': lat,
        'depth': depth_sub,
        'edge_index': edge_index,
        'elevation': elevation,
        'times': times,
        'original_indices': subset_indices[good_indices],
    }


class MidAtlanticDataset(Dataset):
    """Dataset for Mid-Atlantic CWL time series."""

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

        # Node features (normalized)
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
        logger.info(f"Elevation range: [{self.elevation.min():.3f}, {self.elevation.max():.3f}] m")

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


def rollout_prediction(model, dataset, start_idx, num_steps, device):
    """Run multi-step autoregressive prediction."""
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)
    node_features = dataset.node_features.to(device)

    current = torch.tensor(
        dataset.elevation[start_idx] / dataset.eta_scale,
        dtype=torch.float32
    ).to(device)

    predictions = [current.cpu().numpy() * dataset.eta_scale]
    ground_truth = [dataset.elevation[start_idx]]

    with torch.no_grad():
        for step in range(num_steps):
            x = current.unsqueeze(1)
            next_state = model(x, node_features, edge_index, edge_attr).squeeze()
            current = next_state

            predictions.append(current.cpu().numpy() * dataset.eta_scale)
            if start_idx + step + 1 < len(dataset.elevation):
                ground_truth.append(dataset.elevation[start_idx + step + 1])

    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Visualization
# ============================================================

def plot_training_curves(train_losses, val_losses, output_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.semilogy(epochs, train_losses, 'b-', label='Train Loss')
    ax.semilogy(epochs, val_losses, 'r-', label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Mid-Atlantic CWL Model Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_domain(lon, lat, depth, elevation, output_path):
    """Plot the Mid-Atlantic domain."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bathymetry
    ax = axes[0]
    cf = ax.scatter(lon, lat, c=depth, s=1, cmap='Blues_r', vmin=0, vmax=100)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Mid-Atlantic Domain\n{len(lon):,} nodes')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Depth (m)')

    # Sample elevation
    ax = axes[1]
    t = len(elevation) // 2
    cf = ax.scatter(lon, lat, c=elevation[t], s=1, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'CWL at t={t}')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='CWL (m)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


def plot_rollout(lon, lat, predictions, ground_truth, output_path):
    """Plot rollout comparison."""
    timesteps = [0, 6, 12, 24]
    num_times = len(timesteps)

    fig, axes = plt.subplots(num_times, 3, figsize=(15, 4*num_times))

    vmax = 1.0
    s = 2

    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        pred = predictions[t]
        gt = ground_truth[t] if t < len(ground_truth) else predictions[t]
        diff = pred - gt

        # Ground truth
        ax = axes[i, 0]
        cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Ground Truth (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Prediction
        ax = axes[i, 1]
        cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Prediction (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')

        # Error
        ax = axes[i, 2]
        rmse = np.sqrt(np.mean(diff**2))
        cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'Error (RMSE={rmse:.3f}m)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Error (m)')

    plt.suptitle('Mid-Atlantic CWL Model Rollout', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    logger.info("="*60)
    logger.info("Mid-Atlantic CWL GNN Training")
    logger.info("="*60)
    logger.info(f"Domain: [{BBOX['lon_min']}, {BBOX['lon_max']}] x [{BBOX['lat_min']}, {BBOX['lat_max']}]")
    logger.info(f"Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")

    # Check for existing processed data
    mesh_path = f'{OUTPUT_DIR}/data/processed/midatlantic_mesh.npz'
    elev_path = f'{OUTPUT_DIR}/data/processed/midatlantic_elevation.npz'

    if os.path.exists(mesh_path) and os.path.exists(elev_path):
        logger.info("\nLoading existing processed data...")
        mesh_data = np.load(mesh_path)
        elev_data = np.load(elev_path)

        data_dict = {
            'lon': mesh_data['lon'],
            'lat': mesh_data['lat'],
            'depth': mesh_data['depth'],
            'edge_index': mesh_data['edge_index'],
            'elevation': elev_data['elevation'],
            'times': elev_data['times'],
        }
    else:
        logger.info("\nExtracting Mid-Atlantic data...")
        data_dict = extract_midatlantic(CWL_FILE, BBOX)

        # Save processed data
        os.makedirs(f'{OUTPUT_DIR}/data/processed', exist_ok=True)

        np.savez(mesh_path,
                 lon=data_dict['lon'],
                 lat=data_dict['lat'],
                 depth=data_dict['depth'],
                 edge_index=data_dict['edge_index'],
                 original_indices=data_dict['original_indices'])

        np.savez(elev_path,
                 elevation=data_dict['elevation'],
                 times=data_dict['times'])

        logger.info(f"Saved mesh to {mesh_path}")
        logger.info(f"Saved elevation to {elev_path}")

    # Plot domain
    plot_domain(
        data_dict['lon'], data_dict['lat'],
        data_dict['depth'], data_dict['elevation'],
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_domain.png'
    )

    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = MidAtlanticDataset(data_dict, eta_scale=ETA_SCALE)

    # Train/val split
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train: {train_size}, Val: {val_size}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    model = MidAtlanticGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Training loop
    logger.info("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    os.makedirs(f'{OUTPUT_DIR}/outputs/checkpoints', exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
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
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'eta_scale': ETA_SCALE,
                'bbox': BBOX,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
            }, f'{OUTPUT_DIR}/outputs/checkpoints/best_midatlantic_model.pt')

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch:3d}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.2e}, best={best_val_loss:.6f}")

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses,
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_training.png'
    )

    # Load best model and do rollout
    logger.info("\nLoading best model for rollout...")
    checkpoint = torch.load(f'{OUTPUT_DIR}/outputs/checkpoints/best_midatlantic_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    start_idx = train_size + 10  # Start in validation set
    predictions, ground_truth = rollout_prediction(model, dataset, start_idx, 36, device)

    plot_rollout(
        dataset.lon, dataset.lat,
        predictions, ground_truth,
        f'{OUTPUT_DIR}/outputs/figures/midatlantic_rollout.png'
    )

    # Print final metrics
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")

    for t in [1, 6, 12, 24]:
        if t < len(predictions):
            rmse = np.sqrt(np.mean((predictions[t] - ground_truth[t])**2))
            logger.info(f"Rollout t+{t}h RMSE: {rmse:.4f} m")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
