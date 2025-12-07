#!/usr/bin/env python3
"""
Train GNN surrogate on US East Coast with REAL STOFS elevation data.

This script uses actual STOFS water elevation time series for training,
producing a surrogate that can predict real storm surge dynamics.

Usage:
    python scripts/train_us_east_coast_real.py
    python scripts/train_us_east_coast_real.py --epochs 100 --batch-size 4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealSTOFSDataset(Dataset):
    """
    Dataset using REAL STOFS elevation time series.

    Creates input-output pairs for autoregressive training:
    State(t) -> State(t+1)
    """

    def __init__(
        self,
        mesh_path: str = "data/processed/us_east_coast_mesh.npz",
        elevation_path: str = "data/processed/us_east_coast_elevation.npz",
        time_stride: int = 1,
        normalize: bool = True,
        eta_scale: float = 2.0,
    ):
        super().__init__()

        self.time_stride = time_stride
        self.normalize = normalize
        self.eta_scale = eta_scale

        # Load mesh
        logger.info(f"Loading mesh from {mesh_path}")
        mesh = np.load(mesh_path)

        self.lon = mesh['lon']
        self.lat = mesh['lat']
        self.depth = mesh['depth']
        self.edge_index = torch.tensor(mesh['edge_index'], dtype=torch.long)

        self.num_nodes = len(self.lon)
        logger.info(f"Mesh: {self.num_nodes:,} nodes, {self.edge_index.shape[1]:,} edges")

        # Load elevation data
        logger.info(f"Loading elevation from {elevation_path}")
        elev_data = np.load(elevation_path)

        self.elevation = elev_data['elevation']  # [time, nodes]
        self.times = elev_data['times']

        self.num_timesteps = len(self.elevation)
        self.num_samples = (self.num_timesteps - time_stride) // time_stride

        logger.info(f"Elevation: {self.elevation.shape}")
        logger.info(f"Training samples: {self.num_samples}")

        # Convert to Cartesian
        self.x_cart, self.y_cart = self._to_cartesian()

        # Prepare features
        self.node_features = self._prepare_node_features()
        self.edge_attr = self._compute_edge_features()

        # Compute normalization stats
        if normalize:
            self.eta_mean = self.elevation.mean()
            self.eta_std = self.elevation.std()
            logger.info(f"Elevation stats: mean={self.eta_mean:.3f}, std={self.eta_std:.3f}")

    def _to_cartesian(self):
        """Convert lon/lat to Cartesian (meters)."""
        ref_lon = self.lon.mean()
        ref_lat = self.lat.mean()
        R = 6371000.0

        lon_rad = np.radians(self.lon)
        lat_rad = np.radians(self.lat)
        ref_lon_rad = np.radians(ref_lon)
        ref_lat_rad = np.radians(ref_lat)

        x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        y = R * (lat_rad - ref_lat_rad)

        return x.astype(np.float32), y.astype(np.float32)

    def _prepare_node_features(self):
        """Prepare static node features."""
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1

        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

        features = np.stack([x_norm, y_norm, depth_norm], axis=1)
        return torch.tensor(features, dtype=torch.float32)

    def _compute_edge_features(self):
        """Compute edge features."""
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()

        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)

        char_length = np.median(dist) + 1e-8
        dx_norm = dx / char_length
        dy_norm = dy / char_length
        dist_norm = dist / char_length

        edge_attr = np.stack([dx_norm, dy_norm, dist_norm], axis=1)
        return torch.tensor(edge_attr, dtype=torch.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = idx * self.time_stride

        # Get elevation at t and t+1
        eta_in = self.elevation[t].copy()
        eta_out = self.elevation[t + self.time_stride].copy()

        # Normalize
        if self.normalize:
            eta_in = eta_in / self.eta_scale
            eta_out = eta_out / self.eta_scale

        # Create state (just elevation for now, can add velocity later)
        input_state = eta_in[:, np.newaxis]  # [nodes, 1]
        target_state = eta_out[:, np.newaxis]

        data = Data(
            x=torch.tensor(input_state, dtype=torch.float32),
            y=torch.tensor(target_state, dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_features=self.node_features,
            pos=torch.tensor(np.stack([self.x_cart, self.y_cart], axis=1), dtype=torch.float32),
            depth=torch.tensor(self.depth, dtype=torch.float32),
            timestep=t,
        )

        return data


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
        state_dim: int = 1,  # Just elevation
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

        logger.info(f"RealSTOFSGNN: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x, node_features, edge_index, edge_attr):
        h = self.node_encoder(torch.cat([x, node_features], dim=-1))
        e = self.edge_encoder(edge_attr)

        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        return self.decoder(h)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--time-stride', type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("STOFS US East Coast - Real Data GNN Training")
    print("=" * 70)

    # Check data exists
    mesh_path = Path("data/processed/us_east_coast_mesh.npz")
    elev_path = Path("data/processed/us_east_coast_elevation.npz")

    if not mesh_path.exists():
        print(f"ERROR: Mesh not found: {mesh_path}")
        print("Run: python scripts/extract_us_east_coast.py")
        return

    if not elev_path.exists():
        print(f"ERROR: Elevation data not found: {elev_path}")
        print("Run: python scripts/extract_elevation_subset.py")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataset
    print("\n1. Loading real STOFS data...")
    dataset = RealSTOFSDataset(
        mesh_path=str(mesh_path),
        elevation_path=str(elev_path),
        time_stride=args.time_stride,
        normalize=True,
        eta_scale=2.0,
    )

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Data loaders
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model (state_dim=1 for elevation only)
    print("\n2. Creating model...")
    model = RealSTOFSGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n3. Training for {args.epochs} epochs...")
    print("-" * 50)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    output_dir = Path("outputs")
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        if (epoch + 1) % 5 == 0:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'eta_scale': dataset.eta_scale,
                }, output_dir / 'checkpoints' / 'best_real_stofs.pt')

            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f}")

        scheduler.step()

    print("-" * 50)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'eta_scale': dataset.eta_scale,
    }, output_dir / 'checkpoints' / 'final_real_stofs.pt')

    # Plot training
    print("\n4. Creating visualizations...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    val_epochs = [i * 5 for i in range(1, len(val_losses) + 1)]
    ax.plot(val_epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Real STOFS Data Training - US East Coast')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'real_stofs_training.png', dpi=150)
    plt.close()

    # Test rollout
    print("\n5. Testing rollout...")
    model.eval()

    test_sample = dataset[0]
    initial_state = test_sample.x.to(device)
    node_features = test_sample.node_features.to(device)
    edge_index = test_sample.edge_index.to(device)
    edge_attr = test_sample.edge_attr.to(device)

    predictions = [initial_state.cpu().numpy()]
    current = initial_state

    num_rollout = min(20, len(dataset) - 1)

    with torch.no_grad():
        for _ in range(num_rollout):
            pred = model(current, node_features, edge_index, edge_attr)
            predictions.append(pred.cpu().numpy())
            current = pred

    predictions = np.array(predictions) * dataset.eta_scale  # Denormalize

    # Plot rollout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    lon = dataset.lon
    lat = dataset.lat

    steps = [0, 4, 8, 12, 16, 20]
    steps = [s for s in steps if s < len(predictions)]

    vmax = np.percentile(np.abs(predictions[:, :, 0]), 95)

    for ax, step in zip(axes, steps):
        eta = predictions[step, :, 0]
        scatter = ax.scatter(lon, lat, c=eta, cmap='RdBu_r', s=1, vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Step {step} (t+{step}h)')
        ax.set_aspect('equal')

    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Water Elevation (m)')

    plt.suptitle('Real STOFS Surrogate - Rollout Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'real_stofs_rollout.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"""
Results:
  - Final train loss: {train_losses[-1]:.6f}
  - Best val loss: {best_val_loss:.6f}
  - Model saved: outputs/checkpoints/best_real_stofs.pt

This model was trained on REAL STOFS water elevation data!
It can now predict storm surge dynamics on the US East Coast.

For ensemble generation (Phase 2):
  - Perturb initial conditions
  - Run multiple forward passes
  - Compute ensemble statistics
""")


if __name__ == '__main__':
    main()
