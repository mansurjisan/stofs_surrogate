#!/usr/bin/env python3
"""
Train GNN surrogate on US East Coast STOFS mesh.

This script uses the real STOFS mesh structure from the US East Coast
with synthetic dynamics for initial testing. Once elevation data is
downloaded, it can be switched to real data.

Usage:
    python scripts/train_us_east_coast.py
    python scripts/train_us_east_coast.py --epochs 100 --batch-size 4
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


class USEastCoastDataset(Dataset):
    """
    Dataset using real STOFS US East Coast mesh with synthetic dynamics.

    The mesh structure (nodes, edges) is from real STOFS data.
    Dynamics are synthetic until real elevation data is available.
    """

    def __init__(
        self,
        mesh_path: str = "data/processed/us_east_coast_mesh.npz",
        num_samples: int = 500,
        use_real_data: bool = False,
        elevation_path: str = None,
        seed: int = 42,
    ):
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load mesh
        logger.info(f"Loading mesh from {mesh_path}")
        mesh = np.load(mesh_path)

        self.lon = mesh['lon']
        self.lat = mesh['lat']
        self.depth = mesh['depth']
        self.edge_index = torch.tensor(mesh['edge_index'], dtype=torch.long)
        self.original_indices = mesh['original_indices']

        self.num_nodes = len(self.lon)
        self.num_samples = num_samples

        logger.info(f"Mesh: {self.num_nodes:,} nodes, {self.edge_index.shape[1]:,} edges")

        # Convert to Cartesian coordinates
        self.x_cart, self.y_cart = self._to_cartesian()

        # Prepare node features (normalized position + depth)
        self.node_features = self._prepare_node_features()

        # Compute edge features
        self.edge_attr = self._compute_edge_features()

        # Generate synthetic samples or load real data
        if use_real_data and elevation_path:
            self._load_real_data(elevation_path)
        else:
            logger.info("Generating synthetic dynamics on real mesh...")
            self.samples = self._generate_synthetic_samples()

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
        # Normalize position
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1

        # Normalize depth (log scale)
        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

        features = np.stack([x_norm, y_norm, depth_norm], axis=1)
        return torch.tensor(features, dtype=torch.float32)

    def _compute_edge_features(self):
        """Compute edge features (relative position, distance)."""
        src = self.edge_index[0].numpy()
        dst = self.edge_index[1].numpy()

        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)

        # Normalize
        char_length = np.median(dist) + 1e-8
        dx_norm = dx / char_length
        dy_norm = dy / char_length
        dist_norm = dist / char_length

        edge_attr = np.stack([dx_norm, dy_norm, dist_norm], axis=1)
        return torch.tensor(edge_attr, dtype=torch.float32)

    def _generate_synthetic_samples(self):
        """Generate synthetic SWE-like dynamics on real mesh."""
        samples = []
        g = 9.81
        mean_depth = np.maximum(self.depth, 1.0).mean()
        c = np.sqrt(g * mean_depth)

        for _ in range(self.num_samples):
            # Random storm center (within domain)
            lon_range = self.lon.max() - self.lon.min()
            lat_range = self.lat.max() - self.lat.min()

            center_lon = self.lon.min() + np.random.uniform(0.2, 0.8) * lon_range
            center_lat = self.lat.min() + np.random.uniform(0.2, 0.8) * lat_range

            # Storm parameters
            sigma_deg = np.random.uniform(1.0, 3.0)  # degrees
            amplitude = np.random.uniform(0.5, 2.5)  # meters

            # Initial perturbation (Gaussian)
            r2 = (self.lon - center_lon)**2 + (self.lat - center_lat)**2
            sigma2 = sigma_deg**2
            eta_t = amplitude * np.exp(-r2 / (2 * sigma2))

            # Initial velocity (from shallow water relation)
            u_t = 0.1 * (g / c) * eta_t * (self.lon - center_lon) / sigma_deg
            v_t = 0.1 * (g / c) * eta_t * (self.lat - center_lat) / sigma_deg

            # Forward model (decay + spread)
            decay = np.random.uniform(0.90, 0.95)
            spread = np.random.uniform(1.05, 1.15)

            new_sigma2 = sigma2 * spread**2
            eta_t1 = amplitude * decay * np.exp(-r2 / (2 * new_sigma2))
            u_t1 = 0.1 * decay * (g / c) * eta_t1 * (self.lon - center_lon) / (sigma_deg * spread)
            v_t1 = 0.1 * decay * (g / c) * eta_t1 * (self.lat - center_lat) / (sigma_deg * spread)

            # Stack state vectors
            input_state = np.stack([eta_t, u_t, v_t], axis=1).astype(np.float32)
            target_state = np.stack([eta_t1, u_t1, v_t1], axis=1).astype(np.float32)

            samples.append({
                'input': input_state,
                'target': target_state,
            })

        return samples

    def _load_real_data(self, elevation_path: str):
        """Load real elevation data (placeholder for future implementation)."""
        raise NotImplementedError("Real data loading not yet implemented. Download surf.63.nc first.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]

        data = Data(
            x=torch.tensor(sample['input'], dtype=torch.float32),
            y=torch.tensor(sample['target'], dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_features=self.node_features,
            pos=torch.tensor(np.stack([self.x_cart, self.y_cart], axis=1), dtype=torch.float32),
            depth=torch.tensor(self.depth, dtype=torch.float32),
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

        # Edge update
        edge_input = torch.cat([edge_attr, h[row], h[col]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)

        # Aggregate
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_attr_new)

        # Node update
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)

        return h_new, edge_attr_new


class STOFSRegionalGNN(nn.Module):
    """GNN for regional STOFS prediction."""

    def __init__(
        self,
        state_dim: int = 3,
        node_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 6,
    ):
        super().__init__()

        # Encoder
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

        # Processor
        self.layers = nn.ModuleList([
            MeshGraphNetBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        logger.info(f"STOFSRegionalGNN: {sum(p.numel() for p in self.parameters()):,} parameters")

    def forward(self, x, node_features, edge_index, edge_attr):
        # Encode
        h = self.node_encoder(torch.cat([x, node_features], dim=-1))
        e = self.edge_encoder(edge_attr)

        # Process
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        # Decode
        return self.decoder(h)


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(
            batch.x,
            batch.node_features,
            batch.edge_index,
            batch.edge_attr,
        )

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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--samples', type=int, default=500)
    args = parser.parse_args()

    print("=" * 70)
    print("STOFS US East Coast - GNN Surrogate Training")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataset
    print("\n1. Creating dataset...")
    dataset = USEastCoastDataset(
        mesh_path="data/processed/us_east_coast_mesh.npz",
        num_samples=args.samples,
        use_real_data=False,
    )

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Data loaders
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\n2. Creating model...")
    model = STOFSRegionalGNN(
        state_dim=3,
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
    output_dir.mkdir(exist_ok=True)

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
                }, output_dir / 'checkpoints' / 'best_us_east_coast.pt')

            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f}")

        scheduler.step()

    print("-" * 50)
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, output_dir / 'checkpoints' / 'final_us_east_coast.pt')

    # Plot training curve
    print("\n4. Creating visualizations...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    val_epochs = [i * 5 for i in range(1, len(val_losses) + 1)]
    ax.plot(val_epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('US East Coast STOFS Surrogate Training')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'us_east_coast_training.png', dpi=150)
    plt.close()

    # Test rollout
    print("\n5. Testing rollout...")
    model.eval()

    test_sample = dataset[len(dataset) - 1]
    initial_state = test_sample.x.to(device)
    node_features = test_sample.node_features.to(device)
    edge_index = test_sample.edge_index.to(device)
    edge_attr = test_sample.edge_attr.to(device)

    predictions = [initial_state.cpu().numpy()]
    current = initial_state

    with torch.no_grad():
        for _ in range(10):
            pred = model(current, node_features, edge_index, edge_attr)
            predictions.append(pred.cpu().numpy())
            current = pred

    predictions = np.array(predictions)

    # Plot rollout on map
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    lon = dataset.lon
    lat = dataset.lat

    steps = [0, 2, 4, 6, 8, 10]
    vmax = np.abs(predictions[:, :, 0]).max()

    for ax, step in zip(axes, steps):
        eta = predictions[step, :, 0]
        scatter = ax.scatter(lon, lat, c=eta, cmap='RdBu_r', s=1, vmin=-vmax, vmax=vmax)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Step {step}')
        ax.set_aspect('equal')

    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label('Water Elevation (normalized)')

    plt.suptitle('US East Coast STOFS Surrogate - Rollout Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'us_east_coast_rollout.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"""
Results:
  - Final train loss: {train_losses[-1]:.6f}
  - Best val loss: {best_val_loss:.6f}
  - Model saved: outputs/checkpoints/best_us_east_coast.pt

Visualizations:
  - outputs/figures/us_east_coast_training.png
  - outputs/figures/us_east_coast_rollout.png

Next steps for real data:
  1. Download: stofs_2d_glo_surf.63.nc (~14 GB)
  2. Extract elevation time series for US East Coast nodes
  3. Re-train with real dynamics
""")


if __name__ == '__main__':
    main()
