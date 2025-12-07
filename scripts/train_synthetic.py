#!/usr/bin/env python3
"""
Train STOFS surrogate on synthetic data.

This script validates the full pipeline before using real STOFS data.

Usage:
    python scripts/train_synthetic.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dataset import SyntheticSWEDataset
from model import STOFSSurrogateGNN, SimpleMeshGraphNet
from trainer import train_stofs_surrogate


def main():
    print("=" * 60)
    print("STOFS Surrogate - Synthetic Data Training Test")
    print("=" * 60)

    # Configuration
    NUM_NODES = 2500       # ~50x50 grid
    NUM_SAMPLES = 500      # Training samples
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    HIDDEN_DIM = 64
    NUM_LAYERS = 4
    LEARNING_RATE = 1e-3

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    dataset = SyntheticSWEDataset(
        num_nodes=NUM_NODES,
        num_samples=NUM_SAMPLES,
        domain_size=(100000, 100000),
        include_velocity=True,
        include_forcing=False,
        seed=42,
    )

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")

    # Get sample to check dimensions
    sample = dataset[0]
    state_dim = sample.x.shape[1]
    node_feature_dim = sample.node_features.shape[1]
    print(f"   State dim: {state_dim} (eta, u, v)")
    print(f"   Node feature dim: {node_feature_dim}")
    print(f"   Num edges: {sample.edge_index.shape[1]}")

    # Create model
    print("\n2. Creating model...")

    # Use simpler model for quick testing
    model = SimpleMeshGraphNet(
        input_dim=state_dim,
        output_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: SimpleMeshGraphNet")
    print(f"   Parameters: {num_params:,}")

    # Train
    print("\n3. Training...")
    output_dir = Path(__file__).parent.parent / 'outputs'

    history = train_stofs_surrogate(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        output_dir=str(output_dir),
        save_every=25,
        eval_every=5,
        mixed_precision=(device == 'cuda'),
        log_every=20,
    )

    # Plot training curves
    print("\n4. Plotting results...")
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_train = list(range(1, len(history['train_losses']) + 1))
    ax.plot(epochs_train, history['train_losses'], 'b-', label='Train Loss', linewidth=2)

    if history['val_losses']:
        epochs_val = [i * 5 for i in range(1, len(history['val_losses']) + 1)]
        ax.plot(epochs_val, history['val_losses'], 'r-', label='Val Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('STOFS Surrogate Training (Synthetic Data)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.tight_layout()
    fig_path = output_dir / 'figures' / 'synthetic_training.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"   Saved training plot: {fig_path}")

    # Test rollout
    print("\n5. Testing rollout...")
    model.eval()
    model = model.to(device)

    # Get test sample
    test_sample = dataset[len(dataset) - 1]
    initial_state = test_sample.x.to(device)
    pos = test_sample.pos.to(device)
    edge_index = test_sample.edge_index.to(device)
    depth = test_sample.depth.to(device)

    # Perform rollout
    num_rollout_steps = 10
    predictions = [initial_state.cpu().numpy()]

    current_state = initial_state
    with torch.no_grad():
        for _ in range(num_rollout_steps):
            pred = model(current_state, pos, edge_index, depth)
            predictions.append(pred.cpu().numpy())
            current_state = pred

    predictions = np.array(predictions)
    print(f"   Rollout shape: {predictions.shape}")

    # Plot rollout
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    node_coords = test_sample.pos.numpy()
    steps_to_show = [0, 2, 4, 6, 8, 10]

    eta_all = predictions[:, :, 0]
    vmax = max(abs(eta_all.min()), abs(eta_all.max()))
    vmin = -vmax

    for ax, step in zip(axes, steps_to_show):
        if step < len(predictions):
            eta = predictions[step, :, 0]
            scatter = ax.scatter(
                node_coords[:, 0] / 1000,
                node_coords[:, 1] / 1000,
                c=eta,
                cmap='RdBu_r',
                vmin=vmin,
                vmax=vmax,
                s=3,
            )
            ax.set_xlabel('X (km)')
            ax.set_ylabel('Y (km)')
            ax.set_title(f'Step {step}')
            ax.set_aspect('equal')

    cbar = fig.colorbar(scatter, ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.1, aspect=40)
    cbar.set_label('Water Surface Elevation (normalized)')

    plt.suptitle('GNN Rollout Prediction', fontsize=14, y=1.02)
    plt.tight_layout()

    rollout_path = output_dir / 'figures' / 'synthetic_rollout.png'
    fig.savefig(rollout_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved rollout plot: {rollout_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal train loss: {history['train_losses'][-1]:.6f}")
    if history['val_losses']:
        print(f"Best val loss: {min(history['val_losses']):.6f}")

    print(f"\nOutputs saved to: {output_dir}")
    print("  - checkpoints/best_model.pt")
    print("  - checkpoints/final_model.pt")
    print("  - figures/synthetic_training.png")
    print("  - figures/synthetic_rollout.png")

    print("\nNext steps:")
    print("  1. Obtain real STOFS/ADCIRC output data")
    print("  2. Place fort.14, fort.63.nc, fort.64.nc in data/raw/")
    print("  3. Run: python scripts/train_stofs.py")


if __name__ == '__main__':
    main()
