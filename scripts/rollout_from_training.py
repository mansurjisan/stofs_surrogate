#!/usr/bin/env python3
"""
Rollout Visualization using the exact model and data processing from training script.
This ensures feature computation matches exactly.
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.tri as mtri
from pathlib import Path
import argparse

# Import model and dataset from training script
from train_80k_h100_fixed import (
    BatchedTemporalMemoryGNN,
    InMemoryDataset,
    ETA_SCALE,
    WIND_SCALE,
)


def run_rollout(model, dataset, start_idx, num_steps, device='cpu'):
    """
    Run multi-step rollout using the exact same data processing as training.

    Args:
        model: The trained GNN model
        dataset: InMemoryDataset instance
        start_idx: Starting sample index
        num_steps: Number of rollout steps
        device: Computing device

    Returns:
        predictions: list of [N] arrays
        ground_truth: list of [N] arrays
    """
    model.eval()

    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)

    predictions = []
    ground_truth = []

    # Get initial sample
    sample = dataset[start_idx]

    # Current state (from dataset)
    x = sample['x'].unsqueeze(0).to(device)  # [1, N, 1]
    x_prev = sample['x_prev'].unsqueeze(0).to(device)

    with torch.no_grad():
        for step in range(num_steps):
            sample_idx = start_idx + step
            if sample_idx >= len(dataset):
                break

            sample = dataset[sample_idx]

            # Use model's current state but dataset's forcing/tidal
            if step > 0:
                # Update x_prev and x from predictions
                x_prev = x.clone()
                x = pred.clone()
            else:
                x = sample['x'].unsqueeze(0).to(device)
                x_prev = sample['x_prev'].unsqueeze(0).to(device)

            dxdt = x - x_prev

            tidal = sample['tidal_harmonics'].unsqueeze(0).to(device)
            static = sample['static'].unsqueeze(0).to(device)
            forcing = sample['forcing'].unsqueeze(0).to(device)

            # Forward pass
            pred = model(x, x_prev, dxdt, tidal, static, forcing, edge_index, edge_attr)

            # Store results (denormalize)
            pred_np = pred.squeeze().cpu().numpy() * ETA_SCALE
            target_np = sample['y'].squeeze().cpu().numpy() * ETA_SCALE

            predictions.append(pred_np)
            ground_truth.append(target_np)

    return predictions, ground_truth


def plot_error_over_time(predictions, ground_truth, output_dir):
    """Plot RMSE over rollout steps."""
    num_steps = len(predictions)

    rmse = []
    mae = []

    for t in range(num_steps):
        error = predictions[t] - ground_truth[t]
        rmse.append(np.sqrt((error**2).mean()))
        mae.append(np.abs(error).mean())

    hours = np.arange(1, num_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hours, rmse, 'b-o', label='RMSE', linewidth=2, markersize=4)
    ax.plot(hours, mae, 'g-s', label='MAE', linewidth=2, markersize=4)

    ax.set_xlabel('Forecast Hour', fontsize=12)
    ax.set_ylabel('Error (m)', fontsize=12)
    ax.set_title('80k Node GNN Rollout Error vs Forecast Hour', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add cm scale on right axis
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: x*100, lambda x: x/100))
    ax2.set_ylabel('Error (cm)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'rollout_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    return rmse, mae


def plot_spatial_comparison(dataset, predictions, ground_truth, timesteps, output_dir):
    """Create spatial comparison plots."""
    lon = dataset.lon
    lat = dataset.lat

    # Create triangulation
    triang = mtri.Triangulation(lon, lat)

    for idx, t in enumerate(timesteps):
        if t >= len(predictions):
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        pred = predictions[t]
        truth = ground_truth[t]
        error = pred - truth

        # Color scale
        vmin = min(np.nanpercentile(pred, 1), np.nanpercentile(truth, 1))
        vmax = max(np.nanpercentile(pred, 99), np.nanpercentile(truth, 99))

        # Ground truth
        ax = axes[0]
        tcf = ax.tricontourf(triang, truth, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label='Water Level (m)')
        ax.set_title(f'Ground Truth (t+{t+1}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Prediction
        ax = axes[1]
        tcf = ax.tricontourf(triang, pred, levels=50, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(tcf, ax=ax, label='Water Level (m)')
        ax.set_title(f'GNN Prediction (t+{t+1}h)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Error
        ax = axes[2]
        err_max = max(np.abs(np.nanpercentile(error, 1)), np.abs(np.nanpercentile(error, 99)), 0.01)
        norm = TwoSlopeNorm(vmin=-err_max, vcenter=0, vmax=err_max)
        tcf = ax.tricontourf(triang, error, levels=50, cmap='RdBu_r', norm=norm)
        plt.colorbar(tcf, ax=ax, label='Error (m)')
        rmse = np.sqrt(np.nanmean(error**2))
        ax.set_title(f'Error | RMSE: {rmse:.4f}m ({rmse*100:.2f}cm)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.tight_layout()
        plt.savefig(output_dir / f'spatial_t{t+1:02d}h.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved spatial_t{t+1:02d}h.png")


def main():
    parser = argparse.ArgumentParser(description='Rollout using training script')
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_80k_h100/best_model.pt')
    parser.add_argument('--data-dir', type=str,
                        default='/mnt/f/STOFS_TRAINING_DATA/processed_80k_option_a')
    parser.add_argument('--num-steps', type=int, default=48)
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/d/AI_4_STOFS/stofs_surrogate/plots/rollout_80k_v2')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("80k Node Rollout - Using Training Script Classes")
    print("=" * 60)

    # Load checkpoint
    print("\nLoading checkpoint...")
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    print(f"  Epoch: {ckpt.get('epoch')}")
    print(f"  Val loss: {ckpt.get('val_loss'):.6f}")

    # Create model
    model = BatchedTemporalMemoryGNN(
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data using training script's InMemoryDataset
    print("\nLoading data...")
    data_dir = Path(args.data_dir)
    mesh_path = data_dir / 'mesh.npz'
    mesh = np.load(mesh_path)
    mesh_data = {
        'lon': mesh['lon'],
        'lat': mesh['lat'],
        'depth': mesh['depth'],
        'edge_index': mesh['edge_index'],
    }

    # Find validation dates (last 30%)
    npz_files = sorted(data_dir.glob('processed_*.npz'))
    all_dates = [f.stem.replace('processed_', '') for f in npz_files]
    split_idx = int(len(all_dates) * 0.7)
    val_dates = all_dates[split_idx:][:30]  # Use first 30 val dates

    print(f"  Using {len(val_dates)} validation dates")

    # Load date data
    date_data_list = []
    for date in val_dates[:5]:  # Load 5 dates for testing
        data = np.load(data_dir / f'processed_{date}.npz')
        date_data_list.append({
            'date': date,
            'elevation': data['elevation'],
            'forcing': {
                'u10': data['u10'],
                'v10': data['v10'],
                'pressure': data['pressure'],
            }
        })

    # Create dataset
    dataset = InMemoryDataset(mesh_data, date_data_list)
    print(f"  Dataset: {len(dataset)} samples")

    # Run rollout from middle of first date
    start_idx = 50  # Start from timestep 50
    print(f"\nRunning {args.num_steps}-step rollout from sample {start_idx}...")

    predictions, ground_truth = run_rollout(model, dataset, start_idx, args.num_steps, device)
    print(f"  Completed {len(predictions)} steps")

    # Statistics
    print("\nRollout Statistics:")
    for t in [0, 5, 11, 23, 47]:
        if t < len(predictions):
            error = predictions[t] - ground_truth[t]
            rmse = np.sqrt(np.nanmean(error**2))
            print(f"  t+{t+1:2d}h: RMSE = {rmse:.4f}m ({rmse*100:.2f}cm)")

    # Generate plots
    print("\nGenerating plots...")
    rmse, mae = plot_error_over_time(predictions, ground_truth, output_dir)
    plot_spatial_comparison(dataset, predictions, ground_truth, [0, 5, 11, 23], output_dir)

    print(f"\nPlots saved to {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
