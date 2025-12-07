#!/usr/bin/env python3
"""
Compute probability of exceedance for ensemble forecasts.

For storm surge applications, key thresholds might be:
- Minor flooding: 0.5 m above normal
- Moderate flooding: 1.0 m above normal
- Major flooding: 1.5 m above normal

This script computes P(CWL > threshold) at each node and timestep.
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Import model classes from generate_ensemble
from generate_ensemble import CWLGNN, MeshGraphNetBlock, load_model_and_data, run_ensemble_forecast


def compute_exceedance_probability(ensemble_forecasts, thresholds):
    """
    Compute probability of exceeding each threshold.

    Args:
        ensemble_forecasts: (num_members, num_steps, num_nodes)
        thresholds: list of threshold values

    Returns:
        exceedance_probs: dict mapping threshold -> (num_steps, num_nodes) array
    """
    num_members = ensemble_forecasts.shape[0]

    exceedance_probs = {}

    for thresh in thresholds:
        # Count how many members exceed threshold
        exceed_count = np.sum(ensemble_forecasts > thresh, axis=0)
        # Convert to probability
        exceedance_probs[thresh] = exceed_count / num_members

    return exceedance_probs


def compute_return_levels(ensemble_forecasts, return_periods=[2, 5, 10, 20]):
    """
    Compute return levels from ensemble.

    Args:
        ensemble_forecasts: (num_members, num_steps, num_nodes)
        return_periods: list of return periods (in terms of ensemble size)

    Returns:
        return_levels: dict mapping return_period -> (num_steps, num_nodes) array
    """
    num_members = ensemble_forecasts.shape[0]

    return_levels = {}

    for rp in return_periods:
        # Percentile corresponding to return period
        # e.g., 10-member return period = 90th percentile
        percentile = 100 * (1 - 1/rp)
        percentile = min(percentile, 100 * (num_members - 0.5) / num_members)

        return_levels[rp] = np.percentile(ensemble_forecasts, percentile, axis=0)

    return return_levels


def plot_exceedance_maps(data, exceedance_probs, timestep=24, output_path=None):
    """Plot exceedance probability maps for different thresholds."""

    lon = data['lon']
    lat = data['lat']

    thresholds = sorted(exceedance_probs.keys())
    num_thresh = len(thresholds)

    # Custom colormap for probabilities
    colors = ['white', 'lightblue', 'blue', 'yellow', 'orange', 'red', 'darkred']
    prob_cmap = LinearSegmentedColormap.from_list('prob', colors)

    fig, axes = plt.subplots(1, num_thresh, figsize=(5*num_thresh, 5))

    if num_thresh == 1:
        axes = [axes]

    s = 2  # point size

    for i, thresh in enumerate(thresholds):
        ax = axes[i]
        prob = exceedance_probs[thresh][timestep]

        cf = ax.scatter(lon, lat, c=prob, s=s, cmap=prob_cmap, vmin=0, vmax=1)
        ax.set_title(f'P(CWL > {thresh}m)\nt+{timestep}h')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Probability')

    plt.suptitle('Exceedance Probability Maps', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def plot_exceedance_timeseries(exceedance_probs, node_indices, data, output_path=None):
    """Plot exceedance probability time series at selected nodes."""

    lon = data['lon']
    lat = data['lat']
    depth = data['depth']

    thresholds = sorted(exceedance_probs.keys())
    num_nodes = len(node_indices)

    fig, axes = plt.subplots(num_nodes, 1, figsize=(12, 4*num_nodes))

    if num_nodes == 1:
        axes = [axes]

    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(thresholds)))

    for i, node_idx in enumerate(node_indices):
        ax = axes[i]

        for j, thresh in enumerate(thresholds):
            prob = exceedance_probs[thresh][:, node_idx]
            times = np.arange(len(prob))
            ax.plot(times, prob, color=colors[j], linewidth=2, label=f'>{thresh}m')

        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('Probability')
        ax.set_title(f'Node {node_idx}: ({lon[node_idx]:.2f}°, {lat[node_idx]:.2f}°), Depth={depth[node_idx]:.1f}m')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle('Exceedance Probability Time Series', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def plot_max_probability_map(data, exceedance_probs, threshold, output_path=None):
    """Plot maximum probability over all forecast times."""

    lon = data['lon']
    lat = data['lat']

    # Max probability over time
    max_prob = np.max(exceedance_probs[threshold], axis=0)

    # Time of max probability
    time_of_max = np.argmax(exceedance_probs[threshold], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    s = 2

    # Max probability
    ax = axes[0]
    colors = ['white', 'lightblue', 'blue', 'yellow', 'orange', 'red', 'darkred']
    prob_cmap = LinearSegmentedColormap.from_list('prob', colors)
    cf = ax.scatter(lon, lat, c=max_prob, s=s, cmap=prob_cmap, vmin=0, vmax=1)
    ax.set_title(f'Maximum P(CWL > {threshold}m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Max Probability')

    # Time of max
    ax = axes[1]
    cf = ax.scatter(lon, lat, c=time_of_max, s=s, cmap='viridis')
    ax.set_title(f'Hour of Max P(CWL > {threshold}m)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax, label='Forecast Hour')

    plt.suptitle(f'Threshold: {threshold}m', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def identify_high_risk_areas(data, exceedance_probs, threshold, prob_threshold=0.5):
    """
    Identify nodes with high probability of exceeding threshold.

    Args:
        data: Data dictionary
        exceedance_probs: Exceedance probability arrays
        threshold: Water level threshold (m)
        prob_threshold: Probability threshold for "high risk"

    Returns:
        Dictionary with high-risk node information
    """
    # Max probability over forecast period
    max_prob = np.max(exceedance_probs[threshold], axis=0)

    # Find nodes exceeding probability threshold
    high_risk_mask = max_prob >= prob_threshold
    high_risk_indices = np.where(high_risk_mask)[0]

    if len(high_risk_indices) == 0:
        return {
            'num_nodes': 0,
            'indices': [],
            'lon': [],
            'lat': [],
            'max_prob': [],
            'time_of_max': [],
        }

    # Get info for high-risk nodes
    time_of_max = np.argmax(exceedance_probs[threshold], axis=0)

    return {
        'num_nodes': len(high_risk_indices),
        'indices': high_risk_indices,
        'lon': data['lon'][high_risk_indices],
        'lat': data['lat'][high_risk_indices],
        'depth': data['depth'][high_risk_indices],
        'max_prob': max_prob[high_risk_indices],
        'time_of_max': time_of_max[high_risk_indices],
    }


def main():
    parser = argparse.ArgumentParser(description='Compute exceedance probabilities')
    parser.add_argument('--members', type=int, default=50, help='Number of ensemble members')
    parser.add_argument('--steps', type=int, default=48, help='Number of forecast steps')
    parser.add_argument('--start', type=int, default=100, help='Starting timestep index')
    parser.add_argument('--ic-noise', type=float, default=0.1, help='IC noise std (m)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0],
                       help='Exceedance thresholds (m)')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("CWL Exceedance Probability Analysis")
    logger.info("="*60)

    # Load model and data
    logger.info("\nLoading model and data...")
    data = load_model_and_data()

    # Run ensemble
    logger.info(f"\nGenerating {args.members}-member ensemble...")
    results = run_ensemble_forecast(
        data,
        start_idx=args.start,
        num_steps=args.steps,
        num_members=args.members,
        ic_noise_std=args.ic_noise,
        model_noise_std=0.0,
        use_spatial_correlation=True,
        correlation_length=1.0,
    )

    # Compute exceedance probabilities
    logger.info(f"\nComputing exceedance probabilities for thresholds: {args.thresholds}")
    exceedance_probs = compute_exceedance_probability(
        results['ensemble_forecasts'],
        args.thresholds
    )

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXCEEDANCE SUMMARY (t+24h)")
    logger.info("="*60)

    t = min(24, args.steps)
    for thresh in args.thresholds:
        prob = exceedance_probs[thresh][t]
        pct_above_50 = 100 * np.mean(prob > 0.5)
        pct_above_90 = 100 * np.mean(prob > 0.9)
        logger.info(f"  Threshold {thresh:0.1f}m: {pct_above_50:.1f}% nodes P>50%, {pct_above_90:.1f}% nodes P>90%")

    # Identify high-risk areas
    logger.info("\n" + "="*60)
    logger.info("HIGH RISK AREAS (P > 50% for threshold 1.0m)")
    logger.info("="*60)

    if 1.0 in args.thresholds:
        high_risk = identify_high_risk_areas(data, exceedance_probs, threshold=1.0, prob_threshold=0.5)
        logger.info(f"  Number of high-risk nodes: {high_risk['num_nodes']}")
        if high_risk['num_nodes'] > 0:
            logger.info(f"  Longitude range: [{high_risk['lon'].min():.2f}, {high_risk['lon'].max():.2f}]")
            logger.info(f"  Latitude range: [{high_risk['lat'].min():.2f}, {high_risk['lat'].max():.2f}]")

    # Generate plots
    logger.info("\nGenerating plots...")
    output_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures'

    plot_exceedance_maps(
        data, exceedance_probs, timestep=24,
        output_path=f'{output_dir}/exceedance_maps_t24.png'
    )

    # Select some coastal nodes for time series
    coastal_mask = data['depth'] < 20
    coastal_indices = np.where(coastal_mask)[0]
    if len(coastal_indices) > 3:
        sample_nodes = coastal_indices[::len(coastal_indices)//3][:3]
    else:
        sample_nodes = coastal_indices[:3] if len(coastal_indices) > 0 else [0, 1, 2]

    plot_exceedance_timeseries(
        exceedance_probs, sample_nodes.tolist(), data,
        output_path=f'{output_dir}/exceedance_timeseries.png'
    )

    if 1.0 in args.thresholds:
        plot_max_probability_map(
            data, exceedance_probs, threshold=1.0,
            output_path=f'{output_dir}/max_exceedance_1m.png'
        )

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
