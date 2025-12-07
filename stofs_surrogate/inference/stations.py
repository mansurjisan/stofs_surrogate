#!/usr/bin/env python3
"""
Extract ensemble forecasts at specific station locations.

This script:
1. Loads station locations (tide gauges, buoys, etc.)
2. Finds nearest model nodes to each station
3. Runs ensemble forecast
4. Extracts and saves ensemble data at station locations
5. Generates station-specific plots
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import logging
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Import from generate_ensemble
from generate_ensemble import CWLGNN, MeshGraphNetBlock, load_model_and_data, run_ensemble_forecast


# ============================================================
# Station Definitions
# ============================================================

# Default stations: Major US East Coast tide gauge locations
DEFAULT_STATIONS = {
    'Boston': {'lon': -71.0503, 'lat': 42.3554, 'id': '8443970'},
    'New_York': {'lon': -74.0142, 'lat': 40.7003, 'id': '8518750'},
    'Atlantic_City': {'lon': -74.4181, 'lat': 39.3550, 'id': '8534720'},
    'Philadelphia': {'lon': -75.1417, 'lat': 39.9333, 'id': '8545240'},
    'Baltimore': {'lon': -76.5783, 'lat': 39.2667, 'id': '8574680'},
    'Norfolk': {'lon': -76.3300, 'lat': 36.9467, 'id': '8638610'},
    'Wilmington_NC': {'lon': -77.9533, 'lat': 34.2267, 'id': '8658120'},
    'Charleston': {'lon': -79.9250, 'lat': 32.7817, 'id': '8665530'},
    'Savannah': {'lon': -80.9017, 'lat': 32.0333, 'id': '8670870'},
    'Jacksonville': {'lon': -81.4317, 'lat': 30.4000, 'id': '8720218'},
    'Miami': {'lon': -80.1317, 'lat': 25.7683, 'id': '8723214'},
    'Key_West': {'lon': -81.8079, 'lat': 24.5508, 'id': '8724580'},
}


def find_nearest_nodes(lon_grid, lat_grid, station_coords):
    """
    Find nearest grid nodes to station locations.

    Args:
        lon_grid, lat_grid: Grid coordinates
        station_coords: List of (lon, lat) tuples

    Returns:
        List of nearest node indices
    """
    # Build KD-tree
    grid_coords = np.column_stack([lon_grid, lat_grid])
    tree = cKDTree(grid_coords)

    nearest_indices = []
    distances = []

    for slon, slat in station_coords:
        dist, idx = tree.query([slon, slat])
        nearest_indices.append(idx)
        distances.append(dist)

    return nearest_indices, distances


def extract_station_data(results, station_indices, station_names):
    """
    Extract ensemble data at station locations.

    Args:
        results: Ensemble forecast results
        station_indices: List of node indices for stations
        station_names: List of station names

    Returns:
        Dictionary with station data
    """
    station_data = {}

    for i, (name, idx) in enumerate(zip(station_names, station_indices)):
        station_data[name] = {
            'node_index': idx,
            'ensemble_mean': results['ensemble_mean'][:, idx],
            'ensemble_std': results['ensemble_std'][:, idx],
            'ensemble_p10': results['ensemble_p10'][:, idx],
            'ensemble_p25': results['ensemble_p25'][:, idx],
            'ensemble_p50': results['ensemble_p50'][:, idx],
            'ensemble_p75': results['ensemble_p75'][:, idx],
            'ensemble_p90': results['ensemble_p90'][:, idx],
            'control': results['control'][:, idx],
            'ground_truth': results['ground_truth'][:, idx],
            'all_members': results['ensemble_forecasts'][:, :, idx],  # (members, time)
        }

    return station_data


def plot_station_ensemble(station_data, station_name, station_info, output_path=None):
    """Plot ensemble forecast for a single station."""

    data = station_data[station_name]
    times = np.arange(len(data['ensemble_mean']))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Panel 1: Spaghetti plot with percentiles
    ax = axes[0]

    # Plot all members
    num_members = data['all_members'].shape[0]
    for m in range(num_members):
        ax.plot(times, data['all_members'][m], color='lightblue', alpha=0.3, linewidth=0.5)

    # Confidence intervals
    ax.fill_between(times, data['ensemble_p10'], data['ensemble_p90'],
                   alpha=0.2, color='blue', label='10-90% CI')
    ax.fill_between(times, data['ensemble_p25'], data['ensemble_p75'],
                   alpha=0.3, color='blue', label='25-75% CI')

    # Mean, control, truth
    ax.plot(times, data['ensemble_mean'], 'b-', linewidth=2, label='Ensemble Mean')
    ax.plot(times, data['control'], 'g--', linewidth=1.5, label='Control')
    ax.plot(times, data['ground_truth'], 'r-', linewidth=2, label='Ground Truth')

    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Water Level (m)')
    ax.set_title(f'{station_name} ({station_info["id"]})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Spread and skill metrics
    ax = axes[1]

    # Compute running RMSE vs truth
    rmse = np.abs(data['ensemble_mean'] - data['ground_truth'])
    spread = data['ensemble_std']

    ax.plot(times, rmse, 'r-', linewidth=2, label='|Mean - Truth|')
    ax.plot(times, spread, 'b-', linewidth=2, label='Ensemble Spread')
    ax.plot(times, data['ensemble_p90'] - data['ensemble_p10'], 'g--',
           linewidth=1.5, label='80% CI Width')

    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Error / Spread (m)')
    ax.set_title('Forecast Uncertainty')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def plot_all_stations_summary(station_data, stations, output_path=None):
    """Plot summary of all stations."""

    num_stations = len(stations)
    times = np.arange(len(list(station_data.values())[0]['ensemble_mean']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Ensemble mean at all stations
    ax = axes[0, 0]
    for name in stations.keys():
        if name in station_data:
            ax.plot(times, station_data[name]['ensemble_mean'], linewidth=1.5, label=name)
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Ensemble Mean (m)')
    ax.set_title('Ensemble Mean Water Level')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: Spread at all stations
    ax = axes[0, 1]
    for name in stations.keys():
        if name in station_data:
            ax.plot(times, station_data[name]['ensemble_std'], linewidth=1.5, label=name)
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('Ensemble Spread (m)')
    ax.set_title('Ensemble Spread (Std Dev)')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 3: Max water level (P90) at each station
    ax = axes[1, 0]
    station_names = list(station_data.keys())
    max_p90 = [np.max(station_data[n]['ensemble_p90']) for n in station_names]
    max_mean = [np.max(station_data[n]['ensemble_mean']) for n in station_names]

    x = np.arange(len(station_names))
    width = 0.35
    ax.bar(x - width/2, max_mean, width, label='Max Mean', color='blue', alpha=0.7)
    ax.bar(x + width/2, max_p90, width, label='Max P90', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(station_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Water Level (m)')
    ax.set_title('Maximum Forecast Water Level')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Average spread vs latitude
    ax = axes[1, 1]
    lats = [stations[n]['lat'] for n in station_names if n in stations]
    avg_spread = [np.mean(station_data[n]['ensemble_std']) for n in station_names]

    ax.scatter(lats, avg_spread, s=100, c='blue', alpha=0.7)
    for i, name in enumerate(station_names):
        if name in stations:
            ax.annotate(name, (lats[i], avg_spread[i]), fontsize=8, ha='left')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Average Spread (m)')
    ax.set_title('Spread vs Latitude')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Station Ensemble Summary', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")

    return fig


def save_station_csv(station_data, stations, output_path):
    """Save station ensemble data to CSV."""

    rows = []

    for name, data in station_data.items():
        times = np.arange(len(data['ensemble_mean']))

        for t in times:
            row = {
                'station': name,
                'station_id': stations.get(name, {}).get('id', ''),
                'lon': stations.get(name, {}).get('lon', np.nan),
                'lat': stations.get(name, {}).get('lat', np.nan),
                'forecast_hour': t,
                'ensemble_mean': data['ensemble_mean'][t],
                'ensemble_std': data['ensemble_std'][t],
                'ensemble_p10': data['ensemble_p10'][t],
                'ensemble_p50': data['ensemble_p50'][t],
                'ensemble_p90': data['ensemble_p90'][t],
                'control': data['control'][t],
                'ground_truth': data['ground_truth'][t],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CSV: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Extract ensemble at stations')
    parser.add_argument('--members', type=int, default=30, help='Number of ensemble members')
    parser.add_argument('--steps', type=int, default=48, help='Number of forecast steps')
    parser.add_argument('--start', type=int, default=100, help='Starting timestep index')
    parser.add_argument('--ic-noise', type=float, default=0.1, help='IC noise std (m)')
    parser.add_argument('--stations-file', type=str, default=None,
                       help='JSON file with custom station definitions')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Station Ensemble Extraction")
    logger.info("="*60)

    # Load stations
    if args.stations_file:
        with open(args.stations_file, 'r') as f:
            stations = json.load(f)
        logger.info(f"Loaded {len(stations)} stations from {args.stations_file}")
    else:
        stations = DEFAULT_STATIONS
        logger.info(f"Using {len(stations)} default East Coast stations")

    # Load model and data
    logger.info("\nLoading model and data...")
    data = load_model_and_data()

    # Find nearest nodes to stations
    station_coords = [(s['lon'], s['lat']) for s in stations.values()]
    station_indices, distances = find_nearest_nodes(data['lon'], data['lat'], station_coords)

    logger.info("\nStation-Node Mapping:")
    for name, idx, dist in zip(stations.keys(), station_indices, distances):
        logger.info(f"  {name}: Node {idx}, Distance {dist:.4f} deg")

    # Filter stations within domain
    valid_stations = {}
    valid_indices = []
    for name, idx, dist in zip(stations.keys(), station_indices, distances):
        if dist < 0.5:  # Within 0.5 degrees
            valid_stations[name] = stations[name]
            valid_indices.append(idx)
        else:
            logger.warning(f"  {name} too far from nearest node ({dist:.2f} deg) - skipping")

    if len(valid_stations) == 0:
        logger.error("No valid stations found within domain!")
        return

    logger.info(f"\n{len(valid_stations)} valid stations found")

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

    # Extract station data
    logger.info("\nExtracting station data...")
    station_data = extract_station_data(results, valid_indices, list(valid_stations.keys()))

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("STATION FORECAST SUMMARY")
    logger.info("="*60)

    for name in valid_stations.keys():
        d = station_data[name]
        max_mean = np.max(d['ensemble_mean'])
        max_p90 = np.max(d['ensemble_p90'])
        avg_spread = np.mean(d['ensemble_std'])
        logger.info(f"  {name:15s}: Max Mean={max_mean:6.3f}m, Max P90={max_p90:6.3f}m, Avg Spread={avg_spread:.3f}m")

    # Generate plots
    logger.info("\nGenerating plots...")
    output_dir = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures'

    # Individual station plots (first 3)
    for i, name in enumerate(list(valid_stations.keys())[:3]):
        plot_station_ensemble(
            station_data, name, valid_stations[name],
            output_path=f'{output_dir}/station_{name.lower()}_ensemble.png'
        )

    # Summary plot
    plot_all_stations_summary(
        station_data, valid_stations,
        output_path=f'{output_dir}/station_summary.png'
    )

    # Save CSV
    if args.save_csv:
        save_station_csv(
            station_data, valid_stations,
            f'{output_dir}/../station_ensemble.csv'
        )

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
