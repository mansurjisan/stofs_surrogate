#!/usr/bin/env python3
"""
Visualize forcing data from preprocessed NPZ files.

Creates plots of:
- Elevation field
- Wind u/v components
- Pressure field
- Wind magnitude and direction
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import matplotlib.tri as mtri


def plot_forcing_snapshot(npz_path: str, timestep: int = 0, output_dir: str = None):
    """
    Create visualization plots for a preprocessed NPZ file.

    Args:
        npz_path: Path to preprocessed NPZ file
        timestep: Which timestep to visualize
        output_dir: Directory to save plots (uses npz_path directory if None)
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    # Extract data
    lon = data['lon']
    lat = data['lat']
    elevation = data['elevation']
    u10 = data['u10']
    v10 = data['v10']
    pressure = data['pressure']
    date = str(data['date'])

    n_times = elevation.shape[0]
    n_nodes = elevation.shape[1]

    print(f"  Date: {date}")
    print(f"  Timesteps: {n_times}")
    print(f"  Nodes: {n_nodes}")
    print(f"  Plotting timestep {timestep}")

    # Get timestep data
    elev = elevation[timestep]
    u = u10[timestep]
    v = v10[timestep]
    p = pressure[timestep]

    # Wind speed
    wind_speed = np.sqrt(u**2 + v**2)

    # Create output directory
    if output_dir is None:
        output_dir = Path(npz_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create triangulation for plotting
    triang = mtri.Triangulation(lon, lat)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'STOFS Forcing Data - {date} (t={timestep})', fontsize=14)

    # 1. Elevation
    ax = axes[0, 0]
    valid_elev = elev[~np.isnan(elev)]
    if len(valid_elev) > 0:
        vmin, vmax = np.percentile(valid_elev, [5, 95])
        tc = ax.tricontourf(triang, np.nan_to_num(elev, nan=0), levels=50,
                            cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(tc, ax=ax, label='Elevation (m)')
    ax.set_title('Water Surface Elevation')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # 2. Wind U component
    ax = axes[0, 1]
    vmax = np.abs(u).max()
    tc = ax.tricontourf(triang, u, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(tc, ax=ax, label='U10 (m/s)')
    ax.set_title('Wind U Component')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # 3. Wind V component
    ax = axes[0, 2]
    vmax = np.abs(v).max()
    tc = ax.tricontourf(triang, v, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(tc, ax=ax, label='V10 (m/s)')
    ax.set_title('Wind V Component')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # 4. Wind Speed
    ax = axes[1, 0]
    tc = ax.tricontourf(triang, wind_speed, levels=50, cmap='YlOrRd')
    plt.colorbar(tc, ax=ax, label='Speed (m/s)')
    ax.set_title('Wind Speed')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    # 5. Wind vectors (subsampled)
    ax = axes[1, 1]
    step = max(1, n_nodes // 500)  # Subsample for clarity
    ax.quiver(lon[::step], lat[::step], u[::step], v[::step],
              wind_speed[::step], cmap='YlOrRd', scale=100)
    ax.set_title('Wind Vectors')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())

    # 6. Pressure (normalized)
    ax = axes[1, 2]
    # Pressure is normalized: (P - 101325) / 3000
    # Convert back to hPa for display
    p_hPa = (p * 30 + 1013.25)  # Scale back to hPa
    tc = ax.tricontourf(triang, p_hPa, levels=50, cmap='viridis')
    plt.colorbar(tc, ax=ax, label='Pressure (hPa)')
    ax.set_title('Surface Pressure')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'forcing_{date}_t{timestep:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return output_path


def plot_forcing_timeseries(npz_path: str, output_dir: str = None):
    """
    Create timeseries plots showing forcing evolution.
    """
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    date = str(data['date'])
    elevation = data['elevation']
    u10 = data['u10']
    v10 = data['v10']
    pressure = data['pressure']

    n_times = elevation.shape[0]

    # Calculate domain-wide statistics
    times = np.arange(n_times)

    # Mean values
    elev_mean = np.nanmean(elevation, axis=1)
    elev_std = np.nanstd(elevation, axis=1)
    u_mean = np.mean(u10, axis=1)
    v_mean = np.mean(v10, axis=1)
    speed_mean = np.mean(np.sqrt(u10**2 + v10**2), axis=1)
    p_mean = np.mean(pressure, axis=1)

    # Create output directory
    if output_dir is None:
        output_dir = Path(npz_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f'Forcing Time Series - {date}', fontsize=14)

    # 1. Elevation
    ax = axes[0, 0]
    ax.fill_between(times, elev_mean - elev_std, elev_mean + elev_std, alpha=0.3)
    ax.plot(times, elev_mean, 'b-', linewidth=1)
    ax.set_xlabel('Timestep (hours)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Water Surface Elevation (mean +/- std)')
    ax.grid(True, alpha=0.3)

    # 2. Wind Speed
    ax = axes[0, 1]
    ax.plot(times, speed_mean, 'r-', linewidth=1)
    ax.set_xlabel('Timestep (hours)')
    ax.set_ylabel('Wind Speed (m/s)')
    ax.set_title('Mean Wind Speed')
    ax.grid(True, alpha=0.3)

    # 3. Wind Components
    ax = axes[1, 0]
    ax.plot(times, u_mean, 'b-', label='U', linewidth=1)
    ax.plot(times, v_mean, 'r-', label='V', linewidth=1)
    ax.set_xlabel('Timestep (hours)')
    ax.set_ylabel('Wind (m/s)')
    ax.set_title('Mean Wind Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Pressure
    ax = axes[1, 1]
    # Convert normalized pressure to hPa
    p_hPa = p_mean * 30 + 1013.25
    ax.plot(times, p_hPa, 'g-', linewidth=1)
    ax.set_xlabel('Timestep (hours)')
    ax.set_ylabel('Pressure (hPa)')
    ax.set_title('Mean Surface Pressure')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = output_dir / f'forcing_timeseries_{date}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Visualize forcing data from preprocessed NPZ files')
    parser.add_argument('--npz', type=str, required=True, help='Path to NPZ file')
    parser.add_argument('--timestep', type=int, default=0, help='Timestep to visualize')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--timeseries', action='store_true', help='Generate timeseries plot')
    args = parser.parse_args()

    if args.timeseries:
        plot_forcing_timeseries(args.npz, args.output)
    else:
        plot_forcing_snapshot(args.npz, args.timestep, args.output)


if __name__ == '__main__':
    main()
