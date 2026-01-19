#!/usr/bin/env python3
"""
Plot GFS forcing data from NPZ files.

Usage:
    python scripts/plot_gfs_npz.py /path/to/gfs_YYYYMMDD_regional.npz
    python scripts/plot_gfs_npz.py /path/to/date_folder/
    python scripts/plot_gfs_npz.py /path/to/gfs_forcing --date 20230108
    python scripts/plot_gfs_npz.py /path/to/npz --timestep 0 12 24 48
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_gfs_data(npz_path):
    """Load GFS data from NPZ file."""
    data = np.load(npz_path)
    return {
        'u10': data['u10'],
        'v10': data['v10'],
        'sp': data['sp'],
        'fhr': data['fhr'],
        'lat': data['lat'],
        'lon': data['lon'],
        'date': str(data['date']) if 'date' in data.files else npz_path.stem.split('_')[1],
    }


def plot_timestep(data, timestep_idx, output_path=None):
    """Plot wind and pressure for a single timestep."""
    lat = data['lat']
    lon = data['lon']
    fhr = data['fhr'][timestep_idx]
    date = data['date']

    u10 = data['u10'][timestep_idx]
    v10 = data['v10'][timestep_idx]
    sp = data['sp'][timestep_idx]

    # Calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'GFS Forcing - {date} f{fhr:03d}', fontsize=14, fontweight='bold')

    # 1. Wind Speed
    ax = axes[0, 0]
    im = ax.pcolormesh(lon, lat, wind_speed, cmap='viridis', shading='auto')
    plt.colorbar(im, ax=ax, label='m/s')
    ax.set_title(f'Wind Speed (max: {wind_speed.max():.1f} m/s)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. Surface Pressure (convert to hPa)
    ax = axes[0, 1]
    sp_hpa = sp / 100  # Pa to hPa
    im = ax.pcolormesh(lon, lat, sp_hpa, cmap='coolwarm', shading='auto')
    plt.colorbar(im, ax=ax, label='hPa')
    ax.set_title(f'Surface Pressure ({sp_hpa.min():.0f}-{sp_hpa.max():.0f} hPa)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 3. U10 wind component
    ax = axes[1, 0]
    im = ax.pcolormesh(lon, lat, u10, cmap='RdBu_r', shading='auto',
                       vmin=-np.abs(u10).max(), vmax=np.abs(u10).max())
    plt.colorbar(im, ax=ax, label='m/s')
    ax.set_title('U10 (East-West Wind)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 4. V10 wind component
    ax = axes[1, 1]
    im = ax.pcolormesh(lon, lat, v10, cmap='RdBu_r', shading='auto',
                       vmin=-np.abs(v10).max(), vmax=np.abs(v10).max())
    plt.colorbar(im, ax=ax, label='m/s')
    ax.set_title('V10 (North-South Wind)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_wind_vectors(data, timestep_idx, output_path=None, skip=3):
    """Plot wind vectors with pressure contours."""
    lat = data['lat']
    lon = data['lon']
    fhr = data['fhr'][timestep_idx]
    date = data['date']

    u10 = data['u10'][timestep_idx]
    v10 = data['v10'][timestep_idx]
    sp = data['sp'][timestep_idx]
    wind_speed = np.sqrt(u10**2 + v10**2)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Pressure contours
    sp_hpa = sp / 100
    contours = ax.contour(lon, lat, sp_hpa, levels=15, colors='gray', linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

    # Wind speed color mesh
    im = ax.pcolormesh(lon, lat, wind_speed, cmap='YlOrRd', shading='auto', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Wind Speed (m/s)', shrink=0.8)

    # Wind vectors (subsampled)
    ax.quiver(lon[::skip, ::skip], lat[::skip, ::skip],
              u10[::skip, ::skip], v10[::skip, ::skip],
              scale=200, width=0.003, color='black', alpha=0.8)

    ax.set_title(f'GFS Wind & Pressure - {date} f{fhr:03d}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_timeseries(data, output_path=None):
    """Plot time series of domain-averaged values."""
    fhr = data['fhr']

    # Domain averages
    wind_speed = np.sqrt(data['u10']**2 + data['v10']**2)
    mean_wind = wind_speed.mean(axis=(1, 2))
    max_wind = wind_speed.max(axis=(1, 2))
    mean_pressure = data['sp'].mean(axis=(1, 2)) / 100  # hPa

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'GFS Forcing Time Series - {data["date"]}', fontsize=14, fontweight='bold')

    # Wind
    ax = axes[0]
    ax.plot(fhr, mean_wind, 'b-', label='Mean Wind Speed', linewidth=2)
    ax.plot(fhr, max_wind, 'r--', label='Max Wind Speed', linewidth=1)
    ax.set_ylabel('Wind Speed (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Wind Speed')

    # Pressure
    ax = axes[1]
    ax.plot(fhr, mean_pressure, 'g-', linewidth=2)
    ax.set_ylabel('Pressure (hPa)')
    ax.set_xlabel('Forecast Hour')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Mean Surface Pressure ({mean_pressure.min():.0f}-{mean_pressure.max():.0f} hPa)')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot GFS forcing data from NPZ files')
    parser.add_argument('path', help='Path to NPZ file, date folder, or GFS directory')
    parser.add_argument('--date', type=str, help='Date to plot (YYYYMMDD)')
    parser.add_argument('--timestep', type=int, nargs='+', default=[0],
                        help='Timestep indices to plot (default: 0)')
    parser.add_argument('--output-dir', type=str, help='Output directory for plots')
    parser.add_argument('--vectors', action='store_true', help='Plot wind vectors')
    parser.add_argument('--timeseries', action='store_true', help='Plot time series')
    parser.add_argument('--all-timesteps', action='store_true', help='Plot all timesteps')
    args = parser.parse_args()

    path = Path(args.path)

    # Find NPZ file
    if path.suffix == '.npz':
        npz_file = path
    elif path.is_dir():
        # Check if it's a date folder or parent folder
        npz_files = list(path.glob('gfs_*_regional.npz'))
        if npz_files:
            npz_file = npz_files[0]
        elif args.date:
            npz_file = path / args.date / f'gfs_{args.date}_regional.npz'
        else:
            # Find first available
            for d in sorted(path.iterdir()):
                if d.is_dir():
                    npz_files = list(d.glob('gfs_*_regional.npz'))
                    if npz_files:
                        npz_file = npz_files[0]
                        break
            else:
                print("No NPZ files found!")
                return
    else:
        print(f"Path not found: {path}")
        return

    print(f"Loading: {npz_file}")
    data = load_gfs_data(npz_file)

    print(f"Date: {data['date']}")
    print(f"Shape: {data['u10'].shape} (time, lat, lon)")
    print(f"Forecast hours: {data['fhr'][0]} to {data['fhr'][-1]}")
    print(f"Pressure range: {data['sp'].min():.0f} - {data['sp'].max():.0f} Pa")

    # Setup output
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # Determine timesteps to plot
    if args.all_timesteps:
        timesteps = list(range(len(data['fhr'])))
    else:
        timesteps = args.timestep

    # Plot
    for t_idx in timesteps:
        if t_idx >= len(data['fhr']):
            print(f"Timestep {t_idx} out of range (max: {len(data['fhr'])-1})")
            continue

        fhr = data['fhr'][t_idx]

        if output_dir:
            out_path = output_dir / f"gfs_{data['date']}_f{fhr:03d}.png"
        else:
            out_path = None

        print(f"Plotting timestep {t_idx} (f{fhr:03d})...")

        if args.vectors:
            plot_wind_vectors(data, t_idx, out_path)
        else:
            plot_timestep(data, t_idx, out_path)

    # Time series
    if args.timeseries:
        if output_dir:
            out_path = output_dir / f"gfs_{data['date']}_timeseries.png"
        else:
            out_path = None
        plot_timeseries(data, out_path)


if __name__ == '__main__':
    main()
