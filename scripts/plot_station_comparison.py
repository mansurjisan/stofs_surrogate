#!/usr/bin/env python3
"""
Plot station timeseries comparison: STOFS vs GNN predictions with datetime x-axis.

Usage:
    python scripts/plot_station_comparison.py --date 20251129
    python scripts/plot_station_comparison.py --date 20251128 --stations Atlantic_City Sandy_Hook
    python scripts/plot_station_comparison.py --date 20251129 --obs  # Include CO-OPS observations
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path
import requests

STATIONS = ['Atlantic_City', 'Sandy_Hook', 'The_Battery', 'Lewes_DE', 'Cape_May']

# CO-OPS station IDs
COOPS_IDS = {
    'Atlantic_City': '8534720',
    'Sandy_Hook': '8531680',
    'The_Battery': '8518750',
    'Lewes_DE': '8557380',
    'Cape_May': '8536110',
}

def load_timeseries(ts_dir, station):
    """Load timeseries data from text file."""
    ts_file = ts_dir / f'{station}_temporal_memory_rollout.txt'

    datetimes = []
    stofs_wl = []
    gnn_wl = []

    with open(ts_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                dt_str = f"{parts[0]} {parts[1]}"
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                datetimes.append(dt)
                stofs_wl.append(float(parts[3]))
                gnn_wl.append(float(parts[4]))

    return np.array(datetimes), np.array(stofs_wl), np.array(gnn_wl)


def fetch_coops_observations(station_id, start_date, end_date):
    """Fetch observations from CO-OPS API."""
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'begin_date': start_date.strftime('%Y%m%d %H:%M'),
        'end_date': end_date.strftime('%Y%m%d %H:%M'),
        'station': station_id,
        'product': 'water_level',
        'datum': 'MSL',
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'json',
        'application': 'stofs_gnn'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if 'data' in data:
            times = []
            values = []
            for record in data['data']:
                try:
                    t = datetime.strptime(record['t'], '%Y-%m-%d %H:%M')
                    v = float(record['v'])
                    times.append(t)
                    values.append(v)
                except (ValueError, KeyError):
                    continue
            return times, values
    except Exception as e:
        print(f"    Warning: Could not fetch obs for {station_id}: {e}")

    return None, None


def plot_station_timeseries(date_str, output_dir, ts_dir, stations=None, include_obs=False):
    """Generate individual station timeseries plots."""
    if stations is None:
        stations = STATIONS

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch observations if requested
    obs_data = {}
    if include_obs:
        print("Fetching CO-OPS observations...")
        start_dt = datetime.strptime(date_str, '%Y%m%d')
        end_dt = start_dt + timedelta(hours=48)
        for station in stations:
            if station in COOPS_IDS:
                times, values = fetch_coops_observations(COOPS_IDS[station], start_dt, end_dt)
                if times:
                    obs_data[station] = {'times': times, 'values': values}
                    print(f"  {station}: {len(values)} observations")

    for station in stations:
        try:
            datetimes, stofs_wl, gnn_wl = load_timeseries(ts_dir, station)
        except FileNotFoundError:
            print(f"Warning: No data for {station}, skipping")
            continue

        # Calculate metrics (GNN vs STOFS)
        valid = ~(np.isnan(stofs_wl) | np.isnan(gnn_wl))
        rmse_stofs = np.sqrt(np.mean((stofs_wl[valid] - gnn_wl[valid])**2))
        corr_stofs = np.corrcoef(stofs_wl[valid], gnn_wl[valid])[0, 1]
        bias_stofs = np.mean(gnn_wl[valid] - stofs_wl[valid])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot STOFS and GNN
        ax.plot(datetimes, stofs_wl, 'k-', linewidth=2, label='STOFS Ground Truth')
        ax.plot(datetimes, gnn_wl, 'b--', linewidth=1.5, label='GNN Prediction')

        # Plot observations if available
        obs_metrics_str = ""
        if station in obs_data:
            obs_times = obs_data[station]['times']
            obs_values = obs_data[station]['values']
            ax.plot(obs_times, obs_values, 'g.', alpha=0.6, markersize=4, label='CO-OPS Obs')

            # Calculate GNN vs Obs metrics (interpolate GNN to obs times)
            gnn_interp = np.interp(
                [t.timestamp() for t in obs_times],
                [t.timestamp() for t in datetimes],
                gnn_wl
            )
            obs_arr = np.array(obs_values)
            valid_obs = ~np.isnan(obs_arr) & ~np.isnan(gnn_interp)
            if valid_obs.sum() > 10:
                rmse_obs = np.sqrt(np.mean((gnn_interp[valid_obs] - obs_arr[valid_obs])**2))
                corr_obs = np.corrcoef(gnn_interp[valid_obs], obs_arr[valid_obs])[0, 1]
                obs_metrics_str = f"\nGNN vs Obs: RMSE {rmse_obs:.3f}m | R {corr_obs:.3f}"

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45, ha='right')

        # Labels and title
        ax.set_xlabel('Date/Time (UTC)', fontsize=11)
        ax.set_ylabel('Water Level (m, MSL)', fontsize=11)
        ax.set_title(f'{station.replace("_", " ")} - STOFS vs GNN Comparison\n'
                     f'GNN vs STOFS: RMSE {rmse_stofs:.3f}m | R {corr_stofs:.3f} | Bias {bias_stofs:.3f}m'
                     f'{obs_metrics_str}',
                     fontsize=12, fontweight='bold')

        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Set y-limits with some padding
        all_values = [stofs_wl.min(), stofs_wl.max(), gnn_wl.min(), gnn_wl.max()]
        if station in obs_data:
            all_values.extend([min(obs_data[station]['values']), max(obs_data[station]['values'])])
        y_min = min(all_values) - 0.2
        y_max = max(all_values) + 0.2
        ax.set_ylim(y_min, y_max)

        plt.tight_layout()

        # Save
        suffix = '_with_obs' if include_obs else ''
        output_path = output_dir / f'{station}_comparison_{date_str}{suffix}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved: {output_path}")
        print(f"  GNN vs STOFS: RMSE {rmse_stofs:.3f}m, R {corr_stofs:.3f}, Bias {bias_stofs:.3f}m")


def plot_all_stations_combined(date_str, output_dir, ts_dir, include_obs=False):
    """Generate combined plot with all stations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch observations if requested
    obs_data = {}
    if include_obs:
        print("Fetching CO-OPS observations for combined plot...")
        start_dt = datetime.strptime(date_str, '%Y%m%d')
        end_dt = start_dt + timedelta(hours=48)
        for station in STATIONS:
            if station in COOPS_IDS:
                times, values = fetch_coops_observations(COOPS_IDS[station], start_dt, end_dt)
                if times:
                    obs_data[station] = {'times': times, 'values': values}

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, station in enumerate(STATIONS):
        if i >= 5:
            break

        ax = axes[i]

        try:
            datetimes, stofs_wl, gnn_wl = load_timeseries(ts_dir, station)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f'No data for {station}', ha='center', va='center')
            continue

        # Calculate metrics
        valid = ~(np.isnan(stofs_wl) | np.isnan(gnn_wl))
        rmse = np.sqrt(np.mean((stofs_wl[valid] - gnn_wl[valid])**2))
        corr = np.corrcoef(stofs_wl[valid], gnn_wl[valid])[0, 1]

        # Plot
        ax.plot(datetimes, stofs_wl, 'k-', linewidth=1.5, label='STOFS')
        ax.plot(datetimes, gnn_wl, 'b--', linewidth=1.2, label='GNN')

        # Plot observations if available
        if station in obs_data:
            ax.plot(obs_data[station]['times'], obs_data[station]['values'],
                   'g.', alpha=0.5, markersize=3, label='Obs')

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_xlabel('Date/Time (UTC)')
        ax.set_ylabel('Water Level (m)')
        ax.set_title(f'{station.replace("_", " ")}\nRMSE: {rmse:.3f}m | R: {corr:.3f}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[5].axis('off')

    obs_str = ' (with Observations)' if include_obs else ''
    plt.suptitle(f'STOFS vs GNN Timeseries Comparison{obs_str} - {date_str}\nModel: best_temporal_memory_model.pt',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    suffix = '_with_obs' if include_obs else ''
    output_path = output_dir / f'all_stations_comparison_{date_str}{suffix}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot station timeseries comparison')
    parser.add_argument('--date', type=str, required=True, help='Date (YYYYMMDD)')
    parser.add_argument('--stations', type=str, nargs='+', default=None,
                        help='Specific stations to plot (default: all)')
    parser.add_argument('--combined-only', action='store_true',
                        help='Only generate combined plot')
    parser.add_argument('--individual-only', action='store_true',
                        help='Only generate individual plots')
    parser.add_argument('--obs', action='store_true',
                        help='Include CO-OPS observations in plots')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    ts_dir = project_root / 'outputs' / 'timeseries' / args.date
    output_dir = project_root / 'outputs' / 'figures' / 'station_comparison'

    if not ts_dir.exists():
        print(f"Error: Timeseries directory not found: {ts_dir}")
        print("Run rollout_temporal_memory_model.py with --save-ts first")
        return

    if not args.combined_only:
        print(f"\nGenerating individual station plots for {args.date}...")
        plot_station_timeseries(args.date, output_dir, ts_dir, args.stations, include_obs=args.obs)

    if not args.individual_only:
        print(f"\nGenerating combined plot for {args.date}...")
        plot_all_stations_combined(args.date, output_dir, ts_dir, include_obs=args.obs)


if __name__ == '__main__':
    main()
