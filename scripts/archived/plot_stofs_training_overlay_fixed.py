#!/usr/bin/env python3
"""
Plot STOFS Original (CWL), Training Data (25k NPZ), and CO-OPS Observations with CORRECT time alignment.

Key insight: The preprocessing skips NOWCAST_HOURS=5 timesteps, so NPZ data corresponds
to CWL data starting from timestep 5 (index 5).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.spatial import KDTree
import requests

# Configuration
FORECAST_DATE = "20251128"
CWL_PATH = f"/mnt/e/Drive2/Good/STOFS_TRAINING_DATA/{FORECAST_DATE}/stofs_2d_glo.t00z.fields.cwl.nc"
NPZ_PATH = f"/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_25k/processed_{FORECAST_DATE}.npz"
MESH_PATH = "/mnt/d/AI_4_STOFS/stofs_surrogate/data/processed_25k/mesh_25k.npz"
NOWCAST_HOURS = 5  # First 5 hours skipped in preprocessing

# Station info
STATIONS = {
    'Atlantic City': {'id': '8534720', 'lon': -74.4187, 'lat': 39.3583},
    'Sandy Hook': {'id': '8531680', 'lon': -74.0073, 'lat': 40.4619},
}

def parse_nc_time(nc_file):
    """Parse NetCDF time to datetime array."""
    time_var = nc_file.variables['time']
    time_units = time_var.units  # "seconds since 2024-04-04 12:00:00"

    # Parse base date from units string
    parts = time_units.split('since')
    base_str = parts[1].strip()
    # Handle STOFS format: "seconds since 2024-04-04 12:00:00         ! NCDASE - BASE_DAT"
    if '!' in base_str:
        base_str = base_str.split('!')[0].strip()
    base_date = datetime.strptime(base_str, "%Y-%m-%d %H:%M:%S")

    # Convert seconds to datetime
    time_seconds = time_var[:]
    datetimes = [base_date + timedelta(seconds=float(s)) for s in time_seconds]
    return datetimes

def fetch_coops_data(station_id, start_dt, end_dt):
    """Fetch CO-OPS observations."""
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        'begin_date': start_dt.strftime('%Y%m%d %H:%M'),
        'end_date': end_dt.strftime('%Y%m%d %H:%M'),
        'station': station_id,
        'product': 'water_level',
        'datum': 'MSL',
        'units': 'metric',
        'time_zone': 'gmt',
        'format': 'json',
        'application': 'stofs_validation'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()

        if 'data' not in data:
            return None, None

        times = []
        values = []
        for record in data['data']:
            try:
                t = datetime.strptime(record['t'], '%Y-%m-%d %H:%M')
                v = float(record['v'])
                times.append(t)
                values.append(v)
            except:
                continue

        return np.array(times), np.array(values)
    except Exception as e:
        print(f"Error fetching CO-OPS data: {e}")
        return None, None

def main():
    print("Loading CWL NetCDF...")
    nc = Dataset(CWL_PATH, 'r')
    cwl_times = parse_nc_time(nc)
    cwl_lon = nc.variables['x'][:]
    cwl_lat = nc.variables['y'][:]
    cwl_zeta = nc.variables['zeta']

    print(f"CWL timesteps: {len(cwl_times)}")
    print(f"CWL time range: {cwl_times[0]} to {cwl_times[-1]}")

    print("\nLoading 25k mesh...")
    mesh = np.load(MESH_PATH)
    global_indices = mesh['global_indices']
    mesh_lon = mesh['lon']
    mesh_lat = mesh['lat']

    print(f"25k mesh nodes: {len(global_indices)}")

    print("\nLoading NPZ training data...")
    npz = np.load(NPZ_PATH)
    npz_elevation = npz['elevation']

    print(f"NPZ shape: {npz_elevation.shape}")

    # NPZ times correspond to CWL times starting from NOWCAST_HOURS
    # Debug verified: NPZ[0] matches CWL[5] exactly (CWL has 1-hour intervals)
    nowcast_timesteps = NOWCAST_HOURS  # 5 hours = 5 timesteps (CWL has 1-hour intervals)
    npz_times = cwl_times[nowcast_timesteps:nowcast_timesteps + len(npz_elevation)]

    print(f"NPZ time range: {npz_times[0]} to {npz_times[-1]}")

    # Find station nodes in 25k mesh
    mesh_coords = np.column_stack([mesh_lon, mesh_lat])
    tree_25k = KDTree(mesh_coords)

    # Also find in full CWL mesh for direct comparison
    cwl_coords = np.column_stack([cwl_lon, cwl_lat])
    tree_cwl = KDTree(cwl_coords)

    station_nodes_25k = {}
    station_nodes_cwl = {}

    for name, info in STATIONS.items():
        coord = np.array([[info['lon'], info['lat']]])

        # Find in 25k mesh
        dist_25k, idx_25k = tree_25k.query(coord)
        station_nodes_25k[name] = idx_25k[0]

        # Find in CWL mesh
        dist_cwl, idx_cwl = tree_cwl.query(coord)
        station_nodes_cwl[name] = idx_cwl[0]

        print(f"\n{name}:")
        print(f"  25k mesh: node {idx_25k[0]}, dist {dist_25k[0]*111:.1f} km")
        print(f"  CWL mesh: node {idx_cwl[0]}, dist {dist_cwl[0]*111:.1f} km")
        print(f"  Global index from 25k: {global_indices[idx_25k[0]]}")

    # Extract time series
    print("\nExtracting time series...")

    # CWL data - use first 48 hours (96 timesteps at 30 min)
    hours_48 = min(96, len(cwl_times))
    cwl_times_48 = cwl_times[:hours_48]

    # NPZ data - first 48 hours from its start
    npz_hours_48 = min(96, len(npz_times))
    npz_times_48 = npz_times[:npz_hours_48]

    # Fetch observations for full range
    start_dt = cwl_times[0]
    end_dt = cwl_times[min(95, len(cwl_times)-1)]

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'STOFS vs Training Data vs Observations (First 48 Hours) - {FORECAST_DATE}', fontsize=14)

    for ax_idx, (name, info) in enumerate(STATIONS.items()):
        ax = axes[ax_idx]

        node_25k = station_nodes_25k[name]
        node_cwl = station_nodes_cwl[name]

        # Extract CWL data (first 48h)
        cwl_series = np.array([cwl_zeta[t, node_cwl] for t in range(hours_48)])

        # Extract NPZ data (first 48h from NPZ)
        npz_series = npz_elevation[:npz_hours_48, node_25k]

        # Verify alignment by computing correlation for overlapping period
        # NPZ starts at timestep 5 of CWL (00:00 UTC on forecast date)
        cwl_overlap = np.array([cwl_zeta[t, node_cwl] for t in range(nowcast_timesteps, nowcast_timesteps + npz_hours_48)])

        rmse = np.sqrt(np.mean((cwl_overlap - npz_series)**2))
        corr = np.corrcoef(cwl_overlap, npz_series)[0, 1]

        print(f"\n{name} verification:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Correlation: {corr:.4f}")

        # Plot CWL (full 48h)
        ax.plot(cwl_times_48, cwl_series, 'b-', linewidth=2, label='STOFS Original (CWL)')

        # Plot NPZ (aligned with correct times)
        ax.plot(npz_times_48, npz_series, 'g--', linewidth=2, label='Training Data (25k NPZ)')

        # Fetch and plot observations
        obs_times, obs_values = fetch_coops_data(info['id'], start_dt, end_dt)
        if obs_times is not None and len(obs_times) > 0:
            ax.plot(obs_times, obs_values, 'r:', linewidth=1.5, alpha=0.8, label='CO-OPS Observations')

        ax.set_title(f'{name} (25k node={node_25k})')
        ax.set_xlabel('Date/Time (UTC)')
        ax.set_ylabel('Water Level (m, MSL)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add stats to plot
        ax.text(0.02, 0.98, f'CWL vs Training: RMSE={rmse:.4f}m, Corr={corr:.4f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    output_path = '/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/figures/stofs_training_aligned_48h.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    nc.close()
    plt.close()

if __name__ == "__main__":
    main()
