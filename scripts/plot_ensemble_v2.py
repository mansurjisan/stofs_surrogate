#!/usr/bin/env python3
"""
Publication-quality ensemble plots from saved ensemble_results.npz.

Creates:
  1. Station spaghetti plots (individual, high-quality)
  2. Spatial maps: mean, spread, exceedance at multiple lead times
  3. Spread-skill reliability diagram
  4. Rank histogram (calibration)
  5. Multi-station summary panel

Usage:
    python scripts/plot_ensemble_v2.py [--run_dir path/to/run_dir]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Stations
STATIONS = {
    'The_Battery': (-74.003, 40.704),
    'Sandy_Hook': (-74.025, 40.474),
    'Atlantic_City': (-74.406, 39.349),
    'Philadelphia_PA': (-75.225, 39.856),
    'Cape_May': (-74.972, 38.974),
    'Lewes_DE': (-75.131, 38.797),
    'Baltimore': (-76.578, 39.267),
    'Annapolis': (-76.480, 38.983),
}

# Color palette
C_CONTROL = '#1f77b4'
C_TRUTH = '#2ca02c'
C_MEAN = '#ff7f0e'
C_MEMBER = '#a0c4e8'
C_CI_INNER = '#5b9bd5'
C_CI_OUTER = '#bdd7ee'


def load_data(run_dir):
    """Load ensemble results."""
    data = np.load(run_dir / 'ensemble_results.npz')
    return {
        'predictions': data['predictions'],   # [members, timesteps, nodes]
        'mean': data['mean'],                  # [timesteps, nodes]
        'std': data['std'],
        'ground_truth': data['ground_truth'],
        'lon': data['lon'],
        'lat': data['lat'],
    }


def get_station_indices(lon, lat):
    indices = {}
    for name, (slon, slat) in STATIONS.items():
        dist = np.sqrt((lon - slon)**2 + (lat - slat)**2)
        indices[name] = np.argmin(dist)
    return indices


def compute_station_metrics(mean_ts, truth_ts):
    """Compute RMSE, correlation, bias for a station timeseries."""
    rmse = np.sqrt(np.mean((mean_ts - truth_ts)**2))
    if np.std(mean_ts) > 1e-8 and np.std(truth_ts) > 1e-8:
        corr = np.corrcoef(mean_ts, truth_ts)[0, 1]
    else:
        corr = 0.0
    bias = np.mean(mean_ts - truth_ts)
    return rmse, corr, bias


# ============================================================
# Plot 1: Individual station spaghetti (publication quality)
# ============================================================
def plot_station_spaghetti(data, station_indices, run_dir):
    """Create individual spaghetti plot for each station.
    Uses control member (m0) as primary forecast line since ensemble
    mean destroys tidal phase coherence."""
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    n_members, n_times, _ = predictions.shape
    hours = np.arange(1, n_times + 1)

    out_dir = run_dir / 'station_plots'
    out_dir.mkdir(exist_ok=True)

    for name, idx in station_indices.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1],
                                        gridspec_kw={'hspace': 0.08})

        truth = ground_truth[:, idx]
        control_ts = predictions[0, :, idx]  # Member 0 = unperturbed control
        std_ts = data['std'][:, idx]

        # Top: spaghetti + CI
        for m in range(1, n_members):  # Skip control in spaghetti
            ax1.plot(hours, predictions[m, :, idx], color=C_MEMBER, alpha=0.25,
                     linewidth=0.6, zorder=1)

        # Percentile-based CI
        p5 = np.percentile(predictions[:, :, idx], 5, axis=0)
        p25 = np.percentile(predictions[:, :, idx], 25, axis=0)
        p75 = np.percentile(predictions[:, :, idx], 75, axis=0)
        p95 = np.percentile(predictions[:, :, idx], 95, axis=0)

        ax1.fill_between(hours, p5, p95, alpha=0.15, color=C_CI_OUTER, label='90% CI', zorder=2)
        ax1.fill_between(hours, p25, p75, alpha=0.25, color=C_CI_INNER, label='50% CI', zorder=3)
        ax1.plot(hours, control_ts, color=C_CONTROL, linewidth=2.0, label='Control (det.)', zorder=5)
        ax1.plot(hours, truth, color=C_TRUTH, linewidth=2.0, label='STOFS Truth', zorder=4)
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle='-', zorder=0)

        rmse, corr, bias = compute_station_metrics(control_ts, truth)
        ax1.set_title(f'{name.replace("_", " ")}', fontsize=14, fontweight='bold')
        ax1.text(0.02, 0.97, f'RMSE: {rmse*100:.1f} cm  |  R: {corr:.3f}  |  Bias: {bias*100:+.1f} cm',
                 transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax1.set_ylabel('Water Level (m)', fontsize=11)
        ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax1.grid(True, alpha=0.2)
        ax1.set_xlim(1, n_times)
        ax1.tick_params(labelbottom=False)

        # Bottom: spread and error (control vs truth)
        error = np.abs(control_ts - truth)
        spread = std_ts
        ax2.plot(hours, error * 100, color='red', linewidth=1.5, alpha=0.8, label='|Error|')
        ax2.plot(hours, spread * 100, color='blue', linewidth=1.5, alpha=0.8, label='Spread (σ)')
        ax2.set_xlabel('Forecast Hour', fontsize=11)
        ax2.set_ylabel('cm', fontsize=11)
        ax2.legend(fontsize=8, loc='upper left', ncol=2)
        ax2.grid(True, alpha=0.2)
        ax2.set_xlim(1, n_times)
        ax2.set_ylim(bottom=0)

        plt.savefig(out_dir / f'station_{name}.png', dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved {len(station_indices)} station plots to {out_dir}")


# ============================================================
# Plot 2: Multi-station summary panel
# ============================================================
def plot_station_panel(data, station_indices, run_dir):
    """8-station summary panel using control member + CI."""
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    n_members, n_times, _ = predictions.shape
    hours = np.arange(1, n_times + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    names = list(station_indices.keys())
    for i, name in enumerate(names[:8]):
        ax = axes[i]
        idx = station_indices[name]

        truth = ground_truth[:, idx]
        control_ts = predictions[0, :, idx]

        p5 = np.percentile(predictions[:, :, idx], 5, axis=0)
        p25 = np.percentile(predictions[:, :, idx], 25, axis=0)
        p75 = np.percentile(predictions[:, :, idx], 75, axis=0)
        p95 = np.percentile(predictions[:, :, idx], 95, axis=0)

        ax.fill_between(hours, p5, p95, alpha=0.15, color=C_CI_OUTER)
        ax.fill_between(hours, p25, p75, alpha=0.25, color=C_CI_INNER)
        ax.plot(hours, control_ts, color=C_CONTROL, linewidth=1.5, label='Control')
        ax.plot(hours, truth, color=C_TRUTH, linewidth=1.5, label='Truth')
        ax.axhline(0, color='gray', linewidth=0.4, linestyle='-')

        rmse, corr, _ = compute_station_metrics(control_ts, truth)
        ax.set_title(f'{name.replace("_", " ")}\nRMSE={rmse*100:.0f}cm  R={corr:.2f}',
                     fontsize=10)
        ax.grid(True, alpha=0.15)
        ax.set_xlim(1, n_times)
        if i >= 4:
            ax.set_xlabel('Hour')
        if i % 4 == 0:
            ax.set_ylabel('WL (m)')
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(f'Ensemble Station Forecasts ({n_members} members)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(run_dir / 'ensemble_station_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_station_panel.png")


# ============================================================
# Plot 3: Spatial maps at multiple lead times
# ============================================================
def plot_spatial_evolution(data, run_dir):
    """Spatial maps of mean, spread, truth at t+6, 12, 24, 48h."""
    lon, lat = data['lon'], data['lat']
    ens_mean = data['mean']
    ens_std = data['std']
    truth = data['ground_truth']
    n_times = ens_mean.shape[0]

    lead_times = [h for h in [6, 12, 24, 48] if h <= n_times]

    fig, axes = plt.subplots(3, len(lead_times), figsize=(4.5 * len(lead_times), 12))
    if len(lead_times) == 1:
        axes = axes[:, np.newaxis]

    vmax_wl = 1.0  # ±1m for water level
    vmax_std = np.percentile(ens_std, 98)  # dynamic for spread

    for j, h in enumerate(lead_times):
        t = h - 1  # 0-indexed

        # Row 1: Ground truth
        ax = axes[0, j]
        sc = ax.scatter(lon, lat, c=truth[t], cmap='RdBu_r', s=0.3,
                        vmin=-vmax_wl, vmax=vmax_wl, rasterized=True)
        ax.set_title(f't+{h}h', fontsize=12, fontweight='bold')
        if j == 0:
            ax.set_ylabel('STOFS Truth', fontsize=11)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

        # Row 2: Ensemble mean
        ax = axes[1, j]
        sc = ax.scatter(lon, lat, c=ens_mean[t], cmap='RdBu_r', s=0.3,
                        vmin=-vmax_wl, vmax=vmax_wl, rasterized=True)
        if j == 0:
            ax.set_ylabel('Ensemble Mean', fontsize=11)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

        # Row 3: Ensemble spread
        ax = axes[2, j]
        sc2 = ax.scatter(lon, lat, c=ens_std[t], cmap='YlOrRd', s=0.3,
                         vmin=0, vmax=vmax_std, rasterized=True)
        if j == 0:
            ax.set_ylabel('Ensemble Spread (σ)', fontsize=11)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)

    # Colorbars
    fig.colorbar(plt.cm.ScalarMappable(cmap='RdBu_r',
                 norm=plt.Normalize(-vmax_wl, vmax_wl)),
                 ax=axes[:2, :].ravel().tolist(), shrink=0.6, label='Water Level (m)',
                 pad=0.02)
    fig.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd',
                 norm=plt.Normalize(0, vmax_std)),
                 ax=axes[2, :].ravel().tolist(), shrink=0.6, label='Std (m)',
                 pad=0.02)

    plt.suptitle('Spatial Evolution: Truth vs Ensemble Mean vs Spread', fontsize=14, fontweight='bold')
    plt.savefig(run_dir / 'ensemble_spatial_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_spatial_evolution.png")


# ============================================================
# Plot 4: Exceedance probability maps
# ============================================================
def plot_exceedance_maps(data, run_dir):
    """Exceedance probability P(η > threshold) at t+6, 24, 48h."""
    predictions = data['predictions']
    lon, lat = data['lon'], data['lat']
    n_times = predictions.shape[1]

    thresholds = [0.3, 0.5, 1.0]
    lead_times = [h for h in [6, 24, 48] if h <= n_times]

    fig, axes = plt.subplots(len(thresholds), len(lead_times),
                              figsize=(4.5 * len(lead_times), 4 * len(thresholds)))
    if len(lead_times) == 1:
        axes = axes[:, np.newaxis]
    if len(thresholds) == 1:
        axes = axes[np.newaxis, :]

    for i, thresh in enumerate(thresholds):
        for j, h in enumerate(lead_times):
            ax = axes[i, j]
            t = h - 1
            prob = (predictions[:, t, :] > thresh).mean(axis=0)

            sc = ax.scatter(lon, lat, c=prob, cmap='YlOrRd', s=0.3, vmin=0, vmax=1,
                           rasterized=True)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=8)

            if i == 0:
                ax.set_title(f't+{h}h', fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'P(η > {thresh}m)', fontsize=11)

    fig.colorbar(plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 1)),
                 ax=axes.ravel().tolist(), shrink=0.6, label='Probability', pad=0.02)

    plt.suptitle('Exceedance Probability Maps', fontsize=14, fontweight='bold')
    plt.savefig(run_dir / 'ensemble_exceedance_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_exceedance_maps.png")


# ============================================================
# Plot 5: Spread-skill diagram
# ============================================================
def plot_spread_skill(data, station_indices, run_dir):
    """Spread-skill reliability: does ensemble spread match actual error?"""
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    n_members, n_times, _ = predictions.shape
    hours = np.arange(1, n_times + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: domain-averaged spread vs RMSE over lead time
    domain_rmse = np.sqrt(np.mean((data['mean'] - ground_truth)**2, axis=1))
    domain_spread = np.mean(data['std'], axis=1)

    ax1.plot(hours, domain_rmse * 100, 'r-', linewidth=2, label='RMSE (mean vs truth)')
    ax1.plot(hours, domain_spread * 100, 'b-', linewidth=2, label='Ensemble spread (σ)')
    ax1.set_xlabel('Forecast Hour', fontsize=11)
    ax1.set_ylabel('cm', fontsize=11)
    ax1.set_title('Domain-Averaged Spread vs Skill', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(1, n_times)
    ax1.set_ylim(bottom=0)

    # Right: station-level scatter at multiple lead times
    lead_times_h = [h for h in [6, 12, 24, 48] if h <= n_times]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lead_times_h)))

    for k, h in enumerate(lead_times_h):
        t = h - 1
        spreads = []
        errors = []
        for name, idx in station_indices.items():
            spreads.append(data['std'][t, idx] * 100)
            errors.append(np.abs(data['mean'][t, idx] - ground_truth[t, idx]) * 100)
        ax2.scatter(spreads, errors, c=[colors[k]], s=40, alpha=0.8, label=f't+{h}h', zorder=3)

    # 1:1 line
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='1:1 line')
    ax2.set_xlabel('Ensemble Spread (cm)', fontsize=11)
    ax2.set_ylabel('Absolute Error (cm)', fontsize=11)
    ax2.set_title('Station Spread vs Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(run_dir / 'ensemble_spread_skill.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_spread_skill.png")


# ============================================================
# Plot 6: Rank histogram (calibration)
# ============================================================
def plot_rank_histogram(data, station_indices, run_dir):
    """Rank histogram to assess ensemble calibration."""
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    n_members = predictions.shape[0]

    # Collect ranks across all stations and times
    ranks = []
    for name, idx in station_indices.items():
        for t in range(predictions.shape[1]):
            member_vals = predictions[:, t, idx]
            truth_val = ground_truth[t, idx]
            rank = np.sum(member_vals < truth_val)
            ranks.append(rank)

    ranks = np.array(ranks)
    bins = np.arange(n_members + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ranks, bins=bins, color='steelblue', edgecolor='black', alpha=0.8, density=True)
    ax.axhline(1.0 / (n_members + 1), color='red', linestyle='--', linewidth=1.5,
               label=f'Uniform (1/{n_members+1} = {1/(n_members+1):.3f})')
    ax.set_xlabel('Rank of Truth in Ensemble', fontsize=11)
    ax.set_ylabel('Relative Frequency', fontsize=11)
    ax.set_title('Rank Histogram (All Stations, All Lead Times)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(run_dir / 'ensemble_rank_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_rank_histogram.png")


# ============================================================
# Plot 7: Improved dashboard
# ============================================================
def plot_dashboard(data, station_indices, run_dir):
    """Clean summary dashboard."""
    predictions = data['predictions']
    ground_truth = data['ground_truth']
    lon, lat = data['lon'], data['lat']
    n_members, n_times, n_nodes = predictions.shape
    hours = np.arange(1, n_times + 1)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Top left: spatial mean at t+6h
    t_map = min(5, n_times - 1)
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(lon, lat, c=data['mean'][t_map], cmap='RdBu_r', s=0.5,
                    vmin=-1, vmax=1, rasterized=True)
    ax.set_title(f'Ensemble Mean — t+{t_map+1}h', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='WL (m)', shrink=0.8)

    # Top middle: spatial spread at t+6h
    ax = fig.add_subplot(gs[0, 1])
    vmax_std = np.percentile(data['std'], 98)
    sc = ax.scatter(lon, lat, c=data['std'][t_map], cmap='YlOrRd', s=0.5,
                    vmin=0, vmax=vmax_std, rasterized=True)
    ax.set_title(f'Ensemble Spread — t+{t_map+1}h', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    fig.colorbar(sc, ax=ax, label='σ (m)', shrink=0.8)

    # Top right: domain-avg spread vs RMSE
    ax = fig.add_subplot(gs[0, 2])
    domain_rmse = np.sqrt(np.mean((data['mean'] - ground_truth)**2, axis=1))
    domain_spread = np.mean(data['std'], axis=1)
    ax.plot(hours, domain_rmse * 100, 'r-', linewidth=2, label='RMSE')
    ax.plot(hours, domain_spread * 100, 'b-', linewidth=2, label='Spread')
    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('cm')
    ax.set_title('Spread vs Skill (Domain Avg)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1, n_times)
    ax.set_ylim(bottom=0)

    # Bottom row: 3 key stations
    key_stations = ['The_Battery', 'Baltimore', 'Atlantic_City']
    for i, name in enumerate(key_stations):
        ax = fig.add_subplot(gs[1, i])
        idx = station_indices[name]
        truth = ground_truth[:, idx]
        control_ts = predictions[0, :, idx]

        p5 = np.percentile(predictions[:, :, idx], 5, axis=0)
        p25 = np.percentile(predictions[:, :, idx], 25, axis=0)
        p75 = np.percentile(predictions[:, :, idx], 75, axis=0)
        p95 = np.percentile(predictions[:, :, idx], 95, axis=0)

        for m in range(1, n_members):
            ax.plot(hours, predictions[m, :, idx], color=C_MEMBER, alpha=0.2, linewidth=0.4)
        ax.fill_between(hours, p5, p95, alpha=0.12, color=C_CI_OUTER, label='90% CI')
        ax.fill_between(hours, p25, p75, alpha=0.2, color=C_CI_INNER, label='50% CI')
        ax.plot(hours, control_ts, color=C_CONTROL, linewidth=2, label='Control')
        ax.plot(hours, truth, color=C_TRUTH, linewidth=2, label='Truth')
        ax.axhline(0, color='gray', linewidth=0.4)

        rmse, corr, _ = compute_station_metrics(control_ts, truth)
        ax.set_title(f'{name.replace("_", " ")}\nRMSE={rmse*100:.0f}cm  R={corr:.2f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Forecast Hour')
        if i == 0:
            ax.set_ylabel('Water Level (m)')
        ax.grid(True, alpha=0.15)
        ax.set_xlim(1, n_times)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.suptitle(f'Ensemble Forecast Dashboard ({n_members} members, {n_times}h)',
                 fontsize=14, fontweight='bold')
    plt.savefig(run_dir / 'ensemble_dashboard_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: ensemble_dashboard_v2.png")


def main():
    parser = argparse.ArgumentParser(description='Generate ensemble plots from saved results')
    parser.add_argument('--run_dir', type=str,
                        default='/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/ensemble_v2/run_20250120_checkpoint_epoch_95')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logger.info(f"Loading data from {run_dir}")
    data = load_data(run_dir)

    n_members, n_times, n_nodes = data['predictions'].shape
    logger.info(f"Ensemble: {n_members} members, {n_times} timesteps, {n_nodes} nodes")

    station_indices = get_station_indices(data['lon'], data['lat'])

    logger.info("Generating plots...")
    plot_station_spaghetti(data, station_indices, run_dir)
    plot_station_panel(data, station_indices, run_dir)
    plot_spatial_evolution(data, run_dir)
    plot_exceedance_maps(data, run_dir)
    plot_spread_skill(data, station_indices, run_dir)
    plot_rank_histogram(data, station_indices, run_dir)
    plot_dashboard(data, station_indices, run_dir)

    logger.info(f"\nAll plots saved to: {run_dir}")


if __name__ == '__main__':
    main()
