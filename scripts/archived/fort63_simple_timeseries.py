#!/usr/bin/env python3
"""
Generate simple timeseries plots from fort.63 style files at arbitrary coastal locations.

Shows bias-corrected vs non-bias-corrected STOFS2D output on SINGLE plot
(both lines overlaid, no legend needed - simple and clean format).

Usage:
    python fort63_simple_timeseries.py \
        --cwl /path/to/stofs_2d_glo.t12z.fields.cwl.nc \
        --noanomaly /path/to/stofs_2d_glo.t12z.fields.cwl.noanomaly.nc \
        --output-dir offshore_timeseries

Author: Generated for STOFS2D analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import netCDF4 as nc
from datetime import datetime, timedelta
from scipy.spatial import cKDTree
import pandas as pd
import argparse
import os
import sys

# Cartopy for map plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, map panels will be disabled")

# CO-OPS station matching
HAS_COOPS = False
COOPSMatcher = None
try:
    _stofs_path = '/mnt/d/STOFS2D-Analysis/stofs2d-obs-main/stofs2d-obs-main'
    if _stofs_path not in sys.path:
        sys.path.insert(0, _stofs_path)
    from stofs2d_obs.observations import COOPSMatcher
    HAS_COOPS = True
except ImportError as e:
    print(f"Warning: COOPSMatcher not available ({e})")


class Fort63Reader:
    """
    Read and extract timeseries data from fort.63 style NetCDF files.
    """

    def __init__(self, nc_file, build_tree=True):
        self.nc_file = nc_file
        self.ds = nc.Dataset(nc_file, 'r')
        self.x = self.ds.variables['x'][:]
        self.y = self.ds.variables['y'][:]
        self.n_nodes = len(self.x)
        self._parse_time()
        self.tree = None
        if build_tree:
            self._build_tree()

    def _parse_time(self):
        time_var = self.ds.variables['time']
        time_units = time_var.units

        if 'since' in time_units:
            base_date_str = time_units.split('since ')[-1].strip()
            if '!' in base_date_str:
                base_date_str = base_date_str.split('!')[0].strip()
            base_date_str = base_date_str.split('+')[0].strip()

            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
                try:
                    self.base_date = datetime.strptime(base_date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                self.base_date = datetime(1990, 1, 1)
        else:
            self.base_date = datetime(1990, 1, 1)

        time_seconds = time_var[:]
        self.datetimes = [self.base_date + timedelta(seconds=float(t)) for t in time_seconds]
        self.n_times = len(self.datetimes)
        print(f"Time range: {self.datetimes[0]} to {self.datetimes[-1]}")

    def _build_tree(self):
        print(f"Building KDTree for {self.n_nodes:,} nodes...")
        coords = np.column_stack([self.x, self.y])
        self.tree = cKDTree(coords)

    def find_nearest_node(self, lon, lat):
        if self.tree is None:
            self._build_tree()
        dist, idx = self.tree.query([lon, lat])
        return {
            'node_idx': idx,
            'lon': float(self.x[idx]),
            'lat': float(self.y[idx]),
            'distance_km': dist * 111.0
        }

    def get_timeseries(self, lon, lat, location_name=None):
        node_info = self.find_nearest_node(lon, lat)
        node_idx = node_info['node_idx']

        zeta_var = self.ds.variables['zeta']
        zeta_values = zeta_var[:, node_idx]
        valid_mask = ~np.isnan(zeta_values) & ~np.isclose(zeta_values, -99999.0)

        valid_times = np.array(self.datetimes)[valid_mask]
        valid_zeta = zeta_values[valid_mask]

        df = pd.DataFrame({
            'water_level': valid_zeta
        }, index=pd.DatetimeIndex(valid_times, name='time'))

        return {
            'data': df,
            'node_info': node_info,
            'location_name': location_name or f"({lon:.3f}, {lat:.3f})",
            'n_valid': len(valid_zeta),
        }

    def close(self):
        self.ds.close()


def create_simple_plot(location_key, location_info, reader_cwl, reader_noanomaly,
                       output_dir, show_map=True):
    """
    Create a simple plot with both lines overlaid and legend.
    Shows nearest CO-OPS station on map.
    """
    lon = location_info['lon']
    lat = location_info['lat']
    name = location_info['name']

    print(f"\nProcessing: {name} ({lon:.3f}, {lat:.3f})")

    # Extract timeseries
    ts_cwl = reader_cwl.get_timeseries(lon, lat, name)
    ts_noanomaly = reader_noanomaly.get_timeseries(lon, lat, name)

    if ts_cwl['n_valid'] == 0 or ts_noanomaly['n_valid'] == 0:
        print(f"  X No valid data")
        return None

    data_cwl = ts_cwl['data']
    data_noanomaly = ts_noanomaly['data']

    # Find nearest CO-OPS station
    coops_info = None
    if HAS_COOPS:
        try:
            matcher = COOPSMatcher(search_radius=1.5)
            coops_info = matcher.get_best_match(lon, lat)
            if coops_info:
                print(f"  Nearest CO-OPS: {coops_info['name']} ({coops_info['nos_id']}), "
                      f"dist={coops_info['distance']*111:.1f} km")
        except Exception as e:
            print(f"  Warning: Could not find CO-OPS station: {e}")

    # Create figure - timeseries on left, map on right (reduced whitespace)
    if show_map and HAS_CARTOPY:
        fig = plt.figure(figsize=(14, 5))
        # Use gridspec for tighter control of spacing
        gs = fig.add_gridspec(1, 2, wspace=0.05)
        ax_ts = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

        # Map panel - clean, no gridlines
        buffer = 2.0
        ax_map.set_extent([lon - buffer, lon + buffer, lat - buffer, lat + buffer],
                          crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax_map.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

        # Plot timeseries location (red star) - use model node location
        ax_map.plot(ts_cwl['node_info']['lon'], ts_cwl['node_info']['lat'],
                   'r*', markersize=15, transform=ccrs.PlateCarree(),
                   zorder=10, label='Timeseries Location')

        # Plot nearest CO-OPS station (green circle)
        if coops_info:
            ax_map.plot(coops_info['lon'], coops_info['lat'],
                       'go', markersize=10, transform=ccrs.PlateCarree(),
                       zorder=8, label=f"CO-OPS: {coops_info['nos_id']}")
            # Add label for CO-OPS station
            ax_map.annotate(f"{coops_info['nos_id']}",
                           (coops_info['lon'], coops_info['lat']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, transform=ccrs.PlateCarree(),
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))

        ax_map.legend(loc='lower left', fontsize=8)
        ax_map.set_title(f'{name}\n({lon:.3f}, {lat:.3f})', fontsize=11, fontweight='bold')
    else:
        fig, ax_ts = plt.subplots(1, 1, figsize=(12, 5))

    # Timeseries plot - both lines overlaid with legend
    ax_ts.plot(data_noanomaly.index, data_noanomaly['water_level'],
               'r-', linewidth=1.2, alpha=0.9, label='Without Bias Correction')
    ax_ts.plot(data_cwl.index, data_cwl['water_level'],
               'b-', linewidth=1.2, alpha=0.9, label='With Bias Correction')

    ax_ts.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_ts.set_ylabel('Water Level (m)', fontsize=11)
    ax_ts.set_xlabel('Date/Time', fontsize=11)
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc='upper right', fontsize=9)

    # Format x-axis
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Title is just station name
    ax_ts.set_title(name, fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    plot_file = os.path.join(output_dir, f'{location_key}_timeseries.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  + Saved: {plot_file}")

    return {'location_key': location_key, 'name': name}


def combine_plots_to_pdf(plots_dir, output_pdf):
    """Combine all PNG plots into a single PDF file."""
    from PIL import Image

    png_files = sorted([f for f in os.listdir(plots_dir) if f.endswith('.png')])

    if len(png_files) == 0:
        print("No PNG files found")
        return False

    print(f"\nCombining {len(png_files)} plots into PDF...")

    images = []
    for png_file in png_files:
        img_path = os.path.join(plots_dir, png_file)
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    if images:
        images[0].save(output_pdf, save_all=True, append_images=images[1:])
        print(f"PDF saved: {output_pdf}")
        return True

    return False


# Offshore locations (away from observation stations)
OFFSHORE_EAST_COAST = {
    # Gulf of Maine / New England
    'Offshore_Maine_1': {'lon': -68.5, 'lat': 43.5, 'name': 'Offshore Maine 1'},
    'Offshore_Maine_2': {'lon': -69.0, 'lat': 43.0, 'name': 'Offshore Maine 2'},
    'Offshore_Maine_3': {'lon': -67.0, 'lat': 44.0, 'name': 'Offshore Maine 3'},
    'Georges_Bank_1': {'lon': -67.5, 'lat': 41.5, 'name': 'Georges Bank 1'},
    'Georges_Bank_2': {'lon': -68.0, 'lat': 41.0, 'name': 'Georges Bank 2'},
    'Offshore_Cape_Cod_1': {'lon': -69.5, 'lat': 41.0, 'name': 'Offshore Cape Cod 1'},
    'Offshore_Cape_Cod_2': {'lon': -70.0, 'lat': 40.5, 'name': 'Offshore Cape Cod 2'},
    # Mid-Atlantic
    'Mid_Atlantic_Bight_1': {'lon': -73.5, 'lat': 39.0, 'name': 'Mid-Atlantic Bight 1'},
    'Mid_Atlantic_Bight_2': {'lon': -72.5, 'lat': 39.5, 'name': 'Mid-Atlantic Bight 2'},
    'Mid_Atlantic_Bight_3': {'lon': -74.0, 'lat': 38.5, 'name': 'Mid-Atlantic Bight 3'},
    'Delaware_Bay_Offshore': {'lon': -75.0, 'lat': 38.5, 'name': 'Delaware Bay Offshore'},
    'Offshore_New_Jersey': {'lon': -73.5, 'lat': 40.0, 'name': 'Offshore New Jersey'},
    # Virginia / Carolinas
    'Offshore_Virginia_1': {'lon': -75.0, 'lat': 37.0, 'name': 'Offshore Virginia 1'},
    'Offshore_Virginia_2': {'lon': -75.5, 'lat': 36.5, 'name': 'Offshore Virginia 2'},
    'Cape_Hatteras_Offshore_1': {'lon': -75.0, 'lat': 35.5, 'name': 'Cape Hatteras Offshore 1'},
    'Cape_Hatteras_Offshore_2': {'lon': -75.5, 'lat': 35.0, 'name': 'Cape Hatteras Offshore 2'},
    'Offshore_Carolinas_1': {'lon': -77.5, 'lat': 33.5, 'name': 'Offshore Carolinas 1'},
    'Offshore_Carolinas_2': {'lon': -78.0, 'lat': 33.0, 'name': 'Offshore Carolinas 2'},
    'Offshore_Carolinas_3': {'lon': -79.0, 'lat': 32.5, 'name': 'Offshore Carolinas 3'},
    # South Atlantic
    'South_Atlantic_Bight_1': {'lon': -79.5, 'lat': 31.5, 'name': 'South Atlantic Bight 1'},
    'South_Atlantic_Bight_2': {'lon': -80.0, 'lat': 31.0, 'name': 'South Atlantic Bight 2'},
    'Offshore_Georgia': {'lon': -80.5, 'lat': 31.0, 'name': 'Offshore Georgia'},
    'Offshore_Florida_Atlantic': {'lon': -80.0, 'lat': 29.5, 'name': 'Offshore Florida Atlantic'},
}

OFFSHORE_WEST_COAST = {
    # Pacific Northwest
    'Strait_Juan_de_Fuca_1': {'lon': -124.0, 'lat': 48.2, 'name': 'Strait of Juan de Fuca 1'},
    'Strait_Juan_de_Fuca_2': {'lon': -124.5, 'lat': 48.5, 'name': 'Strait of Juan de Fuca 2'},
    'Offshore_Washington_1': {'lon': -125.0, 'lat': 47.5, 'name': 'Offshore Washington 1'},
    'Offshore_Washington_2': {'lon': -124.5, 'lat': 47.0, 'name': 'Offshore Washington 2'},
    'Offshore_Washington_3': {'lon': -125.0, 'lat': 46.5, 'name': 'Offshore Washington 3'},
    # Oregon
    'Offshore_Oregon_1': {'lon': -124.5, 'lat': 44.5, 'name': 'Offshore Oregon 1'},
    'Offshore_Oregon_2': {'lon': -125.0, 'lat': 45.0, 'name': 'Offshore Oregon 2'},
    'Offshore_Oregon_3': {'lon': -124.5, 'lat': 43.5, 'name': 'Offshore Oregon 3'},
    'Offshore_Oregon_4': {'lon': -125.0, 'lat': 42.5, 'name': 'Offshore Oregon 4'},
    # Northern California
    'Offshore_Humboldt_1': {'lon': -124.5, 'lat': 40.5, 'name': 'Offshore Humboldt 1'},
    'Offshore_Humboldt_2': {'lon': -124.5, 'lat': 41.0, 'name': 'Offshore Humboldt 2'},
    'Offshore_Mendocino': {'lon': -124.0, 'lat': 39.5, 'name': 'Offshore Mendocino'},
    'Offshore_Point_Reyes_1': {'lon': -123.5, 'lat': 38.5, 'name': 'Offshore Point Reyes 1'},
    'Offshore_Point_Reyes_2': {'lon': -123.5, 'lat': 38.0, 'name': 'Offshore Point Reyes 2'},
    # Central California
    'Monterey_Bay_Offshore_1': {'lon': -122.5, 'lat': 36.5, 'name': 'Monterey Bay Offshore 1'},
    'Monterey_Bay_Offshore_2': {'lon': -122.0, 'lat': 36.8, 'name': 'Monterey Bay Offshore 2'},
    'Offshore_Big_Sur': {'lon': -122.0, 'lat': 36.0, 'name': 'Offshore Big Sur'},
    'Offshore_Morro_Bay': {'lon': -121.5, 'lat': 35.5, 'name': 'Offshore Morro Bay'},
    # Southern California
    'Santa_Barbara_Channel_1': {'lon': -120.0, 'lat': 34.0, 'name': 'Santa Barbara Channel 1'},
    'Santa_Barbara_Channel_2': {'lon': -119.5, 'lat': 34.0, 'name': 'Santa Barbara Channel 2'},
    'San_Pedro_Channel_1': {'lon': -118.5, 'lat': 33.5, 'name': 'San Pedro Channel 1'},
    'San_Pedro_Channel_2': {'lon': -118.0, 'lat': 33.3, 'name': 'San Pedro Channel 2'},
    'Offshore_San_Diego_1': {'lon': -117.5, 'lat': 32.5, 'name': 'Offshore San Diego 1'},
    'Offshore_San_Diego_2': {'lon': -117.3, 'lat': 32.8, 'name': 'Offshore San Diego 2'},
}


def main():
    parser = argparse.ArgumentParser(
        description='Generate simple overlaid timeseries plots from fort.63 files')

    parser.add_argument('--cwl', required=True,
                        help='Path to bias-corrected (WITH anomaly) NetCDF file')
    parser.add_argument('--noanomaly', required=True,
                        help='Path to non-bias-corrected (WITHOUT anomaly) NetCDF file')
    parser.add_argument('--output-dir', default='offshore_timeseries',
                        help='Output directory for plots')
    parser.add_argument('--coasts', nargs='+', default=['east', 'west'],
                        choices=['east', 'west', 'all'],
                        help='Which coasts to process')
    parser.add_argument('--custom-locations', nargs='+',
                        help='Custom locations as "Name:lon,lat"')
    parser.add_argument('--no-map', action='store_true',
                        help='Omit the map panel')
    parser.add_argument('--output-pdf', default=None,
                        help='Output PDF filename')

    args = parser.parse_args()

    # Build location dictionary
    locations = {}

    if args.custom_locations:
        for loc_str in args.custom_locations:
            try:
                name, coords = loc_str.split(':')
                lon, lat = map(float, coords.split(','))
                locations[name.replace(' ', '_')] = {
                    'lon': lon, 'lat': lat, 'name': name
                }
            except:
                print(f"Warning: Could not parse '{loc_str}'")
    else:
        if 'all' in args.coasts:
            args.coasts = ['east', 'west']
        if 'east' in args.coasts:
            locations.update(OFFSHORE_EAST_COAST)
        if 'west' in args.coasts:
            locations.update(OFFSHORE_WEST_COAST)

    print("="*70)
    print("Fort63 Simple Timeseries (Overlaid)")
    print("="*70)
    print(f"Locations: {len(locations)}")
    print("="*70)

    # Open files
    reader_cwl = Fort63Reader(args.cwl)
    reader_noanomaly = Fort63Reader(args.noanomaly)

    # Process each location
    success = 0
    for loc_key, loc_info in locations.items():
        try:
            result = create_simple_plot(
                loc_key, loc_info,
                reader_cwl, reader_noanomaly,
                args.output_dir,
                show_map=not args.no_map
            )
            if result:
                success += 1
        except Exception as e:
            print(f"  X Error: {e}")

    reader_cwl.close()
    reader_noanomaly.close()

    # Create PDF
    if success > 0:
        pdf_file = args.output_pdf or os.path.join(args.output_dir, 'offshore_timeseries.pdf')
        combine_plots_to_pdf(args.output_dir, pdf_file)

    print(f"\nCompleted: {success}/{len(locations)} locations")
    print(f"Plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
