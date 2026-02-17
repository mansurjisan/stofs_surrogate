#!/usr/bin/env python3
"""
Visualize Long-Range Edges Added to the 25k Mesh

Shows the original mesh with new long-range connections overlaid.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
MESH_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2_longrange')
ORIGINAL_MESH = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2/mesh.npz')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get('STOFS_OUTPUT_DIR', PROJECT_ROOT / 'plots'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Region definitions for coloring
REGIONS = {
    'chesapeake_mouth': {'lon': (-76.2, -75.8), 'lat': (36.9, 37.3), 'color': '#e41a1c'},
    'chesapeake_inner': {'lon': (-76.8, -76.2), 'lat': (38.8, 39.5), 'color': '#377eb8'},
    'chesapeake_mid': {'lon': (-76.6, -76.0), 'lat': (38.2, 38.8), 'color': '#4daf4a'},
    'delaware_mouth': {'lon': (-75.2, -74.8), 'lat': (38.7, 39.1), 'color': '#984ea3'},
    'delaware_inner': {'lon': (-75.6, -75.0), 'lat': (39.5, 40.1), 'color': '#ff7f00'},
    'nj_coast': {'lon': (-74.5, -73.8), 'lat': (39.0, 40.8), 'color': '#ffff33'},
    'ny_harbor': {'lon': (-74.2, -73.8), 'lat': (40.4, 40.8), 'color': '#a65628'},
}


def get_node_region(lon, lat):
    """Determine which region a node belongs to."""
    for name, region in REGIONS.items():
        lon_min, lon_max = region['lon']
        lat_min, lat_max = region['lat']
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return name
    return None


def main():
    print("Loading meshes...")

    # Load enhanced mesh
    enhanced = dict(np.load(MESH_DIR / 'mesh.npz', allow_pickle=True))
    lon = enhanced['lon']
    lat = enhanced['lat']
    depth = enhanced['depth']

    # Load long-range edges info
    lr_data = dict(np.load(MESH_DIR / 'longrange_edges.npz', allow_pickle=True))
    longrange_edges = lr_data['longrange_edges']

    n_nodes = len(lon)
    n_original = int(lr_data['original_edge_count'])
    n_longrange = int(lr_data['longrange_edge_count'])

    print(f"Nodes: {n_nodes:,}")
    print(f"Original edges: {n_original:,}")
    print(f"Long-range edges: {n_longrange:,}")

    # Compute edge distances for long-range edges
    src = longrange_edges[0]
    dst = longrange_edges[1]

    # Haversine distance
    R = 6371  # km
    lat1, lat2 = np.radians(lat[src]), np.radians(lat[dst])
    dlat = lat2 - lat1
    dlon = np.radians(lon[dst] - lon[src])
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 2 * R * np.arcsin(np.sqrt(a))

    # ========================================
    # Figure 1: Overview with all long-range edges
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Full domain
    ax = axes[0]

    # Plot nodes colored by depth
    scatter = ax.scatter(lon, lat, c=np.abs(depth), cmap='Blues', s=0.5, alpha=0.3)

    # Sample long-range edges for visibility (plot every Nth edge)
    sample_rate = max(1, len(src) // 5000)
    for i in range(0, len(src), sample_rate):
        ax.plot([lon[src[i]], lon[dst[i]]], [lat[src[i]], lat[dst[i]]],
                'r-', alpha=0.1, linewidth=0.3)

    # Mark regions
    for name, region in REGIONS.items():
        lon_min, lon_max = region['lon']
        lat_min, lat_max = region['lat']
        rect = plt.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                             fill=False, edgecolor=region['color'], linewidth=2, linestyle='--')
        ax.add_patch(rect)
        ax.text((lon_min + lon_max)/2, lat_max + 0.05, name.replace('_', ' ').title(),
               ha='center', fontsize=7, color=region['color'])

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Long-Range Edges Overview\n({n_longrange:,} new edges, showing 1/{sample_rate})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: Distance histogram
    ax = axes[1]
    ax.hist(distances, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.median(distances), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(distances):.1f} km')
    ax.axvline(np.mean(distances), color='orange', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(distances):.1f} km')
    ax.set_xlabel('Edge Distance (km)')
    ax.set_ylabel('Count')
    ax.set_title('Long-Range Edge Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'longrange_edges_overview.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'longrange_edges_overview.png'}")
    plt.close()

    # ========================================
    # Figure 2: Zoom views of key regions
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    zoom_regions = [
        ('Chesapeake Bay', (-77.0, -75.5), (36.8, 39.6)),
        ('Delaware Bay', (-75.8, -74.5), (38.5, 40.2)),
        ('NY/NJ Coast', (-74.5, -73.5), (39.5, 41.0)),
        ('Bay Mouths', (-76.5, -74.5), (36.5, 39.5)),
    ]

    for ax, (title, lon_range, lat_range) in zip(axes.flat, zoom_regions):
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        # Filter nodes in this region
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

        # Plot nodes
        ax.scatter(lon[mask], lat[mask], c=np.abs(depth[mask]), cmap='Blues', s=2, alpha=0.5)

        # Plot long-range edges in this region
        edge_count = 0
        for i in range(len(src)):
            s, d = src[i], dst[i]
            # Check if either endpoint is in the region
            if ((lon_min <= lon[s] <= lon_max and lat_min <= lat[s] <= lat_max) or
                (lon_min <= lon[d] <= lon_max and lat_min <= lat[d] <= lat_max)):

                # Color by distance
                dist = distances[i]
                if dist < 30:
                    color, alpha = '#2ca02c', 0.4  # Green - short
                elif dist < 80:
                    color, alpha = '#ff7f0e', 0.3  # Orange - medium
                else:
                    color, alpha = '#d62728', 0.2  # Red - long

                ax.plot([lon[s], lon[d]], [lat[s], lat[d]],
                        color=color, alpha=alpha, linewidth=0.5)
                edge_count += 1

        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'{title}\n({edge_count:,} edges shown)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ca02c', linewidth=2, label='< 30 km'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2, label='30-80 km'),
        Line2D([0], [0], color='#d62728', linewidth=2, label='> 80 km'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3,
               title='Edge Distance', bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_DIR / 'longrange_edges_zoom.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'longrange_edges_zoom.png'}")
    plt.close()

    # ========================================
    # Figure 3: Bay connectivity visualization
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all nodes faintly
    ax.scatter(lon, lat, c='lightgray', s=0.5, alpha=0.3)

    # Highlight region nodes
    for name, region in REGIONS.items():
        lon_min, lon_max = region['lon']
        lat_min, lat_max = region['lat']
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        ax.scatter(lon[mask], lat[mask], c=region['color'], s=5, alpha=0.8, label=name.replace('_', ' ').title())

    # Plot inter-region connections
    inter_region_count = 0
    for i in range(len(src)):
        s, d = src[i], dst[i]
        src_region = get_node_region(lon[s], lat[s])
        dst_region = get_node_region(lon[d], lat[d])

        if src_region and dst_region and src_region != dst_region:
            ax.plot([lon[s], lon[d]], [lat[s], lat[d]],
                    'k-', alpha=0.15, linewidth=0.5)
            inter_region_count += 1

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Inter-Region Long-Range Connections\n({inter_region_count:,} connections between defined regions)')
    ax.legend(loc='upper right', markerscale=3, fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'longrange_edges_regions.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'longrange_edges_regions.png'}")
    plt.close()

    # ========================================
    # Summary statistics
    # ========================================
    print("\n" + "="*60)
    print("LONG-RANGE EDGE STATISTICS")
    print("="*60)
    print(f"Total long-range edges: {n_longrange:,}")
    print(f"\nDistance statistics:")
    print(f"  Min:    {distances.min():.1f} km")
    print(f"  Max:    {distances.max():.1f} km")
    print(f"  Mean:   {distances.mean():.1f} km")
    print(f"  Median: {np.median(distances):.1f} km")
    print(f"\nDistance distribution:")
    print(f"  < 30 km:   {(distances < 30).sum():,} edges ({100*(distances < 30).sum()/len(distances):.1f}%)")
    print(f"  30-80 km:  {((distances >= 30) & (distances < 80)).sum():,} edges ({100*((distances >= 30) & (distances < 80)).sum()/len(distances):.1f}%)")
    print(f"  > 80 km:   {(distances >= 80).sum():,} edges ({100*(distances >= 80).sum()/len(distances):.1f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
