#!/usr/bin/env python3
"""
Extract US East Coast subset from STOFS 2D Global mesh.

This script:
1. Loads the full STOFS mesh from NetCDF
2. Extracts nodes within the US East Coast bounding box
3. Subsamples to a manageable size for GPU training
4. Saves the subset for future use
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
MESH_PATH = Path("data/raw/stofs_2d_glo.20251127/stofs_2d_glo_maxele.63.nc")
OUTPUT_DIR = Path("data/processed")

# US East Coast bounding box
# Covers from Florida Keys to Maine, including the continental shelf
US_EAST_COAST_BBOX = {
    'lon_min': -82.0,   # Western bound (Gulf Stream)
    'lon_max': -65.0,   # Eastern bound (offshore)
    'lat_min': 24.0,    # Southern bound (Florida Keys)
    'lat_max': 46.0,    # Northern bound (Maine/Canada)
}

# Target number of nodes for GPU training
TARGET_NODES = 50000


def fast_subsample(coords: np.ndarray, target_n: int, seed: int = 42) -> np.ndarray:
    """
    Fast subsampling using stratified random sampling.

    Divides domain into grid cells and samples proportionally from each.
    Much faster than farthest point sampling for large datasets.
    """
    np.random.seed(seed)
    n_points = len(coords)

    if n_points <= target_n:
        return np.arange(n_points)

    # Create grid for stratified sampling
    n_cells = int(np.sqrt(target_n))

    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

    lon_edges = np.linspace(lon_min, lon_max, n_cells + 1)
    lat_edges = np.linspace(lat_min, lat_max, n_cells + 1)

    # Assign points to cells
    lon_idx = np.digitize(coords[:, 0], lon_edges) - 1
    lat_idx = np.digitize(coords[:, 1], lat_edges) - 1

    # Clip to valid range
    lon_idx = np.clip(lon_idx, 0, n_cells - 1)
    lat_idx = np.clip(lat_idx, 0, n_cells - 1)

    cell_id = lon_idx * n_cells + lat_idx

    # Sample from each cell
    selected = []
    samples_per_cell = max(1, target_n // (n_cells * n_cells))

    for cell in range(n_cells * n_cells):
        cell_mask = cell_id == cell
        cell_indices = np.where(cell_mask)[0]

        if len(cell_indices) > 0:
            n_sample = min(len(cell_indices), samples_per_cell)
            sampled = np.random.choice(cell_indices, n_sample, replace=False)
            selected.extend(sampled)

    # If we need more points, add randomly
    selected = np.array(selected)
    if len(selected) < target_n:
        remaining = np.setdiff1d(np.arange(n_points), selected)
        extra = np.random.choice(remaining, target_n - len(selected), replace=False)
        selected = np.concatenate([selected, extra])
    elif len(selected) > target_n:
        selected = np.random.choice(selected, target_n, replace=False)

    return np.sort(selected)


def main():
    print("=" * 70)
    print("STOFS 2D Global - US East Coast Subset Extraction")
    print("=" * 70)
    print(f"\nMesh file: {MESH_PATH}")
    print(f"Bounding box: {US_EAST_COAST_BBOX}")
    print(f"Target nodes: {TARGET_NODES:,}")

    # Load full mesh
    print("\n1. Loading full STOFS mesh...")
    ds = xr.open_dataset(MESH_PATH)

    lon = ds['x'].values
    lat = ds['y'].values
    depth = ds['depth'].values
    elements = ds['element'].values

    # Check if 1-based indexing
    if elements.min() >= 1:
        elements = elements - 1

    print(f"   Full mesh: {len(lon):,} nodes, {len(elements):,} elements")
    print(f"   Longitude range: [{lon.min():.2f}, {lon.max():.2f}]")
    print(f"   Latitude range: [{lat.min():.2f}, {lat.max():.2f}]")

    # Extract bounding box
    print("\n2. Extracting US East Coast region...")
    bbox = US_EAST_COAST_BBOX

    mask = (
        (lon >= bbox['lon_min']) & (lon <= bbox['lon_max']) &
        (lat >= bbox['lat_min']) & (lat <= bbox['lat_max'])
    )

    node_indices = np.where(mask)[0]
    print(f"   Nodes in bbox: {len(node_indices):,}")

    # Extract subset
    lon_bbox = lon[node_indices]
    lat_bbox = lat[node_indices]
    depth_bbox = depth[node_indices]

    # Subsample if needed
    if len(lon_bbox) > TARGET_NODES:
        print(f"\n3. Subsampling from {len(lon_bbox):,} to {TARGET_NODES:,} nodes...")
        print("   Using stratified random sampling (fast method)...")

        coords = np.stack([lon_bbox, lat_bbox], axis=1)
        subsample_indices = fast_subsample(coords, TARGET_NODES)

        # Map back to original indices
        selected_original = node_indices[subsample_indices]

        lon_final = lon_bbox[subsample_indices]
        lat_final = lat_bbox[subsample_indices]
        depth_final = depth_bbox[subsample_indices]

        print(f"   Selected {len(lon_final):,} nodes")

    else:
        lon_final = lon_bbox
        lat_final = lat_bbox
        depth_final = depth_bbox
        selected_original = node_indices

    # Create mapping for elements
    print("\n4. Filtering elements and building graph...")
    final_to_new = {old: new for new, old in enumerate(selected_original)}

    valid_elements = []
    for elem in elements:
        if all(n in final_to_new for n in elem):
            new_elem = [final_to_new[n] for n in elem]
            valid_elements.append(new_elem)

    elements_final = np.array(valid_elements, dtype=np.int32) if valid_elements else np.zeros((0, 3), dtype=np.int32)
    print(f"   Elements retained: {len(elements_final):,}")

    # Build graph edge index
    edges = set()
    for elem in elements_final:
        for i in range(3):
            n1, n2 = elem[i], elem[(i + 1) % 3]
            edges.add((min(n1, n2), max(n1, n2)))

    edge_list = []
    for n1, n2 in edges:
        edge_list.append([n1, n2])
        edge_list.append([n2, n1])

    if edge_list:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        # If no elements, build edges using k-nearest neighbors
        print("   No elements retained, building graph from k-NN...")
        from scipy.spatial import cKDTree
        tree = cKDTree(np.stack([lon_final, lat_final], axis=1))
        k = 8  # Connect to 8 nearest neighbors
        _, indices = tree.query(np.stack([lon_final, lat_final], axis=1), k=k+1)

        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                edge_list.append([i, j])
        edge_index = np.array(edge_list, dtype=np.int64).T

    print(f"   Graph edges: {edge_index.shape[1]:,}")

    # Save subset
    print("\n5. Saving subset...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "us_east_coast_mesh.npz"
    np.savez_compressed(
        output_file,
        lon=lon_final.astype(np.float32),
        lat=lat_final.astype(np.float32),
        depth=depth_final.astype(np.float32),
        elements=elements_final,
        edge_index=edge_index,
        original_indices=selected_original,
        bbox=np.array([bbox['lon_min'], bbox['lon_max'], bbox['lat_min'], bbox['lat_max']]),
    )
    print(f"   Saved: {output_file}")
    print(f"   File size: {output_file.stat().st_size / 1e6:.1f} MB")

    # Create visualization
    print("\n6. Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Full mesh extent with bbox
    ax1 = axes[0]
    # Sample for plotting (every 100th point)
    sample_idx = np.arange(0, len(lon), 100)
    ax1.scatter(lon[sample_idx], lat[sample_idx], s=0.1, c='lightblue', alpha=0.5, label='Full mesh')
    ax1.plot([bbox['lon_min'], bbox['lon_max'], bbox['lon_max'], bbox['lon_min'], bbox['lon_min']],
             [bbox['lat_min'], bbox['lat_min'], bbox['lat_max'], bbox['lat_max'], bbox['lat_min']],
             'r-', linewidth=2, label='US East Coast bbox')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('STOFS 2D Global Mesh (Full Domain)')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)

    # Right: US East Coast subset colored by depth
    ax2 = axes[1]
    scatter = ax2.scatter(lon_final, lat_final, c=np.clip(depth_final, 0, 500),
                          cmap='viridis_r', s=1, vmin=0, vmax=500)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'US East Coast Subset ({len(lon_final):,} nodes)')
    ax2.set_aspect('equal')
    cbar = plt.colorbar(scatter, ax=ax2, label='Depth (m, clipped at 500)')

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "us_east_coast_mesh.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {fig_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"""
Subset Statistics:
  - Nodes: {len(lon_final):,}
  - Elements: {len(elements_final):,}
  - Edges: {edge_index.shape[1]:,}
  - Longitude: [{lon_final.min():.2f}, {lon_final.max():.2f}]
  - Latitude: [{lat_final.min():.2f}, {lat_final.max():.2f}]
  - Depth: [{depth_final.min():.1f}, {depth_final.max():.1f}] m

Files saved:
  - {output_file}
  - {fig_path}

Next steps:
  1. Download elevation data: stofs_2d_glo_surf.63.nc (~14 GB)
  2. Train GNN on US East Coast subset
""")

    ds.close()


if __name__ == '__main__':
    main()
