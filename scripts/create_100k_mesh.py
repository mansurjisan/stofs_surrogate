#!/usr/bin/env python3
"""
Create 100k node mesh for Mid-Atlantic region.
Subsamples STOFS-2D Global nodes to create a higher resolution mesh
than the current 25k while staying within single-GPU memory limits.
"""

import numpy as np
import netCDF4 as nc
from scipy.spatial import Delaunay, cKDTree
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mid-Atlantic domain (same as 25k)
LON_MIN, LON_MAX = -77.0, -72.0
LAT_MIN, LAT_MAX = 37.0, 42.0

def create_mesh(stofs_file: str, output_dir: str, target_nodes: int = 100000,
                max_edge_km: float = 50.0):
    """
    Create a subsampled mesh from STOFS-2D Global.

    Args:
        stofs_file: Path to a STOFS NetCDF file (any date, just for coordinates)
        output_dir: Where to save mesh.npz
        target_nodes: Target number of nodes (default 100k)
        max_edge_km: Maximum edge length in km (for filtering bad triangles)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading STOFS mesh from: {stofs_file}")

    # Load STOFS coordinates
    with nc.Dataset(stofs_file) as ds:
        lon_full = ds.variables['x'][:]
        lat_full = ds.variables['y'][:]
        depth_full = ds.variables['depth'][:]

    logger.info(f"Full STOFS mesh: {len(lon_full):,} nodes")

    # Filter to domain
    mask = (lon_full >= LON_MIN) & (lon_full <= LON_MAX) & \
           (lat_full >= LAT_MIN) & (lat_full <= LAT_MAX)

    lon_domain = lon_full[mask]
    lat_domain = lat_full[mask]
    depth_domain = depth_full[mask]

    # Store original indices for later data extraction
    original_indices = np.where(mask)[0]

    logger.info(f"Nodes in domain: {len(lon_domain):,}")
    logger.info(f"  Lon: {lon_domain.min():.2f} to {lon_domain.max():.2f}")
    logger.info(f"  Lat: {lat_domain.min():.2f} to {lat_domain.max():.2f}")

    # Subsample to target number of nodes
    if len(lon_domain) <= target_nodes:
        logger.info(f"Domain has fewer nodes than target, using all {len(lon_domain):,}")
        indices = np.arange(len(lon_domain))
    else:
        # Use farthest point sampling for uniform distribution
        logger.info(f"Subsampling to {target_nodes:,} nodes using farthest point sampling...")
        indices = farthest_point_sampling(lon_domain, lat_domain, target_nodes)

    lon = lon_domain[indices]
    lat = lat_domain[indices]
    depth = depth_domain[indices]
    original_idx = original_indices[indices]  # Map back to full STOFS mesh

    logger.info(f"Subsampled mesh: {len(lon):,} nodes")

    # Create Delaunay triangulation for edges
    logger.info("Creating Delaunay triangulation...")
    points = np.column_stack([lon, lat])
    tri = Delaunay(points)
    triangles = tri.simplices

    logger.info(f"  Triangles: {len(triangles):,}")

    # Extract edges from triangles
    edges = set()
    for t in triangles:
        edges.add((min(t[0], t[1]), max(t[0], t[1])))
        edges.add((min(t[1], t[2]), max(t[1], t[2])))
        edges.add((min(t[0], t[2]), max(t[0], t[2])))

    edges = np.array(list(edges))
    logger.info(f"  Edges (before filtering): {len(edges):,}")

    # Filter edges that are too long (boundary artifacts)
    R = 6371.0  # Earth radius in km
    lat_mid = np.radians(lat.mean())

    src, dst = edges[:, 0], edges[:, 1]
    dx = (lon[dst] - lon[src]) * np.cos(lat_mid) * R * np.pi / 180
    dy = (lat[dst] - lat[src]) * R * np.pi / 180
    edge_lengths = np.sqrt(dx**2 + dy**2)

    valid_edges = edge_lengths < max_edge_km
    edges = edges[valid_edges]
    edge_lengths = edge_lengths[valid_edges]

    logger.info(f"  Edges (after filtering): {len(edges):,}")
    logger.info(f"  Edge lengths: {edge_lengths.min():.2f} to {edge_lengths.max():.2f} km")
    logger.info(f"  Mean edge length: {edge_lengths.mean():.2f} km")

    # Create bidirectional edge index
    edge_index = np.vstack([
        np.concatenate([edges[:, 0], edges[:, 1]]),
        np.concatenate([edges[:, 1], edges[:, 0]])
    ])

    logger.info(f"  Edge index shape: {edge_index.shape}")

    # Save mesh
    output_file = output_dir / 'mesh.npz'
    np.savez(
        output_file,
        lon=lon.astype(np.float32),
        lat=lat.astype(np.float32),
        depth=depth.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
        triangles=triangles.astype(np.int64),
        original_stofs_indices=original_idx.astype(np.int64),  # For data extraction
    )

    logger.info(f"\nMesh saved to: {output_file}")
    logger.info(f"  Nodes: {len(lon):,}")
    logger.info(f"  Edges: {edge_index.shape[1]:,}")

    # Calculate resolution stats
    domain_area = (LON_MAX - LON_MIN) * (LAT_MAX - LAT_MIN)
    node_density = len(lon) / domain_area
    avg_spacing = np.sqrt(1 / node_density) * 111  # Approximate km

    logger.info(f"\nResolution stats:")
    logger.info(f"  Domain: {LON_MIN}° to {LON_MAX}° lon, {LAT_MIN}° to {LAT_MAX}° lat")
    logger.info(f"  Node density: {node_density:.0f} nodes/deg²")
    logger.info(f"  Avg spacing: {avg_spacing:.2f} km")
    logger.info(f"  Original STOFS in domain: 1,126,497 nodes")
    logger.info(f"  Resolution captured: {100 * len(lon) / 1126497:.1f}%")

    return output_file


def farthest_point_sampling(lon, lat, n_samples):
    """
    Farthest point sampling for uniform spatial distribution.
    """
    n_points = len(lon)

    # Normalize coordinates
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min() + 1e-8)
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min() + 1e-8)
    points = np.column_stack([lon_norm, lat_norm])

    # Start with random point
    selected = [np.random.randint(n_points)]
    min_distances = np.full(n_points, np.inf)

    for i in range(1, n_samples):
        # Update distances to nearest selected point
        last_selected = points[selected[-1]]
        distances = np.sqrt(np.sum((points - last_selected)**2, axis=1))
        min_distances = np.minimum(min_distances, distances)

        # Select farthest point
        min_distances[selected] = -1  # Exclude already selected
        next_point = np.argmax(min_distances)
        selected.append(next_point)

        if i % 10000 == 0:
            logger.info(f"    Sampled {i:,}/{n_samples:,} points...")

    return np.array(selected)


def main():
    parser = argparse.ArgumentParser(description='Create 100k mesh for Mid-Atlantic')
    parser.add_argument('--stofs-file', type=str, required=True,
                        help='Path to any STOFS NetCDF file (for coordinates)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for mesh.npz')
    parser.add_argument('--target-nodes', type=int, default=100000,
                        help='Target number of nodes (default: 100000)')
    parser.add_argument('--max-edge-km', type=float, default=50.0,
                        help='Maximum edge length in km (default: 50)')

    args = parser.parse_args()

    create_mesh(
        stofs_file=args.stofs_file,
        output_dir=args.output_dir,
        target_nodes=args.target_nodes,
        max_edge_km=args.max_edge_km,
    )


if __name__ == '__main__':
    main()
