#!/usr/bin/env python3
"""
Create Long-Range Enhanced Mesh for STOFS-GNN

Adds strategic long-range edges to the existing 25k mesh to improve
information propagation without changing the model architecture.

Strategies:
1. Bay mouth to inner bay connections (tidal propagation)
2. Along-coast connections (storm surge propagation)
3. Cross-bay connections (wind-driven setup)
4. Sparse k-nearest distant neighbors

This is simpler than multi-scale GNN and can be used with existing trained models.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
OUTPUT_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2_longrange')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Long-range edge parameters
MIN_DISTANCE_KM = 20      # Minimum distance for "long-range" (skip local)
MAX_DISTANCE_KM = 150     # Maximum distance for connections
K_NEIGHBORS = 5           # Number of distant neighbors per node
COASTAL_BOOST = 2         # Extra connections for coastal nodes

# Region definitions (approximate bounding boxes)
REGIONS = {
    'chesapeake_mouth': {'lon': (-76.2, -75.8), 'lat': (36.9, 37.3)},
    'chesapeake_inner': {'lon': (-76.8, -76.2), 'lat': (38.8, 39.5)},
    'chesapeake_mid': {'lon': (-76.6, -76.0), 'lat': (38.2, 38.8)},
    'delaware_mouth': {'lon': (-75.2, -74.8), 'lat': (38.7, 39.1)},
    'delaware_inner': {'lon': (-75.6, -75.0), 'lat': (39.5, 40.1)},
    'nj_coast': {'lon': (-74.5, -73.8), 'lat': (39.0, 40.8)},
    'ny_harbor': {'lon': (-74.2, -73.8), 'lat': (40.4, 40.8)},
}


def get_nodes_in_region(lon, lat, region):
    """Get node indices within a region bounding box"""
    lon_min, lon_max = region['lon']
    lat_min, lat_max = region['lat']

    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    return np.where(mask)[0]


def compute_distance_km(lon1, lat1, lon2, lat2):
    """Compute distance in km using Haversine formula"""
    R = 6371  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    return R * c


def identify_coastal_nodes(lon, lat, depth, threshold=-5):
    """Identify nodes near the coast (shallow water)"""
    # Coastal nodes are shallow and near domain edges or have high depth gradient
    is_shallow = np.abs(depth) < 20

    # Also consider nodes near longitude edges (open ocean boundary)
    lon_range = lon.max() - lon.min()
    near_east = lon > (lon.max() - 0.3 * lon_range)

    return np.where(is_shallow | near_east)[0]


def add_region_to_region_edges(lon, lat, src_region, dst_region, max_edges=500):
    """Add edges connecting two regions"""
    src_nodes = get_nodes_in_region(lon, lat, REGIONS[src_region])
    dst_nodes = get_nodes_in_region(lon, lat, REGIONS[dst_region])

    if len(src_nodes) == 0 or len(dst_nodes) == 0:
        logger.warning(f"Empty region: {src_region} ({len(src_nodes)}) or {dst_region} ({len(dst_nodes)})")
        return []

    edges = []

    # Connect each source node to k nearest destination nodes
    dst_coords = np.stack([lon[dst_nodes], lat[dst_nodes]], axis=1)
    tree = cKDTree(dst_coords)

    k = min(3, len(dst_nodes))

    for src_idx in src_nodes:
        src_coord = np.array([[lon[src_idx], lat[src_idx]]])
        _, nearest_indices = tree.query(src_coord, k=k)

        for ni in nearest_indices.flatten():
            dst_idx = dst_nodes[ni]
            if src_idx != dst_idx:
                edges.append((src_idx, dst_idx))
                edges.append((dst_idx, src_idx))  # Bidirectional

    # Limit total edges
    if len(edges) > max_edges * 2:
        indices = np.random.choice(len(edges)//2, max_edges, replace=False)
        edges = [(edges[2*i], edges[2*i+1]) for i in indices]
        edges = [e for pair in edges for e in pair]

    return edges


def add_sparse_longrange_edges(lon, lat, k=5, min_dist_km=20, max_dist_km=150):
    """Add sparse long-range edges using distance-weighted sampling"""
    n_nodes = len(lon)
    edges = []

    # Build KD-tree for efficient neighbor search
    coords = np.stack([lon, lat], axis=1)
    tree = cKDTree(coords)

    # Convert km to approximate degrees (rough approximation)
    min_dist_deg = min_dist_km / 111
    max_dist_deg = max_dist_km / 111

    logger.info(f"Adding sparse long-range edges (k={k}, {min_dist_km}-{max_dist_km} km)...")

    for i in range(n_nodes):
        # Find all neighbors within max distance
        neighbors = tree.query_ball_point(coords[i], max_dist_deg)

        # Filter by minimum distance
        valid_neighbors = []
        for j in neighbors:
            if i != j:
                dist = np.sqrt((lon[i] - lon[j])**2 + (lat[i] - lat[j])**2)
                if dist >= min_dist_deg:
                    valid_neighbors.append((j, dist))

        if len(valid_neighbors) == 0:
            continue

        # Select k neighbors, weighted by distance (prefer medium distance)
        valid_neighbors = sorted(valid_neighbors, key=lambda x: x[1])

        # Take evenly spaced samples across distance range
        n_select = min(k, len(valid_neighbors))
        indices = np.linspace(0, len(valid_neighbors)-1, n_select, dtype=int)

        for idx in indices:
            j, _ = valid_neighbors[idx]
            edges.append((i, j))
            edges.append((j, i))

    return edges


def add_coastal_enhancement_edges(lon, lat, depth, k=3, max_dist_km=100):
    """Add extra long-range edges for coastal nodes"""
    coastal_nodes = identify_coastal_nodes(lon, lat, depth)
    logger.info(f"Identified {len(coastal_nodes)} coastal nodes")

    edges = []

    # Connect coastal nodes to each other
    if len(coastal_nodes) < 2:
        return edges

    coastal_coords = np.stack([lon[coastal_nodes], lat[coastal_nodes]], axis=1)
    tree = cKDTree(coastal_coords)

    max_dist_deg = max_dist_km / 111
    min_dist_deg = 20 / 111

    for i, node_i in enumerate(coastal_nodes):
        # Find distant coastal neighbors
        neighbors = tree.query_ball_point(coastal_coords[i], max_dist_deg)

        valid = []
        for j in neighbors:
            if i != j:
                dist = np.sqrt((coastal_coords[i,0] - coastal_coords[j,0])**2 +
                              (coastal_coords[i,1] - coastal_coords[j,1])**2)
                if dist >= min_dist_deg:
                    valid.append((coastal_nodes[j], dist))

        # Select k neighbors
        valid = sorted(valid, key=lambda x: x[1])
        for j, _ in valid[:k]:
            edges.append((node_i, j))
            edges.append((j, node_i))

    return edges


def create_longrange_mesh():
    """Create enhanced mesh with long-range edges"""

    # Load original mesh
    logger.info("Loading original mesh...")
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))

    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = mesh_data['edge_index']

    n_nodes = len(lon)
    n_original_edges = edge_index.shape[1]

    logger.info(f"Original mesh: {n_nodes:,} nodes, {n_original_edges:,} edges")

    # Collect all new long-range edges
    new_edges = []

    # ========================================
    # 1. Bay mouth to inner bay connections
    # ========================================
    logger.info("\n1. Adding bay mouth → inner bay connections...")

    # Chesapeake: mouth → inner
    edges = add_region_to_region_edges(lon, lat, 'chesapeake_mouth', 'chesapeake_inner', max_edges=300)
    new_edges.extend(edges)
    logger.info(f"   Chesapeake mouth→inner: {len(edges)//2} edges")

    # Chesapeake: mouth → mid
    edges = add_region_to_region_edges(lon, lat, 'chesapeake_mouth', 'chesapeake_mid', max_edges=300)
    new_edges.extend(edges)
    logger.info(f"   Chesapeake mouth→mid: {len(edges)//2} edges")

    # Chesapeake: mid → inner
    edges = add_region_to_region_edges(lon, lat, 'chesapeake_mid', 'chesapeake_inner', max_edges=200)
    new_edges.extend(edges)
    logger.info(f"   Chesapeake mid→inner: {len(edges)//2} edges")

    # Delaware: mouth → inner
    edges = add_region_to_region_edges(lon, lat, 'delaware_mouth', 'delaware_inner', max_edges=300)
    new_edges.extend(edges)
    logger.info(f"   Delaware mouth→inner: {len(edges)//2} edges")

    # ========================================
    # 2. Along-coast connections
    # ========================================
    logger.info("\n2. Adding along-coast connections...")

    # NJ coast → NY harbor
    edges = add_region_to_region_edges(lon, lat, 'nj_coast', 'ny_harbor', max_edges=200)
    new_edges.extend(edges)
    logger.info(f"   NJ coast→NY harbor: {len(edges)//2} edges")

    # Delaware mouth → NJ coast
    edges = add_region_to_region_edges(lon, lat, 'delaware_mouth', 'nj_coast', max_edges=200)
    new_edges.extend(edges)
    logger.info(f"   Delaware→NJ coast: {len(edges)//2} edges")

    # ========================================
    # 3. Sparse global long-range edges
    # ========================================
    logger.info("\n3. Adding sparse global long-range edges...")
    edges = add_sparse_longrange_edges(lon, lat, k=K_NEIGHBORS,
                                        min_dist_km=MIN_DISTANCE_KM,
                                        max_dist_km=MAX_DISTANCE_KM)
    new_edges.extend(edges)
    logger.info(f"   Sparse global: {len(edges)//2} edges")

    # ========================================
    # 4. Coastal enhancement edges
    # ========================================
    logger.info("\n4. Adding coastal enhancement edges...")
    edges = add_coastal_enhancement_edges(lon, lat, depth, k=COASTAL_BOOST, max_dist_km=80)
    new_edges.extend(edges)
    logger.info(f"   Coastal enhancement: {len(edges)//2} edges")

    # ========================================
    # Combine with original edges
    # ========================================
    logger.info("\nCombining edges...")

    # Convert to set for deduplication
    original_edges_set = set()
    for i in range(edge_index.shape[1]):
        original_edges_set.add((edge_index[0, i], edge_index[1, i]))

    new_edges_set = set()
    for e in new_edges:
        if e not in original_edges_set:
            new_edges_set.add(e)

    logger.info(f"New unique long-range edges: {len(new_edges_set)//2}")

    # Combine
    all_edges = list(original_edges_set) + list(new_edges_set)
    combined_edge_index = np.array(all_edges, dtype=np.int64).T

    n_combined_edges = combined_edge_index.shape[1]
    n_new_edges = n_combined_edges - n_original_edges

    logger.info(f"\nFinal mesh:")
    logger.info(f"  Nodes: {n_nodes:,}")
    logger.info(f"  Original edges: {n_original_edges:,}")
    logger.info(f"  New long-range edges: {n_new_edges:,}")
    logger.info(f"  Total edges: {n_combined_edges:,}")
    logger.info(f"  Edge increase: {100*n_new_edges/n_original_edges:.1f}%")

    # ========================================
    # Compute edge attributes for new edges
    # ========================================
    logger.info("\nComputing edge attributes...")

    # Cartesian coordinates for edge features
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)

    src = combined_edge_index[0]
    dst = combined_edge_index[1]

    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)

    # Use median of ORIGINAL edges for normalization
    original_dist = np.sqrt(
        (x_cart[edge_index[1]] - x_cart[edge_index[0]])**2 +
        (y_cart[edge_index[1]] - y_cart[edge_index[0]])**2
    )
    char_length = np.median(original_dist) + 1e-8

    edge_attr = np.stack([
        dx / char_length,
        dy / char_length,
        dist / char_length
    ], axis=1).astype(np.float32)

    # ========================================
    # Save enhanced mesh
    # ========================================
    logger.info(f"\nSaving enhanced mesh to {OUTPUT_DIR}...")

    np.savez(
        OUTPUT_DIR / 'mesh.npz',
        lon=lon,
        lat=lat,
        depth=depth,
        edge_index=combined_edge_index,
        edge_attr=edge_attr,
        # Metadata
        n_original_edges=n_original_edges,
        n_longrange_edges=n_new_edges,
        bbox=mesh_data.get('bbox', None),
        global_indices=mesh_data.get('global_indices', None),
        config=mesh_data.get('config', None),
    )

    # Also save a visualization-friendly version
    np.savez(
        OUTPUT_DIR / 'longrange_edges.npz',
        longrange_edges=np.array(list(new_edges_set), dtype=np.int64).T,
        original_edge_count=n_original_edges,
        longrange_edge_count=n_new_edges,
    )

    logger.info("Done!")

    # ========================================
    # Summary statistics
    # ========================================
    print("\n" + "="*60)
    print("LONG-RANGE MESH SUMMARY")
    print("="*60)
    print(f"Original mesh: {n_nodes:,} nodes, {n_original_edges:,} edges")
    print(f"Enhanced mesh: {n_nodes:,} nodes, {n_combined_edges:,} edges")
    print(f"New edges added: {n_new_edges:,} ({100*n_new_edges/n_original_edges:.1f}% increase)")
    print(f"\nEdge distance statistics (km):")
    dist_km = dist / 1000
    print(f"  Original edges: median={np.median(original_dist)/1000:.1f} km")
    print(f"  All edges: min={dist_km.min():.1f}, median={np.median(dist_km):.1f}, max={dist_km.max():.1f}")
    print(f"\nOutput: {OUTPUT_DIR / 'mesh.npz'}")
    print("="*60)

    return combined_edge_index, edge_attr


if __name__ == '__main__':
    create_longrange_mesh()
