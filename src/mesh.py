"""
Mesh handling utilities for ADCIRC/SCHISM unstructured grids.

Supports:
- ADCIRC fort.14 mesh files
- SCHISM hgrid.gr3 mesh files
- Conversion to PyTorch Geometric graph format
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ADCIRCMesh:
    """
    ADCIRC mesh reader and graph converter.

    Reads fort.14 format and converts to graph representation suitable
    for GNN training.
    """

    def __init__(self, mesh_path: str):
        """
        Initialize mesh from fort.14 file.

        Args:
            mesh_path: Path to fort.14 file
        """
        self.mesh_path = Path(mesh_path)
        self.header: str = ""
        self.num_elements: int = 0
        self.num_nodes: int = 0

        # Node data
        self.node_ids: np.ndarray = None  # [num_nodes]
        self.lon: np.ndarray = None       # [num_nodes] - longitude
        self.lat: np.ndarray = None       # [num_nodes] - latitude
        self.depth: np.ndarray = None     # [num_nodes] - bathymetry (positive down)

        # Element data
        self.elements: np.ndarray = None  # [num_elements, 3] - triangle node indices

        # Graph representation
        self._edge_index: torch.Tensor = None
        self._edge_attr: torch.Tensor = None

        # Load mesh
        self._read_fort14()

    def _read_fort14(self):
        """Read ADCIRC fort.14 mesh file."""
        logger.info(f"Reading mesh from {self.mesh_path}")

        with open(self.mesh_path, 'r') as f:
            # Line 1: Header/title
            self.header = f.readline().strip()

            # Line 2: NE, NP (num elements, num nodes)
            line = f.readline().split()
            self.num_elements = int(line[0])
            self.num_nodes = int(line[1])

            logger.info(f"Mesh: {self.num_nodes:,} nodes, {self.num_elements:,} elements")

            # Read nodes: node_id, lon, lat, depth
            node_ids = np.zeros(self.num_nodes, dtype=np.int32)
            coords = np.zeros((self.num_nodes, 3), dtype=np.float64)

            for i in range(self.num_nodes):
                parts = f.readline().split()
                node_ids[i] = int(parts[0])
                coords[i, 0] = float(parts[1])  # lon
                coords[i, 1] = float(parts[2])  # lat
                coords[i, 2] = float(parts[3])  # depth

            self.node_ids = node_ids
            self.lon = coords[:, 0].astype(np.float32)
            self.lat = coords[:, 1].astype(np.float32)
            self.depth = coords[:, 2].astype(np.float32)

            # Read elements: elem_id, 3, n1, n2, n3
            elements = np.zeros((self.num_elements, 3), dtype=np.int32)

            for i in range(self.num_elements):
                parts = f.readline().split()
                # Convert to 0-indexed
                elements[i, 0] = int(parts[2]) - 1
                elements[i, 1] = int(parts[3]) - 1
                elements[i, 2] = int(parts[4]) - 1

            self.elements = elements

        logger.info("Mesh loaded successfully")

    def to_cartesian(self, ref_lon: float = None, ref_lat: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert lon/lat to approximate Cartesian coordinates (meters).

        Uses simple equirectangular projection centered at reference point.
        Good for regional domains. For global, consider proper projection.

        Args:
            ref_lon: Reference longitude (default: domain center)
            ref_lat: Reference latitude (default: domain center)

        Returns:
            x, y: Cartesian coordinates in meters
        """
        if ref_lon is None:
            ref_lon = self.lon.mean()
        if ref_lat is None:
            ref_lat = self.lat.mean()

        # Earth radius in meters
        R = 6371000.0

        # Convert to radians
        lon_rad = np.radians(self.lon)
        lat_rad = np.radians(self.lat)
        ref_lon_rad = np.radians(ref_lon)
        ref_lat_rad = np.radians(ref_lat)

        # Equirectangular projection
        x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        y = R * (lat_rad - ref_lat_rad)

        return x.astype(np.float32), y.astype(np.float32)

    def build_graph(self, include_edge_features: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build graph edge_index from mesh connectivity.

        Each triangle edge becomes a bidirectional graph edge.

        Args:
            include_edge_features: Whether to compute edge features (distance, angle)

        Returns:
            edge_index: [2, num_edges] tensor
            edge_attr: [num_edges, num_features] tensor or None
        """
        if self._edge_index is not None:
            return self._edge_index, self._edge_attr

        logger.info("Building graph from mesh...")

        # Collect unique edges from triangles
        edges = set()

        for elem in self.elements:
            # Add three edges per triangle
            for i in range(3):
                n1, n2 = elem[i], elem[(i + 1) % 3]
                # Store as sorted tuple to avoid duplicates
                edges.add((min(n1, n2), max(n1, n2)))

        # Convert to bidirectional edge_index
        edge_list = []
        for n1, n2 in edges:
            edge_list.append([n1, n2])
            edge_list.append([n2, n1])

        self._edge_index = torch.tensor(edge_list, dtype=torch.long).T

        logger.info(f"Graph: {self._edge_index.shape[1]:,} edges (bidirectional)")

        # Compute edge features if requested
        if include_edge_features:
            self._edge_attr = self._compute_edge_features()

        return self._edge_index, self._edge_attr

    def _compute_edge_features(self) -> torch.Tensor:
        """
        Compute edge features: relative position, distance.

        Features:
        - dx, dy: Relative position (normalized)
        - distance: Edge length (normalized)
        """
        x, y = self.to_cartesian()

        src = self._edge_index[0].numpy()
        dst = self._edge_index[1].numpy()

        dx = x[dst] - x[src]
        dy = y[dst] - y[src]
        dist = np.sqrt(dx**2 + dy**2)

        # Normalize by characteristic length
        char_length = np.median(dist)
        dx_norm = dx / char_length
        dy_norm = dy / char_length
        dist_norm = dist / char_length

        edge_attr = np.stack([dx_norm, dy_norm, dist_norm], axis=1)

        return torch.tensor(edge_attr, dtype=torch.float32)

    def get_node_features(self, normalize: bool = True) -> torch.Tensor:
        """
        Get static node features.

        Features:
        - x, y: Cartesian position (normalized)
        - depth: Bathymetry (normalized)

        Args:
            normalize: Whether to normalize features

        Returns:
            node_features: [num_nodes, 3] tensor
        """
        x, y = self.to_cartesian()

        if normalize:
            # Normalize position to [-1, 1]
            x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1
            y_norm = 2 * (y - y.min()) / (y.max() - y.min()) - 1

            # Normalize depth (log-scale for wide range)
            depth_positive = np.maximum(self.depth, 0.1)  # Avoid log(0)
            depth_log = np.log10(depth_positive)
            depth_norm = (depth_log - depth_log.mean()) / depth_log.std()
        else:
            x_norm, y_norm = x, y
            depth_norm = self.depth

        features = np.stack([x_norm, y_norm, depth_norm], axis=1)

        return torch.tensor(features, dtype=torch.float32)

    def to_pyg_data(self) -> Data:
        """
        Convert mesh to PyTorch Geometric Data object.

        Returns:
            PyG Data object with node positions and graph structure
        """
        edge_index, edge_attr = self.build_graph()
        node_features = self.get_node_features()

        x, y = self.to_cartesian()
        pos = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            num_nodes=self.num_nodes,
        )

        # Add metadata
        data.lon = torch.tensor(self.lon, dtype=torch.float32)
        data.lat = torch.tensor(self.lat, dtype=torch.float32)
        data.depth = torch.tensor(self.depth, dtype=torch.float32)

        return data

    def subsample(self, target_nodes: int, method: str = 'farthest') -> 'ADCIRCMesh':
        """
        Subsample mesh for memory-efficient training.

        Args:
            target_nodes: Target number of nodes
            method: 'farthest' (farthest point sampling) or 'random'

        Returns:
            Subsampled mesh
        """
        if target_nodes >= self.num_nodes:
            return self

        logger.info(f"Subsampling mesh from {self.num_nodes:,} to {target_nodes:,} nodes")

        if method == 'random':
            indices = np.random.choice(self.num_nodes, target_nodes, replace=False)
            indices = np.sort(indices)
        elif method == 'farthest':
            indices = self._farthest_point_sampling(target_nodes)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Create new mesh with subsampled nodes
        submesh = ADCIRCMesh.__new__(ADCIRCMesh)
        submesh.mesh_path = self.mesh_path
        submesh.header = self.header + " (subsampled)"
        submesh.num_nodes = target_nodes

        # Create mapping from old to new indices
        old_to_new = {old: new for new, old in enumerate(indices)}

        submesh.node_ids = np.arange(target_nodes)
        submesh.lon = self.lon[indices]
        submesh.lat = self.lat[indices]
        submesh.depth = self.depth[indices]

        # Filter elements that have all nodes in subsampled set
        valid_elements = []
        for elem in self.elements:
            if all(n in old_to_new for n in elem):
                new_elem = [old_to_new[n] for n in elem]
                valid_elements.append(new_elem)

        submesh.elements = np.array(valid_elements, dtype=np.int32)
        submesh.num_elements = len(valid_elements)

        submesh._edge_index = None
        submesh._edge_attr = None

        logger.info(f"Subsampled mesh: {submesh.num_nodes:,} nodes, {submesh.num_elements:,} elements")

        return submesh

    def _farthest_point_sampling(self, target_nodes: int) -> np.ndarray:
        """Farthest point sampling for mesh subsampling."""
        from scipy.spatial import cKDTree

        x, y = self.to_cartesian()
        coords = np.stack([x, y], axis=1)

        # Start with random point
        indices = [np.random.randint(self.num_nodes)]

        for _ in range(target_nodes - 1):
            # Build tree of selected points
            tree = cKDTree(coords[indices])

            # Find distances to nearest selected point
            dists, _ = tree.query(coords, k=1)

            # Select farthest point
            farthest = np.argmax(dists)
            indices.append(farthest)

        return np.array(sorted(indices))


class SCHISMMesh:
    """
    SCHISM mesh reader (hgrid.gr3 format).

    Placeholder for STOFS 3D Atlantic support.
    """

    def __init__(self, mesh_path: str):
        raise NotImplementedError("SCHISM mesh support coming soon. Use ADCIRC mesh for now.")


def create_mesh(mesh_path: str, mesh_type: str = 'auto') -> ADCIRCMesh:
    """
    Factory function to create mesh object.

    Args:
        mesh_path: Path to mesh file
        mesh_type: 'adcirc', 'schism', or 'auto' (detect from extension)

    Returns:
        Mesh object
    """
    path = Path(mesh_path)

    if mesh_type == 'auto':
        if path.suffix == '.14' or 'fort.14' in path.name:
            mesh_type = 'adcirc'
        elif path.suffix == '.gr3':
            mesh_type = 'schism'
        else:
            mesh_type = 'adcirc'  # Default

    if mesh_type == 'adcirc':
        return ADCIRCMesh(mesh_path)
    elif mesh_type == 'schism':
        return SCHISMMesh(mesh_path)
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
