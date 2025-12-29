"""
STOFS NetCDF Data Loader

Loads STOFS 2D Global output directly from NetCDF files,
including mesh extraction and regional subsetting.

Key files:
- stofs_2d_glo_maxele.63.nc: Mesh + max elevation (smallest file with mesh)
- stofs_2d_glo_surf.63.nc: Water elevation time series
- stofs_2d_glo_surf.64.nc: Velocity time series
- stofs_2d_glo_surf.68.nc: Meteorological forcing
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, Optional, Dict, List, Union
from pathlib import Path
import logging
import xarray as xr
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class STOFSNetCDFMesh:
    """
    Load STOFS mesh from NetCDF output files.

    The mesh can be extracted from any .63.nc file that contains
    the element connectivity array.
    """

    def __init__(
        self,
        mesh_nc_path: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        max_nodes: Optional[int] = None,
    ):
        """
        Initialize mesh from STOFS NetCDF file.

        Args:
            mesh_nc_path: Path to NetCDF file containing mesh (e.g., maxele.63.nc)
            bbox: Optional bounding box (lon_min, lon_max, lat_min, lat_max)
            max_nodes: Maximum number of nodes (subsample if exceeded)
        """
        self.mesh_path = Path(mesh_nc_path)
        self.bbox = bbox
        self.max_nodes = max_nodes

        # Mesh data
        self.lon: np.ndarray = None
        self.lat: np.ndarray = None
        self.depth: np.ndarray = None
        self.elements: np.ndarray = None
        self.num_nodes: int = 0
        self.num_elements: int = 0

        # Index mapping for subsets
        self._original_indices: np.ndarray = None
        self._edge_index: torch.Tensor = None

        # Load mesh
        self._load_mesh()

    def _load_mesh(self):
        """Load mesh from NetCDF file."""
        logger.info(f"Loading STOFS mesh from {self.mesh_path}")

        ds = xr.open_dataset(self.mesh_path)

        # Load coordinates
        self.lon = ds['x'].values.astype(np.float32)
        self.lat = ds['y'].values.astype(np.float32)

        # Load bathymetry
        if 'depth' in ds:
            self.depth = ds['depth'].values.astype(np.float32)
        else:
            self.depth = np.zeros(len(self.lon), dtype=np.float32)

        # Load element connectivity (0-indexed in output)
        if 'element' in ds:
            # ADCIRC uses 1-based indexing in fort.14, but NetCDF may be 0 or 1 based
            elements = ds['element'].values
            # Check if 1-based
            if elements.min() >= 1:
                elements = elements - 1
            self.elements = elements.astype(np.int32)
        else:
            raise ValueError("No 'element' variable found in NetCDF file")

        ds.close()

        self.num_nodes = len(self.lon)
        self.num_elements = len(self.elements)

        logger.info(f"Full mesh: {self.num_nodes:,} nodes, {self.num_elements:,} elements")

        # Apply bounding box filter
        if self.bbox is not None:
            self._apply_bbox()

        # Subsample if too large
        if self.max_nodes is not None and self.num_nodes > self.max_nodes:
            self._subsample()

    def _apply_bbox(self):
        """Filter mesh to bounding box region."""
        lon_min, lon_max, lat_min, lat_max = self.bbox

        logger.info(f"Applying bounding box: [{lon_min}, {lon_max}] x [{lat_min}, {lat_max}]")

        # Find nodes within bbox
        mask = (
            (self.lon >= lon_min) & (self.lon <= lon_max) &
            (self.lat >= lat_min) & (self.lat <= lat_max)
        )

        if mask.sum() == 0:
            raise ValueError(f"No nodes found within bounding box {self.bbox}")

        # Get indices of nodes in bbox
        old_indices = np.where(mask)[0]
        self._original_indices = old_indices

        # Create mapping from old to new indices
        old_to_new = {old: new for new, old in enumerate(old_indices)}

        # Filter nodes
        self.lon = self.lon[mask]
        self.lat = self.lat[mask]
        self.depth = self.depth[mask]

        # Filter elements (keep only elements with all nodes in bbox)
        valid_elements = []
        for elem in self.elements:
            if all(n in old_to_new for n in elem):
                new_elem = [old_to_new[n] for n in elem]
                valid_elements.append(new_elem)

        self.elements = np.array(valid_elements, dtype=np.int32)

        self.num_nodes = len(self.lon)
        self.num_elements = len(self.elements)

        logger.info(f"After bbox filter: {self.num_nodes:,} nodes, {self.num_elements:,} elements")

    def _subsample(self):
        """Subsample mesh using farthest point sampling."""
        logger.info(f"Subsampling from {self.num_nodes:,} to {self.max_nodes:,} nodes")

        # Use farthest point sampling
        coords = np.stack([self.lon, self.lat], axis=1)

        # Start with random point
        indices = [np.random.randint(self.num_nodes)]

        # Iteratively add farthest point
        for _ in range(self.max_nodes - 1):
            tree = cKDTree(coords[indices])
            dists, _ = tree.query(coords, k=1)
            farthest = np.argmax(dists)
            indices.append(farthest)

            if len(indices) % 10000 == 0:
                logger.info(f"  Selected {len(indices):,} nodes...")

        indices = np.array(sorted(indices))

        # Update original indices tracking
        if self._original_indices is not None:
            self._original_indices = self._original_indices[indices]
        else:
            self._original_indices = indices

        # Create mapping
        old_to_new = {old: new for new, old in enumerate(indices)}

        # Filter nodes
        self.lon = self.lon[indices]
        self.lat = self.lat[indices]
        self.depth = self.depth[indices]

        # Filter elements
        valid_elements = []
        for elem in self.elements:
            if all(n in old_to_new for n in elem):
                new_elem = [old_to_new[n] for n in elem]
                valid_elements.append(new_elem)

        self.elements = np.array(valid_elements, dtype=np.int32) if valid_elements else np.array([], dtype=np.int32).reshape(0, 3)

        self.num_nodes = len(self.lon)
        self.num_elements = len(self.elements)

        logger.info(f"After subsampling: {self.num_nodes:,} nodes, {self.num_elements:,} elements")

    def to_cartesian(self, ref_lon: float = None, ref_lat: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat to Cartesian coordinates (meters)."""
        if ref_lon is None:
            ref_lon = self.lon.mean()
        if ref_lat is None:
            ref_lat = self.lat.mean()

        R = 6371000.0  # Earth radius in meters

        lon_rad = np.radians(self.lon)
        lat_rad = np.radians(self.lat)
        ref_lon_rad = np.radians(ref_lon)
        ref_lat_rad = np.radians(ref_lat)

        x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        y = R * (lat_rad - ref_lat_rad)

        return x.astype(np.float32), y.astype(np.float32)

    def build_edge_index(self) -> torch.Tensor:
        """Build graph edge_index from mesh elements."""
        if self._edge_index is not None:
            return self._edge_index

        logger.info("Building graph from mesh...")

        edges = set()
        for elem in self.elements:
            for i in range(3):
                n1, n2 = elem[i], elem[(i + 1) % 3]
                edges.add((min(n1, n2), max(n1, n2)))

        # Bidirectional
        edge_list = []
        for n1, n2 in edges:
            edge_list.append([n1, n2])
            edge_list.append([n2, n1])

        self._edge_index = torch.tensor(edge_list, dtype=torch.long).T

        logger.info(f"Graph: {self._edge_index.shape[1]:,} edges")

        return self._edge_index

    def get_node_features(self, normalize: bool = True) -> torch.Tensor:
        """Get static node features (position + depth)."""
        x, y = self.to_cartesian()

        if normalize:
            x_norm = 2 * (x - x.min()) / (x.max() - x.min() + 1e-8) - 1
            y_norm = 2 * (y - y.min()) / (y.max() - y.min() + 1e-8) - 1

            depth_safe = np.maximum(np.abs(self.depth), 0.1)
            depth_log = np.log10(depth_safe)
            depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
        else:
            x_norm, y_norm = x, y
            depth_norm = self.depth

        features = np.stack([x_norm, y_norm, depth_norm], axis=1)
        return torch.tensor(features, dtype=torch.float32)

    def to_pyg_data(self) -> Data:
        """Convert mesh to PyTorch Geometric Data object."""
        edge_index = self.build_edge_index()
        node_features = self.get_node_features()

        x, y = self.to_cartesian()
        pos = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            pos=pos,
            num_nodes=self.num_nodes,
        )

        data.lon = torch.tensor(self.lon, dtype=torch.float32)
        data.lat = torch.tensor(self.lat, dtype=torch.float32)
        data.depth = torch.tensor(self.depth, dtype=torch.float32)

        return data

    def get_original_indices(self) -> np.ndarray:
        """Get indices mapping to original full mesh."""
        if self._original_indices is not None:
            return self._original_indices
        return np.arange(self.num_nodes)


class STOFSDataset:
    """
    Dataset for STOFS 2D Global NetCDF output.

    Handles:
    - Loading mesh from maxele.63.nc or any .63.nc file
    - Loading time series from surf.63.nc, surf.64.nc
    - Regional subsetting for manageable training
    """

    def __init__(
        self,
        mesh_path: str,
        elevation_path: str,
        velocity_path: Optional[str] = None,
        forcing_path: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        max_nodes: int = 50000,
        time_stride: int = 1,
        normalize: bool = True,
        eta_scale: float = 3.0,
        vel_scale: float = 2.0,
    ):
        """
        Initialize STOFS dataset.

        Args:
            mesh_path: Path to NetCDF with mesh (e.g., maxele.63.nc)
            elevation_path: Path to elevation time series (surf.63.nc)
            velocity_path: Path to velocity time series (surf.64.nc)
            forcing_path: Path to forcing data (surf.68.nc)
            bbox: Bounding box (lon_min, lon_max, lat_min, lat_max)
            max_nodes: Maximum nodes after subsetting
            time_stride: Temporal stride for samples
            normalize: Whether to normalize data
            eta_scale: Scale for elevation normalization
            vel_scale: Scale for velocity normalization
        """
        self.normalize = normalize
        self.eta_scale = eta_scale
        self.vel_scale = vel_scale
        self.time_stride = time_stride

        # Load mesh with subsetting
        logger.info("Loading mesh...")
        self.mesh = STOFSNetCDFMesh(
            mesh_nc_path=mesh_path,
            bbox=bbox,
            max_nodes=max_nodes,
        )

        # Get indices for data extraction
        self.node_indices = self.mesh.get_original_indices()

        # Load elevation time series
        logger.info("Loading elevation data...")
        self.elevation = self._load_timeseries(elevation_path, 'zeta')

        # Load velocity if provided
        self.velocity = None
        if velocity_path:
            logger.info("Loading velocity data...")
            self.velocity = self._load_velocity(velocity_path)

        # Load forcing if provided
        self.forcing = None
        if forcing_path:
            logger.info("Loading forcing data...")
            self.forcing = self._load_forcing(forcing_path)

        # Setup dataset
        self.num_timesteps = len(self.elevation)
        self.num_samples = (self.num_timesteps - 1) // time_stride

        logger.info(f"Dataset ready: {self.num_samples} samples, {self.mesh.num_nodes:,} nodes")

    def _load_timeseries(self, path: str, var_name: str) -> np.ndarray:
        """Load time series data, extracting subset of nodes."""
        ds = xr.open_dataset(path)

        # Find variable
        if var_name in ds:
            data = ds[var_name]
        else:
            # Try alternatives
            alternatives = ['zeta', 'elevation', 'surf_el', 'sea_surface_height']
            for alt in alternatives:
                if alt in ds:
                    data = ds[alt]
                    break
            else:
                raise ValueError(f"Could not find elevation variable in {path}")

        # Extract subset of nodes
        # Data shape is typically (time, node)
        if 'node' in data.dims:
            values = data.isel(node=self.node_indices).values
        else:
            values = data[:, self.node_indices].values

        ds.close()

        values = np.nan_to_num(values, nan=0.0)
        logger.info(f"Loaded {var_name}: shape {values.shape}")

        return values.astype(np.float32)

    def _load_velocity(self, path: str) -> np.ndarray:
        """Load velocity time series."""
        ds = xr.open_dataset(path)

        # Try different variable names
        u_names = ['u-vel', 'u', 'eastward_velocity']
        v_names = ['v-vel', 'v', 'northward_velocity']

        u_data = v_data = None
        for name in u_names:
            if name in ds:
                u_data = ds[name]
                break
        for name in v_names:
            if name in ds:
                v_data = ds[name]
                break

        if u_data is None or v_data is None:
            ds.close()
            return None

        # Extract subset
        if 'node' in u_data.dims:
            u_vals = u_data.isel(node=self.node_indices).values
            v_vals = v_data.isel(node=self.node_indices).values
        else:
            u_vals = u_data[:, self.node_indices].values
            v_vals = v_data[:, self.node_indices].values

        ds.close()

        u_vals = np.nan_to_num(u_vals, nan=0.0)
        v_vals = np.nan_to_num(v_vals, nan=0.0)

        return np.stack([u_vals, v_vals], axis=-1).astype(np.float32)

    def _load_forcing(self, path: str) -> Dict[str, np.ndarray]:
        """Load meteorological forcing."""
        ds = xr.open_dataset(path)

        forcing = {}

        # Try to find wind and pressure
        for var_name in ds.data_vars:
            if 'wind' in var_name.lower() or 'pressure' in var_name.lower():
                data = ds[var_name]
                if 'node' in data.dims:
                    vals = data.isel(node=self.node_indices).values
                else:
                    vals = data[:, self.node_indices].values

                forcing[var_name] = np.nan_to_num(vals, nan=0.0).astype(np.float32)

        ds.close()
        return forcing if forcing else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """Get a training sample."""
        t = idx * self.time_stride

        # Input state
        eta_in = self.elevation[t]
        eta_out = self.elevation[t + self.time_stride]

        if self.velocity is not None:
            vel_in = self.velocity[t]
            vel_out = self.velocity[t + self.time_stride]
            input_state = np.concatenate([eta_in[:, None], vel_in], axis=1)
            target_state = np.concatenate([eta_out[:, None], vel_out], axis=1)
        else:
            input_state = eta_in[:, None]
            target_state = eta_out[:, None]

        # Normalize
        if self.normalize:
            input_state[:, 0] /= self.eta_scale
            target_state[:, 0] /= self.eta_scale
            if input_state.shape[1] > 1:
                input_state[:, 1:] /= self.vel_scale
                target_state[:, 1:] /= self.vel_scale

        # Build PyG data
        data = Data(
            x=torch.tensor(input_state, dtype=torch.float32),
            y=torch.tensor(target_state, dtype=torch.float32),
            edge_index=self.mesh.build_edge_index(),
            node_features=self.mesh.get_node_features(),
            pos=self.mesh.to_pyg_data().pos,
        )

        if self.forcing is not None:
            forcing_list = []
            for name, vals in self.forcing.items():
                forcing_list.append(vals[t])
            if forcing_list:
                data.forcing = torch.tensor(np.stack(forcing_list, axis=1), dtype=torch.float32)

        return data


# Predefined regions for subsetting
REGIONS = {
    'us_east_coast': (-82, -65, 24, 46),
    'gulf_of_mexico': (-98, -80, 18, 31),
    'us_west_coast': (-130, -115, 30, 50),
    'caribbean': (-90, -60, 10, 25),
    'atlantic_basin': (-100, -10, 0, 60),
    'north_atlantic': (-80, 0, 30, 60),
}


def get_region_bbox(region_name: str) -> Tuple[float, float, float, float]:
    """Get bounding box for predefined region."""
    if region_name not in REGIONS:
        available = ', '.join(REGIONS.keys())
        raise ValueError(f"Unknown region: {region_name}. Available: {available}")
    return REGIONS[region_name]
