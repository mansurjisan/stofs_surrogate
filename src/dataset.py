"""
Dataset classes for STOFS/ADCIRC data.

Provides:
- ADCIRCDataset: Load real ADCIRC output (fort.63, fort.64)
- SyntheticSWEDataset: Generate synthetic data for testing
- ForcingConditionedDataset: Include atmospheric forcing
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path
import logging
import xarray as xr

try:
    from .mesh import ADCIRCMesh
except ImportError:
    from mesh import ADCIRCMesh

logger = logging.getLogger(__name__)


class ADCIRCDataset(Dataset):
    """
    Dataset for ADCIRC/STOFS output.

    Loads water elevation (fort.63) and optionally velocity (fort.64)
    to create input-output pairs for autoregressive training.

    The model learns: State(t) + [Forcing(t)] -> State(t+1)
    """

    def __init__(
        self,
        mesh: ADCIRCMesh,
        elevation_path: str,
        velocity_path: Optional[str] = None,
        forcing_paths: Optional[Dict[str, str]] = None,
        time_stride: int = 1,
        normalize: bool = True,
        eta_scale: float = 3.0,
        vel_scale: float = 2.0,
        cache_data: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            mesh: ADCIRCMesh object
            elevation_path: Path to fort.63.nc (water elevation)
            velocity_path: Path to fort.64.nc (velocity, optional)
            forcing_paths: Dict of forcing variable paths (e.g., {'wind_u': path, 'wind_v': path})
            time_stride: Stride between samples (1 = every timestep)
            normalize: Whether to normalize state variables
            eta_scale: Scale factor for elevation (m)
            vel_scale: Scale factor for velocity (m/s)
            cache_data: Whether to cache data in memory
        """
        super().__init__()

        self.mesh = mesh
        self.normalize = normalize
        self.eta_scale = eta_scale
        self.vel_scale = vel_scale
        self.time_stride = time_stride

        # Load data
        self.elevation = self._load_elevation(elevation_path)
        self.velocity = self._load_velocity(velocity_path) if velocity_path else None
        self.forcing = self._load_forcing(forcing_paths) if forcing_paths else None

        # Determine number of samples
        self.num_timesteps = len(self.elevation)
        self.num_samples = (self.num_timesteps - 1) // time_stride

        logger.info(f"Dataset: {self.num_samples} samples from {self.num_timesteps} timesteps")

        # Get graph structure from mesh
        self.edge_index, self.edge_attr = mesh.build_graph()
        self.node_features = mesh.get_node_features()

        # Cache if requested
        if cache_data:
            self._cache_data()

    def _load_elevation(self, path: str) -> np.ndarray:
        """Load water elevation from NetCDF."""
        logger.info(f"Loading elevation from {path}")

        ds = xr.open_dataset(path)

        # Handle different variable names
        var_names = ['zeta', 'elevation', 'surf_el', 'sea_surface_height']
        data = None

        for name in var_names:
            if name in ds:
                data = ds[name].values
                logger.info(f"Found elevation variable: {name}")
                break

        if data is None:
            # Try first data variable
            var_name = list(ds.data_vars)[0]
            data = ds[var_name].values
            logger.warning(f"Using first variable as elevation: {var_name}")

        ds.close()

        # Handle fill values
        data = np.nan_to_num(data, nan=0.0)

        logger.info(f"Elevation shape: {data.shape}, range: [{data.min():.2f}, {data.max():.2f}] m")

        return data.astype(np.float32)

    def _load_velocity(self, path: str) -> np.ndarray:
        """Load velocity from NetCDF."""
        logger.info(f"Loading velocity from {path}")

        ds = xr.open_dataset(path)

        # Try different variable name conventions
        u_names = ['u-vel', 'u', 'eastward_velocity', 'water_u']
        v_names = ['v-vel', 'v', 'northward_velocity', 'water_v']

        u_data = v_data = None

        for name in u_names:
            if name in ds:
                u_data = ds[name].values
                break

        for name in v_names:
            if name in ds:
                v_data = ds[name].values
                break

        ds.close()

        if u_data is None or v_data is None:
            logger.warning("Could not find velocity variables, skipping")
            return None

        # Stack [time, nodes, 2]
        data = np.stack([u_data, v_data], axis=-1)
        data = np.nan_to_num(data, nan=0.0)

        logger.info(f"Velocity shape: {data.shape}")

        return data.astype(np.float32)

    def _load_forcing(self, paths: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Load atmospheric forcing data."""
        forcing = {}

        for name, path in paths.items():
            logger.info(f"Loading forcing {name} from {path}")
            ds = xr.open_dataset(path)

            # Get first data variable
            var_name = list(ds.data_vars)[0]
            data = ds[var_name].values
            data = np.nan_to_num(data, nan=0.0)

            forcing[name] = data.astype(np.float32)
            ds.close()

        return forcing

    def _cache_data(self):
        """Pre-compute all samples for faster access."""
        logger.info("Caching dataset samples...")
        self._cached_samples = []

        for idx in range(self.num_samples):
            sample = self._create_sample(idx)
            self._cached_samples.append(sample)

        logger.info(f"Cached {len(self._cached_samples)} samples")

    def _create_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Create a single training sample."""
        t = idx * self.time_stride

        # Input state at time t
        eta_in = self.elevation[t]

        # Target state at time t+1
        eta_out = self.elevation[t + self.time_stride]

        # Build input features
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
            input_state[:, 0] = input_state[:, 0] / self.eta_scale
            target_state[:, 0] = target_state[:, 0] / self.eta_scale

            if input_state.shape[1] > 1:
                input_state[:, 1:] = input_state[:, 1:] / self.vel_scale
                target_state[:, 1:] = target_state[:, 1:] / self.vel_scale

        sample = {
            'input': torch.tensor(input_state, dtype=torch.float32),
            'target': torch.tensor(target_state, dtype=torch.float32),
            'timestep': t,
        }

        # Add forcing if available
        if self.forcing is not None:
            forcing_features = []
            for name, data in self.forcing.items():
                forcing_features.append(data[t])

            forcing_array = np.stack(forcing_features, axis=1)
            sample['forcing'] = torch.tensor(forcing_array, dtype=torch.float32)

        return sample

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """Get a sample as PyG Data object."""
        if hasattr(self, '_cached_samples'):
            sample = self._cached_samples[idx]
        else:
            sample = self._create_sample(idx)

        # Create PyG Data object
        data = Data(
            x=sample['input'],
            y=sample['target'],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            pos=self.mesh.to_pyg_data().pos,
            node_features=self.node_features,
        )

        if 'forcing' in sample:
            data.forcing = sample['forcing']

        data.timestep = sample['timestep']

        return data


class SyntheticSWEDataset(Dataset):
    """
    Synthetic Shallow Water Equations dataset for testing.

    Generates Gaussian perturbations that propagate and decay,
    mimicking storm surge behavior without real data.
    """

    def __init__(
        self,
        num_nodes: int = 2500,
        num_samples: int = 500,
        domain_size: Tuple[float, float] = (100000, 100000),
        include_velocity: bool = True,
        include_forcing: bool = False,
        seed: int = 42,
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_nodes: Approximate number of mesh nodes
            num_samples: Number of training samples
            domain_size: Domain size in meters (Lx, Ly)
            include_velocity: Whether to include velocity in state
            include_forcing: Whether to generate synthetic forcing
            seed: Random seed
        """
        super().__init__()

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.include_velocity = include_velocity
        self.include_forcing = include_forcing

        # Create simple grid mesh
        nx = int(np.sqrt(num_nodes))
        ny = nx
        self.num_nodes = nx * ny

        Lx, Ly = domain_size
        dx = Lx / (nx - 1)

        # Node coordinates
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        xx, yy = np.meshgrid(x, y)
        self.node_coords = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)

        # Bathymetry (coastal slope)
        self.depth = 5.0 + 0.0003 * self.node_coords[:, 0]
        self.depth = self.depth.astype(np.float32)

        # Build graph (4-connectivity)
        self.edge_index = self._build_graph(nx, ny)

        # Pre-generate samples
        self.samples = self._generate_samples()

        logger.info(f"Synthetic dataset: {self.num_nodes} nodes, {self.num_samples} samples")

    def _build_graph(self, nx: int, ny: int) -> torch.Tensor:
        """Build graph edges for regular grid."""
        edges = []

        for i in range(ny):
            for j in range(nx):
                node_id = i * nx + j

                # Right neighbor
                if j < nx - 1:
                    edges.append([node_id, node_id + 1])
                    edges.append([node_id + 1, node_id])

                # Top neighbor
                if i < ny - 1:
                    edges.append([node_id, node_id + nx])
                    edges.append([node_id + nx, node_id])

        return torch.tensor(edges, dtype=torch.long).T

    def _generate_samples(self) -> List[Dict[str, np.ndarray]]:
        """Generate synthetic SWE samples."""
        samples = []

        g = 9.81
        c = np.sqrt(g * self.depth.mean())  # Wave speed

        for _ in range(self.num_samples):
            # Random Gaussian perturbation
            x0 = np.random.uniform(0.2, 0.8) * self.node_coords[:, 0].max()
            y0 = np.random.uniform(0.2, 0.8) * self.node_coords[:, 1].max()
            sigma = np.random.uniform(5000, 20000)
            amplitude = np.random.uniform(0.3, 2.0)

            r2 = (self.node_coords[:, 0] - x0)**2 + (self.node_coords[:, 1] - y0)**2
            eta_t = amplitude * np.exp(-r2 / (2 * sigma**2))

            # Velocity from shallow water relation
            u_t = (g / c) * eta_t * (self.node_coords[:, 0] - x0) / sigma * 0.1
            v_t = (g / c) * eta_t * (self.node_coords[:, 1] - y0) / sigma * 0.1

            # Simple forward model (decay + spread)
            decay = 0.92
            spread = 1.08
            new_sigma = sigma * spread

            eta_t1 = amplitude * decay * np.exp(-r2 / (2 * new_sigma**2))
            u_t1 = decay * (g / c) * eta_t1 * (self.node_coords[:, 0] - x0) / new_sigma * 0.1
            v_t1 = decay * (g / c) * eta_t1 * (self.node_coords[:, 1] - y0) / new_sigma * 0.1

            # Build state vectors
            if self.include_velocity:
                input_state = np.stack([eta_t, u_t, v_t], axis=1)
                target_state = np.stack([eta_t1, u_t1, v_t1], axis=1)
            else:
                input_state = eta_t[:, None]
                target_state = eta_t1[:, None]

            sample = {
                'input': input_state.astype(np.float32),
                'target': target_state.astype(np.float32),
            }

            # Add synthetic forcing
            if self.include_forcing:
                # Synthetic wind field
                wind_speed = np.random.uniform(5, 30)
                wind_dir = np.random.uniform(0, 2 * np.pi)
                wind_u = wind_speed * np.cos(wind_dir) * np.ones(self.num_nodes)
                wind_v = wind_speed * np.sin(wind_dir) * np.ones(self.num_nodes)
                pressure = 101325 - amplitude * 2000  # Lower pressure at surge center

                sample['forcing'] = np.stack([wind_u, wind_v, np.full(self.num_nodes, pressure)], axis=1).astype(np.float32)

            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """Get sample as PyG Data object."""
        sample = self.samples[idx]

        # Normalize node coordinates
        pos = self.node_coords.copy()
        pos[:, 0] = 2 * (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min()) - 1
        pos[:, 1] = 2 * (pos[:, 1] - pos[:, 1].min()) / (pos[:, 1].max() - pos[:, 1].min()) - 1

        # Node features: normalized position + bathymetry
        depth_norm = (self.depth - self.depth.mean()) / self.depth.std()
        node_features = np.stack([pos[:, 0], pos[:, 1], depth_norm], axis=1)

        data = Data(
            x=torch.tensor(sample['input'], dtype=torch.float32),
            y=torch.tensor(sample['target'], dtype=torch.float32),
            edge_index=self.edge_index,
            pos=torch.tensor(self.node_coords, dtype=torch.float32),
            node_features=torch.tensor(node_features, dtype=torch.float32),
            depth=torch.tensor(self.depth, dtype=torch.float32),
        )

        if 'forcing' in sample:
            data.forcing = torch.tensor(sample['forcing'], dtype=torch.float32)

        return data


class MultiEventDataset(Dataset):
    """
    Dataset combining multiple storm events for training.

    Useful for training on historical hurricane data.
    """

    def __init__(
        self,
        mesh: ADCIRCMesh,
        event_paths: List[Dict[str, str]],
        **kwargs
    ):
        """
        Initialize multi-event dataset.

        Args:
            mesh: ADCIRCMesh object
            event_paths: List of dicts with 'elevation_path' and optional 'velocity_path'
            **kwargs: Additional arguments passed to ADCIRCDataset
        """
        super().__init__()

        self.datasets = []
        self.cumulative_lengths = [0]

        for event in event_paths:
            ds = ADCIRCDataset(
                mesh=mesh,
                elevation_path=event['elevation_path'],
                velocity_path=event.get('velocity_path'),
                **kwargs
            )
            self.datasets.append(ds)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))

        logger.info(f"Multi-event dataset: {len(self.datasets)} events, {len(self)} total samples")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> Data:
        """Get sample from appropriate event dataset."""
        # Find which event this index belongs to
        for i, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if start <= idx < end:
                return self.datasets[i][idx - start]

        raise IndexError(f"Index {idx} out of range")
