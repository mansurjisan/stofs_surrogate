#!/usr/bin/env python3
"""
Ensemble Inference Script for STOFS CWL GNN Surrogate

This script enables rapid ensemble generation for storm surge forecasting
using your trained GNN model. Key features:

1. Meteorological forcing perturbations (wind speed, direction, pressure)
2. Initial condition perturbations
3. Track perturbations (for hurricane scenarios)
4. Parallel ensemble execution
5. Comprehensive statistics and visualization
6. Exceedance probability maps

Performance: ~50 ensemble members in 1-2 minutes on RTX 3050

Usage:
    python ensemble_inference.py --checkpoint best_physics_informed_model.pt \
                                 --n_members 50 \
                                 --forecast_hours 48

Author: STOFS Surrogate Project
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import gc
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
import logging
import pandas as pd

# For fetching CO-OPS observations
try:
    from searvey import fetch_coops_station
    SEARVEY_AVAILABLE = True
except ImportError:
    SEARVEY_AVAILABLE = False
    print("Warning: searvey not available. Observations will not be plotted.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Memory Management Utilities
# ============================================================

def get_gpu_memory_info() -> Dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
    }


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def check_memory_warning(config: Dict = None) -> bool:
    """Check if memory usage is approaching limits. Returns True if warning."""
    config = config or MEMORY_CONFIG
    if not config.get('enable_memory_monitoring', True):
        return False

    if not torch.cuda.is_available():
        return False

    mem_info = get_gpu_memory_info()
    if mem_info['allocated_gb'] > config.get('max_vram_usage_gb', 3.5):
        logger.warning(
            f"⚠️  High VRAM usage: {mem_info['allocated_gb']:.2f}GB / "
            f"{mem_info['total_gb']:.1f}GB"
        )
        return True
    return False


# ============================================================
# Configuration
# ============================================================

# Paths
DATA_DIR = '/mnt/e/Drive2/Good/STOFS_TRAINING_DATA'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'
CHECKPOINT_DIR = f'{OUTPUT_DIR}/outputs/checkpoints'
ENSEMBLE_OUTPUT_DIR = f'{OUTPUT_DIR}/outputs/ensemble'

# Domain (must match training)
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Normalization constants (must match training)
ETA_SCALE = 2.0
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0

# Default ensemble settings
DEFAULT_N_MEMBERS = 50
DEFAULT_FORECAST_HOURS = 48

# ============================================================
# Memory Management Settings (RTX 3050 4GB VRAM optimized)
# ============================================================
MEMORY_CONFIG = {
    'cache_clear_interval': 5,      # Clear GPU cache every N members
    'gc_collect_interval': 10,      # Run gc.collect() every N members
    'use_float16_storage': True,    # Store results in float16
    'max_vram_usage_gb': 3.5,       # Warn if VRAM usage exceeds this
    'enable_memory_monitoring': True,
}

# Perturbation parameters
PERTURBATION_CONFIG = {
    'wind_speed_std': 0.15,      # 15% standard deviation
    'wind_direction_std': 10.0,  # 10 degrees standard deviation
    'pressure_std': 300.0,       # 300 Pa (3 hPa) standard deviation
    'initial_cwl_std': 0.02,     # 2 cm standard deviation
    'spatial_correlation': 3.0,  # Gaussian smoothing sigma for correlated noise
}

# Exceedance thresholds (meters)
EXCEEDANCE_THRESHOLDS = [0.3, 0.5, 1.0, 1.5, 2.0]

# Key stations for time series output (approximate locations)
# You can customize these for your domain
# Added CO-OPS station IDs for observation fetching
KEY_STATIONS = {
    'Atlantic_City': {'lon': -74.42, 'lat': 39.36, 'coops_id': '8534720'},
    'Sandy_Hook': {'lon': -74.01, 'lat': 40.47, 'coops_id': '8531680'},
    'The_Battery': {'lon': -74.01, 'lat': 40.70, 'coops_id': '8518750'},
    'Lewes_DE': {'lon': -75.12, 'lat': 38.78, 'coops_id': '8557380'},
    'Cape_May': {'lon': -74.96, 'lat': 38.97, 'coops_id': '8536110'},
}


# ============================================================
# Model Architecture (must match training)
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
    """Message passing block with SWE-inspired gradient term."""
    
    def __init__(self, hidden_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.gradient_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, h, edge_index, edge_attr):
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        h_gradient = h_dst - h_src
        
        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)
        
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)
        
        return h_new, edge_attr


class PhysicsInformedCWLModel(nn.Module):
    """GNN model for CWL prediction."""
    
    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int = 6,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        node_input_dim = state_dim + static_feature_dim + forcing_feature_dim
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        self.layers = nn.ModuleList([
            SWEInspiredGraphBlock(hidden_dim, use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
    
    def forward(self, x, static_features, forcing_features, edge_index, edge_attr):
        node_input = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_input)
        e = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        
        return self.decoder(h)


# ============================================================
# Perturbation Generators
# ============================================================

class MeteorologicalPerturbationGenerator:
    """
    Generate perturbed meteorological forcing for ensemble members.
    
    Perturbation types:
    1. Wind speed scaling (multiplicative)
    2. Wind direction rotation
    3. Pressure offset
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or PERTURBATION_CONFIG
        self.rng = np.random.default_rng()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
    
    def generate(self, base_forcing: Dict, n_members: int) -> List[Dict]:
        """
        Generate n_members perturbed forcing scenarios.
        
        Args:
            base_forcing: Dictionary with u10, v10, pressure arrays [T, N]
            n_members: Number of ensemble members
            
        Returns:
            List of perturbed forcing dictionaries
        """
        ensembles = []
        
        for i in range(n_members):
            perturbed = self._perturb_single(base_forcing, i)
            ensembles.append(perturbed)
        
        return ensembles
    
    def _perturb_single(self, base_forcing: Dict, member_idx: int) -> Dict:
        """Generate single perturbed forcing."""
        perturbed = {}
        
        u10 = base_forcing['u10'].astype(np.float32)
        v10 = base_forcing['v10'].astype(np.float32)
        pressure = base_forcing['pressure'].astype(np.float32)
        
        # 1. Wind speed perturbation (multiplicative)
        wind_scale = 1.0 + self.rng.normal(0, self.config['wind_speed_std'])
        wind_scale = np.clip(wind_scale, 0.5, 1.5)  # Limit to ±50%
        
        u10_scaled = u10 * wind_scale
        v10_scaled = v10 * wind_scale
        
        # 2. Wind direction perturbation (rotation)
        angle_deg = self.rng.normal(0, self.config['wind_direction_std'])
        angle_rad = np.radians(angle_deg)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        u10_rot = u10_scaled * cos_a - v10_scaled * sin_a
        v10_rot = u10_scaled * sin_a + v10_scaled * cos_a
        
        perturbed['u10'] = u10_rot
        perturbed['v10'] = v10_rot
        
        # 3. Pressure perturbation (additive, spatially correlated)
        pressure_offset = self.rng.normal(0, self.config['pressure_std'])
        
        # Add some spatial variation
        if pressure.ndim == 2:
            spatial_noise = self.rng.normal(0, self.config['pressure_std'] * 0.3, 
                                            size=pressure.shape[1])
            spatial_noise = gaussian_filter(spatial_noise, 
                                           sigma=self.config['spatial_correlation'])
            perturbed['pressure'] = pressure + (pressure_offset + spatial_noise) / PRESSURE_SCALE
        else:
            perturbed['pressure'] = pressure + pressure_offset / PRESSURE_SCALE
        
        # Store perturbation parameters for reference
        perturbed['_perturbation_params'] = {
            'member_idx': member_idx,
            'wind_scale': wind_scale,
            'wind_rotation_deg': angle_deg,
            'pressure_offset_pa': pressure_offset,
        }
        
        return perturbed


class InitialConditionPerturbationGenerator:
    """
    Generate perturbed initial water level conditions.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or PERTURBATION_CONFIG
        self.rng = np.random.default_rng()
    
    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
    
    def generate(self, base_cwl: np.ndarray, n_members: int) -> List[np.ndarray]:
        """
        Generate perturbed initial conditions.
        
        Args:
            base_cwl: Base initial water level [N]
            n_members: Number of ensemble members
            
        Returns:
            List of perturbed initial conditions
        """
        ensembles = []
        
        for i in range(n_members):
            # Spatially correlated noise
            noise = self.rng.normal(0, self.config['initial_cwl_std'], 
                                    size=base_cwl.shape)
            
            # Apply spatial smoothing for correlation
            if self.config['spatial_correlation'] > 0:
                noise = gaussian_filter(noise, sigma=self.config['spatial_correlation'])
                # Renormalize after smoothing
                noise = noise * self.config['initial_cwl_std'] / (noise.std() + 1e-8)
            
            perturbed = base_cwl + noise
            ensembles.append(perturbed)
        
        return ensembles


class HurricaneTrackPerturbationGenerator:
    """
    Generate perturbed hurricane tracks for ensemble forcing.
    
    Based on NHC forecast cone uncertainty.
    """
    
    def __init__(self):
        self.rng = np.random.default_rng()
        
        # NHC cone radii (nautical miles) by forecast hour
        # Based on 5-year average errors
        self.cross_track_error_nm = {
            0: 0, 12: 25, 24: 40, 36: 55, 48: 70, 72: 100, 96: 130, 120: 160
        }
        
        # Along-track timing uncertainty (hours)
        self.along_track_error_hr = {
            0: 0, 24: 3, 48: 6, 72: 9, 96: 12, 120: 15
        }
    
    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
    
    def perturb_track(
        self, 
        track_lons: np.ndarray, 
        track_lats: np.ndarray,
        track_times: np.ndarray,  # hours from t=0
        n_members: int
    ) -> List[Dict]:
        """
        Generate perturbed hurricane tracks.
        
        Args:
            track_lons: Storm center longitudes
            track_lats: Storm center latitudes
            track_times: Forecast hours
            n_members: Number of ensemble members
            
        Returns:
            List of perturbed track dictionaries
        """
        ensembles = []
        
        for i in range(n_members):
            # Cross-track perturbation
            cross_track_nm = np.array([
                self.rng.normal(0, self._get_cross_track_error(t))
                for t in track_times
            ])
            
            # Convert nm to degrees (approximate)
            cross_track_deg = cross_track_nm / 60.0
            
            # Compute track heading for perpendicular offset
            heading = np.arctan2(
                np.gradient(track_lons),
                np.gradient(track_lats)
            )
            
            # Apply perpendicular offset
            perturbed_lons = track_lons + cross_track_deg * np.cos(heading + np.pi/2)
            perturbed_lats = track_lats + cross_track_deg * np.sin(heading + np.pi/2)
            
            # Along-track (timing) perturbation
            timing_offset = self.rng.normal(0, 3)  # hours
            
            ensembles.append({
                'lons': perturbed_lons,
                'lats': perturbed_lats,
                'times': track_times,
                'timing_offset_hr': timing_offset,
                'cross_track_nm': cross_track_nm,
            })
        
        return ensembles
    
    def _get_cross_track_error(self, forecast_hour: float) -> float:
        """Interpolate cross-track error for given forecast hour."""
        hours = sorted(self.cross_track_error_nm.keys())
        errors = [self.cross_track_error_nm[h] for h in hours]
        return np.interp(forecast_hour, hours, errors)


# ============================================================
# Ensemble Runner
# ============================================================

class EnsembleForecaster:
    """
    Main class for running ensemble forecasts.
    """
    
    def __init__(
        self,
        model: nn.Module,
        mesh_data: Dict,
        device: torch.device = None,
    ):
        self.model = model
        self.mesh_data = mesh_data
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        # Precompute static mesh features
        self._prepare_mesh_features()
        
        # Perturbation generators
        self.met_perturber = MeteorologicalPerturbationGenerator()
        self.ic_perturber = InitialConditionPerturbationGenerator()
        
        logger.info(f"EnsembleForecaster initialized on {self.device}")
        logger.info(f"Mesh: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges")
    
    def _prepare_mesh_features(self):
        """Prepare static mesh features for inference."""
        self.lon = self.mesh_data['lon'].astype(np.float32)
        self.lat = self.mesh_data['lat'].astype(np.float32)
        self.depth = self.mesh_data['depth'].astype(np.float32)
        self.num_nodes = len(self.lon)
        
        # Edge index
        self.edge_index = torch.tensor(
            self.mesh_data['edge_index'], dtype=torch.long
        ).to(self.device)
        
        # Cartesian coordinates
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = 6371000.0
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)
        
        # Normalized positions
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1
        
        # Normalized depth
        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
        
        # Base static features (without water level - added per timestep)
        self.static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)
        
        # Edge features
        src, dst = self.mesh_data['edge_index']
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)
        char_length = np.median(dist) + 1e-8
        
        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        ).to(self.device)
    
    def _prepare_static_features(self, cwl: np.ndarray) -> torch.Tensor:
        """Prepare static features including water level."""
        water_level = self.depth + cwl * ETA_SCALE
        water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        
        static_features = np.concatenate([
            self.static_base,
            water_level_norm[:, np.newaxis]
        ], axis=1)
        
        return torch.tensor(static_features, dtype=torch.float32).to(self.device)
    
    def _prepare_forcing_features(self, forcing: Dict, time_idx: int) -> torch.Tensor:
        """Prepare forcing features for a single timestep."""
        u10 = forcing['u10'][time_idx].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][time_idx].astype(np.float32) / WIND_SCALE
        pressure = forcing['pressure'][time_idx].astype(np.float32)
        
        forcing_features = np.stack([u10, v10, pressure], axis=1)
        return torch.tensor(forcing_features, dtype=torch.float32).to(self.device)
    
    def run_single_forecast(
        self,
        initial_cwl: np.ndarray,
        forcing: Dict,
        forecast_hours: int,
    ) -> np.ndarray:
        """
        Run single deterministic forecast.
        
        Args:
            initial_cwl: Initial water level [N]
            forcing: Forcing dictionary with u10, v10, pressure [T, N]
            forecast_hours: Number of hours to forecast
            
        Returns:
            predictions: [forecast_hours + 1, N]
        """
        predictions = [initial_cwl.copy()]
        
        current_cwl = initial_cwl / ETA_SCALE
        current_cwl_tensor = torch.tensor(
            current_cwl, dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            for t in range(forecast_hours):
                # Prepare features
                static_features = self._prepare_static_features(current_cwl)
                forcing_features = self._prepare_forcing_features(forcing, t)
                
                # Model prediction
                x = current_cwl_tensor.unsqueeze(1)
                pred = self.model(
                    x, static_features, forcing_features,
                    self.edge_index, self.edge_attr
                ).squeeze()
                
                # Update state
                current_cwl_tensor = pred
                current_cwl = pred.cpu().numpy()
                predictions.append(current_cwl * ETA_SCALE)
        
        return np.array(predictions)
    
    def run_ensemble(
        self,
        initial_cwl: np.ndarray,
        base_forcing: Dict,
        n_members: int = 50,
        forecast_hours: int = 48,
        perturb_forcing: bool = True,
        perturb_ic: bool = False,
        seed: int = None,
        show_progress: bool = True,
        memory_config: Dict = None,
    ) -> Dict:
        """
        Run ensemble forecast with memory-safe execution.

        Args:
            initial_cwl: Initial water level [N]
            base_forcing: Base forcing dictionary
            n_members: Number of ensemble members
            forecast_hours: Forecast length in hours
            perturb_forcing: Whether to perturb meteorological forcing
            perturb_ic: Whether to perturb initial conditions
            seed: Random seed for reproducibility
            show_progress: Whether to show progress bar
            memory_config: Memory management configuration

        Returns:
            Dictionary with ensemble results and statistics
        """
        start_time = time.time()
        mem_cfg = memory_config or MEMORY_CONFIG

        # Log initial memory state
        if self.device.type == 'cuda':
            mem_info = get_gpu_memory_info()
            logger.info(f"Initial GPU memory: {mem_info['allocated_gb']:.2f}GB allocated")

        if seed is not None:
            self.met_perturber.set_seed(seed)
            self.ic_perturber.set_seed(seed + 1000)

        # Generate perturbations
        if perturb_forcing:
            forcing_ensembles = self.met_perturber.generate(base_forcing, n_members)
        else:
            forcing_ensembles = [base_forcing] * n_members

        if perturb_ic:
            ic_ensembles = self.ic_perturber.generate(initial_cwl, n_members)
        else:
            ic_ensembles = [initial_cwl] * n_members

        # Storage - use float16 if configured to save memory
        storage_dtype = np.float16 if mem_cfg.get('use_float16_storage', True) else np.float32
        ensemble_predictions = np.zeros(
            (n_members, forecast_hours + 1, self.num_nodes),
            dtype=storage_dtype
        )

        # Run ensemble members
        logger.info(f"Running {n_members} ensemble members for {forecast_hours}h forecast...")
        logger.info(f"Storage dtype: {storage_dtype}, Memory clearing every {mem_cfg['cache_clear_interval']} members")

        cache_interval = mem_cfg.get('cache_clear_interval', 5)
        gc_interval = mem_cfg.get('gc_collect_interval', 10)

        for member_idx in range(n_members):
            predictions = self.run_single_forecast(
                ic_ensembles[member_idx],
                forcing_ensembles[member_idx],
                forecast_hours
            )
            ensemble_predictions[member_idx] = predictions.astype(storage_dtype)

            # Explicit cleanup of predictions array
            del predictions

            if show_progress and (member_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (member_idx + 1) / elapsed
                remaining = (n_members - member_idx - 1) / rate

                mem_str = ""
                if self.device.type == 'cuda':
                    mem_info = get_gpu_memory_info()
                    mem_str = f" | VRAM: {mem_info['allocated_gb']:.2f}GB"

                logger.info(
                    f"  Completed {member_idx + 1}/{n_members} members "
                    f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining){mem_str}"
                )

            # Aggressive memory management (key for WSL stability)
            if self.device.type == 'cuda':
                # Clear cache frequently
                if (member_idx + 1) % cache_interval == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Check for memory warnings
                if check_memory_warning(mem_cfg):
                    logger.warning("Forcing aggressive memory cleanup...")
                    clear_gpu_memory()

            # Periodic garbage collection
            if (member_idx + 1) % gc_interval == 0:
                gc.collect()

        # Final cleanup before computing statistics
        clear_gpu_memory()

        elapsed_total = time.time() - start_time
        logger.info(f"Ensemble complete in {elapsed_total:.1f}s ({elapsed_total/n_members:.2f}s per member)")
        
        # Compute statistics
        stats = self._compute_statistics(ensemble_predictions)
        
        # Store perturbation parameters
        perturbation_params = []
        if perturb_forcing:
            for f in forcing_ensembles:
                if '_perturbation_params' in f:
                    perturbation_params.append(f['_perturbation_params'])
        
        return {
            'predictions': ensemble_predictions,
            'statistics': stats,
            'metadata': {
                'n_members': n_members,
                'forecast_hours': forecast_hours,
                'perturb_forcing': perturb_forcing,
                'perturb_ic': perturb_ic,
                'seed': seed,
                'elapsed_seconds': elapsed_total,
            },
            'perturbation_params': perturbation_params,
        }
    
    def _compute_statistics(self, ensemble_predictions: np.ndarray) -> Dict:
        """Compute ensemble statistics."""
        stats = {}
        
        # Central tendency
        stats['mean'] = np.mean(ensemble_predictions, axis=0)
        stats['median'] = np.median(ensemble_predictions, axis=0)
        
        # Spread
        stats['std'] = np.std(ensemble_predictions, axis=0)
        stats['min'] = np.min(ensemble_predictions, axis=0)
        stats['max'] = np.max(ensemble_predictions, axis=0)
        
        # Percentiles
        for p in [5, 10, 25, 75, 90, 95]:
            stats[f'p{p}'] = np.percentile(ensemble_predictions, p, axis=0)
        
        # Interquartile range
        stats['iqr'] = stats['p75'] - stats['p25']
        
        # Exceedance probabilities
        for thresh in EXCEEDANCE_THRESHOLDS:
            exceed = (ensemble_predictions > thresh).astype(float)
            stats[f'prob_exceed_{thresh}m'] = np.mean(exceed, axis=0)
        
        return stats
    
    def find_nearest_node(self, lon: float, lat: float) -> int:
        """Find index of nearest mesh node to given coordinates."""
        dist = np.sqrt((self.lon - lon)**2 + (self.lat - lat)**2)
        return np.argmin(dist)
    
    def get_station_indices(self, stations: Dict = None) -> Dict[str, int]:
        """Get mesh indices for named stations."""
        stations = stations or KEY_STATIONS
        indices = {}
        for name, coords in stations.items():
            indices[name] = self.find_nearest_node(coords['lon'], coords['lat'])
        return indices


# ============================================================
# Visualization Functions
# ============================================================

def find_valid_water_node(
    lon: np.ndarray,
    lat: np.ndarray,
    predictions: np.ndarray,
    station_lon: float,
    station_lat: float,
    search_radius: float = 0.2,
) -> int:
    """
    Find the nearest valid water node (with non-zero signal) near a station.

    Args:
        lon, lat: Mesh coordinates
        predictions: Ensemble predictions [members, times, nodes]
        station_lon, station_lat: Station coordinates
        search_radius: Search radius in degrees (~0.1 = 10km)

    Returns:
        Index of best node (valid water with signal)
    """
    dist = np.sqrt((lon - station_lon)**2 + (lat - station_lat)**2)

    # Find nodes within search radius
    nearby_mask = dist < search_radius

    if not nearby_mask.any():
        # No nodes nearby, return nearest
        return np.argmin(dist)

    nearby_indices = np.where(nearby_mask)[0]

    # For each nearby node, compute the signal range (across ensemble mean)
    mean_preds = np.mean(predictions, axis=0)  # [times, nodes]

    best_idx = None
    best_score = -1

    for nidx in nearby_indices:
        ts = mean_preds[:, nidx]

        # Check if this node has valid signal (not all zeros/near-zeros)
        ts_range = np.ptp(ts)  # peak-to-peak range
        ts_std = np.std(ts)

        # Score: prefer nodes with larger range, weighted by proximity
        if ts_range > 0.01:  # Minimum 1cm range to be "valid"
            proximity = 1.0 / (dist[nidx] + 0.01)  # Inverse distance weight
            score = ts_range * proximity

            if score > best_score:
                best_score = score
                best_idx = nidx

    if best_idx is not None:
        return best_idx
    else:
        # No valid nodes found, return nearest
        return np.argmin(dist)


def fetch_coops_observations(
    station_id: str,
    start_time: datetime,
    end_time: datetime,
    datum: str = 'MSL',
) -> Optional[pd.DataFrame]:
    """
    Fetch CO-OPS water level observations for a station.

    Args:
        station_id: CO-OPS station ID (7 digits)
        start_time: Start datetime
        end_time: End datetime
        datum: Vertical datum (default: MSL)

    Returns:
        DataFrame with water level observations or None if fetch fails
    """
    if not SEARVEY_AVAILABLE:
        return None

    try:
        obs_data = fetch_coops_station(
            station_id=station_id,
            start_date=start_time,
            end_date=end_time,
            product='water_level',
            datum=datum,
        )

        if obs_data is not None and len(obs_data) > 0:
            logger.info(f"  Fetched {len(obs_data)} observations for station {station_id}")
            return obs_data
        else:
            logger.warning(f"  No observations available for station {station_id}")
            return None

    except Exception as e:
        logger.warning(f"  Failed to fetch observations for station {station_id}: {e}")
        return None


def plot_ensemble_spaghetti(
    ensemble_results: Dict,
    lon: np.ndarray,
    lat: np.ndarray,
    station_coords: Dict,
    output_dir: str,
    forecast_start_time: datetime = None,
    fetch_obs: bool = True,
    datum: str = 'MSL',
):
    """
    Create spaghetti plots for key stations with optional observations.

    Args:
        ensemble_results: Dictionary with ensemble predictions
        lon: Mesh longitude coordinates
        lat: Mesh latitude coordinates
        station_coords: Dictionary of station info (lon, lat, coops_id)
        output_dir: Output directory for plots
        forecast_start_time: Start time of forecast (for fetching obs)
        fetch_obs: Whether to fetch and plot observations
        datum: Vertical datum for observations
    """
    os.makedirs(output_dir, exist_ok=True)

    predictions = ensemble_results['predictions']
    n_members, n_times, n_nodes = predictions.shape

    hours = np.arange(n_times)

    # Create time axis if start time provided
    if forecast_start_time is not None:
        times = [forecast_start_time + timedelta(hours=h) for h in range(n_times)]
        use_datetime_axis = True
    else:
        times = hours
        use_datetime_axis = False

    for station_name, coords in station_coords.items():
        # Find valid water node (not just nearest - may be on land in subsampled mesh)
        node_idx = find_valid_water_node(
            lon=lon,
            lat=lat,
            predictions=predictions,
            station_lon=coords['lon'],
            station_lat=coords['lat'],
            search_radius=0.3,  # ~30km search radius for subsampled mesh
        )

        # Log the node selection
        dist = np.sqrt((lon - coords['lon'])**2 + (lat - coords['lat'])**2)
        nearest_idx = np.argmin(dist)
        if node_idx != nearest_idx:
            logger.info(f"  {station_name}: Using node {node_idx} (valid water) instead of nearest {nearest_idx} (land/zero signal)")

        fig, ax = plt.subplots(figsize=(12, 6))

        # Fetch observations if available
        obs_data = None
        if fetch_obs and SEARVEY_AVAILABLE and forecast_start_time is not None:
            coops_id = coords.get('coops_id')
            if coops_id:
                logger.info(f"Fetching observations for {station_name} (CO-OPS: {coops_id})...")
                obs_data = fetch_coops_observations(
                    station_id=coops_id,
                    start_time=forecast_start_time,
                    end_time=forecast_start_time + timedelta(hours=n_times),
                    datum=datum,
                )

        # Plot observations first (so they appear behind ensemble)
        obs_plotted = False
        if obs_data is not None and len(obs_data) > 0:
            # Get water level column
            if 'water_level' in obs_data.columns:
                wl_col = 'water_level'
            elif 'v' in obs_data.columns:
                wl_col = 'v'
            else:
                wl_col = obs_data.columns[0]

            ax.plot(obs_data.index, obs_data[wl_col],
                   'ko', markersize=3, alpha=0.6, label='CO-OPS Obs', zorder=5)
            obs_plotted = True

        # Individual ensemble members
        if use_datetime_axis:
            for m in range(n_members):
                ax.plot(times, predictions[m, :, node_idx],
                       color='gray', alpha=0.2, linewidth=0.5)
        else:
            for m in range(n_members):
                ax.plot(hours, predictions[m, :, node_idx],
                       color='gray', alpha=0.2, linewidth=0.5)

        # Statistics
        mean = np.mean(predictions[:, :, node_idx], axis=0)
        p10 = np.percentile(predictions[:, :, node_idx], 10, axis=0)
        p90 = np.percentile(predictions[:, :, node_idx], 90, axis=0)
        p25 = np.percentile(predictions[:, :, node_idx], 25, axis=0)
        p75 = np.percentile(predictions[:, :, node_idx], 75, axis=0)

        # Shaded uncertainty
        x_axis = times if use_datetime_axis else hours
        ax.fill_between(x_axis, p10, p90, alpha=0.2, color='blue', label='10-90%')
        ax.fill_between(x_axis, p25, p75, alpha=0.3, color='blue', label='25-75%')

        # Mean and median
        ax.plot(x_axis, mean, 'b-', linewidth=2, label='Ensemble Mean')
        ax.plot(x_axis, np.median(predictions[:, :, node_idx], axis=0),
               'g--', linewidth=1.5, label='Ensemble Median')

        # Reference lines
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        for thresh in [0.5, 1.0, 1.5]:
            ax.axhline(y=thresh, color='orange', linestyle='--',
                      linewidth=0.5, alpha=0.5)

        # Formatting
        if use_datetime_axis:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_xlabel('Date/Time (UTC)', fontsize=12)
        else:
            ax.set_xlabel('Forecast Hour', fontsize=12)

        ax.set_ylabel(f'Coastal Water Level (m, {datum})', fontsize=12)

        # Title with obs info
        title = f'Ensemble Forecast - {station_name}\n{n_members} members'
        if obs_plotted:
            title += f' | CO-OPS: {coords.get("coops_id", "N/A")}'
        ax.set_title(title, fontsize=14)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if not use_datetime_axis:
            ax.set_xlim(0, n_times - 1)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/spaghetti_{station_name}.png', dpi=150)
        plt.close()

    logger.info(f"Saved spaghetti plots to {output_dir}")


def plot_exceedance_maps(
    ensemble_results: Dict,
    lon: np.ndarray,
    lat: np.ndarray,
    output_dir: str,
    time_indices: List[int] = None,
):
    """
    Create exceedance probability maps.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = ensemble_results['statistics']
    n_times = stats['mean'].shape[0]
    
    time_indices = time_indices or [12, 24, 36, 48]
    time_indices = [t for t in time_indices if t < n_times]
    
    for thresh in EXCEEDANCE_THRESHOLDS:
        prob_key = f'prob_exceed_{thresh}m'
        if prob_key not in stats:
            continue
        
        prob = stats[prob_key]
        
        fig, axes = plt.subplots(1, len(time_indices), figsize=(4*len(time_indices), 5))
        if len(time_indices) == 1:
            axes = [axes]
        
        for ax, t_idx in zip(axes, time_indices):
            sc = ax.scatter(lon, lat, c=prob[t_idx], cmap='YlOrRd', 
                           s=3, vmin=0, vmax=1)
            ax.set_title(f't+{t_idx}h')
            ax.set_aspect('equal')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        
        # Colorbar
        cbar = fig.colorbar(sc, ax=axes, shrink=0.8, label=f'P(CWL > {thresh}m)')
        
        plt.suptitle(f'Exceedance Probability - {thresh}m Threshold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/exceedance_prob_{thresh}m.png', dpi=150)
        plt.close()
    
    logger.info(f"Saved exceedance maps to {output_dir}")


def plot_ensemble_spread(
    ensemble_results: Dict,
    lon: np.ndarray,
    lat: np.ndarray,
    output_dir: str,
    time_indices: List[int] = None,
):
    """
    Create ensemble spread (standard deviation) maps.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = ensemble_results['statistics']
    std = stats['std']
    n_times = std.shape[0]
    
    time_indices = time_indices or [12, 24, 36, 48]
    time_indices = [t for t in time_indices if t < n_times]
    
    fig, axes = plt.subplots(1, len(time_indices), figsize=(4*len(time_indices), 5))
    if len(time_indices) == 1:
        axes = [axes]
    
    vmax = np.percentile(std, 95)
    
    for ax, t_idx in zip(axes, time_indices):
        sc = ax.scatter(lon, lat, c=std[t_idx], cmap='viridis', 
                       s=3, vmin=0, vmax=vmax)
        ax.set_title(f't+{t_idx}h')
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    cbar = fig.colorbar(sc, ax=axes, shrink=0.8, label='Std Dev (m)')
    
    plt.suptitle('Ensemble Spread (Standard Deviation)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_spread.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved spread map to {output_dir}")


def plot_summary_dashboard(
    ensemble_results: Dict,
    lon: np.ndarray,
    lat: np.ndarray,
    station_coords: Dict,
    output_dir: str,
):
    """
    Create comprehensive summary dashboard.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = ensemble_results['predictions']
    stats = ensemble_results['statistics']
    metadata = ensemble_results['metadata']
    
    n_members, n_times, n_nodes = predictions.shape
    hours = np.arange(n_times)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Row 1: Maps at peak time ---
    # Find time of maximum mean surge
    max_mean = np.max(stats['mean'], axis=1)
    peak_time = np.argmax(max_mean)
    
    # Mean at peak
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(lon, lat, c=stats['mean'][peak_time], cmap='RdBu_r', 
                    s=3, vmin=-1, vmax=1)
    ax1.set_title(f'Mean CWL at t+{peak_time}h')
    ax1.set_aspect('equal')
    plt.colorbar(sc, ax=ax1, label='CWL (m)')
    
    # Std at peak
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(lon, lat, c=stats['std'][peak_time], cmap='viridis', 
                    s=3, vmin=0, vmax=0.5)
    ax2.set_title(f'Uncertainty (Std) at t+{peak_time}h')
    ax2.set_aspect('equal')
    plt.colorbar(sc, ax=ax2, label='Std (m)')
    
    # P(exceed 0.5m) at peak
    ax3 = fig.add_subplot(gs[0, 2])
    prob_key = 'prob_exceed_0.5m'
    if prob_key in stats:
        sc = ax3.scatter(lon, lat, c=stats[prob_key][peak_time], cmap='YlOrRd', 
                        s=3, vmin=0, vmax=1)
        ax3.set_title(f'P(CWL > 0.5m) at t+{peak_time}h')
        ax3.set_aspect('equal')
        plt.colorbar(sc, ax=ax3, label='Probability')
    
    # --- Row 2: Time series at key stations ---
    station_indices = {}
    for name, coords in list(station_coords.items())[:3]:
        dist = np.sqrt((lon - coords['lon'])**2 + (lat - coords['lat'])**2)
        station_indices[name] = np.argmin(dist)
    
    for i, (name, node_idx) in enumerate(station_indices.items()):
        ax = fig.add_subplot(gs[1, i])
        
        mean = np.mean(predictions[:, :, node_idx], axis=0)
        p10 = np.percentile(predictions[:, :, node_idx], 10, axis=0)
        p90 = np.percentile(predictions[:, :, node_idx], 90, axis=0)
        
        ax.fill_between(hours, p10, p90, alpha=0.3, color='blue')
        ax.plot(hours, mean, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.set_xlabel('Forecast Hour')
        ax.set_ylabel('CWL (m)')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
    
    # --- Row 3: Ensemble diagnostics ---
    # Max surge distribution
    ax7 = fig.add_subplot(gs[2, 0])
    max_surge_per_member = np.max(predictions, axis=(1, 2))
    ax7.hist(max_surge_per_member, bins=20, edgecolor='black', alpha=0.7)
    ax7.axvline(np.mean(max_surge_per_member), color='r', linestyle='--', 
               label=f'Mean: {np.mean(max_surge_per_member):.2f}m')
    ax7.set_xlabel('Maximum Surge (m)')
    ax7.set_ylabel('Count')
    ax7.set_title('Distribution of Peak Surge')
    ax7.legend()
    
    # Domain-mean time series
    ax8 = fig.add_subplot(gs[2, 1])
    domain_mean = np.mean(predictions, axis=2)  # [members, times]
    for m in range(min(20, n_members)):
        ax8.plot(hours, domain_mean[m], color='gray', alpha=0.3, linewidth=0.5)
    ax8.plot(hours, np.mean(domain_mean, axis=0), 'b-', linewidth=2, label='Ensemble Mean')
    ax8.set_xlabel('Forecast Hour')
    ax8.set_ylabel('Domain-Mean CWL (m)')
    ax8.set_title('Domain-Averaged Response')
    ax8.grid(True, alpha=0.3)
    
    # Metadata text
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    info_text = (
        f"Ensemble Summary\n"
        f"{'─' * 30}\n"
        f"Members: {metadata['n_members']}\n"
        f"Forecast length: {metadata['forecast_hours']}h\n"
        f"Runtime: {metadata['elapsed_seconds']:.1f}s\n"
        f"Per member: {metadata['elapsed_seconds']/metadata['n_members']:.2f}s\n"
        f"{'─' * 30}\n"
        f"Peak surge time: t+{peak_time}h\n"
        f"Max ensemble mean: {np.max(stats['mean']):.2f}m\n"
        f"Max ensemble max: {np.max(stats['max']):.2f}m\n"
        f"Max uncertainty: {np.max(stats['std']):.2f}m\n"
    )
    ax9.text(0.1, 0.9, info_text, transform=ax9.transAxes, 
            fontfamily='monospace', fontsize=11, verticalalignment='top')
    
    plt.suptitle('Storm Surge Ensemble Forecast Summary', fontsize=16, fontweight='bold')
    plt.savefig(f'{output_dir}/ensemble_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dashboard to {output_dir}")


# ============================================================
# Data Loading Utilities
# ============================================================

def load_mesh_data(mesh_path: str) -> Dict:
    """Load preprocessed mesh data."""
    mesh_np = np.load(mesh_path)
    mesh_data = {k: mesh_np[k] for k in mesh_np.files}
    mesh_np.close()
    return mesh_data


def load_forcing_for_ensemble(
    date_dir: str,
    met_dir: str,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    num_hours: int = 48,
) -> Dict:
    """
    Load and interpolate forcing data for ensemble run.
    
    This is a simplified version - in production you'd connect
    to real-time GFS/HRRR data.
    """
    base_path = f'{DATA_DIR}/{date_dir}/{met_dir}'
    
    # Load grid coordinates
    sample_file = f'{base_path}/stofs_2d_glo_ncst.222.nc'
    if not os.path.exists(sample_file):
        sample_file = f'{base_path}/stofs_2d_glo_fcst1.222.nc'
    
    nc = NCDataset(sample_file)
    grid_lon = np.array(nc.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc.variables['grid_yt'][:], dtype=np.float32)
    nc.close()
    
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    # Load forcing data
    all_u, all_v, all_p = [], [], []
    
    for file_type in ['ncst', 'fcst1', 'fcst2']:
        wind_file = f'{base_path}/stofs_2d_glo_{file_type}.222.nc'
        pres_file = f'{base_path}/stofs_2d_glo_{file_type}.221.nc'
        
        if os.path.exists(wind_file):
            nc = NCDataset(wind_file)
            all_u.append(np.array(nc.variables['ugrd10m'][:], dtype=np.float32))
            all_v.append(np.array(nc.variables['vgrd10m'][:], dtype=np.float32))
            nc.close()
        
        if os.path.exists(pres_file):
            nc = NCDataset(pres_file)
            all_p.append(np.array(nc.variables['pressfc'][:], dtype=np.float32))
            nc.close()
    
    u_all = np.concatenate(all_u, axis=0)[:num_hours]
    v_all = np.concatenate(all_v, axis=0)[:num_hours]
    p_all = np.concatenate(all_p, axis=0)[:num_hours]
    
    # Interpolate to mesh nodes
    lon_sort_idx = np.argsort(grid_lon)
    lat_sort_idx = np.argsort(grid_lat)
    grid_lon_sorted = grid_lon[lon_sort_idx]
    grid_lat_sorted = grid_lat[lat_sort_idx]
    
    num_times = u_all.shape[0]
    num_nodes = len(node_lon)
    
    result = {
        'u10': np.zeros((num_times, num_nodes), dtype=np.float32),
        'v10': np.zeros((num_times, num_nodes), dtype=np.float32),
        'pressure': np.zeros((num_times, num_nodes), dtype=np.float32),
    }
    
    for t in range(num_times):
        for var, data in [('u10', u_all), ('v10', v_all), ('pressure', p_all)]:
            data_t = data[t]
            data_sorted = data_t[lat_sort_idx][:, lon_sort_idx]
            
            interp = RegularGridInterpolator(
                (grid_lat_sorted, grid_lon_sorted),
                data_sorted,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            
            values = interp(np.column_stack([node_lat, node_lon]))
            if np.any(np.isnan(values)):
                values[np.isnan(values)] = np.nanmean(values)
            
            if var == 'pressure':
                values = (values - PRESSURE_MEAN) / PRESSURE_SCALE
            
            result[var][t] = values
    
    return result


def load_initial_condition(
    cwl_file: str,
    global_indices: np.ndarray,
    time_idx: int = 0,
) -> np.ndarray:
    """Load initial water level condition."""
    nc = NCDataset(cwl_file, 'r')
    zeta = nc.variables['zeta'][time_idx, global_indices]
    nc.close()

    zeta = np.where(zeta < -9000, 0.0, zeta)
    return zeta.astype(np.float32)


def load_from_preprocessed(
    preprocessed_path: str,
    forecast_hours: int = 48,
) -> Tuple[np.ndarray, Dict]:
    """
    Load initial condition and forcing from preprocessed data.

    Args:
        preprocessed_path: Path to preprocessed .npz file
        forecast_hours: Number of forecast hours to load

    Returns:
        initial_cwl: Initial water level [N] - in physical units (meters)
        forcing: Dictionary with u10, v10, pressure [T, N]
    """
    data = np.load(preprocessed_path)

    # Get initial condition (first timestep)
    # The preprocessed elevation is in PHYSICAL UNITS (meters), NOT normalized
    # The range is approximately -2m to +3m for Mid-Atlantic tides/surge
    elevation = data['elevation'].astype(np.float32)
    initial_cwl = elevation[0]

    # Replace NaN with 0
    initial_cwl = np.where(np.isnan(initial_cwl), 0.0, initial_cwl)

    # NO scaling needed - data is already in physical units (meters)
    # The run_single_forecast will divide by ETA_SCALE before feeding to model

    # Get forcing data - limit to forecast hours
    num_hours = min(forecast_hours, elevation.shape[0] - 1)

    # The preprocessed data has:
    # - u10, v10: in m/s (physical units)
    # - pressure: already normalized by (p - PRESSURE_MEAN) / PRESSURE_SCALE
    # The _prepare_forcing_features method expects:
    # - u10, v10: in m/s, will divide by WIND_SCALE
    # - pressure: already normalized

    u10 = data['u10'][:num_hours].astype(np.float32)
    v10 = data['v10'][:num_hours].astype(np.float32)
    pressure = data['pressure'][:num_hours].astype(np.float32)

    # Replace NaN with 0 in forcing
    u10 = np.where(np.isnan(u10), 0.0, u10)
    v10 = np.where(np.isnan(v10), 0.0, v10)
    pressure = np.where(np.isnan(pressure), 0.0, pressure)

    forcing = {
        'u10': u10,
        'v10': v10,
        'pressure': pressure,
    }

    data.close()
    return initial_cwl, forcing


# ============================================================
# Main Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run ensemble storm surge forecast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Memory-safe options for WSL/limited VRAM:
  --device cpu                  Run on CPU only (slower but stable)
  --n_members 20                Reduce ensemble size
  --forecast_hours 24           Shorter forecast
  --cache_clear_interval 3      More frequent GPU cache clearing

Example for RTX 3050 (4GB):
  python ensemble_inference.py --n_members 30 --cache_clear_interval 3
        """
    )
    parser.add_argument('--checkpoint', type=str,
                       default='best_physics_informed_model.pt',
                       help='Model checkpoint file')
    parser.add_argument('--n_members', type=int, default=DEFAULT_N_MEMBERS,
                       help='Number of ensemble members (default: 50)')
    parser.add_argument('--forecast_hours', type=int, default=DEFAULT_FORECAST_HOURS,
                       help='Forecast length in hours (default: 48)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--perturb_ic', action='store_true',
                       help='Also perturb initial conditions')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')

    # Memory management options
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use: auto, cuda, or cpu (default: auto)')
    parser.add_argument('--cache_clear_interval', type=int, default=5,
                       help='Clear GPU cache every N members (default: 5, lower=safer)')
    parser.add_argument('--no_float16', action='store_true',
                       help='Disable float16 storage (uses more memory)')

    # Observation options
    parser.add_argument('--fetch_obs', action='store_true', default=True,
                       help='Fetch CO-OPS observations for comparison (default: True)')
    parser.add_argument('--no_obs', action='store_true',
                       help='Disable observation fetching')
    parser.add_argument('--datum', type=str, default='MSL',
                       choices=['MSL', 'MLLW', 'NAVD', 'STND'],
                       help='Vertical datum for observations (default: MSL)')
    parser.add_argument('--forecast_date', type=str, default='20251128',
                       help='Forecast date in YYYYMMDD format (default: 20251128)')
    parser.add_argument('--forecast_cycle', type=str, default='00',
                       choices=['00', '06', '12', '18'],
                       help='Forecast cycle hour (default: 00)')
    parser.add_argument('--preprocessed', type=str, default=None,
                       help='Path to preprocessed .npz file (overrides forecast_date/cycle)')

    args = parser.parse_args()

    # Setup memory configuration from args
    memory_config = MEMORY_CONFIG.copy()
    memory_config['cache_clear_interval'] = args.cache_clear_interval
    memory_config['use_float16_storage'] = not args.no_float16

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or f'{ENSEMBLE_OUTPUT_DIR}/run_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ENSEMBLE STORM SURGE FORECAST")
    logger.info("=" * 60)
    logger.info(f"Members: {args.n_members}")
    logger.info(f"Forecast hours: {args.forecast_hours}")
    logger.info(f"Output: {output_dir}")

    # Device selection with memory safety
    if args.device == 'auto':
        if torch.cuda.is_available():
            # Check available VRAM before using GPU
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_vram < 4.0:
                logger.warning(f"Low VRAM detected ({total_vram:.1f}GB). Consider --device cpu")
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({total_vram:.1f}GB VRAM)")
        logger.info(f"Memory config: cache_clear_interval={memory_config['cache_clear_interval']}, "
                   f"float16={memory_config['use_float16_storage']}")

    # Load mesh - use optimized mesh if available
    mesh_path_optimized = f'{OUTPUT_DIR}/data/processed_optimized/mesh_optimized.npz'
    mesh_path_default = f'{OUTPUT_DIR}/data/processed/midatlantic_mesh_v5.npz'
    mesh_path = mesh_path_optimized if os.path.exists(mesh_path_optimized) else mesh_path_default
    logger.info(f"\nLoading mesh from {mesh_path}")
    mesh_data = load_mesh_data(mesh_path)

    # Load model
    checkpoint_path = f'{CHECKPOINT_DIR}/{args.checkpoint}'
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    model = PhysicsInformedCWLModel(
        state_dim=1,
        static_feature_dim=config['static_features'],
        forcing_feature_dim=config['forcing_features'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        use_checkpointing=False,  # Not needed for inference
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize forecaster
    forecaster = EnsembleForecaster(model, mesh_data, device)

    # Load forcing and initial condition
    # Check if preprocessed data is provided or use default
    preprocessed_path = args.preprocessed
    if preprocessed_path is None:
        # Try to find preprocessed data for the forecast date
        preprocessed_path = f'{OUTPUT_DIR}/data/processed_optimized/processed_{args.forecast_date}.npz'

    # Parse forecast start time for observation fetching
    forecast_start_time = datetime.strptime(
        f"{args.forecast_date}{args.forecast_cycle}", "%Y%m%d%H"
    )
    logger.info(f"Forecast start time: {forecast_start_time}")

    if os.path.exists(preprocessed_path):
        logger.info(f"\nLoading from preprocessed: {preprocessed_path}")
        initial_cwl, forcing = load_from_preprocessed(
            preprocessed_path, args.forecast_hours
        )
        logger.info(f"  Initial CWL shape: {initial_cwl.shape}")
        logger.info(f"  Forcing u10 shape: {forcing['u10'].shape}")
    else:
        # Fall back to original loading method
        date_dir = f'stofs_2d_glo.{args.forecast_date}'
        met_dir = f'met_forcing_{args.forecast_cycle}z'
        cwl_file = f'{DATA_DIR}/{date_dir}/stofs_2d_glo.t{args.forecast_cycle}z.fields.cwl.nc'

        logger.info(f"\nLoading forcing from {date_dir}/{met_dir}")
        forcing = load_forcing_for_ensemble(
            date_dir, met_dir,
            mesh_data['lon'], mesh_data['lat'],
            num_hours=args.forecast_hours
        )

        logger.info(f"Loading initial condition from {cwl_file}")
        initial_cwl = load_initial_condition(
            cwl_file, mesh_data['global_indices'], time_idx=0
        )
    
    # Run ensemble
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING ENSEMBLE")
    logger.info("=" * 60)

    results = forecaster.run_ensemble(
        initial_cwl=initial_cwl,
        base_forcing=forcing,
        n_members=args.n_members,
        forecast_hours=args.forecast_hours,
        perturb_forcing=True,
        perturb_ic=args.perturb_ic,
        seed=args.seed,
        memory_config=memory_config,
    )
    
    # Save results
    logger.info("\nSaving results...")
    
    # Save numpy arrays
    np.savez_compressed(
        f'{output_dir}/ensemble_predictions.npz',
        predictions=results['predictions'],
        lon=mesh_data['lon'],
        lat=mesh_data['lat'],
    )
    
    # Save statistics
    np.savez_compressed(
        f'{output_dir}/ensemble_statistics.npz',
        **results['statistics']
    )
    
    # Save metadata
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(results['metadata'], f, indent=2)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_ensemble_spaghetti(
        results, mesh_data['lon'], mesh_data['lat'],
        KEY_STATIONS, f'{output_dir}/timeseries',
        forecast_start_time=forecast_start_time,
        fetch_obs=not args.no_obs,
        datum=args.datum,
    )
    
    plot_exceedance_maps(
        results, mesh_data['lon'], mesh_data['lat'],
        f'{output_dir}/maps'
    )
    
    plot_ensemble_spread(
        results, mesh_data['lon'], mesh_data['lat'],
        f'{output_dir}/maps'
    )
    
    plot_summary_dashboard(
        results, mesh_data['lon'], mesh_data['lat'],
        KEY_STATIONS, output_dir
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Runtime: {results['metadata']['elapsed_seconds']:.1f}s")
    logger.info(f"Per member: {results['metadata']['elapsed_seconds']/args.n_members:.2f}s")
    logger.info(f"\nPeak surge statistics:")
    logger.info(f"  Mean of max: {np.max(results['statistics']['mean']):.3f}m")
    logger.info(f"  Max of max: {np.max(results['statistics']['max']):.3f}m")
    logger.info(f"  Max uncertainty: {np.max(results['statistics']['std']):.3f}m")
    
    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
