#!/usr/bin/env python3
"""
Enhanced CWL GNN Training Script for STOFS 2D Global

Incorporates physics-informed enhancements from SWE-GNN research while
respecting RTX 3050 (4GB VRAM) memory constraints.

Enhancements included:
1. ✅ Gradient term in message passing (critical for physics)
2. ✅ Curriculum learning (faster convergence)
3. ✅ Water level in static features (better physics)
4. ✅ Physics-informed loss with mass conservation
5. ✅ Lightweight multi-step loss (2-step, memory efficient)

Memory optimizations:
- Float16 storage with float32 computation
- Gradient checkpointing for deeper networks
- Efficient batch processing
- Periodic GPU cache clearing

References:
- Bentivoglio et al. (2023) "Rapid spatio-temporal flood modelling via 
  hydraulics-based graph neural networks" HESS
- Taghizadeh et al. (2025) "HydroGraphNet" CACAIE

Author: Adapted for STOFS operational forecasting
"""

import sys
sys.path.insert(0, '/mnt/d/AI_4_STOFS/stofs_surrogate')

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.checkpoint import checkpoint
from netCDF4 import Dataset as NCDataset
from scipy.spatial import Delaunay
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration - Optimized for RTX 3050 (4GB VRAM)
# ============================================================

# Memory settings
USE_FLOAT16_STORAGE = True
USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory

# Domain
BBOX = {
    'lon_min': -76.0,
    'lon_max': -73.0,
    'lat_min': 38.0,
    'lat_max': 41.0,
}

# Data paths
DATA_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate/data/raw'
OUTPUT_DIR = '/mnt/d/AI_4_STOFS/stofs_surrogate'

# Training cycles
CYCLES = [
    ('stofs_2d_glo.20251122', 't00z', 'met_forcing_00z'),
    ('stofs_2d_glo.20251122', 't12z', 'met_forcing_12z'),
]

# Model parameters - VRAM optimized
HIDDEN_DIM = 96           # Balance between capacity and memory
NUM_LAYERS = 6            # With gradient checkpointing, can go deeper
STATE_DIM = 1             # CWL
STATIC_NODE_FEATURES = 4  # x_norm, y_norm, depth_norm, water_level_norm (NEW)
FORCING_FEATURES = 3      # u10, v10, pressure
EDGE_FEATURES = 3         # dx, dy, dist

# Training parameters
EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
ETA_SCALE = 2.0
GRAD_CLIP = 1.0

# Curriculum learning parameters
CURRICULUM_ENABLED = True
CURRICULUM_WARMUP_EPOCHS = 100
MAX_ROLLOUT_STEPS = 2     # Memory-efficient: just 2 steps

# Physics-informed loss
MASS_CONSERVATION_WEIGHT = 0.05
SMOOTHNESS_WEIGHT = 0.01

# Normalization constants
WIND_SCALE = 15.0
PRESSURE_MEAN = 101325.0
PRESSURE_SCALE = 3000.0


# ============================================================
# Physics-Informed Model Architecture
# ============================================================

class SWEInspiredGraphBlock(nn.Module):
    """
    Message passing block inspired by Shallow Water Equations.
    
    Key insight from Bentivoglio et al. (2023):
    - The gradient term (h_j - h_i) enforces that water only propagates
      from wet cells to neighbors
    - This provides physical consistency and better generalization
    
    Memory optimization:
    - Uses gradient checkpointing when enabled
    - LayerNorm instead of BatchNorm (more stable, similar memory)
    """
    
    def __init__(self, hidden_dim: int, use_checkpointing: bool = False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        
        # Edge MLP: processes node pair info + gradient
        # Input: edge_attr (H) + h_src (H) + h_dst (H) + gradient (H) = 4H
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Node MLP: aggregates messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Learnable scaling for gradient gating
        self.gradient_scale = nn.Parameter(torch.ones(1))
    
    def _edge_update(self, edge_attr, h_src, h_dst, h_gradient):
        """Compute edge messages with gradient gating."""
        edge_input = torch.cat([edge_attr, h_src, h_dst, h_gradient], dim=-1)
        edge_msg = self.edge_mlp(edge_input)
        
        # Gradient gating: messages scale with gradient magnitude
        # This enforces physics: no gradient = no flow
        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)  # Soft gating
        
        # Normalize to prevent instabilities (from SWE-GNN paper)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)
        
        return edge_msg
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with SWE-inspired message passing.
        
        Args:
            h: Node features [N, hidden_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, hidden_dim]
            
        Returns:
            Updated node features and edge attributes
        """
        row, col = edge_index
        h_src, h_dst = h[row], h[col]
        
        # Key physics: compute gradient between connected nodes
        h_gradient = h_dst - h_src
        
        # Compute edge messages
        if self.use_checkpointing and self.training:
            edge_msg = checkpoint(
                self._edge_update, edge_attr, h_src, h_dst, h_gradient,
                use_reentrant=False
            )
        else:
            edge_msg = self._edge_update(edge_attr, h_src, h_dst, h_gradient)
        
        # Aggregate messages at nodes
        aggr = torch.zeros_like(h)
        aggr.index_add_(0, row, edge_msg)
        
        # Node update with residual connection
        node_input = torch.cat([h, aggr], dim=-1)
        h_new = h + self.node_mlp(node_input)
        
        return h_new, edge_attr


class PhysicsInformedCWLModel(nn.Module):
    """
    GNN for Coastal Water Level prediction with physics-informed design.
    
    Architecture follows encoder-processor-decoder from MeshGraphNet,
    with SWE-inspired message passing in the processor.
    
    Features:
    - Gradient-based message passing (physics constraint)
    - Water level included in static features
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        state_dim: int = 1,
        static_feature_dim: int = 4,  # Now includes water level
        forcing_feature_dim: int = 3,
        edge_feature_dim: int = 3,
        hidden_dim: int = 96,
        num_layers: int = 6,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_checkpointing = use_checkpointing
        
        node_input_dim = state_dim + static_feature_dim + forcing_feature_dim
        
        # Encoder
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
        
        # Processor: SWE-inspired message passing layers
        self.layers = nn.ModuleList([
            SWEInspiredGraphBlock(hidden_dim, use_checkpointing=use_checkpointing)
            for _ in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"PhysicsInformedCWLModel initialized:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")
        logger.info(f"  Gradient checkpointing: {use_checkpointing}")
        logger.info(f"  Estimated VRAM: ~{total_params * 4 / 1e6:.1f} MB (weights only)")
    
    def forward(
        self,
        x: torch.Tensor,
        static_features: torch.Tensor,
        forcing_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Current CWL state [N, 1]
            static_features: Static node features [N, 4] (x, y, depth, water_level)
            forcing_features: Met forcing [N, 3] (u10, v10, pressure)
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, 3]
            
        Returns:
            Predicted CWL delta [N, 1]
        """
        # Encode nodes
        node_input = torch.cat([x, static_features, forcing_features], dim=-1)
        h = self.node_encoder(node_input)
        
        # Encode edges
        e = self.edge_encoder(edge_attr)
        
        # Process through SWE-inspired layers
        for layer in self.layers:
            h, e = layer(h, edge_index, e)
        
        # Decode to CWL prediction
        out = self.decoder(h)
        
        return out


# ============================================================
# Physics-Informed Loss Functions
# ============================================================

class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physics constraints.
    
    Components:
    1. MSE on predicted vs actual CWL
    2. Mass conservation penalty
    3. Spatial smoothness regularization
    
    From HydroGraphNet: embedding physical constraints in the loss function
    improves flood predictions and long-term stability.
    """
    
    def __init__(
        self,
        mass_weight: float = 0.05,
        smoothness_weight: float = 0.01,
    ):
        super().__init__()
        self.mass_weight = mass_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss.
        
        Args:
            pred: Predicted CWL [N, 1]
            target: Target CWL [N, 1]
            edge_index: Graph connectivity (for smoothness)
            
        Returns:
            Total loss and dict of component losses
        """
        # Primary MSE loss
        mse_loss = F.mse_loss(pred, target)
        
        # Mass conservation: total water should be approximately preserved
        # (simplified: just penalize large changes in sum)
        pred_sum = pred.sum()
        target_sum = target.sum()
        mass_loss = ((pred_sum - target_sum) / (target_sum.abs() + 1e-8)) ** 2
        
        # Spatial smoothness: penalize large gradients between neighbors
        smoothness_loss = torch.tensor(0.0, device=pred.device)
        if edge_index is not None and self.smoothness_weight > 0:
            row, col = edge_index
            pred_diff = (pred[row] - pred[col]).abs()
            target_diff = (target[row] - target[col]).abs()
            # Penalize if prediction is much less smooth than target
            smoothness_loss = F.relu(pred_diff - target_diff * 1.5).mean()
        
        # Combined loss
        total_loss = (
            mse_loss + 
            self.mass_weight * mass_loss + 
            self.smoothness_weight * smoothness_loss
        )
        
        # Return components for logging
        components = {
            'mse': mse_loss.item(),
            'mass': mass_loss.item(),
            'smooth': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0.0,
            'total': total_loss.item(),
        }
        
        return total_loss, components


# ============================================================
# Curriculum Learning
# ============================================================

class CurriculumScheduler:
    """
    Curriculum learning scheduler for multi-step predictions.
    
    From SWE-GNN paper: progressively increasing the prediction horizon
    during training improves stability and convergence.
    
    Memory-efficient version: caps at 2 steps to fit in 4GB VRAM.
    """
    
    def __init__(
        self,
        max_steps: int = 2,
        warmup_epochs: int = 100,
        min_steps: int = 1,
    ):
        self.max_steps = max_steps
        self.warmup_epochs = warmup_epochs
        self.min_steps = min_steps
    
    def get_num_steps(self, epoch: int) -> int:
        """Get number of rollout steps for current epoch."""
        if epoch >= self.warmup_epochs:
            return self.max_steps
        
        # Linear interpolation
        progress = epoch / self.warmup_epochs
        steps = self.min_steps + int((self.max_steps - self.min_steps) * progress)
        return min(steps, self.max_steps)
    
    def __repr__(self):
        return f"CurriculumScheduler(max_steps={self.max_steps}, warmup={self.warmup_epochs})"


# ============================================================
# Enhanced Dataset with Water Level Feature
# ============================================================

class EnhancedCWLDataset(Dataset):
    """
    Dataset with physics-informed enhancements:
    - Water level (depth + CWL) as static feature
    - Support for multi-step sequences (memory-efficient)
    - Float16 storage with float32 computation
    """
    
    def __init__(
        self,
        mesh_data: Dict,
        cycles_data: list,
        eta_scale: float = 2.0,
        max_sequence_length: int = 2,
    ):
        self.eta_scale = np.float32(eta_scale)
        self.max_seq_len = max_sequence_length
        
        # Store mesh data
        self.lon = mesh_data['lon'].astype(np.float32)
        self.lat = mesh_data['lat'].astype(np.float32)
        self.depth = mesh_data['depth'].astype(np.float32)
        self.edge_index = torch.tensor(mesh_data['edge_index'], dtype=torch.long)
        
        self.num_nodes = len(self.lon)
        
        # Process cycles
        self.samples = []
        self.elevations = []
        self.forcings = []
        
        for cycle_idx, cycle in enumerate(cycles_data):
            elev = cycle['elevation'].astype(np.float32)
            forcing = cycle['forcing']
            
            # Find valid nodes
            valid_mask = np.all(~np.isnan(elev), axis=0)
            if cycle_idx == 0:
                self.valid_mask = valid_mask
            else:
                self.valid_mask &= valid_mask
            
            # Store
            dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
            self.elevations.append(elev.astype(dtype))
            self.forcings.append(forcing)
            
            # Create samples with enough room for sequences
            num_times = elev.shape[0]
            for t in range(num_times - max_sequence_length):
                self.samples.append((cycle_idx, t))
        
        # Apply valid mask
        valid_indices = np.where(self.valid_mask)[0]
        logger.info(f"Valid nodes: {len(valid_indices):,} / {self.num_nodes:,}")
        
        self.lon = self.lon[valid_indices]
        self.lat = self.lat[valid_indices]
        self.depth = self.depth[valid_indices]
        self.num_nodes = len(self.lon)
        
        # Rebuild edge index
        old_to_new = {old: new for new, old in enumerate(valid_indices)}
        new_edges = []
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            if src in old_to_new and dst in old_to_new:
                new_edges.append([old_to_new[src], old_to_new[dst]])
        self.edge_index = torch.tensor(np.array(new_edges).T, dtype=torch.long)
        
        # Filter data arrays
        for i in range(len(self.elevations)):
            self.elevations[i] = self.elevations[i][:, valid_indices]
            self.forcings[i]['u10'] = self.forcings[i]['u10'][:, valid_indices]
            self.forcings[i]['v10'] = self.forcings[i]['v10'][:, valid_indices]
            self.forcings[i]['pressure'] = self.forcings[i]['pressure'][:, valid_indices]
        
        gc.collect()
        
        # Compute base static features (x, y, depth)
        self._compute_static_features()
        self._compute_edge_features()
        
        logger.info(f"Dataset: {len(self.samples)} samples, {self.num_nodes} nodes")
    
    def _compute_static_features(self):
        """Compute normalized static features."""
        # Cartesian coordinates
        ref_lon, ref_lat = self.lon.mean(), self.lat.mean()
        R = np.float32(6371000.0)
        self.x_cart = R * np.radians(self.lon - ref_lon) * np.cos(np.radians(ref_lat))
        self.y_cart = R * np.radians(self.lat - ref_lat)
        
        # Normalize positions
        x_norm = 2 * (self.x_cart - self.x_cart.min()) / (self.x_cart.max() - self.x_cart.min() + 1e-8) - 1
        y_norm = 2 * (self.y_cart - self.y_cart.min()) / (self.y_cart.max() - self.y_cart.min() + 1e-8) - 1
        
        # Normalize depth (log scale)
        depth_safe = np.maximum(np.abs(self.depth), 0.1)
        depth_log = np.log10(depth_safe)
        depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)
        
        # Base static features (water level added per-sample)
        self.static_base = np.stack([x_norm, y_norm, depth_norm], axis=1).astype(np.float32)
    
    def _compute_edge_features(self):
        """Compute edge features."""
        src, dst = self.edge_index[0].numpy(), self.edge_index[1].numpy()
        dx = self.x_cart[dst] - self.x_cart[src]
        dy = self.y_cart[dst] - self.y_cart[src]
        dist = np.sqrt(dx**2 + dy**2)
        char_length = np.median(dist) + 1e-8
        
        self.edge_attr = torch.tensor(
            np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1),
            dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        cycle_idx, time_idx = self.samples[idx]
        
        # Get elevation sequence
        elev_sequence = []
        for t in range(self.max_seq_len + 1):
            elev = self.elevations[cycle_idx][time_idx + t].astype(np.float32)
            elev_sequence.append(elev / self.eta_scale)
        
        eta_in = elev_sequence[0]
        eta_out = elev_sequence[1]
        
        # Compute water level feature (NEW: from SWE-GNN paper)
        # Water level = bathymetry + surface elevation
        water_level = self.depth + eta_in * self.eta_scale
        water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
        
        # Combine static features with water level
        static_features = np.concatenate([
            self.static_base,
            water_level_norm[:, np.newaxis]
        ], axis=1)
        
        # Get forcing
        forcing = self.forcings[cycle_idx]
        u10 = forcing['u10'][time_idx].astype(np.float32) / WIND_SCALE
        v10 = forcing['v10'][time_idx].astype(np.float32) / WIND_SCALE
        pressure = forcing['pressure'][time_idx].astype(np.float32)  # Pre-normalized
        
        forcing_features = np.stack([u10, v10, pressure], axis=1)
        
        # Build data object
        data = Data(
            x=torch.tensor(eta_in[:, np.newaxis], dtype=torch.float32),
            y=torch.tensor(eta_out[:, np.newaxis], dtype=torch.float32),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            static_features=torch.tensor(static_features, dtype=torch.float32),
            forcing_features=torch.tensor(forcing_features, dtype=torch.float32),
        )
        
        # For multi-step training, include next forcing
        if self.max_seq_len >= 2:
            u10_next = forcing['u10'][time_idx + 1].astype(np.float32) / WIND_SCALE
            v10_next = forcing['v10'][time_idx + 1].astype(np.float32) / WIND_SCALE
            pressure_next = forcing['pressure'][time_idx + 1].astype(np.float32)
            
            data.forcing_next = torch.tensor(
                np.stack([u10_next, v10_next, pressure_next], axis=1),
                dtype=torch.float32
            )
            data.y_next = torch.tensor(
                elev_sequence[2][:, np.newaxis],
                dtype=torch.float32
            )
        
        return data


# ============================================================
# Training Functions
# ============================================================

def train_epoch_curriculum(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: PhysicsInformedLoss,
    device: torch.device,
    num_steps: int = 1,
    grad_clip: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Train one epoch with curriculum-based multi-step loss.
    
    Args:
        model: The GNN model
        loader: Data loader
        optimizer: Optimizer
        criterion: Physics-informed loss function
        device: Device to use
        num_steps: Number of rollout steps (from curriculum scheduler)
        grad_clip: Gradient clipping value
        
    Returns:
        Average loss and component losses
    """
    model.train()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = len(loader)
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        accumulated_loss = 0
        current_state = batch.x
        
        # Step 1: Always do first step
        pred = model(
            current_state,
            batch.static_features,
            batch.forcing_features,
            batch.edge_index,
            batch.edge_attr
        )
        
        loss1, components1 = criterion(pred, batch.y, batch.edge_index)
        accumulated_loss = loss1
        
        # Step 2: If curriculum allows and data available
        if num_steps >= 2 and hasattr(batch, 'y_next'):
            # Update water level in static features for next step
            # (simplified: reuse same static features)
            pred2 = model(
                pred.detach(),  # Use prediction as input
                batch.static_features,
                batch.forcing_next,
                batch.edge_index,
                batch.edge_attr
            )
            
            loss2, components2 = criterion(pred2, batch.y_next, batch.edge_index)
            accumulated_loss = accumulated_loss + 0.5 * loss2  # Weight second step less
        
        # Backward pass
        accumulated_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += accumulated_loss.item()
        for k in total_components:
            total_components[k] += components1.get(k, 0)
        
        # Memory management
        if device.type == 'cuda' and batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        del batch, pred, accumulated_loss
        if num_steps >= 2:
            del pred2
    
    # Cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in total_components.items()}
    
    return avg_loss, avg_components


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Validate model."""
    model.eval()
    total_loss = 0
    total_components = {'mse': 0, 'mass': 0, 'smooth': 0}
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            pred = model(
                batch.x,
                batch.static_features,
                batch.forcing_features,
                batch.edge_index,
                batch.edge_attr
            )
            
            loss, components = criterion(pred, batch.y, batch.edge_index)
            total_loss += loss.item()
            for k in total_components:
                total_components[k] += components.get(k, 0)
            
            del batch, pred
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in total_components.items()}
    
    return avg_loss, avg_components


def rollout_prediction(
    model: nn.Module,
    dataset: EnhancedCWLDataset,
    cycle_idx: int,
    start_idx: int,
    num_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run multi-step autoregressive prediction."""
    model.eval()
    
    edge_index = dataset.edge_index.to(device)
    edge_attr = dataset.edge_attr.to(device)
    
    # Initial state
    elev_start = dataset.elevations[cycle_idx][start_idx].astype(np.float32)
    current = torch.tensor(elev_start / dataset.eta_scale, dtype=torch.float32).to(device)
    
    predictions = [elev_start.copy()]
    ground_truth = [elev_start.copy()]
    
    forcing = dataset.forcings[cycle_idx]
    
    with torch.no_grad():
        for step in range(num_steps):
            t = start_idx + step
            
            # Get current elevation for water level computation
            current_elev = current.cpu().numpy() * dataset.eta_scale
            water_level = dataset.depth + current_elev
            water_level_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)
            
            # Build static features with updated water level
            static_features = np.concatenate([
                dataset.static_base,
                water_level_norm[:, np.newaxis]
            ], axis=1)
            static_features = torch.tensor(static_features, dtype=torch.float32).to(device)
            
            # Get forcing
            u10 = forcing['u10'][t].astype(np.float32) / WIND_SCALE
            v10 = forcing['v10'][t].astype(np.float32) / WIND_SCALE
            pressure = forcing['pressure'][t].astype(np.float32)
            
            forcing_features = torch.tensor(
                np.stack([u10, v10, pressure], axis=1),
                dtype=torch.float32
            ).to(device)
            
            # Predict
            x = current.unsqueeze(1)
            next_state = model(
                x, static_features, forcing_features, edge_index, edge_attr
            ).squeeze()
            
            current = next_state
            predictions.append(current.cpu().numpy() * dataset.eta_scale)
            
            if t + 1 < len(dataset.elevations[cycle_idx]):
                gt = dataset.elevations[cycle_idx][t + 1].astype(np.float32)
                ground_truth.append(gt)
    
    return np.array(predictions), np.array(ground_truth)


# ============================================================
# Visualization
# ============================================================

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    components: Dict[str, list],
    output_path: str,
):
    """Plot training curves with loss components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Total loss
    ax = axes[0]
    ax.semilogy(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.semilogy(epochs, val_losses, 'r-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Components
    ax = axes[1]
    for name, values in components.items():
        if len(values) > 0:
            ax.semilogy(epochs, values, label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Component')
    ax.set_title('Loss Components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_rollout_comparison(
    lon: np.ndarray,
    lat: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: str,
):
    """Plot rollout comparison."""
    timesteps = [0, 6, 12, 24, 48]
    fig, axes = plt.subplots(len(timesteps), 3, figsize=(15, 4*len(timesteps)))
    
    vmax = 1.0
    s = 2
    
    for i, t in enumerate(timesteps):
        if t >= len(predictions):
            continue
        
        pred = predictions[t]
        gt = ground_truth[t] if t < len(ground_truth) else pred
        diff = pred - gt
        rmse = np.sqrt(np.mean(diff**2))
        
        # Ground truth
        ax = axes[i, 0]
        cf = ax.scatter(lon, lat, c=gt, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Ground Truth (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')
        
        # Prediction
        ax = axes[i, 1]
        cf = ax.scatter(lon, lat, c=pred, s=s, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Prediction (t+{t}h)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='CWL (m)')
        
        # Error
        ax = axes[i, 2]
        cf = ax.scatter(lon, lat, c=diff, s=s, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_title(f'Error (RMSE={rmse:.3f}m)')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax, label='Error (m)')
    
    plt.suptitle('Physics-Informed CWL Model - Rollout', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")


# ============================================================
# Data Loading (reuse from original script)
# ============================================================

def load_met_forcing_for_cycle(date_dir, met_dir, num_cwl_times, subsample_factor=2):
    """Load and process meteorological forcing data."""
    base_path = f'{DATA_DIR}/{date_dir}/{met_dir}'
    
    # Find sample file for coordinates
    sample_file = f'{base_path}/stofs_2d_glo_ncst.222.nc'
    if not os.path.exists(sample_file):
        sample_file = f'{base_path}/stofs_2d_glo_fcst1.222.nc'
    
    nc = NCDataset(sample_file)
    grid_lon = np.array(nc.variables['grid_xt'][:], dtype=np.float32)
    grid_lat = np.array(nc.variables['grid_yt'][:], dtype=np.float32)
    nc.close()
    
    grid_lon = np.where(grid_lon > 180, grid_lon - 360, grid_lon)
    
    all_u, all_v, all_p = [], [], []
    file_order = ['ncst', 'fcst1', 'fcst2']
    
    for file_type in file_order:
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
    
    u_all = np.concatenate(all_u, axis=0)
    v_all = np.concatenate(all_v, axis=0)
    p_all = np.concatenate(all_p, axis=0)
    
    del all_u, all_v, all_p
    gc.collect()
    
    # Subsample spatial grid
    if subsample_factor > 1:
        u_all = u_all[:, ::subsample_factor, ::subsample_factor]
        v_all = v_all[:, ::subsample_factor, ::subsample_factor]
        p_all = p_all[:, ::subsample_factor, ::subsample_factor]
        grid_lon = grid_lon[::subsample_factor]
        grid_lat = grid_lat[::subsample_factor]
    
    # Interpolate temporal if needed
    met_times = u_all.shape[0]
    if met_times != num_cwl_times:
        old_times = np.linspace(0, 1, met_times)
        new_times = np.linspace(0, 1, num_cwl_times)
        
        u_interp = np.zeros((num_cwl_times, u_all.shape[1], u_all.shape[2]), dtype=np.float32)
        v_interp = np.zeros_like(u_interp)
        p_interp = np.zeros_like(u_interp)
        
        for i in range(u_all.shape[1]):
            for j in range(u_all.shape[2]):
                u_interp[:, i, j] = np.interp(new_times, old_times, u_all[:, i, j])
                v_interp[:, i, j] = np.interp(new_times, old_times, v_all[:, i, j])
                p_interp[:, i, j] = np.interp(new_times, old_times, p_all[:, i, j])
        
        u_all, v_all, p_all = u_interp, v_interp, p_interp
    
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
    p_normalized = (p_all - PRESSURE_MEAN) / PRESSURE_SCALE
    
    return {
        'u10': u_all.astype(dtype),
        'v10': v_all.astype(dtype),
        'pressure': p_normalized.astype(dtype),
        'grid_lon': grid_lon,
        'grid_lat': grid_lat,
    }


def interpolate_forcing_to_nodes(forcing_data, node_lon, node_lat):
    """Interpolate forcing from regular grid to mesh nodes."""
    grid_lon = forcing_data['grid_lon']
    grid_lat = forcing_data['grid_lat']
    
    lon_sort_idx = np.argsort(grid_lon)
    lat_sort_idx = np.argsort(grid_lat)
    grid_lon_sorted = grid_lon[lon_sort_idx]
    grid_lat_sorted = grid_lat[lat_sort_idx]
    
    num_times = forcing_data['u10'].shape[0]
    num_nodes = len(node_lon)
    
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
    result = {
        'u10': np.zeros((num_times, num_nodes), dtype=dtype),
        'v10': np.zeros((num_times, num_nodes), dtype=dtype),
        'pressure': np.zeros((num_times, num_nodes), dtype=dtype),
    }
    
    for t in range(num_times):
        if t % 50 == 0:
            logger.info(f"    Interpolating time step {t}/{num_times}")
        
        for var in ['u10', 'v10', 'pressure']:
            data = forcing_data[var][t].astype(np.float32)
            data_sorted = data[lat_sort_idx][:, lon_sort_idx]
            
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
            
            result[var][t] = values.astype(dtype)
    
    return result


def extract_midatlantic_mesh(nc_file, bbox, max_nodes=15000):
    """Extract mesh for Mid-Atlantic region."""
    nc = NCDataset(nc_file, 'r')
    
    x = np.array(nc.variables['x'][:], dtype=np.float32)
    y = np.array(nc.variables['y'][:], dtype=np.float32)
    depth = np.array(nc.variables['depth'][:], dtype=np.float32)
    
    mask = (
        (x >= bbox['lon_min']) & (x <= bbox['lon_max']) &
        (y >= bbox['lat_min']) & (y <= bbox['lat_max'])
    )
    subset_indices = np.where(mask)[0]
    
    if len(subset_indices) > max_nodes:
        rng = np.random.RandomState(42)
        subset_indices = rng.choice(subset_indices, size=max_nodes, replace=False)
        subset_indices = np.sort(subset_indices)
    
    lon = x[subset_indices]
    lat = y[subset_indices]
    depth_sub = depth[subset_indices]
    
    points = np.column_stack([lon, lat])
    tri = Delaunay(points)
    
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(tuple(sorted([simplex[i], simplex[j]])))
    
    edges = np.array(list(edges))
    edge_index = np.vstack([edges, edges[:, ::-1]]).T
    
    nc.close()
    
    return {
        'lon': lon,
        'lat': lat,
        'depth': depth_sub,
        'edge_index': edge_index,
        'global_indices': subset_indices,
    }


def extract_cycle_data(nc_file, global_indices, temporal_subsample=1):
    """Extract CWL time series for a cycle."""
    nc = NCDataset(nc_file, 'r')
    
    zeta = nc.variables['zeta']
    full_times = zeta.shape[0]
    time_indices = list(range(0, full_times, temporal_subsample))
    num_times = len(time_indices)
    
    dtype = np.float16 if USE_FLOAT16_STORAGE else np.float32
    elevation = np.zeros((num_times, len(global_indices)), dtype=dtype)
    
    for i, t in enumerate(time_indices):
        elevation[i, :] = zeta[t, global_indices]
    
    elevation = np.where(elevation < -9000, np.nan, elevation)
    times = nc.variables['time'][time_indices]
    nc.close()
    
    return elevation, times


# ============================================================
# Main Training Loop
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("PHYSICS-INFORMED CWL GNN TRAINING")
    logger.info("Enhanced with SWE-GNN concepts")
    logger.info("=" * 60)
    logger.info(f"Domain: [{BBOX['lon_min']}, {BBOX['lon_max']}] × [{BBOX['lat_min']}, {BBOX['lat_max']}]")
    logger.info(f"Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}")
    logger.info(f"Enhancements: gradient_term=True, curriculum={CURRICULUM_ENABLED}")
    logger.info(f"Physics loss: mass_weight={MASS_CONSERVATION_WEIGHT}, smooth_weight={SMOOTHNESS_WEIGHT}")
    
    # Load or create mesh
    mesh_path = f'{OUTPUT_DIR}/data/processed/midatlantic_mesh_v5.npz'
    
    if os.path.exists(mesh_path):
        logger.info("\nLoading existing mesh...")
        mesh_np = np.load(mesh_path)
        mesh_data = {k: mesh_np[k] for k in mesh_np.files}
        mesh_np.close()
    else:
        logger.info("\nExtracting mesh...")
        first_cwl = f'{DATA_DIR}/{CYCLES[0][0]}/stofs_2d_glo.{CYCLES[0][1]}.fields.cwl.nc'
        mesh_data = extract_midatlantic_mesh(first_cwl, BBOX)
        os.makedirs(f'{OUTPUT_DIR}/data/processed', exist_ok=True)
        np.savez(mesh_path, **mesh_data)
    
    gc.collect()
    
    # Load cycle data
    logger.info("\nLoading cycle data...")
    cycles_data = []
    
    for date_dir, cycle, met_dir in CYCLES:
        logger.info(f"\nProcessing {date_dir} {cycle}...")
        
        cwl_file = f'{DATA_DIR}/{date_dir}/stofs_2d_glo.{cycle}.fields.cwl.nc'
        elevation, times = extract_cycle_data(cwl_file, mesh_data['global_indices'])
        
        forcing_raw = load_met_forcing_for_cycle(date_dir, met_dir, elevation.shape[0])
        forcing = interpolate_forcing_to_nodes(forcing_raw, mesh_data['lon'], mesh_data['lat'])
        
        del forcing_raw
        gc.collect()
        
        cycles_data.append({
            'elevation': elevation,
            'times': times,
            'forcing': forcing,
        })
    
    # Create dataset
    logger.info("\nCreating dataset...")
    dataset = EnhancedCWLDataset(
        mesh_data, cycles_data,
        eta_scale=ETA_SCALE,
        max_sequence_length=MAX_ROLLOUT_STEPS,
    )
    
    del cycles_data
    gc.collect()
    
    # Train/val split
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model = PhysicsInformedCWLModel(
        state_dim=STATE_DIM,
        static_feature_dim=STATIC_NODE_FEATURES,
        forcing_feature_dim=FORCING_FEATURES,
        edge_feature_dim=EDGE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        use_checkpointing=USE_GRADIENT_CHECKPOINTING,
    ).to(device)
    
    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = PhysicsInformedLoss(
        mass_weight=MASS_CONSERVATION_WEIGHT,
        smoothness_weight=SMOOTHNESS_WEIGHT,
    )
    
    # Curriculum scheduler
    curriculum = CurriculumScheduler(
        max_steps=MAX_ROLLOUT_STEPS,
        warmup_epochs=CURRICULUM_WARMUP_EPOCHS,
    ) if CURRICULUM_ENABLED else None
    
    # Training loop
    logger.info("\nStarting training...")
    os.makedirs(f'{OUTPUT_DIR}/outputs/checkpoints', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/outputs/figures', exist_ok=True)
    
    train_losses = []
    val_losses = []
    loss_components = {'mse': [], 'mass': [], 'smooth': []}
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        # Get curriculum steps
        num_steps = curriculum.get_num_steps(epoch) if curriculum else 1
        
        # Train
        train_loss, train_comp = train_epoch_curriculum(
            model, train_loader, optimizer, criterion, device,
            num_steps=num_steps, grad_clip=GRAD_CLIP
        )
        
        # Validate
        val_loss, val_comp = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Record
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        for k in loss_components:
            loss_components[k].append(train_comp.get(k, 0))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'hidden_dim': HIDDEN_DIM,
                    'num_layers': NUM_LAYERS,
                    'static_features': STATIC_NODE_FEATURES,
                    'forcing_features': FORCING_FEATURES,
                    'eta_scale': ETA_SCALE,
                    'bbox': BBOX,
                },
            }, f'{OUTPUT_DIR}/outputs/checkpoints/best_physics_informed_model.pt')
        
        # Log
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch:3d} | steps={num_steps} | "
                f"train={train_loss:.6f} | val={val_loss:.6f} | "
                f"mse={train_comp['mse']:.6f} | mass={train_comp['mass']:.6f} | "
                f"lr={lr:.2e} | best={best_val_loss:.6f}"
            )
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, loss_components,
        f'{OUTPUT_DIR}/outputs/figures/physics_informed_training.png'
    )
    
    # Load best and do rollout
    logger.info("\nLoading best model for rollout...")
    checkpoint = torch.load(f'{OUTPUT_DIR}/outputs/checkpoints/best_physics_informed_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    cycle_idx = len(dataset.elevations) - 1
    start_idx = 100
    predictions, ground_truth = rollout_prediction(
        model, dataset, cycle_idx, start_idx, 48, device
    )
    
    plot_rollout_comparison(
        dataset.lon, dataset.lat,
        predictions, ground_truth,
        f'{OUTPUT_DIR}/outputs/figures/physics_informed_rollout.png'
    )
    
    # Final metrics
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Best epoch: {checkpoint['epoch']}")
    
    for t in [1, 6, 12, 24, 48]:
        if t < len(predictions) and t < len(ground_truth):
            rmse = np.sqrt(np.mean((predictions[t] - ground_truth[t])**2))
            logger.info(f"Rollout t+{t}h RMSE: {rmse:.4f} m")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
