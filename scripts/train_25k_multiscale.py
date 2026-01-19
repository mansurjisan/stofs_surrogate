#!/usr/bin/env python3
"""
Hierarchical Multi-Scale GNN Training for STOFS

Uses pretrained 25k model as frozen fine-scale backbone,
adds trainable coarse (1k) and medium (5k) scales to capture
long-range patterns and correct amplitude damping.

Architecture:
    COARSE (1k) → MEDIUM (5k) → FINE (25k, frozen) → Fusion → Output
    [trainable]   [trainable]   [pretrained]        [trainable]

Benefits:
    - Preserves learned local/tidal dynamics from 25k model
    - Coarse scale captures basin-wide storm surge propagation
    - Medium scale bridges regional patterns
    - Only ~20% parameters need training
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import Delaunay
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path('/mnt/f/STOFS_TRAINING_DATA/processed_25k_v2')
CHECKPOINT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_25k_v2')
OUTPUT_DIR = Path('/mnt/d/AI_4_STOFS/stofs_surrogate/outputs/checkpoints_multiscale')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained model to use as fine-scale backbone
PRETRAINED_25K = CHECKPOINT_DIR / 'checkpoint_epoch_55.pt'  # Update as needed

# Model configuration
HIDDEN_DIM = 128
NUM_LAYERS_FINE = 6  # Must match pretrained model
NUM_LAYERS_MEDIUM = 4
NUM_LAYERS_COARSE = 3

# Multi-scale mesh sizes
N_FINE = 25000
N_MEDIUM = 5000
N_COARSE = 1000

# Feature dimensions (must match pretrained)
STATE_DIM = 1
FORCING_FEATURES = 8
TEMPORAL_FEATURES = 12
STATIC_NODE_FEATURES = 4

ETA_SCALE = 2.0
DT_HOURS = 1.0
EPOCH_DATETIME = datetime(2023, 1, 1, 0, 0, 0)

# Training configuration
NUM_EPOCHS = 30
BASE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Rollout schedule (shorter since fine-scale is pretrained)
ROLLOUT_SCHEDULE = {
    1: (1, 5, 4),    # 1-step: epochs 1-5, batch_mult=4
    3: (6, 15, 2),   # 3-step: epochs 6-15, batch_mult=2
    6: (16, 25, 2),  # 6-step: epochs 16-25, batch_mult=2
    12: (26, 30, 1), # 12-step: epochs 26-30, batch_mult=1
}

# Tidal constituent periods
TIDAL_PERIODS = {
    'M2': 12.4206, 'S2': 12.0000, 'N2': 12.6583,
    'K1': 23.9345, 'O1': 25.8193, 'M4': 6.2103,
}


# ============================================================
# Pretrained Fine-Scale Model (from existing training)
# ============================================================

class BatchedSWEGraphBlock(nn.Module):
    """Graph block from pretrained model - DO NOT MODIFY"""
    def __init__(self, hidden_dim: int):
        super().__init__()
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
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]

        h_src = h[:, row, :]
        h_dst = h[:, col, :]
        h_gradient = h_dst - h_src

        edge_attr_batch = edge_attr.unsqueeze(0).expand(B, -1, -1)
        edge_input = torch.cat([edge_attr_batch, h_src, h_dst, h_gradient], dim=-1)

        edge_input_flat = edge_input.reshape(B * E, -1)
        edge_msg_flat = self.edge_mlp(edge_input_flat)
        edge_msg = edge_msg_flat.reshape(B, E, F)

        gradient_gate = torch.tanh(self.gradient_scale * h_gradient)
        edge_msg = edge_msg * (1.0 + gradient_gate)
        edge_msg = edge_msg / (torch.norm(edge_msg, dim=-1, keepdim=True) + 1e-8)

        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, E, F)
        aggr.scatter_add_(1, row_expanded, edge_msg)

        node_input = torch.cat([h, aggr], dim=-1)
        node_input_flat = node_input.reshape(B * N, -1)
        node_out_flat = self.node_mlp(node_input_flat)
        node_out = node_out_flat.reshape(B, N, F)

        return h + node_out, edge_attr


class BatchedTemporalMemoryGNN(nn.Module):
    """Pretrained fine-scale model - DO NOT MODIFY architecture"""
    def __init__(self, state_dim=1, temporal_dim=12, static_feature_dim=4,
                 forcing_feature_dim=8, edge_feature_dim=3, hidden_dim=128, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        node_input_dim = 3 * state_dim + temporal_dim + static_feature_dim + forcing_feature_dim

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
        self.gnn_layers = nn.ModuleList([
            BatchedSWEGraphBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)
        B, N, F_in = node_features.shape

        node_flat = node_features.reshape(B * N, F_in)
        h_flat = self.node_encoder(node_flat)
        h = h_flat.reshape(B, N, self.hidden_dim)

        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        h_flat = h.reshape(B * N, self.hidden_dim)
        delta_flat = self.decoder(h_flat)
        delta = delta_flat.reshape(B, N, -1)

        return x + delta

    def get_hidden_features(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr):
        """Return hidden features instead of prediction (for fusion)"""
        node_features = torch.cat([x, x_prev, dxdt, tidal_harmonics, static_features, forcing], dim=-1)
        B, N, F_in = node_features.shape

        node_flat = node_features.reshape(B * N, F_in)
        h_flat = self.node_encoder(node_flat)
        h = h_flat.reshape(B, N, self.hidden_dim)

        e = self.edge_encoder(edge_attr)

        for layer in self.gnn_layers:
            h, e = layer(h, edge_index, e)

        return h  # Return hidden features (B, N, hidden_dim)


# ============================================================
# Coarse-Scale GNN (NEW - Trainable)
# ============================================================

class CoarseScaleGNNBlock(nn.Module):
    """Simplified GNN block for coarse scale"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: (B, N, F) node features
            edge_index: (2, E) edge indices
        """
        B, N, F = h.shape
        row, col = edge_index
        E = row.shape[0]

        # Message passing
        h_src = h[:, row, :]  # (B, E, F)
        h_dst = h[:, col, :]  # (B, E, F)

        msg_input = torch.cat([h_src, h_dst, h_dst - h_src], dim=-1)
        messages = self.message_mlp(msg_input.reshape(B * E, -1)).reshape(B, E, F)

        # Aggregation
        aggr = torch.zeros(B, N, F, device=h.device, dtype=h.dtype)
        row_exp = row.unsqueeze(0).unsqueeze(-1).expand(B, E, F)
        aggr.scatter_add_(1, row_exp, messages)

        # Update
        update_input = torch.cat([h, aggr], dim=-1)
        h_new = self.update_mlp(update_input.reshape(B * N, -1)).reshape(B, N, F)

        return h + h_new


class CoarseScaleGNN(nn.Module):
    """Coarse-scale GNN for long-range pattern capture"""
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.gnn_layers = nn.ModuleList([
            CoarseScaleGNNBlock(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x, edge_index):
        """
        Args:
            x: (B, N_coarse, input_dim) coarse node features
            edge_index: (2, E) coarse edges
        Returns:
            h: (B, N_coarse, hidden_dim) coarse hidden features
        """
        B, N, _ = x.shape

        h = self.encoder(x.reshape(B * N, -1)).reshape(B, N, self.hidden_dim)

        for layer in self.gnn_layers:
            h = layer(h, edge_index)

        return h


# ============================================================
# Hierarchical Multi-Scale Model
# ============================================================

class HierarchicalMultiScaleGNN(nn.Module):
    """
    Hierarchical Multi-Scale GNN combining:
    - Pretrained fine-scale (25k) - FROZEN
    - Trainable medium-scale (5k)
    - Trainable coarse-scale (1k)
    - Trainable fusion layers
    """

    def __init__(self, pretrained_path, n_fine, n_medium, n_coarse, hidden_dim=128):
        super().__init__()

        self.n_fine = n_fine
        self.n_medium = n_medium
        self.n_coarse = n_coarse
        self.hidden_dim = hidden_dim

        # ============================================
        # FINE SCALE (25k) - Pretrained, FROZEN
        # ============================================
        self.fine_model = BatchedTemporalMemoryGNN(
            state_dim=STATE_DIM,
            temporal_dim=TEMPORAL_FEATURES,
            static_feature_dim=STATIC_NODE_FEATURES,
            forcing_feature_dim=FORCING_FEATURES,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS_FINE,
        )

        # Load pretrained weights
        if pretrained_path and Path(pretrained_path).exists():
            logger.info(f"Loading pretrained fine-scale model from {pretrained_path}")
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)

            # Remove _orig_mod prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '')
                new_state_dict[new_key] = v

            self.fine_model.load_state_dict(new_state_dict, strict=False)
            logger.info("Pretrained fine-scale model loaded successfully")

        # Freeze fine-scale model
        for param in self.fine_model.parameters():
            param.requires_grad = False
        self.fine_model.eval()

        # ============================================
        # COARSE SCALE (1k) - Trainable
        # ============================================
        # Input: pooled state + forcing + temporal
        coarse_input_dim = STATE_DIM + FORCING_FEATURES + TEMPORAL_FEATURES + 3  # +3 for position
        self.coarse_gnn = CoarseScaleGNN(
            input_dim=coarse_input_dim,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS_COARSE,
        )

        # ============================================
        # MEDIUM SCALE (5k) - Trainable
        # ============================================
        medium_input_dim = STATE_DIM + FORCING_FEATURES + TEMPORAL_FEATURES + 3 + hidden_dim  # +hidden from coarse
        self.medium_gnn = CoarseScaleGNN(
            input_dim=medium_input_dim,
            hidden_dim=hidden_dim,
            num_layers=NUM_LAYERS_MEDIUM,
        )

        # ============================================
        # FUSION LAYERS - Trainable
        # ============================================
        # Combine fine prediction with coarse/medium corrections
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, STATE_DIM),
        )

        # Learnable correction scale (start small)
        self.correction_scale = nn.Parameter(torch.tensor(0.1))

        # Store mesh mappings (set during setup)
        self.register_buffer('fine_to_medium_idx', torch.zeros(n_fine, dtype=torch.long))
        self.register_buffer('medium_to_coarse_idx', torch.zeros(n_medium, dtype=torch.long))
        self.register_buffer('coarse_pos', torch.zeros(n_coarse, 2))
        self.register_buffer('medium_pos', torch.zeros(n_medium, 2))
        self.register_buffer('fine_pos', torch.zeros(n_fine, 2))

    def set_mesh_mappings(self, fine_to_medium, medium_to_coarse,
                          fine_pos, medium_pos, coarse_pos,
                          medium_edge_index, coarse_edge_index):
        """Set the multi-scale mesh mappings"""
        self.fine_to_medium_idx = torch.tensor(fine_to_medium, dtype=torch.long)
        self.medium_to_coarse_idx = torch.tensor(medium_to_coarse, dtype=torch.long)
        self.fine_pos = torch.tensor(fine_pos, dtype=torch.float32)
        self.medium_pos = torch.tensor(medium_pos, dtype=torch.float32)
        self.coarse_pos = torch.tensor(coarse_pos, dtype=torch.float32)

        self.register_buffer('medium_edge_index', torch.tensor(medium_edge_index, dtype=torch.long))
        self.register_buffer('coarse_edge_index', torch.tensor(coarse_edge_index, dtype=torch.long))

    def pool_to_coarse(self, x_fine, forcing_fine, temporal):
        """Pool fine-scale features to coarse scale"""
        B = x_fine.shape[0]
        device = x_fine.device

        # Pool state to medium
        x_medium = torch.zeros(B, self.n_medium, STATE_DIM, device=device)
        counts_medium = torch.zeros(self.n_medium, device=device)

        for i in range(self.n_fine):
            mid = self.fine_to_medium_idx[i]
            x_medium[:, mid] += x_fine[:, i]
            counts_medium[mid] += 1

        x_medium = x_medium / counts_medium.unsqueeze(0).unsqueeze(-1).clamp(min=1)

        # Pool forcing to medium
        forcing_medium = torch.zeros(B, self.n_medium, FORCING_FEATURES, device=device)
        for i in range(self.n_fine):
            mid = self.fine_to_medium_idx[i]
            forcing_medium[:, mid] += forcing_fine[:, i]
        forcing_medium = forcing_medium / counts_medium.unsqueeze(0).unsqueeze(-1).clamp(min=1)

        # Pool medium to coarse
        x_coarse = torch.zeros(B, self.n_coarse, STATE_DIM, device=device)
        forcing_coarse = torch.zeros(B, self.n_coarse, FORCING_FEATURES, device=device)
        counts_coarse = torch.zeros(self.n_coarse, device=device)

        for i in range(self.n_medium):
            cid = self.medium_to_coarse_idx[i]
            x_coarse[:, cid] += x_medium[:, i]
            forcing_coarse[:, cid] += forcing_medium[:, i]
            counts_coarse[cid] += 1

        x_coarse = x_coarse / counts_coarse.unsqueeze(0).unsqueeze(-1).clamp(min=1)
        forcing_coarse = forcing_coarse / counts_coarse.unsqueeze(0).unsqueeze(-1).clamp(min=1)

        # Expand temporal to all coarse nodes
        temporal_coarse = temporal[:, :self.n_coarse, :]
        if temporal_coarse.shape[1] < self.n_coarse:
            temporal_coarse = temporal[:, 0:1, :].expand(B, self.n_coarse, -1)

        # Add normalized positions
        coarse_pos_norm = self.coarse_pos.to(device)
        coarse_pos_norm = (coarse_pos_norm - coarse_pos_norm.mean(0)) / (coarse_pos_norm.std(0) + 1e-8)
        coarse_pos_batch = coarse_pos_norm.unsqueeze(0).expand(B, -1, -1)

        # Position encoding
        pos_enc = torch.zeros(B, self.n_coarse, 3, device=device)
        pos_enc[:, :, 0] = coarse_pos_batch[:, :, 0]
        pos_enc[:, :, 1] = coarse_pos_batch[:, :, 1]
        pos_enc[:, :, 2] = torch.sqrt(coarse_pos_batch[:, :, 0]**2 + coarse_pos_batch[:, :, 1]**2)

        return x_coarse, forcing_coarse, x_medium, forcing_medium, temporal_coarse, pos_enc

    def upsample_to_fine(self, h_coarse, h_medium):
        """Upsample coarse/medium features to fine scale using nearest neighbor"""
        B = h_coarse.shape[0]
        device = h_coarse.device

        # Upsample coarse to medium
        h_coarse_to_medium = h_coarse[:, self.medium_to_coarse_idx, :]

        # Combine with medium features
        h_medium_combined = h_medium + h_coarse_to_medium

        # Upsample medium to fine
        h_fine = h_medium_combined[:, self.fine_to_medium_idx, :]

        return h_fine

    def forward(self, x, x_prev, dxdt, tidal_harmonics, static_features, forcing,
                edge_index, edge_attr):
        """
        Forward pass with multi-scale processing.

        Args:
            x: Current state (B, N_fine, 1)
            x_prev: Previous state (B, N_fine, 1)
            dxdt: Rate of change (B, N_fine, 1)
            tidal_harmonics: Tidal features (B, N_fine, 12)
            static_features: Static node features (B, N_fine, 4)
            forcing: Forcing features (B, N_fine, 8)
            edge_index: Fine-scale edges (2, E)
            edge_attr: Fine-scale edge attributes (E, 3)

        Returns:
            pred: Prediction (B, N_fine, 1)
        """
        B = x.shape[0]
        device = x.device

        # Move buffers to device
        self.fine_to_medium_idx = self.fine_to_medium_idx.to(device)
        self.medium_to_coarse_idx = self.medium_to_coarse_idx.to(device)
        self.coarse_pos = self.coarse_pos.to(device)
        self.medium_pos = self.medium_pos.to(device)
        self.coarse_edge_index = self.coarse_edge_index.to(device)
        self.medium_edge_index = self.medium_edge_index.to(device)

        # ============================================
        # 1. FINE-SCALE PREDICTION (Frozen pretrained)
        # ============================================
        with torch.no_grad():
            self.fine_model.eval()
            fine_pred = self.fine_model(x, x_prev, dxdt, tidal_harmonics,
                                        static_features, forcing, edge_index, edge_attr)
            fine_hidden = self.fine_model.get_hidden_features(
                x, x_prev, dxdt, tidal_harmonics, static_features, forcing, edge_index, edge_attr
            )

        # ============================================
        # 2. POOL TO COARSE SCALES
        # ============================================
        x_coarse, forcing_coarse, x_medium, forcing_medium, temporal_coarse, pos_enc = \
            self.pool_to_coarse(x, forcing, tidal_harmonics)

        # ============================================
        # 3. COARSE-SCALE PROCESSING (1k nodes)
        # ============================================
        coarse_input = torch.cat([x_coarse, forcing_coarse, temporal_coarse, pos_enc], dim=-1)
        h_coarse = self.coarse_gnn(coarse_input, self.coarse_edge_index)

        # ============================================
        # 4. MEDIUM-SCALE PROCESSING (5k nodes)
        # ============================================
        # Upsample coarse hidden to medium
        h_coarse_up = h_coarse[:, self.medium_to_coarse_idx, :]

        # Medium position encoding
        medium_pos_norm = self.medium_pos.to(device)
        medium_pos_norm = (medium_pos_norm - medium_pos_norm.mean(0)) / (medium_pos_norm.std(0) + 1e-8)
        medium_pos_batch = medium_pos_norm.unsqueeze(0).expand(B, -1, -1)
        medium_pos_enc = torch.zeros(B, self.n_medium, 3, device=device)
        medium_pos_enc[:, :, 0] = medium_pos_batch[:, :, 0]
        medium_pos_enc[:, :, 1] = medium_pos_batch[:, :, 1]
        medium_pos_enc[:, :, 2] = torch.sqrt(medium_pos_batch[:, :, 0]**2 + medium_pos_batch[:, :, 1]**2)

        # Temporal for medium (same as coarse, expanded)
        temporal_medium = tidal_harmonics[:, 0:1, :].expand(B, self.n_medium, -1)

        medium_input = torch.cat([x_medium, forcing_medium, temporal_medium, medium_pos_enc, h_coarse_up], dim=-1)
        h_medium = self.medium_gnn(medium_input, self.medium_edge_index)

        # ============================================
        # 5. UPSAMPLE TO FINE SCALE
        # ============================================
        h_multiscale = self.upsample_to_fine(h_coarse, h_medium)

        # ============================================
        # 6. FUSION - Combine fine prediction with multi-scale correction
        # ============================================
        # Concatenate fine hidden with upsampled multi-scale
        fusion_input = torch.cat([fine_hidden, h_multiscale], dim=-1)

        # Compute correction
        correction = self.fusion_mlp(fusion_input.reshape(B * self.n_fine, -1)).reshape(B, self.n_fine, STATE_DIM)

        # Apply scaled correction to fine prediction
        scale = torch.sigmoid(self.correction_scale)  # 0 to 1
        final_pred = fine_pred + scale * correction

        return final_pred


# ============================================================
# Multi-Scale Mesh Creation
# ============================================================

def create_multiscale_mesh(lon, lat, n_medium=5000, n_coarse=1000):
    """
    Create coarsened versions of the fine mesh.

    Args:
        lon, lat: Fine mesh coordinates (N_fine,)
        n_medium: Number of medium-scale nodes
        n_coarse: Number of coarse-scale nodes

    Returns:
        dict with mesh mappings and edges
    """
    logger.info(f"Creating multi-scale mesh: {len(lon)} -> {n_medium} -> {n_coarse}")

    pos = np.stack([lon, lat], axis=1)
    n_fine = len(lon)

    # ========================================
    # Create MEDIUM mesh (5k nodes)
    # ========================================
    logger.info("  Clustering to medium scale...")
    kmeans_medium = MiniBatchKMeans(n_clusters=n_medium, random_state=42, batch_size=1000)
    fine_to_medium_idx = kmeans_medium.fit_predict(pos)
    medium_pos = kmeans_medium.cluster_centers_

    # Create edges for medium mesh using Delaunay
    logger.info("  Creating medium-scale edges...")
    tri = Delaunay(medium_pos)
    medium_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                medium_edges.add((simplex[i], simplex[j]))
                medium_edges.add((simplex[j], simplex[i]))
    medium_edge_index = np.array(list(medium_edges)).T

    # ========================================
    # Create COARSE mesh (1k nodes)
    # ========================================
    logger.info("  Clustering to coarse scale...")
    kmeans_coarse = MiniBatchKMeans(n_clusters=n_coarse, random_state=42, batch_size=500)
    medium_to_coarse_idx = kmeans_coarse.fit_predict(medium_pos)
    coarse_pos = kmeans_coarse.cluster_centers_

    # Create edges for coarse mesh
    logger.info("  Creating coarse-scale edges...")
    tri = Delaunay(coarse_pos)
    coarse_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                coarse_edges.add((simplex[i], simplex[j]))
                coarse_edges.add((simplex[j], simplex[i]))
    coarse_edge_index = np.array(list(coarse_edges)).T

    logger.info(f"  Medium: {n_medium} nodes, {medium_edge_index.shape[1]} edges")
    logger.info(f"  Coarse: {n_coarse} nodes, {coarse_edge_index.shape[1]} edges")

    return {
        'fine_to_medium': fine_to_medium_idx,
        'medium_to_coarse': medium_to_coarse_idx,
        'fine_pos': pos,
        'medium_pos': medium_pos,
        'coarse_pos': coarse_pos,
        'medium_edge_index': medium_edge_index,
        'coarse_edge_index': coarse_edge_index,
    }


# ============================================================
# Dataset (same as V2 training)
# ============================================================

class MultiScaleDataset(Dataset):
    """Dataset for multi-scale training"""

    def __init__(self, data_list, mesh_data):
        self.data_list = data_list
        self.mesh_data = mesh_data
        self.samples = []

        for data in data_list:
            date_str = data['date']
            T = data['elevation'].shape[0]
            for t in range(1, T - 1):
                self.samples.append((data, t))

        logger.info(f"MultiScaleDataset: {len(self.samples)} samples from {len(data_list)} dates")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, t = self.samples[idx]

        elevation = data['elevation']
        forcing = data['forcing']
        date_str = data['date']

        # Compute global hour for tidal harmonics
        date_dt = datetime.strptime(date_str, '%Y%m%d')
        global_hours = (date_dt - EPOCH_DATETIME).total_seconds() / 3600.0 + t * DT_HOURS

        # Tidal harmonics
        harmonics = []
        for name, period in TIDAL_PERIODS.items():
            phase = 2.0 * np.pi * global_hours / period
            harmonics.extend([np.sin(phase), np.cos(phase)])
        tidal = np.array(harmonics, dtype=np.float32)

        # Current and previous state
        cwl_t = np.nan_to_num(elevation[t], nan=0.0).astype(np.float32) / ETA_SCALE
        cwl_prev = np.nan_to_num(elevation[t-1], nan=0.0).astype(np.float32) / ETA_SCALE
        cwl_next = np.nan_to_num(elevation[t+1], nan=0.0).astype(np.float32) / ETA_SCALE

        # Forcing at time t
        forcing_t = np.stack([
            forcing['u10'][t],
            forcing['v10'][t],
            forcing['wind_speed'][t],
            forcing['wind_speed_sq'][t],
            forcing['wind_dir'][t],
            forcing['pressure'][t],
            forcing['dP_dx'][t],
            forcing['dP_dy'][t],
        ], axis=1).astype(np.float32)

        return {
            'state': cwl_t,
            'state_prev': cwl_prev,
            'target': cwl_next,
            'forcing': forcing_t,
            'tidal': tidal,
            'date': date_str,
            't': t,
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'state': torch.tensor(np.stack([b['state'] for b in batch])),
        'state_prev': torch.tensor(np.stack([b['state_prev'] for b in batch])),
        'target': torch.tensor(np.stack([b['target'] for b in batch])),
        'forcing': torch.tensor(np.stack([b['forcing'] for b in batch])),
        'tidal': torch.tensor(np.stack([b['tidal'] for b in batch])),
    }


# ============================================================
# Training Loop
# ============================================================

def train_epoch(model, dataloader, optimizer, device, mesh_tensors,
                rollout_steps=1, grad_accum_steps=8):
    """Train for one epoch"""
    model.train()
    # Keep fine model frozen
    model.fine_model.eval()

    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        state = batch['state'].unsqueeze(-1).to(device)
        state_prev = batch['state_prev'].unsqueeze(-1).to(device)
        target = batch['target'].unsqueeze(-1).to(device)
        forcing = batch['forcing'].to(device)
        tidal = batch['tidal'].to(device)

        B, N = state.shape[:2]

        # Expand tidal to all nodes
        tidal_expanded = tidal.unsqueeze(1).expand(B, N, -1)

        # Static features
        static = mesh_tensors['static'].unsqueeze(0).expand(B, -1, -1).to(device)
        edge_index = mesh_tensors['edge_index'].to(device)
        edge_attr = mesh_tensors['edge_attr'].to(device)

        # Rollout
        current = state
        current_prev = state_prev
        loss = 0.0

        for step in range(rollout_steps):
            # Rate of change
            dxdt = (current - current_prev) / DT_HOURS

            # Forward pass
            pred = model(current, current_prev, dxdt, tidal_expanded,
                        static, forcing, edge_index, edge_attr)

            # Loss (MSE)
            step_loss = F.mse_loss(pred, target)
            loss = loss + step_loss

            # Update state for next step
            current_prev = current
            current = pred.detach()  # Detach to prevent memory buildup

        loss = loss / rollout_steps
        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if batch_idx % 200 == 0:
            logger.info(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item()*grad_accum_steps:.5f}")

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, mesh_tensors, rollout_steps=6):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].unsqueeze(-1).to(device)
            state_prev = batch['state_prev'].unsqueeze(-1).to(device)
            target = batch['target'].unsqueeze(-1).to(device)
            forcing = batch['forcing'].to(device)
            tidal = batch['tidal'].to(device)

            B, N = state.shape[:2]
            tidal_expanded = tidal.unsqueeze(1).expand(B, N, -1)
            static = mesh_tensors['static'].unsqueeze(0).expand(B, -1, -1).to(device)
            edge_index = mesh_tensors['edge_index'].to(device)
            edge_attr = mesh_tensors['edge_attr'].to(device)

            current = state
            current_prev = state_prev
            loss = 0.0

            for step in range(rollout_steps):
                dxdt = (current - current_prev) / DT_HOURS
                pred = model(current, current_prev, dxdt, tidal_expanded,
                            static, forcing, edge_index, edge_attr)
                loss = loss + F.mse_loss(pred, target)
                current_prev = current
                current = pred

            total_loss += (loss / rollout_steps).item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Main Training
# ============================================================

def main():
    logger.info("="*70)
    logger.info("HIERARCHICAL MULTI-SCALE GNN TRAINING")
    logger.info("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================
    # Load fine-scale mesh
    # ========================================
    mesh_path = DATA_DIR / 'mesh.npz'
    mesh_data = dict(np.load(mesh_path, allow_pickle=True))

    lon = mesh_data['lon']
    lat = mesh_data['lat']
    depth = mesh_data['depth']
    edge_index = mesh_data['edge_index']
    n_fine = len(lon)

    logger.info(f"Fine mesh: {n_fine:,} nodes, {edge_index.shape[1]:,} edges")

    # ========================================
    # Create multi-scale mesh
    # ========================================
    multiscale = create_multiscale_mesh(lon, lat, n_medium=N_MEDIUM, n_coarse=N_COARSE)

    # ========================================
    # Prepare mesh tensors
    # ========================================
    # Static features for fine scale
    ref_lon, ref_lat = lon.mean(), lat.mean()
    R = 6371000.0
    x_cart = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y_cart = R * np.radians(lat - ref_lat)
    x_norm = 2 * (x_cart - x_cart.min()) / (x_cart.max() - x_cart.min() + 1e-8) - 1
    y_norm = 2 * (y_cart - y_cart.min()) / (y_cart.max() - y_cart.min() + 1e-8) - 1
    depth_safe = np.maximum(np.abs(depth), 0.1)
    depth_log = np.log10(depth_safe)
    depth_norm = (depth_log - depth_log.mean()) / (depth_log.std() + 1e-8)

    static_features = np.stack([x_norm, y_norm, depth_norm, np.zeros_like(depth_norm)], axis=1).astype(np.float32)

    # Edge features
    src, dst = edge_index[0], edge_index[1]
    dx = x_cart[dst] - x_cart[src]
    dy = y_cart[dst] - y_cart[src]
    dist = np.sqrt(dx**2 + dy**2)
    char_length = np.median(dist) + 1e-8
    edge_attr = np.stack([dx/char_length, dy/char_length, dist/char_length], axis=1).astype(np.float32)

    mesh_tensors = {
        'static': torch.tensor(static_features),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_attr),
    }

    # ========================================
    # Load training data
    # ========================================
    logger.info("\nLoading training data...")
    train_files = sorted([f for f in DATA_DIR.glob('processed_202[34]*.npz') if 'mesh' not in f.stem])[:100]  # Limit for faster iteration
    val_files = sorted([f for f in DATA_DIR.glob('processed_2025*.npz') if 'mesh' not in f.stem])[:20]

    train_data = []
    for f in train_files:
        data = np.load(f)
        train_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    val_data = []
    for f in val_files:
        data = np.load(f)
        val_data.append({
            'date': f.stem.replace('processed_', ''),
            'elevation': data['elevation'],
            'forcing': {k: data[k] for k in ['u10', 'v10', 'wind_speed', 'wind_speed_sq',
                                              'wind_dir', 'pressure', 'dP_dx', 'dP_dy']}
        })

    logger.info(f"Training dates: {len(train_data)}")
    logger.info(f"Validation dates: {len(val_data)}")

    train_dataset = MultiScaleDataset(train_data, mesh_data)
    val_dataset = MultiScaleDataset(val_data, mesh_data)

    # ========================================
    # Create model
    # ========================================
    logger.info("\nCreating hierarchical multi-scale model...")
    model = HierarchicalMultiScaleGNN(
        pretrained_path=PRETRAINED_25K,
        n_fine=n_fine,
        n_medium=N_MEDIUM,
        n_coarse=N_COARSE,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    # Set mesh mappings
    model.set_mesh_mappings(
        fine_to_medium=multiscale['fine_to_medium'],
        medium_to_coarse=multiscale['medium_to_coarse'],
        fine_pos=multiscale['fine_pos'],
        medium_pos=multiscale['medium_pos'],
        coarse_pos=multiscale['coarse_pos'],
        medium_edge_index=multiscale['medium_edge_index'],
        coarse_edge_index=multiscale['coarse_edge_index'],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"Frozen parameters: {frozen_params:,} (fine-scale backbone)")

    # ========================================
    # Optimizer (only trainable params)
    # ========================================
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ========================================
    # Training loop
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)

    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # Determine rollout steps for this epoch
        rollout_steps = 1
        batch_mult = 4
        for steps, (start, end, mult) in ROLLOUT_SCHEDULE.items():
            if start <= epoch <= end:
                rollout_steps = steps
                batch_mult = mult
                break

        batch_size = BASE_BATCH_SIZE * batch_mult

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn, pin_memory=True
        )

        logger.info(f"\nEpoch {epoch}/{NUM_EPOCHS} | rollout={rollout_steps} | batch={batch_size}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 mesh_tensors, rollout_steps, GRAD_ACCUM_STEPS)

        # Validate
        val_loss = validate(model, val_loader, device, mesh_tensors, rollout_steps=6)

        scheduler.step()

        logger.info(f"  train={train_loss:.5f} | val={val_loss:.5f} | lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % 5 == 0 or val_loss < best_val_loss:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'multiscale': multiscale,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, OUTPUT_DIR / 'best_multiscale.pt')
                logger.info(f"  Saved best model (val={val_loss:.5f})")

            if epoch % 5 == 0:
                torch.save(ckpt, OUTPUT_DIR / f'checkpoint_multiscale_epoch_{epoch}.pt')

    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_val_loss:.5f}")


if __name__ == '__main__':
    main()
