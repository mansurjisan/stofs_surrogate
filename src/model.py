"""
Graph Neural Network models for STOFS surrogate.

Implements MeshGraphNet-style architecture for learning
Shallow Water Equation dynamics on unstructured meshes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MLPBlock(nn.Module):
    """Multi-layer perceptron block with residual connection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        activation: str = 'relu',
        layer_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()

        hidden_dim = hidden_dim or output_dim

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))

            if i < num_layers - 1:
                if layer_norm:
                    layers.append(nn.LayerNorm(out_dim))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'silu':
                    layers.append(nn.SiLU())

            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.residual = residual and (input_dim == output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.residual:
            out = out + x
        return out


class GraphNetworkBlock(MessagePassing):
    """
    Single message-passing block for MeshGraphNet.

    Implements:
    1. Edge update: e'_ij = MLP([e_ij, h_i, h_j])
    2. Node aggregation: agg_i = sum(e'_ij for j in N(i))
    3. Node update: h'_i = MLP([h_i, agg_i])
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        activation: str = 'relu',
    ):
        super().__init__(aggr='add')

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Edge update MLP
        self.edge_mlp = MLPBlock(
            input_dim=2 * node_dim + edge_dim,
            output_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            residual=False,
        )

        # Node update MLP
        self.node_mlp = MLPBlock(
            input_dim=node_dim + edge_dim,
            output_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            residual=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features, updated edge features
        """
        row, col = edge_index

        # Edge update
        edge_input = torch.cat([edge_attr, x[row], x[col]], dim=-1)
        edge_attr_new = self.edge_mlp(edge_input)

        # Node update via message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_new)

        return out, edge_attr_new

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Construct messages from neighbors."""
        return edge_attr

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node features."""
        node_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(node_input)


class STOFSSurrogateGNN(nn.Module):
    """
    Graph Neural Network surrogate for STOFS/ADCIRC.

    Architecture:
        Encoder -> Processor (N message-passing layers) -> Decoder

    Supports:
    - Water elevation (η) prediction
    - Velocity (u, v) prediction
    - Atmospheric forcing conditioning (wind, pressure)
    """

    def __init__(
        self,
        state_dim: int = 3,           # η, u, v
        node_feature_dim: int = 3,    # x, y, depth
        edge_feature_dim: int = 3,    # dx, dy, dist
        forcing_dim: int = 0,         # wind_u, wind_v, pressure (optional)
        hidden_dim: int = 128,
        num_layers: int = 8,
        output_dim: int = None,       # Default: same as state_dim
        activation: str = 'relu',
    ):
        """
        Initialize model.

        Args:
            state_dim: Dimension of state variables (η, u, v)
            node_feature_dim: Dimension of static node features
            edge_feature_dim: Dimension of edge features
            forcing_dim: Dimension of forcing inputs (0 = no forcing)
            hidden_dim: Hidden layer dimension
            num_layers: Number of message-passing layers
            output_dim: Output dimension (default: state_dim)
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()

        self.state_dim = state_dim
        self.forcing_dim = forcing_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        output_dim = output_dim or state_dim

        # Encoder: state + node_features [+ forcing] -> hidden
        encoder_input_dim = state_dim + node_feature_dim + forcing_dim
        self.node_encoder = MLPBlock(
            input_dim=encoder_input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            residual=False,
        )

        # Edge encoder
        self.edge_encoder = MLPBlock(
            input_dim=edge_feature_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            residual=False,
        )

        # Processor: stack of message-passing layers
        self.processor = nn.ModuleList([
            GraphNetworkBlock(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        # Decoder: hidden -> output state
        self.decoder = MLPBlock(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            activation=activation,
            residual=False,
        )

        # Initialize weights
        self._init_weights()

        # Log model info
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"STOFSSurrogateGNN: {num_params:,} parameters")
        logger.info(f"  State dim: {state_dim}, Forcing dim: {forcing_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}, Layers: {num_layers}")

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: predict next state.

        Args:
            state: Current state [num_nodes, state_dim] or [batch, num_nodes, state_dim]
            node_features: Static node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            forcing: Atmospheric forcing [num_nodes, forcing_dim] (optional)

        Returns:
            Predicted next state [num_nodes, state_dim]
        """
        # Build encoder input
        if forcing is not None and self.forcing_dim > 0:
            encoder_input = torch.cat([state, node_features, forcing], dim=-1)
        else:
            encoder_input = torch.cat([state, node_features], dim=-1)

        # Encode
        h = self.node_encoder(encoder_input)
        e = self.edge_encoder(edge_attr)

        # Process through message-passing layers
        for layer in self.processor:
            h, e = layer(h, edge_index, e)

        # Decode to output state
        output = self.decoder(h)

        return output

    def forward_pyg(self, data: Data) -> torch.Tensor:
        """
        Forward pass using PyG Data object.

        Convenience method for training loop.
        """
        forcing = data.forcing if hasattr(data, 'forcing') else None

        return self.forward(
            state=data.x,
            node_features=data.node_features,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            forcing=forcing,
        )

    def rollout(
        self,
        initial_state: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_steps: int,
        forcing_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive rollout for multi-step prediction.

        Args:
            initial_state: Initial state [num_nodes, state_dim]
            node_features: Static node features [num_nodes, node_feature_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_feature_dim]
            num_steps: Number of forecast steps
            forcing_sequence: Forcing for each step [num_steps, num_nodes, forcing_dim]

        Returns:
            Predictions [num_steps+1, num_nodes, state_dim]
        """
        self.eval()

        predictions = [initial_state.clone()]
        current_state = initial_state

        with torch.no_grad():
            for t in range(num_steps):
                forcing = None
                if forcing_sequence is not None:
                    forcing = forcing_sequence[t]

                pred = self.forward(
                    state=current_state,
                    node_features=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    forcing=forcing,
                )

                predictions.append(pred)
                current_state = pred

        return torch.stack(predictions, dim=0)


class SimpleMeshGraphNet(nn.Module):
    """
    Simplified MeshGraphNet for small-scale testing.

    Lighter architecture when PhysicsNeMo is not available
    or for quick experiments.
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for position and bathymetry
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message passing layers
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        # Node decoder
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SimpleMeshGraphNet: {num_params:,} parameters")

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        bathymetry: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, input_dim]
            pos: Node positions [num_nodes, 2]
            edge_index: Edge connectivity [2, num_edges]
            bathymetry: Bathymetry at nodes [num_nodes]
        """
        # Normalize position
        pos_norm = pos / pos.abs().max()
        bath_norm = bathymetry / bathymetry.abs().max()

        # Encode nodes
        node_input = torch.cat([x, pos_norm, bath_norm.unsqueeze(-1)], dim=-1)
        h = self.node_encoder(node_input)

        # Message passing
        for processor in self.processors:
            row, col = edge_index
            neighbor_feats = h[col]

            # Mean aggregation
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, neighbor_feats)
            counts = torch.zeros(h.size(0), device=h.device)
            counts.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
            agg = agg / counts.unsqueeze(-1).clamp(min=1)

            # Update
            combined = torch.cat([h, agg], dim=-1)
            h = h + processor(combined)

        # Decode
        output = self.node_decoder(h)

        return output


def create_model(
    model_type: str = 'stofs_gnn',
    **kwargs
) -> nn.Module:
    """
    Factory function to create model.

    Args:
        model_type: 'stofs_gnn' or 'simple'
        **kwargs: Model-specific arguments

    Returns:
        Model instance
    """
    if model_type == 'stofs_gnn':
        return STOFSSurrogateGNN(**kwargs)
    elif model_type == 'simple':
        return SimpleMeshGraphNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
