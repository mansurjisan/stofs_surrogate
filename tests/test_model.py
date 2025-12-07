"""Tests for GNN model architecture."""

import pytest
import torch
import numpy as np


def test_swe_block_shape():
    """Test SWE-inspired graph block output shape."""
    # Import here to avoid import errors if package not installed
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import SWEInspiredGraphBlock

    hidden_dim = 64
    num_nodes = 100
    num_edges = 300

    block = SWEInspiredGraphBlock(hidden_dim)

    # Create dummy inputs
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    # Forward pass
    out = block(x, edge_index, edge_attr)

    assert out.shape == (num_nodes, hidden_dim)


def test_model_forward():
    """Test full model forward pass."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import PhysicsInformedCWLModel

    # Model config
    hidden_dim = 64
    num_layers = 3
    static_features = 4
    forcing_features = 3

    model = PhysicsInformedCWLModel(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        static_features=static_features,
        forcing_features=forcing_features
    )

    # Create dummy inputs
    num_nodes = 100
    num_edges = 300

    cwl = torch.randn(num_nodes, 1)
    static = torch.randn(num_nodes, static_features)
    forcing = torch.randn(num_nodes, forcing_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    # Forward pass
    model.eval()
    with torch.no_grad():
        out = model(cwl, static, forcing, edge_index, edge_attr)

    assert out.shape == (num_nodes, 1)


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import PhysicsInformedCWLModel

    model = PhysicsInformedCWLModel(
        hidden_dim=32,
        num_layers=2,
        static_features=4,
        forcing_features=3
    )

    num_nodes = 50
    num_edges = 150

    cwl = torch.randn(num_nodes, 1, requires_grad=True)
    static = torch.randn(num_nodes, 4)
    forcing = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    out = model(cwl, static, forcing, edge_index, edge_attr)
    loss = out.sum()
    loss.backward()

    assert cwl.grad is not None
    assert not torch.all(cwl.grad == 0)


if __name__ == "__main__":
    test_swe_block_shape()
    test_model_forward()
    test_model_gradient_flow()
    print("All model tests passed!")
