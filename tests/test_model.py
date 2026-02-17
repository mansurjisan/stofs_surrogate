"""Tests for GNN model architecture."""

import pytest
import torch
import numpy as np


def test_graph_block_shape():
    """Test GraphNetworkBlock output shape."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import GraphNetworkBlock

    node_dim = 64
    edge_dim = 64
    hidden_dim = 64
    num_nodes = 100
    num_edges = 300

    block = GraphNetworkBlock(node_dim, edge_dim, hidden_dim)

    # Create dummy inputs
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_dim)

    # Forward pass
    h_out, e_out = block(x, edge_index, edge_attr)

    assert h_out.shape == (num_nodes, node_dim)
    assert e_out.shape == (num_edges, edge_dim)


def test_model_forward():
    """Test full STOFSSurrogateGNN forward pass."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import STOFSSurrogateGNN

    model = STOFSSurrogateGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        forcing_dim=3,
        hidden_dim=64,
        num_layers=3,
    )

    num_nodes = 100
    num_edges = 300

    state = torch.randn(num_nodes, 1)
    node_features = torch.randn(num_nodes, 3)
    forcing = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    model.eval()
    with torch.no_grad():
        out = model(state, node_features, edge_index, edge_attr, forcing=forcing)

    assert out.shape == (num_nodes, 1)


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import STOFSSurrogateGNN

    model = STOFSSurrogateGNN(
        state_dim=1,
        node_feature_dim=3,
        edge_feature_dim=3,
        hidden_dim=32,
        num_layers=2,
    )

    num_nodes = 50
    num_edges = 150

    state = torch.randn(num_nodes, 1, requires_grad=True)
    node_features = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)

    out = model(state, node_features, edge_index, edge_attr)
    loss = out.sum()
    loss.backward()

    assert state.grad is not None
    assert not torch.all(state.grad == 0)


def test_create_model():
    """Test model factory function."""
    import sys
    sys.path.insert(0, '.')

    from stofs_surrogate.models.gnn import create_model

    model = create_model('stofs_gnn', state_dim=1, hidden_dim=32, num_layers=2)
    assert isinstance(model, torch.nn.Module)

    model2 = create_model('simple', input_dim=1, hidden_dim=32, num_layers=2)
    assert isinstance(model2, torch.nn.Module)


if __name__ == "__main__":
    test_graph_block_shape()
    test_model_forward()
    test_model_gradient_flow()
    test_create_model()
    print("All model tests passed!")
