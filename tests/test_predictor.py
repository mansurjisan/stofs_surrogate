"""Tests for the package inference API (Predictor, EnsemblePredictor)."""

import torch

from stofs_surrogate.inference import EnsemblePredictor, Predictor
from stofs_surrogate.models.gnn import create_model


def _small_graph(num_nodes=40, num_edges=120, state_dim=1):
    state = torch.randn(num_nodes, state_dim)
    node_features = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 3)
    return state, node_features, edge_index, edge_attr


def test_predictor_rollout_shape():
    model = create_model("stofs_gnn", state_dim=1, node_feature_dim=3,
                         edge_feature_dim=3, hidden_dim=16, num_layers=2)
    predictor = Predictor(model, device="cpu")
    state, nf, ei, ea = _small_graph(state_dim=1)
    out = predictor.rollout(state, nf, ei, ea, num_steps=5)
    assert out.shape == (6, 40, 1)


def test_predictor_from_checkpoint(tmp_path):
    kwargs = dict(state_dim=1, node_feature_dim=3, edge_feature_dim=3,
                  hidden_dim=16, num_layers=2)
    model = create_model("stofs_gnn", **kwargs)
    ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt)

    predictor = Predictor.from_checkpoint(ckpt, model_type="stofs_gnn", model_kwargs=kwargs)
    state, nf, ei, ea = _small_graph(state_dim=1)
    out = predictor.rollout(state, nf, ei, ea, num_steps=3)
    assert out.shape == (4, 40, 1)


def test_ensemble_shapes_and_stats():
    model = create_model("stofs_gnn", state_dim=1, node_feature_dim=3,
                         edge_feature_dim=3, forcing_dim=3, hidden_dim=16, num_layers=2)
    predictor = Predictor(model, device="cpu")
    ensemble = EnsemblePredictor(predictor, n_members=4, seed=1)

    state, nf, ei, ea = _small_graph(state_dim=1)
    forcing_seq = torch.randn(5, 40, 3)  # [num_steps, num_nodes, forcing_dim]
    result = ensemble.run(state, nf, ei, ea, num_steps=5, forcing_sequence=forcing_seq,
                          channel_scales={0: 2.0, 1: 2.0, 2: 200.0})

    assert result["members"].shape == (4, 6, 40, 1)
    assert result["mean"].shape == (6, 40, 1)
    assert result["std"].shape == (6, 40, 1)
    assert set(result["percentiles"]) == {5, 25, 50, 75, 95}
