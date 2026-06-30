"""Tests for the MLflow model registry + lineage."""

import pytest


def test_data_manifest_hash_handles_empty():
    from stofs_surrogate.training.registry import data_manifest_hash

    assert data_manifest_hash(None) == "unknown"
    assert data_manifest_hash([]) == "unknown"


def test_data_manifest_hash_is_stable(tmp_path):
    from stofs_surrogate.training.registry import data_manifest_hash

    f = tmp_path / "data.npz"
    f.write_bytes(b"x" * 100)
    h1 = data_manifest_hash([str(f)])
    h2 = data_manifest_hash([str(f)])
    assert h1 == h2 and h1 != "unknown"


def test_register_model_with_lineage(tmp_path):
    pytest.importorskip("mlflow")

    from stofs_surrogate.models.gnn import create_model
    from stofs_surrogate.training.registry import ModelRegistry

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # registry requires a DB backend
    registry = ModelRegistry(uri, experiment="test-registry")

    model = create_model("stofs_gnn", state_dim=1, node_feature_dim=3,
                         edge_feature_dim=3, hidden_dim=16, num_layers=2)
    version = registry.register(
        model,
        name="test-model",
        config={"model": {"type": "stofs_gnn"}, "domain": {"name": "mid_atlantic"}},
        metrics={"val_rmse_cm": 16.2},
        alias="staging",
    )
    assert int(version.version) >= 1

    lineage = registry.get_lineage("test-model", version.version)
    assert "git_sha" in lineage["tags"]
    assert lineage["tags"]["data_manifest"] == "unknown"
    assert lineage["params"].get("model.type") == "stofs_gnn"
    assert "val_rmse_cm" in lineage["metrics"]
