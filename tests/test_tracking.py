"""Tests for the experiment-tracking abstraction."""

import pytest


def test_make_noop_tracker_does_nothing():
    from stofs_surrogate.training.tracking import NoOpTracker, make_tracker

    tracker = make_tracker("none")
    assert isinstance(tracker, NoOpTracker)
    # None of these should raise.
    tracker.start_run("run", tags={"a": 1})
    tracker.log_params({"x": 1, "nested": {"y": 2}})
    tracker.log_metrics({"loss": 0.5}, step=0)
    tracker.end_run()


def test_flatten_params_nested_and_lists():
    from stofs_surrogate.training.tracking import _flatten_params

    flat = _flatten_params({"a": 1, "b": {"c": 2, "d": [1, 2, 3]}})
    assert flat == {"a": 1, "b.c": 2, "b.d": "1,2,3"}


def test_make_tracker_rejects_unknown():
    from stofs_surrogate.training.tracking import make_tracker

    with pytest.raises(ValueError):
        make_tracker("not-a-real-backend")


def test_get_git_sha_returns_string():
    from stofs_surrogate.training.tracking import get_git_sha

    sha = get_git_sha()
    assert isinstance(sha, str) and len(sha) > 0


def test_mlflow_smoke_run_logs_params_metrics_artifact(tmp_path):
    """End-to-end: a CPU synthetic train logs params, metrics, lineage, and a model
    artifact to a temporary MLflow file store."""
    mlflow = pytest.importorskip("mlflow")

    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from stofs_surrogate.data.dataset import SyntheticSWEDataset
    from stofs_surrogate.models.gnn import create_model
    from stofs_surrogate.training.tracking import make_tracker
    from stofs_surrogate.training.trainer import Trainer

    torch.manual_seed(0)
    uri = (tmp_path / "mlruns").as_uri()
    tracker = make_tracker("mlflow", experiment="test-smoke", tracking_uri=uri)

    dataset = SyntheticSWEDataset(num_nodes=100, num_samples=12, include_velocity=True)
    loader = PyGDataLoader(dataset, batch_size=4, shuffle=False)
    model = create_model("simple", input_dim=3, output_dim=3, hidden_dim=16, num_layers=2)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device="cpu",
        output_dir=str(tmp_path / "out"),
        num_epochs=1,
        eval_every=1,
        save_every=1,
        mixed_precision=False,
        tracker=tracker,
        config={"model": {"type": "simple"}, "training": {"learning_rate": 1e-3}},
    )
    trainer.train()

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("test-smoke")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    # Params from the resolved config + trainer hyperparameters.
    assert run.data.params.get("model.type") == "simple"
    assert "num_params" in run.data.params
    assert run.data.params.get("optimizer") == "AdamW"

    # Per-epoch metrics.
    assert "train_loss" in run.data.metrics
    assert "val_loss" in run.data.metrics

    # Lineage: git SHA tagged.
    assert "git_sha" in run.data.tags

    # Model artifact logged.
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert any("final_model.pt" in a for a in artifacts)
