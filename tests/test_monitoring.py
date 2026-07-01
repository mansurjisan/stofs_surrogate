"""Tests for monitoring: skill, degradation, drift, and metrics."""

import numpy as np
import pandas as pd
import pytest

from stofs_surrogate.monitoring.drift import compute_drift, drift_report
from stofs_surrogate.monitoring.metrics import skill_to_prometheus
from stofs_surrogate.monitoring.observations import synthetic_observations
from stofs_surrogate.monitoring.skill import (
    compute_skill,
    detect_degradation,
    rolling_skill,
)


def _pred_obs(bias=0.1, n=48):
    obs = synthetic_observations("8574680", periods=n)
    pred = obs.copy()
    pred["water_level"] = obs["water_level"] + bias  # constant offset -> known metrics
    return pred, obs


def test_compute_skill_known_values():
    pred, obs = _pred_obs(bias=0.1, n=48)
    row = compute_skill(pred, obs).loc["8574680"]
    assert row["n"] == 48
    assert abs(row["bias"] - 0.1) < 1e-9   # constant +0.1 offset
    assert abs(row["rmse"] - 0.1) < 1e-9   # error is exactly 0.1 everywhere
    assert abs(row["corr"] - 1.0) < 1e-9   # shifted copy -> perfect correlation


def test_detect_degradation_flags_threshold():
    degraded = detect_degradation(compute_skill(*_pred_obs(bias=0.3)), rmse_threshold=0.15)
    assert bool(degraded.loc["8574680", "degraded"]) is True
    ok = detect_degradation(compute_skill(*_pred_obs(bias=0.02)), rmse_threshold=0.15)
    assert bool(ok.loc["8574680", "degraded"]) is False


def test_rolling_skill_shape():
    roll = rolling_skill(*_pred_obs(bias=0.1, n=48), window="6h")
    assert set(roll.columns) == {"station_id", "time", "rolling_rmse"}
    assert len(roll) == 48


def test_compute_drift_detects_shift():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 300)})
    current = pd.DataFrame({"x": rng.normal(3, 1, 300)})  # large shift
    summary = compute_drift(reference, current)
    assert summary["dataset_drift"] is True
    assert summary["columns"]["x"]["drifted"] is True


def test_drift_report_writes_files(tmp_path):
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"x": rng.normal(2, 1, 200), "y": rng.normal(0, 1, 200)})
    summary = drift_report(reference, current, output_dir=str(tmp_path), use_evidently=False)
    assert (tmp_path / "drift_report.json").exists()
    assert (tmp_path / "drift_report.html").exists()
    assert summary["n_columns"] == 2


def test_evidently_report_optional(tmp_path):
    pytest.importorskip("evidently")
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    current = pd.DataFrame({"x": rng.normal(2, 1, 200)})
    drift_report(reference, current, output_dir=str(tmp_path), use_evidently=True)
    assert (tmp_path / "evidently_drift.html").exists()


def test_skill_to_prometheus_format():
    text = skill_to_prometheus(compute_skill(*_pred_obs(bias=0.1)))
    assert "stofs_forecast_rmse" in text
    assert 'station="8574680"' in text
