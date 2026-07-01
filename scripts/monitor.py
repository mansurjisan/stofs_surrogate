#!/usr/bin/env python3
"""Monitoring: forecast skill vs CO-OPS ground truth, drift, and Prometheus metrics.

Runs on synthetic sample data by default (no network); writes a drift report to
outputs/monitoring/ and emits Prometheus metrics.

    python scripts/monitor.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stofs_surrogate.monitoring.drift import drift_report
from stofs_surrogate.monitoring.metrics import drift_to_prometheus, skill_to_prometheus
from stofs_surrogate.monitoring.observations import STATIONS, synthetic_observations
from stofs_surrogate.monitoring.skill import compute_skill, detect_degradation


def _synthetic_predictions(obs: pd.DataFrame, bias: float = 0.05,
                           noise: float = 0.08, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = obs.copy()
    pred["water_level"] = obs["water_level"] + bias + rng.normal(0, noise, len(obs))
    return pred


def main():
    ap = argparse.ArgumentParser(description="Skill + drift monitoring on sample data")
    ap.add_argument("--output-dir", default="outputs/monitoring")
    ap.add_argument("--rmse-threshold", type=float, default=0.15)
    args = ap.parse_args()

    # Forecast skill vs (synthetic) tide-gauge observations.
    obs = pd.concat([synthetic_observations(sid) for sid in STATIONS], ignore_index=True)
    pred = _synthetic_predictions(obs)
    skill = detect_degradation(compute_skill(pred, obs), args.rmse_threshold)
    print("Per-station skill:")
    print(skill.to_string())

    # Forcing drift: a reference vs a shifted current distribution.
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"wind_speed": rng.normal(8, 2, 300),
                              "pressure": rng.normal(1013, 5, 300)})
    current = pd.DataFrame({"wind_speed": rng.normal(11, 3, 300),
                            "pressure": rng.normal(1007, 6, 300)})
    summary = drift_report(reference, current, output_dir=args.output_dir)
    print(f"\nDrift: dataset_drift={summary['dataset_drift']} "
          f"({summary['n_drifted']}/{summary['n_columns']} columns) -> {args.output_dir}/")

    metrics_path = Path(args.output_dir) / "metrics.prom"
    metrics_path.write_text(skill_to_prometheus(skill) + drift_to_prometheus(summary))
    print(f"Prometheus metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
