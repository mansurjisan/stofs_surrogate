"""Monitoring: forecast-skill tracking against tide-gauge ground truth, and drift reports.

- ``observations``: CO-OPS observed water level (searvey live path + synthetic fallback).
- ``skill``: rolling RMSE / correlation / bias per station and lead time, with degradation flags.
- ``drift``: input-forcing and prediction drift (KS + PSI; optional Evidently backend).
- ``metrics``: render metrics in Prometheus text format.
"""

from stofs_surrogate.monitoring.drift import compute_drift, drift_report
from stofs_surrogate.monitoring.skill import compute_skill, detect_degradation, rolling_skill

__all__ = [
    "compute_skill",
    "rolling_skill",
    "detect_degradation",
    "compute_drift",
    "drift_report",
]
