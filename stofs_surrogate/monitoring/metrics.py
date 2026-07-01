"""Render monitoring metrics in Prometheus text exposition format."""

from typing import Dict

import pandas as pd


def skill_to_prometheus(skill: pd.DataFrame) -> str:
    """Render per-station skill (rmse / corr / bias) as Prometheus gauges."""
    lines = [
        "# HELP stofs_forecast_rmse Forecast RMSE vs observed water level.",
        "# TYPE stofs_forecast_rmse gauge",
    ]
    frame = skill.reset_index()
    for _, row in frame.iterrows():
        station = row.get("station_id", "unknown")
        for metric in ("rmse", "corr", "bias"):
            if metric in row and pd.notna(row[metric]):
                lines.append(
                    f'stofs_forecast_{metric}{{station="{station}"}} {float(row[metric])}'
                )
    return "\n".join(lines) + "\n"


def drift_to_prometheus(summary: Dict) -> str:
    """Render a drift summary as Prometheus gauges."""
    lines = [
        "# HELP stofs_drift_detected Whether dataset drift was detected (1/0).",
        "# TYPE stofs_drift_detected gauge",
        f"stofs_drift_detected {int(bool(summary.get('dataset_drift', False)))}",
        "# HELP stofs_drift_columns Number of drifted columns.",
        "# TYPE stofs_drift_columns gauge",
        f"stofs_drift_columns {int(summary.get('n_drifted', 0))}",
    ]
    return "\n".join(lines) + "\n"
