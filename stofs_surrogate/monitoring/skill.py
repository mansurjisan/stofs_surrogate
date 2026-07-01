"""Forecast-skill tracking against tide-gauge observations.

Aligns predictions with observations and computes RMSE / correlation / bias per station
(and, optionally, per lead time), plus a rolling-window view and a degradation flag.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _align(pred: pd.DataFrame, obs: pd.DataFrame, value: str) -> pd.DataFrame:
    """Inner-join predictions and observations on ``[station_id, time]``."""
    left = pred.rename(columns={value: "pred"})
    right = obs.rename(columns={value: "obs"})[["station_id", "time", "obs"]]
    return left.merge(right, on=["station_id", "time"])


def compute_skill(pred: pd.DataFrame, obs: pd.DataFrame, value: str = "water_level",
                  by: Sequence[str] = ("station_id",)) -> pd.DataFrame:
    """RMSE / correlation / bias between predictions and observations.

    ``pred`` and ``obs`` have columns ``[time, station_id, <value>]`` (``pred`` may also
    carry ``lead_time``). Returns a DataFrame indexed by ``by`` with columns
    ``rmse, corr, bias, n``.
    """
    merged = _align(pred, obs, value)
    by = list(by)
    grouper = by[0] if len(by) == 1 else by
    rows = []
    for key, group in merged.groupby(grouper):
        err = group["pred"].to_numpy() - group["obs"].to_numpy()
        corr = (float(np.corrcoef(group["pred"], group["obs"])[0, 1])
                if len(group) > 1 else float("nan"))
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(rmse=float(np.sqrt(np.mean(err ** 2))),
                   corr=corr, bias=float(np.mean(err)), n=int(len(group)))
        rows.append(row)
    return pd.DataFrame(rows).set_index(by)


def rolling_skill(pred: pd.DataFrame, obs: pd.DataFrame, window: str = "24h",
                  value: str = "water_level") -> pd.DataFrame:
    """Rolling-window RMSE per station over time.

    Returns a long DataFrame ``[station_id, time, rolling_rmse]``.
    """
    merged = _align(pred, obs, value).sort_values("time")
    out = []
    for station, group in merged.groupby("station_id"):
        series = group.set_index("time")
        sq_err = (series["pred"] - series["obs"]) ** 2
        rolling_rmse = np.sqrt(sq_err.rolling(window).mean())
        out.append(pd.DataFrame({
            "station_id": station,
            "time": series.index,
            "rolling_rmse": rolling_rmse.to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def detect_degradation(skill: pd.DataFrame, rmse_threshold: float,
                       metric: str = "rmse") -> pd.DataFrame:
    """Flag rows whose ``metric`` exceeds ``rmse_threshold`` (adds a ``degraded`` column)."""
    flagged = skill.copy()
    flagged["degraded"] = flagged[metric] > rmse_threshold
    return flagged
