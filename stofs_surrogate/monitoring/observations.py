"""CO-OPS tide-gauge observations for validation/monitoring.

The live path fetches observed water level from NOAA CO-OPS via ``searvey`` (optional dep)
and caches to disk. For CI and offline development, :func:`synthetic_observations` generates
a plausible tide + surge signal so the monitoring pipeline runs end-to-end without network.

TODO(user): wire :func:`fetch_observed_water_level` to the real CO-OPS pull (searvey or the
ocean-mcp CO-OPS server) for the validation window, and reconcile the station set with the
stations shown in the README figures (Baltimore, Philadelphia, Annapolis).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Validation stations (CO-OPS station IDs -> names). Mirrors the README station validation.
STATIONS = {
    "8574680": "Baltimore",
    "8545240": "Philadelphia",
    "8575512": "Annapolis",
    "8534720": "Atlantic City",
    "8518750": "The Battery",
}


def synthetic_observations(station_id: str, periods: int = 96, freq: str = "1h",
                           seed: int = 0, start: str = "2025-01-20") -> pd.DataFrame:
    """Synthetic observed water level: M2 + S2 tides + a surge bump + noise.

    Returns a DataFrame with columns ``[time, station_id, water_level]``.
    """
    # Deterministic per-station stream so stations differ but runs are reproducible.
    offset = int(str(station_id)[-4:]) if str(station_id)[-4:].isdigit() else 0
    rng = np.random.default_rng(seed + offset)
    t = np.arange(periods)
    tide = 0.6 * np.sin(2 * np.pi * t / 12.42) + 0.2 * np.sin(2 * np.pi * t / 12.0)
    surge = 0.4 * np.exp(-((t - periods / 2) ** 2) / (2 * (periods / 8) ** 2))
    water_level = tide + surge + rng.normal(0, 0.03, periods)
    return pd.DataFrame({
        "time": pd.date_range(start=start, periods=periods, freq=freq),
        "station_id": str(station_id),
        "water_level": water_level.astype(float),
    })


def fetch_observed_water_level(station_id: str, start: str, end: str,
                               cache_dir: str = "data/obs_cache",
                               use_synthetic: bool = True) -> pd.DataFrame:
    """Observed water level for a station over ``[start, end]``.

    ``use_synthetic=True`` (default; used in CI/offline) returns a synthetic series. The
    live path (``use_synthetic=False``) fetches from CO-OPS via searvey and caches to
    ``cache_dir``.
    """
    if use_synthetic:
        return synthetic_observations(station_id)

    cache = Path(cache_dir) / f"{station_id}_{start}_{end}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # TODO(user): real CO-OPS fetch, e.g. with searvey:
    #   from searvey import coops
    #   df = coops.coops_stations_by_ids([station_id]) ...  # then the 'water_level' product
    # normalize to columns [time, station_id, water_level], write cache, and return.
    raise NotImplementedError(
        "Live CO-OPS fetch is not wired yet; pass use_synthetic=True or implement the "
        "searvey pull (see TODO)."
    )
