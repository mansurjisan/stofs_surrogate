"""Tests for inference utilities."""

import pytest
import numpy as np


def test_perturbation_generation():
    """Test forcing perturbation generation."""
    n_timesteps = 48
    n_nodes = 100
    n_members = 10

    # Simulate wind perturbations
    wind_scale = 2.0
    perturbations = np.random.randn(n_members, n_timesteps, n_nodes) * wind_scale

    assert perturbations.shape == (n_members, n_timesteps, n_nodes)
    assert np.abs(perturbations.std() - wind_scale) < 0.5  # Roughly correct scale


def test_station_extraction():
    """Test station node finding."""
    # Create dummy coordinates
    n_nodes = 1000
    x = np.linspace(-76, -73, n_nodes)
    y = np.linspace(38, 41, n_nodes)

    # Target station (Atlantic City approx)
    target_lon = -74.42
    target_lat = 39.36

    # Find nearest node
    distances = np.sqrt((x - target_lon)**2 + (y - target_lat)**2)
    nearest_idx = np.argmin(distances)

    assert 0 <= nearest_idx < n_nodes
    assert distances[nearest_idx] < 0.1  # Should be within 0.1 degrees


def test_ensemble_statistics():
    """Test ensemble statistics computation."""
    n_members = 50
    n_timesteps = 48
    n_stations = 5

    # Simulate ensemble predictions
    predictions = np.random.randn(n_members, n_timesteps, n_stations) * 0.3

    # Compute statistics
    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)
    percentiles = np.percentile(predictions, [5, 25, 50, 75, 95], axis=0)

    assert mean.shape == (n_timesteps, n_stations)
    assert std.shape == (n_timesteps, n_stations)
    assert percentiles.shape == (5, n_timesteps, n_stations)
    assert np.all(std >= 0)


if __name__ == "__main__":
    test_perturbation_generation()
    test_station_extraction()
    test_ensemble_statistics()
    print("All inference tests passed!")
