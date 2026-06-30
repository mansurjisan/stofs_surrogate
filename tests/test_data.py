"""Tests for data loading and preprocessing."""

import numpy as np


def test_mesh_connectivity():
    """Test mesh edge computation."""
    # Simple triangular mesh
    triangles = np.array([[0, 1, 2], [1, 2, 3]])

    # Extract edges from triangles
    edges = set()
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i+1) % 3]]))
            edges.add(edge)

    assert len(edges) == 5  # 4 nodes, 2 triangles share 1 edge


def test_normalization():
    """Test data normalization."""
    ETA_SCALE = 2.0
    WIND_SCALE = 15.0

    # Test water level normalization
    cwl = np.array([-1.5, 0.0, 1.5, 3.0])
    cwl_norm = cwl / ETA_SCALE
    assert np.allclose(cwl_norm, [-0.75, 0.0, 0.75, 1.5])

    # Test wind normalization
    wind = np.array([0, 10, -10, 20])
    wind_norm = wind / WIND_SCALE
    assert np.allclose(wind_norm, [0, 0.667, -0.667, 1.333], atol=0.01)


def test_feature_computation():
    """Test static feature computation."""
    n_nodes = 10

    # Coordinates
    x = np.linspace(-76, -73, n_nodes)
    y = np.linspace(38, 41, n_nodes)
    depth = np.random.uniform(1, 50, n_nodes)
    cwl = np.random.uniform(-0.5, 0.5, n_nodes)

    # Normalize coordinates to [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    # Normalize depth (log scale)
    depth_norm = np.log1p(depth) / np.log1p(depth.max())

    # Water level
    water_level = depth + cwl
    wl_norm = (water_level - water_level.mean()) / (water_level.std() + 1e-8)

    # Stack features
    static_features = np.stack([x_norm, y_norm, depth_norm, wl_norm], axis=1)

    assert static_features.shape == (n_nodes, 4)
    assert np.all(x_norm >= 0) and np.all(x_norm <= 1)
    assert np.all(y_norm >= 0) and np.all(y_norm <= 1)


def test_edge_features():
    """Test edge feature computation."""
    # Two nodes
    node1 = np.array([0.0, 0.0])
    node2 = np.array([1.0, 1.0])

    # Distance
    dist = np.sqrt(np.sum((node2 - node1)**2))
    assert np.isclose(dist, np.sqrt(2))

    # Direction (unit vector)
    direction = (node2 - node1) / dist
    assert np.allclose(direction, [1/np.sqrt(2), 1/np.sqrt(2)])


if __name__ == "__main__":
    test_mesh_connectivity()
    test_normalization()
    test_feature_computation()
    test_edge_features()
    print("All data tests passed!")
