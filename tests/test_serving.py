"""Tests for the FastAPI serving app (via Starlette's TestClient)."""

import random

from fastapi.testclient import TestClient

from stofs_surrogate.serving.app import app

client = TestClient(app)


def _payload(num_nodes=30, num_edges=90, num_steps=4):
    rng = random.Random(0)
    return {
        "state": [[rng.gauss(0, 1)] for _ in range(num_nodes)],
        "node_features": [[rng.gauss(0, 1) for _ in range(3)] for _ in range(num_nodes)],
        "edge_index": [[rng.randrange(num_nodes) for _ in range(num_edges)] for _ in range(2)],
        "edge_attr": [[rng.gauss(0, 1) for _ in range(3)] for _ in range(num_edges)],
        "num_steps": num_steps,
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_class"] == "STOFSSurrogateGNN"
    assert "git_sha" in body and "num_params" in body
    assert body["source"] == "synthetic-demo"


def test_predict_returns_forecast_shape():
    response = client.post("/predict", json=_payload(num_nodes=30, num_steps=4))
    assert response.status_code == 200
    body = response.json()
    assert body["shape"] == [5, 30, 1]  # [num_steps + 1, num_nodes, state_dim]
    assert body["num_steps"] == 4
    assert len(body["predictions"]) == 5


def test_predict_batch():
    payloads = [_payload(num_nodes=20, num_steps=2), _payload(num_nodes=25, num_steps=3)]
    response = client.post("/predict/batch", json=payloads)
    assert response.status_code == 200
    bodies = response.json()
    assert [b["shape"] for b in bodies] == [[3, 20, 1], [4, 25, 1]]


def test_predict_bad_dims_returns_422():
    # state_dim 2 does not match the demo model (state_dim 1) -> handled as 422.
    bad = {
        "state": [[0.0, 0.0], [0.1, 0.1]],
        "node_features": [[0, 0, 0], [0, 0, 0]],
        "edge_index": [[0, 1], [1, 0]],
        "edge_attr": [[0, 0, 0], [0, 0, 0]],
        "num_steps": 2,
    }
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
