"""FastAPI inference service for the STOFS surrogate.

Endpoints
---------
- ``GET  /health``        liveness probe
- ``GET  /metadata``      served-model identity + lineage (git SHA, params, registry skill)
- ``POST /predict``       single forecast cycle (graph state + optional forcing -> rollout)
- ``POST /predict/batch`` multiple cycles in one request

The served model is configured via environment variables (see :func:`_load_predictor`);
with none set it falls back to a small synthetic demo model, so the service runs and is
testable without a trained checkpoint or a registry. Inputs are JSON (no form handling).
"""

import json
import os
from functools import lru_cache
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stofs_surrogate.inference.predictor import Predictor
from stofs_surrogate.models.gnn import create_model
from stofs_surrogate.training.tracking import get_git_sha

app = FastAPI(title="STOFS-GNN Surrogate", version="0.1.0")

# Small CPU model used when no checkpoint is configured (keeps the service self-contained).
DEMO_MODEL_KWARGS = dict(state_dim=1, node_feature_dim=3, edge_feature_dim=3,
                         forcing_dim=0, hidden_dim=32, num_layers=2)


@lru_cache(maxsize=1)
def _load_predictor() -> Predictor:
    """Load the served model once.

    Environment configuration:
      ``STOFS_MODEL_CHECKPOINT``  path to a ``.pt`` checkpoint
      ``STOFS_MODEL_TYPE``        package model type (default ``stofs_gnn``)
      ``STOFS_MODEL_KWARGS``      JSON model kwargs matching the checkpoint
      ``STOFS_DEVICE``            ``cpu`` (default) or ``cuda``
    With no checkpoint set, a small synthetic demo model is used.
    """
    device = os.environ.get("STOFS_DEVICE", "cpu")
    checkpoint = os.environ.get("STOFS_MODEL_CHECKPOINT")
    if checkpoint:
        kwargs = json.loads(os.environ.get("STOFS_MODEL_KWARGS", "{}"))
        return Predictor.from_checkpoint(
            checkpoint, model_type=os.environ.get("STOFS_MODEL_TYPE", "stofs_gnn"),
            model_kwargs=kwargs, device=device,
        )
    return Predictor(create_model("stofs_gnn", **DEMO_MODEL_KWARGS), device=device)


def get_predictor() -> Predictor:
    return _load_predictor()


def _registry_skill() -> Optional[dict]:
    """Lead-time skill metrics from the model registry, when a model is configured.

    Set ``STOFS_REGISTRY_URI`` (a DB-backed MLflow URI), ``STOFS_REGISTRY_MODEL`` (name)
    and optionally ``STOFS_REGISTRY_ALIAS`` (default ``production``). Returns ``None`` for
    the synthetic demo model.
    """
    uri = os.environ.get("STOFS_REGISTRY_URI")
    name = os.environ.get("STOFS_REGISTRY_MODEL")
    alias = os.environ.get("STOFS_REGISTRY_ALIAS", "production")
    if not (uri and name):
        return None
    try:
        from stofs_surrogate.training.registry import ModelRegistry
        return ModelRegistry(uri).get_lineage_by_alias(name, alias)["metrics"]
    except Exception:
        return None


class PredictRequest(BaseModel):
    state: List[List[float]] = Field(..., description="[num_nodes, state_dim]")
    node_features: List[List[float]] = Field(..., description="[num_nodes, node_feature_dim]")
    edge_index: List[List[int]] = Field(..., description="[2, num_edges]")
    edge_attr: List[List[float]] = Field(..., description="[num_edges, edge_feature_dim]")
    num_steps: int = Field(12, ge=1, le=240)
    forcing_sequence: Optional[List[List[List[float]]]] = Field(
        None, description="[num_steps, num_nodes, forcing_dim]")


class PredictResponse(BaseModel):
    predictions: List = Field(..., description="[num_steps + 1, num_nodes, state_dim]")
    shape: List[int]
    num_steps: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    predictor = get_predictor()
    model = predictor.model
    return {
        "model_class": type(model).__name__,
        "num_params": sum(p.numel() for p in model.parameters()),
        "git_sha": get_git_sha(),
        "device": predictor.device,
        "source": os.environ.get("STOFS_MODEL_CHECKPOINT", "synthetic-demo"),
        # Reported lead-time skill comes from the registry when configured; the demo model
        # has none (so this is null rather than a fabricated number).
        "lead_time_skill": _registry_skill(),
    }


def _run_one(req: PredictRequest, predictor: Predictor) -> PredictResponse:
    try:
        forcing = (torch.tensor(req.forcing_sequence, dtype=torch.float32)
                   if req.forcing_sequence is not None else None)
        preds = predictor.rollout(
            torch.tensor(req.state, dtype=torch.float32),
            torch.tensor(req.node_features, dtype=torch.float32),
            torch.tensor(req.edge_index, dtype=torch.long),
            torch.tensor(req.edge_attr, dtype=torch.float32),
            num_steps=req.num_steps,
            forcing_sequence=forcing,
        )
    except Exception as exc:  # malformed shapes / wrong dims -> 422
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc
    return PredictResponse(predictions=preds.cpu().tolist(),
                           shape=list(preds.shape), num_steps=req.num_steps)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return _run_one(req, get_predictor())


@app.post("/predict/batch", response_model=List[PredictResponse])
def predict_batch(reqs: List[PredictRequest]):
    predictor = get_predictor()
    return [_run_one(r, predictor) for r in reqs]
