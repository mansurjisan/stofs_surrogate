"""MLflow Model Registry + lineage stamping.

Registers a trained model with full lineage -- git commit SHA, the resolved config (as
params + a JSON artifact), key validation metrics, and a training-data manifest hash -- so
a reviewer can open a registered version and see exactly what produced it.

The MLflow Model Registry requires a database-backed tracking store (e.g.
``sqlite:///mlflow.db``); the file store (``./mlruns``) does not support the registry.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from stofs_surrogate.training.tracking import _flatten_params, get_git_sha

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "stofs-gnn-midatlantic"


def data_manifest_hash(paths: Optional[List[str]] = None) -> str:
    """Fingerprint the training-data files (path + size) into a short hash.

    TODO(user): pass the real preprocessed training files (e.g. data/processed_25k/*.npz)
    for a meaningful manifest. With no paths this returns ``"unknown"``.
    """
    if not paths:
        return "unknown"
    digest = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        digest.update(p.encode())
        try:
            digest.update(str(os.stat(p).st_size).encode())
        except OSError:
            pass
    return digest.hexdigest()[:16]


class ModelRegistry:
    """Thin wrapper over the MLflow Model Registry that stamps lineage on each version."""

    def __init__(self, tracking_uri: str, experiment: str = "stofs-registry"):
        import mlflow  # lazy; the registry needs a DB-backed tracking_uri
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def register(self, model, name: str = DEFAULT_MODEL_NAME,
                 config: Optional[Dict] = None, metrics: Optional[Dict] = None,
                 data_paths: Optional[List[str]] = None, alias: Optional[str] = None):
        """Log ``model`` with lineage and register a new version. Returns the ModelVersion.

        ``alias`` (e.g. ``"staging"`` / ``"production"``) is set on the new version using
        MLflow's alias API, which supersedes the deprecated model-stage transitions.
        """
        mlflow = self._mlflow
        with mlflow.start_run():
            mlflow.set_tags({
                "git_sha": get_git_sha(),
                "data_manifest": data_manifest_hash(data_paths),
            })
            if config:
                mlflow.log_params({k: str(v) for k, v in _flatten_params(config).items()})
                with tempfile.TemporaryDirectory() as tmp:
                    cfg = Path(tmp) / "config.json"
                    cfg.write_text(json.dumps(config, indent=2, default=str))
                    mlflow.log_artifact(str(cfg))
            if metrics:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            # serialization_format="pickle": MLflow >= 3 defaults to the "pt2"
            # traced-graph format, which needs an input_example -- impractical for a
            # dynamic-graph GNN. Pickle just needs the model class importable.
            try:
                mlflow.pytorch.log_model(model, name="model", registered_model_name=name,
                                         serialization_format="pickle")
            except TypeError:
                # Older MLflow used `artifact_path` instead of `name`.
                mlflow.pytorch.log_model(model, artifact_path="model",
                                         registered_model_name=name,
                                         serialization_format="pickle")

        versions = self._client.search_model_versions(f"name='{name}'")
        version = max(versions, key=lambda v: int(v.version))
        if alias:
            self._client.set_registered_model_alias(name, alias, version.version)
        return version

    def get_lineage_by_alias(self, name: str, alias: str) -> Dict:
        """Lineage for the version currently holding ``alias`` (e.g. ``"production"``)."""
        model_version = self._client.get_model_version_by_alias(name, alias)
        return self.get_lineage(name, model_version.version)

    def get_lineage(self, name: str, version: str) -> Dict:
        """Return the lineage (tags, params, metrics) recorded for a model version."""
        model_version = self._client.get_model_version(name, version)
        run = self._client.get_run(model_version.run_id)
        return {
            "version": model_version.version,
            "run_id": model_version.run_id,
            "tags": run.data.tags,
            "params": run.data.params,
            "metrics": run.data.metrics,
        }
