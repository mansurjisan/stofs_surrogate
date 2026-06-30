"""Experiment-tracking abstraction.

A thin, backend-agnostic interface so the training code does not depend on a specific
tracking tool. Backends:

  - ``MLflowTracker``  (default)
  - ``WandbTracker``   (optional)
  - ``NoOpTracker``    (tracking disabled)

Construct one with :func:`make_tracker`. If the requested backend's package is not
installed, ``make_tracker`` logs a warning and returns a :class:`NoOpTracker`, so a
training run is never blocked merely because a tracking package is missing.
"""

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_git_sha(short: bool = False) -> str:
    """Return the current git commit SHA, or ``"unknown"`` if this is not a git repo."""
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _flatten_params(params: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts into dotted keys; join lists/tuples into a string."""
    flat: Dict[str, Any] = {}
    for k, v in params.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key, sep))
        elif isinstance(v, (list, tuple)):
            flat[key] = ",".join(str(x) for x in v)
        else:
            flat[key] = v
    return flat


class Tracker(ABC):
    """Minimal experiment-tracking interface used by the trainer."""

    @abstractmethod
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, Any]] = None) -> None:
        ...

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def log_artifact(self, path: str) -> None:
        ...

    @abstractmethod
    def end_run(self) -> None:
        ...

    # Context-manager sugar: `with make_tracker(...) as t: ...`
    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.end_run()


class NoOpTracker(Tracker):
    """Tracker that records nothing (used when tracking is disabled)."""

    def start_run(self, run_name=None, tags=None):
        logger.info("Experiment tracking disabled (NoOpTracker).")

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_artifact(self, path):
        pass

    def end_run(self):
        pass


class MLflowTracker(Tracker):
    """MLflow-backed tracker (the default backend)."""

    def __init__(self, experiment: str = "stofs-surrogate",
                 tracking_uri: Optional[str] = None):
        import mlflow  # lazy import; ImportError is handled by make_tracker
        # MLflow >= 3 puts the filesystem store in "maintenance mode" and raises unless
        # opted out. The file store is fine for this project's local/demo use; honor a
        # user-set value if one is already present.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._active = False

    def start_run(self, run_name=None, tags=None):
        self._mlflow.start_run(run_name=run_name)
        if tags:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
        self._active = True

    def log_params(self, params):
        # MLflow params are immutable strings; flatten + stringify everything.
        self._mlflow.log_params({k: str(v) for k, v in _flatten_params(params).items()})

    def log_metrics(self, metrics, step=None):
        numeric = {k: float(v) for k, v in metrics.items() if v is not None}
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path):
        self._mlflow.log_artifact(str(path))

    def end_run(self):
        if self._active:
            self._mlflow.end_run()
            self._active = False


class WandbTracker(Tracker):
    """Weights & Biases-backed tracker (optional)."""

    def __init__(self, project: str = "stofs-surrogate", entity: Optional[str] = None):
        import wandb  # lazy import; ImportError is handled by make_tracker
        self._wandb = wandb
        self._project = project
        self._entity = entity
        self._run = None

    def start_run(self, run_name=None, tags=None):
        self._run = self._wandb.init(
            project=self._project,
            entity=self._entity,
            name=run_name,
            tags=[f"{k}={v}" for k, v in (tags or {}).items()] or None,
            reinit=True,
        )

    def log_params(self, params):
        self._wandb.config.update(_flatten_params(params), allow_val_change=True)

    def log_metrics(self, metrics, step=None):
        numeric = {k: v for k, v in metrics.items() if v is not None}
        if numeric:
            self._wandb.log(numeric, step=step)

    def log_artifact(self, path):
        self._wandb.save(str(path))

    def end_run(self):
        if self._run is not None:
            self._wandb.finish()
            self._run = None


def make_tracker(
    name: Optional[str] = "mlflow",
    *,
    experiment: str = "stofs-surrogate",
    tracking_uri: Optional[str] = None,
    project: str = "stofs-surrogate",
    entity: Optional[str] = None,
) -> Tracker:
    """Construct a tracker by name (``"mlflow"`` | ``"wandb"`` | ``"none"``).

    Falls back to :class:`NoOpTracker` (with a warning) if the chosen backend's package
    is not installed, so training never fails just because a tracker is unavailable.
    """
    key = (name or "none").lower()
    if key in ("none", "noop", "off", "false"):
        return NoOpTracker()
    if key == "mlflow":
        try:
            return MLflowTracker(experiment=experiment, tracking_uri=tracking_uri)
        except ImportError:
            logger.warning("mlflow not installed; using NoOpTracker. `pip install mlflow` to enable.")
            return NoOpTracker()
    if key == "wandb":
        try:
            return WandbTracker(project=project, entity=entity)
        except ImportError:
            logger.warning("wandb not installed; using NoOpTracker. `pip install wandb` to enable.")
            return NoOpTracker()
    raise ValueError(f"Unknown tracker: {name!r} (expected 'mlflow', 'wandb', or 'none')")
