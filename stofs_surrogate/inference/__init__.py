"""Inference and ensemble forecasting utilities.

- ``Predictor``: load a checkpoint and run autoregressive rollouts.
- ``EnsemblePredictor``: perturbed-forcing ensemble with mean/std/percentile statistics.

The production 25k/ensemble scripts in ``scripts/`` (e.g. rollout_25k_model.py,
ensemble_v2.py) embed their own physics-informed model (``PhysicsInformedCWLModel``)
inline; migrating them to these classes requires promoting that model into the package.
"""

from stofs_surrogate.inference.ensemble import EnsemblePredictor
from stofs_surrogate.inference.predictor import Predictor

__all__ = ["Predictor", "EnsemblePredictor"]
