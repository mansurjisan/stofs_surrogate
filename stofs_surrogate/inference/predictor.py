"""Inference / rollout for the STOFS surrogate.

A thin wrapper that loads a trained checkpoint and runs autoregressive rollouts with the
package model (``STOFSSurrogateGNN``). The production 25k/ensemble scripts in ``scripts/``
embed their own physics-informed model (``PhysicsInformedCWLModel``) inline and are not yet
migrated to this class; see ``scripts/predict.py`` for the package-native pattern.
"""

import logging
from typing import Optional

import torch

from stofs_surrogate.models.gnn import create_model

logger = logging.getLogger(__name__)


def _extract_state_dict(checkpoint):
    """Return the model state_dict from a checkpoint that may be raw or wrapped.

    Handles the package trainer format (``{"model_state_dict": ...}``), the inline-script
    format (same key), and a bare state_dict.
    """
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


class Predictor:
    """Load a trained surrogate and run autoregressive forecasts on CPU or GPU."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu",
                 eta_scale: Optional[float] = None):
        self.model = model.to(device).eval()
        self.device = device
        self.eta_scale = eta_scale  # if set, predictions are denormalized by this factor

    @classmethod
    def from_checkpoint(cls, checkpoint_path, model: Optional[torch.nn.Module] = None,
                        model_type: str = "stofs_gnn", model_kwargs: Optional[dict] = None,
                        device: str = "cpu",
                        eta_scale: Optional[float] = None) -> "Predictor":
        """Build a Predictor from a ``.pt`` checkpoint.

        If ``model`` is given it is used as-is; otherwise a package model is created via
        ``create_model(model_type, **model_kwargs)`` and the checkpoint's weights loaded.
        """
        # weights_only=False: checkpoints may carry a config dict alongside the weights.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if model is None:
            model = create_model(model_type, **(model_kwargs or {}))
        model.load_state_dict(_extract_state_dict(checkpoint))
        return cls(model, device=device, eta_scale=eta_scale)

    def _to(self, tensor):
        return None if tensor is None else tensor.to(self.device)

    @torch.no_grad()
    def rollout(self, initial_state, node_features, edge_index, edge_attr, num_steps,
                forcing_sequence=None):
        """Autoregressive N-step rollout. Returns ``[num_steps + 1, num_nodes, state_dim]``."""
        self.model.eval()
        preds = self.model.rollout(
            initial_state=self._to(initial_state),
            node_features=self._to(node_features),
            edge_index=self._to(edge_index),
            edge_attr=self._to(edge_attr),
            num_steps=num_steps,
            forcing_sequence=self._to(forcing_sequence),
        )
        if self.eta_scale is not None:
            preds = preds * self.eta_scale
        return preds

    @torch.no_grad()
    def predict_step(self, state, node_features, edge_index, edge_attr, forcing=None):
        """Single-step prediction. Returns ``[num_nodes, state_dim]``."""
        self.model.eval()
        return self.model(
            self._to(state), self._to(node_features), self._to(edge_index),
            self._to(edge_attr), forcing=self._to(forcing),
        )
