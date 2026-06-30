"""Perturbed-forcing ensemble forecasting."""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from stofs_surrogate.inference.predictor import Predictor

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Run an ensemble by perturbing the meteorological forcing sequence.

    Mirrors ``configs/ensemble.yaml`` (Gaussian perturbations of the forcing channels) and
    aggregates the member forecasts into mean / std / percentiles.
    """

    def __init__(self, predictor: Predictor, n_members: int = 20, seed: int = 0):
        self.predictor = predictor
        self.n_members = n_members
        self.seed = seed

    def run(self, initial_state, node_features, edge_index, edge_attr, num_steps,
            forcing_sequence, channel_scales: Optional[Dict[int, float]] = None,
            percentiles=(5, 25, 50, 75, 95)) -> Dict[str, object]:
        """Generate ``n_members`` forecasts with perturbed forcing and aggregate them.

        ``channel_scales`` maps a forcing-channel index to the Gaussian std used to perturb
        it (e.g. ``{0: 2.0, 1: 2.0, 2: 200.0}`` for u10/v10/pressure); if ``None`` every
        channel is perturbed with std 1.0. Returns a dict with ``members``
        ``[M, T+1, N, S]``, ``mean``/``std`` ``[T+1, N, S]``, and ``percentiles`` ``{p: arr}``.
        """
        if forcing_sequence is None:
            raise ValueError("EnsemblePredictor requires a forcing_sequence to perturb.")

        generator = torch.Generator().manual_seed(self.seed)
        n_channels = forcing_sequence.shape[-1]
        if channel_scales is None:
            channel_scales = {c: 1.0 for c in range(n_channels)}

        members = []
        for _ in range(self.n_members):
            perturbed = forcing_sequence.clone()
            for channel, scale in channel_scales.items():
                noise = torch.randn(perturbed[..., channel].shape, generator=generator)
                perturbed[..., channel] = perturbed[..., channel] + noise * scale
            pred = self.predictor.rollout(
                initial_state, node_features, edge_index, edge_attr,
                num_steps=num_steps, forcing_sequence=perturbed,
            )
            members.append(pred.cpu().numpy())

        stack = np.stack(members, axis=0)  # [M, T+1, N, S]
        return {
            "members": stack,
            "mean": stack.mean(axis=0),
            "std": stack.std(axis=0),
            "percentiles": {p: np.percentile(stack, p, axis=0) for p in percentiles},
        }
