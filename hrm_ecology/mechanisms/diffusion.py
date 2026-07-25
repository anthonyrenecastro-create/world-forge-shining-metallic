"""Diffusion mechanism.

Spreads field activations to neighbouring nodes using graph Laplacian
diffusion.  The proposal is a weighted, bounded additive delta.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, FieldDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class DiffusionMechanism(BaseMechanism):
    """Laplacian diffusion of field values across the topology.

    Parameters
    ----------
    alpha:
        Diffusion rate in (0, 1].
    max_activation:
        Activation threshold (scaled by mean absolute field value).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_activation: float = 1.0,
    ) -> None:
        super().__init__(
            mechanism_id="diffusion",
            version="0.1.0",
            description="Graph Laplacian diffusion of field activations.",
        )
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self.max_activation = max_activation

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        adj = snapshot.topology.adjacency
        n = snapshot.n_nodes
        values = snapshot.field_state.values  # read-only

        # Degree vector
        degree = np.asarray(adj.sum(axis=1)).ravel()

        # Laplacian diffusion: delta_i = alpha * sum_j A_ij (v_j - v_i)
        # = alpha * (A @ v - D * v)
        neighbour_sum = adj @ values  # (n, n_features)
        delta = self.alpha * (neighbour_sum - degree[:, None] * values)

        # Scale by per-node resource allocation
        delta = delta * snapshot.resources.compute[:, None]

        # Suppress diffusion in high-stress regions
        stress_suppression = 1.0 - snapshot.stress.regions[:, None]
        delta = delta * stress_suppression

        # Sanitise NaN/Inf before computing summary statistics
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        mean_abs = float(np.abs(np.nan_to_num(values)).mean()) or 1.0
        activation = float(np.clip(
            np.abs(delta).mean() / (mean_abs * self.max_activation), 0.0, 1.0
        ))

        cost = float(np.abs(delta).sum()) * self.alpha

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.8,
            activation=activation,
            estimated_cost=cost,
            rationale=(
                f"Laplacian diffusion with alpha={self.alpha:.3f}; "
                f"|delta|_mean={float(np.abs(delta).mean()):.4g}"
            ),
            field_delta=FieldDelta(delta=delta, weight=1.0),
            diagnostics={
                "alpha": self.alpha,
                "mean_degree": float(degree.mean()),
                "delta_norm": float(np.linalg.norm(delta)),
            },
        )
