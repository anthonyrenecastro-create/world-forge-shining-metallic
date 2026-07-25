"""Curiosity and Novelty Detection mechanisms.

Curiosity increases attention on nodes whose current state deviates most
from stored memory traces (high surprise / novelty).  This activates
more compute allocation toward unexplored regions.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import (
    Contribution,
    CognitionDelta,
    ResourceDelta,
    UncertaintyDelta,
)
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class CuriosityMechanism(BaseMechanism):
    """Allocates attention and compute toward high-novelty nodes.

    Parameters
    ----------
    curiosity_strength:
        Scale factor for attention re-weighting.
    resource_boost:
        Extra compute fraction allocated to novel nodes.
    """

    def __init__(
        self,
        curiosity_strength: float = 0.2,
        resource_boost: float = 0.1,
    ) -> None:
        super().__init__(
            mechanism_id="curiosity",
            version="0.1.0",
            description="Attention and resource allocation toward novel regions.",
        )
        self.curiosity_strength = curiosity_strength
        self.resource_boost = resource_boost

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        field_vals = snapshot.field_state.values      # (n, f)
        memory = snapshot.memory
        attention = snapshot.cognition.attention      # (n,)

        # Novelty: distance of each node's current state from the
        # most-similar memory trace (if any salient traces exist)
        if memory.n_traces > 0 and float(memory.salience.max()) > 1e-9:
            # Project mean field of each node onto trace space
            # Approximate: compare global field mean to each trace
            node_norms = np.linalg.norm(field_vals, axis=1) + 1e-9  # (n,)
            trace_norms = np.linalg.norm(memory.traces, axis=1) + 1e-9  # (t,)
            # Similarity: (n, t)
            sim = (field_vals @ memory.traces.T) / (
                node_norms[:, None] * trace_norms[None, :]
            )
            best_sim = sim.max(axis=1)  # (n,) most similar trace per node
            novelty = 1.0 - np.clip(best_sim, 0.0, 1.0)  # (n,)
        else:
            # No memory: uniform novelty
            novelty = np.ones(snapshot.n_nodes)

        # Attention shift toward high-novelty nodes
        delta_attention = self.curiosity_strength * (novelty - attention)

        # Resource boost for novel nodes
        novel_mask = novelty > 0.6
        delta_compute = np.where(
            novel_mask,
            self.resource_boost * novelty,
            -0.01 * snapshot.resources.compute,
        )
        delta_memory = np.zeros(snapshot.n_nodes)

        # Reduce epistemic uncertainty for well-remembered nodes
        delta_epistemic = -0.01 * (1.0 - novelty)
        delta_aleatoric = np.zeros(snapshot.n_nodes)

        activation = float(novelty.mean())

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.65,
            activation=min(activation, 1.0),
            estimated_cost=self.resource_boost * float(novel_mask.sum()),
            rationale=(
                f"Curiosity; mean_novelty={activation:.4g}; "
                f"novel_nodes={int(novel_mask.sum())}"
            ),
            cognition_delta=CognitionDelta(
                delta_attention=delta_attention,
                delta_predictions=np.zeros_like(snapshot.cognition.predictions),
                weight=1.0,
            ),
            resource_delta=ResourceDelta(
                delta_compute=delta_compute,
                delta_memory=delta_memory,
                weight=1.0,
            ),
            uncertainty_delta=UncertaintyDelta(
                delta_epistemic=delta_epistemic,
                delta_aleatoric=delta_aleatoric,
                weight=0.5,
            ),
            diagnostics={
                "mean_novelty": activation,
                "n_novel_nodes": int(novel_mask.sum()),
                "curiosity_strength": self.curiosity_strength,
            },
        )
