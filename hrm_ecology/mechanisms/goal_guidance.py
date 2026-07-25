"""Goal guidance mechanism.

Biases field activations toward the target patterns of active goals,
weighted by goal priority and current attention.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, FieldDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class GoalGuidanceMechanism(BaseMechanism):
    """Pull field activations toward active goal target patterns.

    Parameters
    ----------
    strength:
        Goal-guidance strength in [0, 1].
    """

    def __init__(self, strength: float = 0.1) -> None:
        super().__init__(
            mechanism_id="goal_guidance",
            version="0.1.0",
            description="Biases field toward active goal target patterns.",
        )
        self.strength = strength

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        active_goals = [g for g in snapshot.goals.goals if g.active]
        if not active_goals:
            return self._null_contribution("no active goals")

        field_vals = snapshot.field_state.values     # (n, f)
        attention = snapshot.cognition.attention     # (n,)
        priorities = snapshot.goals.priorities       # (n,)

        # Aggregate goal target as a weighted sum of active goal patterns
        total_priority = sum(g.priority for g in active_goals) or 1.0
        aggregate_target = sum(
            g.priority * g.target_pattern for g in active_goals
        ) / total_priority  # (n_features,)

        # Delta: pull field toward aggregate target, weighted by attention & priority
        guidance_weight = (attention * priorities)[:, None]   # (n, 1)
        field_delta = self.strength * guidance_weight * (
            aggregate_target[None, :] - field_vals
        )

        activation = float(np.abs(field_delta).mean())

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.8,
            activation=min(activation, 1.0),
            estimated_cost=float(np.abs(field_delta).sum()),
            rationale=(
                f"Goal guidance toward {len(active_goals)} active goal(s); "
                f"strength={self.strength}"
            ),
            field_delta=FieldDelta(delta=field_delta, weight=1.0),
            diagnostics={
                "n_active_goals": len(active_goals),
                "aggregate_target_norm": float(np.linalg.norm(aggregate_target)),
                "strength": self.strength,
            },
        )
