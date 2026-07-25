"""Resource allocation mechanism.

Proposes compute and memory reallocation based on node activity,
goal priority, and attention.  Adaptive allocation reduces actual
computation: nodes with low activity receive less budget.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, ResourceDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class ResourceAllocationMechanism(BaseMechanism):
    """Redistribute compute and memory toward active, high-priority nodes.

    Parameters
    ----------
    realloc_rate:
        How aggressively to shift resources each step.
    floor:
        Minimum resource fraction per node (prevents starvation).
    """

    def __init__(self, realloc_rate: float = 0.05, floor: float = 0.01) -> None:
        super().__init__(
            mechanism_id="resource_allocation",
            version="0.1.0",
            description="Activity-based compute and memory reallocation.",
        )
        self.realloc_rate = realloc_rate
        self.floor = floor

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        field_vals = snapshot.field_state.values         # (n, f)
        attention = snapshot.cognition.attention         # (n,)
        goal_priority = snapshot.goals.priorities        # (n,)
        current_compute = snapshot.resources.compute     # (n,)
        current_memory = snapshot.resources.memory_alloc # (n,)

        # Activity: combine field magnitude, attention, and goal priority
        activity = np.abs(field_vals).mean(axis=1)       # (n,)
        activity_norm = activity / (activity.sum() + 1e-9)

        # Target allocation = weighted combination
        target = 0.5 * activity_norm + 0.3 * attention + 0.2 * goal_priority
        target_norm = target / (target.sum() + 1e-9)

        # Reallocation delta (move current toward target)
        delta_compute = self.realloc_rate * (target_norm - current_compute)
        delta_memory = self.realloc_rate * (target_norm - current_memory)

        # Enforce floor: nodes at floor get a small positive boost
        at_floor_compute = current_compute < self.floor
        at_floor_memory = current_memory < self.floor
        delta_compute = np.where(at_floor_compute, self.floor - current_compute, delta_compute)
        delta_memory = np.where(at_floor_memory, self.floor - current_memory, delta_memory)

        activation = float(np.abs(delta_compute).mean())

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.9,
            activation=min(activation * 10.0, 1.0),
            estimated_cost=float(np.abs(delta_compute).sum()) * 0.5,
            rationale=(
                f"Resource reallocation toward active/goal-relevant nodes; "
                f"rate={self.realloc_rate}"
            ),
            resource_delta=ResourceDelta(
                delta_compute=delta_compute,
                delta_memory=delta_memory,
                weight=1.0,
            ),
            diagnostics={
                "realloc_rate": self.realloc_rate,
                "mean_activity": float(activity.mean()),
                "target_entropy": float(
                    -np.sum(target_norm * np.log(target_norm + 1e-9))
                ),
            },
        )
