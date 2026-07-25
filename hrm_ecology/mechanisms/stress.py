"""Stress regulation mechanism.

Increases system stress when novelty or prediction error is high;
decreases stress when the field is stable.  High stress suppresses
risky structural changes (enforced by the safety constitution).
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, StressDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class StressRegulationMechanism(BaseMechanism):
    """Regulate global and regional stress based on field volatility.

    Parameters
    ----------
    rise_rate:
        How quickly stress rises when novelty is detected.
    fall_rate:
        How quickly stress falls when the field is stable.
    stability_window:
        Number of recent steps considered for stability (informational).
    """

    def __init__(
        self,
        rise_rate: float = 0.2,
        fall_rate: float = 0.05,
    ) -> None:
        super().__init__(
            mechanism_id="stress_regulation",
            version="0.1.0",
            description="Regulates global/regional stress from field volatility.",
        )
        self.rise_rate = rise_rate
        self.fall_rate = fall_rate

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        field_vals = snapshot.field_state.values       # (n, f)
        predictions = snapshot.cognition.predictions   # (n, f)
        stress = snapshot.stress

        # Prediction error per node
        error_per_node = np.abs(field_vals - predictions).mean(axis=1)  # (n,)

        # Normalise to [0, 1]
        max_err = float(error_per_node.max()) or 1.0
        novelty_per_node = error_per_node / max_err

        # Regional stress delta
        delta_regions = np.where(
            novelty_per_node > 0.5,
            self.rise_rate * (novelty_per_node - stress.regions),
            -self.fall_rate * stress.regions,
        )

        global_novelty = float(novelty_per_node.mean())
        if global_novelty > 0.5:
            delta_level = self.rise_rate * (global_novelty - stress.level)
        else:
            delta_level = -self.fall_rate * stress.level

        activation = global_novelty

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.75,
            activation=activation,
            estimated_cost=0.1,
            rationale=(
                f"Stress regulation; novelty={global_novelty:.4g}; "
                f"current_stress={stress.level:.4g}"
            ),
            stress_delta=StressDelta(
                delta_level=delta_level,
                delta_regions=delta_regions,
                weight=1.0,
            ),
            diagnostics={
                "global_novelty": global_novelty,
                "current_stress": stress.level,
                "rise_rate": self.rise_rate,
                "fall_rate": self.fall_rate,
            },
        )
