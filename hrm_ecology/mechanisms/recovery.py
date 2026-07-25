"""Damage detection and self-repair mechanisms.

DamageDetectionMechanism identifies corrupted or pathological state
regions (NaN, Inf, field collapse, extreme magnitudes).

RecoveryMechanism proposes restorative field deltas for damaged nodes.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import (
    Contribution,
    FieldDelta,
    UncertaintyDelta,
    StressDelta,
)
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class DamageDetectionMechanism(BaseMechanism):
    """Detect pathological field regions and flag them.

    Parameters
    ----------
    magnitude_threshold:
        Field values exceeding this (L∞) are flagged as damaged.
    collapse_threshold:
        Nodes whose field std across features is below this are flagged
        as collapsed.
    """

    def __init__(
        self,
        magnitude_threshold: float = 10.0,
        collapse_threshold: float = 1e-6,
    ) -> None:
        super().__init__(
            mechanism_id="damage_detection",
            version="0.1.0",
            description="Detects NaN/Inf/extreme field values and field collapse.",
        )
        self.magnitude_threshold = magnitude_threshold
        self.collapse_threshold = collapse_threshold

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        field_vals = snapshot.field_state.values  # (n, f)

        # Detect NaN/Inf
        bad_finite = ~np.isfinite(field_vals).all(axis=1)        # (n,)

        # Detect magnitude overflow
        bad_magnitude = (np.abs(field_vals) > self.magnitude_threshold).any(axis=1)

        # Detect field collapse (all features near zero with no variance)
        feature_std = field_vals.std(axis=1)                      # (n,)
        bad_collapse = feature_std < self.collapse_threshold

        damaged = bad_finite | bad_magnitude | bad_collapse       # (n,)
        n_damaged = int(damaged.sum())

        if n_damaged == 0:
            return self._null_contribution("no damage detected")

        # Increase uncertainty for damaged nodes
        delta_epistemic = np.where(damaged, 0.5, 0.0)
        delta_aleatoric = np.where(damaged, 0.2, 0.0)

        # Raise regional stress at damaged locations
        delta_regions = np.where(damaged, 0.3, 0.0)
        delta_level = 0.1 * n_damaged / max(snapshot.n_nodes, 1)

        activation = min(n_damaged / snapshot.n_nodes, 1.0)

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.95,
            activation=activation,
            estimated_cost=0.1 * n_damaged,
            rationale=(
                f"Detected {n_damaged} damaged node(s); "
                f"nan/inf={int(bad_finite.sum())} "
                f"magnitude={int(bad_magnitude.sum())} "
                f"collapse={int(bad_collapse.sum())}"
            ),
            uncertainty_delta=UncertaintyDelta(
                delta_epistemic=delta_epistemic,
                delta_aleatoric=delta_aleatoric,
                weight=1.0,
            ),
            stress_delta=StressDelta(
                delta_level=delta_level,
                delta_regions=delta_regions,
                weight=1.0,
            ),
            diagnostics={
                "n_damaged": n_damaged,
                "n_bad_finite": int(bad_finite.sum()),
                "n_bad_magnitude": int(bad_magnitude.sum()),
                "n_bad_collapse": int(bad_collapse.sum()),
                "damaged_mask": damaged.tolist(),
            },
        )


class RecoveryMechanism(BaseMechanism):
    """Propose restorative field deltas for high-uncertainty nodes.

    When damage detection has flagged nodes (high epistemic uncertainty),
    this mechanism proposes small field nudges toward the mean of their
    healthy neighbours.

    Parameters
    ----------
    repair_rate:
        Magnitude of the corrective nudge.
    uncertainty_threshold:
        Only nodes with epistemic uncertainty above this are repaired.
    """

    def __init__(
        self,
        repair_rate: float = 0.3,
        uncertainty_threshold: float = 0.7,
    ) -> None:
        super().__init__(
            mechanism_id="recovery",
            version="0.1.0",
            description="Repairs high-uncertainty nodes toward neighbour mean.",
        )
        self.repair_rate = repair_rate
        self.uncertainty_threshold = uncertainty_threshold

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        epistemic = snapshot.uncertainty.epistemic        # (n,)
        field_vals = snapshot.field_state.values          # (n, f)
        adj = snapshot.topology.adjacency

        damaged = epistemic > self.uncertainty_threshold  # (n,) bool

        if not damaged.any():
            return self._null_contribution("no nodes exceed uncertainty threshold")

        n_damaged = int(damaged.sum())

        # Compute neighbour mean for each node
        degree = np.asarray(adj.sum(axis=1)).ravel()     # (n,)
        neighbour_sum = adj @ field_vals                  # (n, f)
        with np.errstate(invalid="ignore", divide="ignore"):
            neighbour_mean = np.where(
                degree[:, None] > 0,
                neighbour_sum / degree[:, None],
                field_vals,
            )

        # Repair: nudge damaged nodes toward neighbour mean
        delta = np.zeros_like(field_vals)
        delta[damaged] = self.repair_rate * (
            neighbour_mean[damaged] - field_vals[damaged]
        )

        # Replace NaN/Inf with zero after computing delta
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        activation = min(n_damaged / snapshot.n_nodes, 1.0)

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.85,
            activation=activation,
            estimated_cost=float(np.abs(delta).sum()),
            rationale=(
                f"Repairing {n_damaged} damaged node(s) toward neighbour mean; "
                f"repair_rate={self.repair_rate}"
            ),
            field_delta=FieldDelta(delta=delta, weight=1.0),
            diagnostics={
                "n_repaired": n_damaged,
                "repair_rate": self.repair_rate,
                "mean_repair_magnitude": float(np.abs(delta[damaged]).mean())
                if n_damaged > 0
                else 0.0,
            },
        )
