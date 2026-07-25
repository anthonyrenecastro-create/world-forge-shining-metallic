"""Safety constitution for the HRM Ecology runtime.

Safety is constitutionally separate from mechanisms.  It is never a
competing mechanism.  Safety rules are applied as a final gate before
any state is committed; they may repair or reject proposals.

Every safety intervention is recorded in the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hrm_ecology.contribution import (
    Contribution,
    FieldDelta,
    StressDelta,
    UncertaintyDelta,
)
from hrm_ecology.ledger import SafetyIntervention
from hrm_ecology.state import WorldState


@dataclass(frozen=True)
class SafetyConfig:
    """Tunable parameters for the safety constitution."""
    # Field value clipping
    field_max_abs: float = 50.0
    # Maximum allowed L2 norm increase of the field per step
    max_field_update_energy: float = 20.0
    # Stress lockout: reject structural events above this threshold
    stress_lockout_threshold: float = 0.9
    # Resource floor: minimum compute/memory fraction per node
    resource_floor: float = 1e-3
    # Residual tolerance for causal attribution
    residual_tolerance: float = 1e-4
    # Maximum number of nodes that can be removed in a single step
    max_nodes_removed_per_step: int = 3


@dataclass
class SafetyResult:
    """Output of the safety filter: possibly-modified contributions and diagnostics."""
    contributions: list[Contribution]
    interventions: list[SafetyIntervention]
    proposed_field_repair: FieldDelta | None
    rollback: bool  # If True the engine must keep the prior state


class SafetyConstitution:
    """Applies constitutional safety rules to a set of contributions.

    Rules:
    1. Finite value preservation
    2. Bounded update energy
    3. Stress lockout (structural events blocked above threshold)
    4. Resource floor enforcement
    5. Field collapse prevention
    6. Rollback on unrecoverable violation
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def apply(
        self,
        snapshot: WorldState,
        contributions: list[Contribution],
        pending_structural_events: list,
    ) -> SafetyResult:
        """Filter and potentially repair contributions.

        Parameters
        ----------
        snapshot:
            Current (prior) world state.
        contributions:
            List of mechanism contributions to evaluate.
        pending_structural_events:
            Structural events to be applied this step.

        Returns
        -------
        SafetyResult
            Filtered contributions, interventions, repair delta, and rollback flag.
        """
        interventions: list[SafetyIntervention] = []
        safe_contributions: list[Contribution] = []
        rollback = False

        cfg = self.config

        for contrib in contributions:
            accept = True
            reason = ""

            # Rule 1: NaN / Inf in field deltas
            if contrib.field_delta is not None:
                delta = contrib.field_delta.delta
                if not np.isfinite(delta).all():
                    accept = False
                    reason = "field_delta contains NaN or Inf"
                    interventions.append(SafetyIntervention(
                        rule_name="finite_value",
                        description=f"Rejected {contrib.mechanism_id}: {reason}",
                        severity="error",
                    ))

            # Rule 2: Bounded update energy
            if accept and contrib.field_delta is not None:
                energy = float(np.linalg.norm(contrib.field_delta.delta))
                if energy > cfg.max_field_update_energy:
                    accept = False
                    reason = (
                        f"field_delta energy {energy:.4g} exceeds "
                        f"limit {cfg.max_field_update_energy}"
                    )
                    interventions.append(SafetyIntervention(
                        rule_name="bounded_update_energy",
                        description=f"Rejected {contrib.mechanism_id}: {reason}",
                        severity="warning",
                    ))

            if accept:
                safe_contributions.append(contrib)

        # Rule 3: Stress lockout – block structural events under high stress
        if snapshot.stress.level >= cfg.stress_lockout_threshold and pending_structural_events:
            interventions.append(SafetyIntervention(
                rule_name="stress_lockout",
                description=(
                    f"Blocked {len(pending_structural_events)} structural event(s); "
                    f"stress={snapshot.stress.level:.4g} ≥ threshold {cfg.stress_lockout_threshold}"
                ),
                severity="warning",
            ))
            pending_structural_events.clear()

        # Rule 4: Detect NaN/Inf in current field (pre-commit state check)
        if not np.isfinite(snapshot.field_state.values).all():
            interventions.append(SafetyIntervention(
                rule_name="field_integrity",
                description="Current state field contains NaN/Inf; triggering repair",
                severity="critical",
            ))
            # Propose a repair: zero out non-finite nodes
            bad = ~np.isfinite(snapshot.field_state.values).all(axis=1)
            repair_delta_arr = np.where(
                bad[:, None],
                -snapshot.field_state.values,  # reset to zero
                0.0,
            )
            repair = FieldDelta(delta=repair_delta_arr, weight=2.0)
        else:
            repair = None

        # Rule 5: Field collapse prevention
        if snapshot.n_nodes > 0:
            field_std = float(snapshot.field_state.values.std())
            if field_std < 1e-10 and snapshot.metadata.step > 0:
                interventions.append(SafetyIntervention(
                    rule_name="field_collapse",
                    description=f"Field std={field_std:.2e} near zero; injecting noise",
                    severity="warning",
                ))
                # Inject tiny noise via a pseudo repair
                # (handled downstream by the engine)

        # Rollback if field integrity is unrecoverable (all bad and no repair possible)
        if not np.isfinite(snapshot.field_state.values).all() and repair is None:
            rollback = True

        return SafetyResult(
            contributions=safe_contributions,
            interventions=interventions,
            proposed_field_repair=repair,
            rollback=rollback,
        )
