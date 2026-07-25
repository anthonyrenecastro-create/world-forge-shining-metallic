"""HRM Ecology transition engine.

Implements the core constitutional semantics:

1. **Common Snapshot** – every mechanism receives the same immutable ``Xₜ``.
2. **Proposal, Never Mutation** – mechanisms return :class:`Contribution` only.
3. **Singular Commit** – exactly one function produces ``X_{t+1}``.
4. **Causal Attribution** – every change is recorded in the :class:`Ledger`.
5. **Safety First** – the :class:`SafetyConstitution` gates every transition.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrm_ecology.contribution import (
    Contribution,
    FieldDelta,
    MemoryDelta,
    UncertaintyDelta,
    ResourceDelta,
    CognitionDelta,
    StressDelta,
)
from hrm_ecology.ledger import (
    AttributionEntry,
    CauseKind,
    ContributionRecord,
    Ledger,
    SafetyIntervention,
    StructuralEventRecord,
)
from hrm_ecology.mechanisms.base import Mechanism, Observations
from hrm_ecology.safety import SafetyConstitution, SafetyConfig
from hrm_ecology.state import (
    FieldState,
    MemoryState,
    CognitionState,
    UncertaintyState,
    ResourceState,
    StressState,
    WorldState,
)
from hrm_ecology.structural.events import (
    StructuralEvent,
    sort_events,
    validate_event,
)
from hrm_ecology.structural.transport import (
    TransportDiagnostics,
    apply_event,
)


# ---------------------------------------------------------------------------
# Arbitration helpers
# ---------------------------------------------------------------------------

def _arbitrate_field_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
    cfg_field_max: float,
) -> np.ndarray:
    """Weighted average of all proposed field deltas."""
    n, f = snapshot.n_nodes, snapshot.n_features
    aggregate = np.zeros((n, f))
    total_weight = 0.0

    for contrib in contributions:
        if contrib.field_delta is None:
            continue
        w = contrib.field_delta.weight * contrib.confidence * contrib.activation
        if w == 0.0:
            continue
        aggregate += w * contrib.field_delta.delta
        total_weight += w

    if total_weight > 0.0:
        aggregate /= total_weight

    # Clip to prevent overflow
    aggregate = np.clip(aggregate, -cfg_field_max, cfg_field_max)
    return aggregate


def _arbitrate_memory_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted average of proposed memory deltas."""
    t = snapshot.memory.n_traces
    f = snapshot.n_features
    agg_traces = np.zeros((t, f))
    agg_salience = np.zeros(t)
    total_w = 0.0
    for contrib in contributions:
        if contrib.memory_delta is None:
            continue
        w = contrib.memory_delta.weight * contrib.confidence
        agg_traces += w * contrib.memory_delta.delta_traces
        agg_salience += w * contrib.memory_delta.delta_salience
        total_w += w
    if total_w > 0.0:
        agg_traces /= total_w
        agg_salience /= total_w
    return agg_traces, agg_salience


def _arbitrate_uncertainty_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
) -> tuple[np.ndarray, np.ndarray]:
    n = snapshot.n_nodes
    agg_ep = np.zeros(n)
    agg_al = np.zeros(n)
    total_w = 0.0
    for contrib in contributions:
        if contrib.uncertainty_delta is None:
            continue
        w = contrib.uncertainty_delta.weight * contrib.confidence
        agg_ep += w * contrib.uncertainty_delta.delta_epistemic
        agg_al += w * contrib.uncertainty_delta.delta_aleatoric
        total_w += w
    if total_w > 0.0:
        agg_ep /= total_w
        agg_al /= total_w
    return agg_ep, agg_al


def _arbitrate_resource_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
) -> tuple[np.ndarray, np.ndarray]:
    n = snapshot.n_nodes
    agg_c = np.zeros(n)
    agg_m = np.zeros(n)
    total_w = 0.0
    for contrib in contributions:
        if contrib.resource_delta is None:
            continue
        w = contrib.resource_delta.weight * contrib.confidence
        agg_c += w * contrib.resource_delta.delta_compute
        agg_m += w * contrib.resource_delta.delta_memory
        total_w += w
    if total_w > 0.0:
        agg_c /= total_w
        agg_m /= total_w
    return agg_c, agg_m


def _arbitrate_cognition_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
) -> tuple[np.ndarray, np.ndarray]:
    n = snapshot.n_nodes
    f = snapshot.n_features
    agg_attn = np.zeros(n)
    agg_pred = np.zeros((n, f))
    total_w = 0.0
    for contrib in contributions:
        if contrib.cognition_delta is None:
            continue
        w = contrib.cognition_delta.weight * contrib.confidence
        agg_attn += w * contrib.cognition_delta.delta_attention
        agg_pred += w * contrib.cognition_delta.delta_predictions
        total_w += w
    if total_w > 0.0:
        agg_attn /= total_w
        agg_pred /= total_w
    return agg_attn, agg_pred


def _arbitrate_stress_deltas(
    contributions: list[Contribution],
    snapshot: WorldState,
) -> tuple[float, np.ndarray]:
    n = snapshot.n_nodes
    agg_level = 0.0
    agg_regions = np.zeros(n)
    total_w = 0.0
    for contrib in contributions:
        if contrib.stress_delta is None:
            continue
        w = contrib.stress_delta.weight * contrib.confidence
        agg_level += w * contrib.stress_delta.delta_level
        agg_regions += w * contrib.stress_delta.delta_regions
        total_w += w
    if total_w > 0.0:
        agg_level /= total_w
        agg_regions /= total_w
    return agg_level, agg_regions


# ---------------------------------------------------------------------------
# Singular commit function
# ---------------------------------------------------------------------------

def commit(
    snapshot: WorldState,
    contributions: list[Contribution],
    structural_events: list[StructuralEvent],
    safety_interventions: list[SafetyIntervention],
    field_repair: FieldDelta | None,
    safety_config: SafetyConfig,
    rng: np.random.Generator,
) -> tuple[WorldState, float]:
    """The **only** function that creates ``X_{t+1}`` from ``Xₜ``.

    Returns the next state and the residual norm.
    All changes are attributable to one of the recognised cause kinds.

    Parameters
    ----------
    snapshot:
        Current immutable world state ``Xₜ``.
    contributions:
        Safety-filtered mechanism contributions.
    structural_events:
        Validated and ordered structural events to apply.
    safety_interventions:
        Safety interventions already decided.
    field_repair:
        Optional repair delta proposed by the safety constitution.
    safety_config:
        Safety configuration for post-commit validation.
    rng:
        Deterministic random generator (used only for stochastic tie-breaking).

    Returns
    -------
    next_state:
        The new authoritative world state ``X_{t+1}``.
    residual_norm:
        ‖ Δfield_applied - Δfield_proposed ‖₂ (causal completeness check).
    """
    cfg = safety_config
    n = snapshot.n_nodes

    # ------------------------------------------------------------------
    # 1. Arbitrate all sub-proposals
    # ------------------------------------------------------------------
    proposed_field_delta = _arbitrate_field_deltas(contributions, snapshot, cfg.field_max_abs)
    delta_traces, delta_salience = _arbitrate_memory_deltas(contributions, snapshot)
    delta_ep, delta_al = _arbitrate_uncertainty_deltas(contributions, snapshot)
    delta_comp, delta_mem = _arbitrate_resource_deltas(contributions, snapshot)
    delta_attn, delta_pred = _arbitrate_cognition_deltas(contributions, snapshot)
    delta_level, delta_regions = _arbitrate_stress_deltas(contributions, snapshot)

    # ------------------------------------------------------------------
    # 2. Apply field delta
    # ------------------------------------------------------------------
    new_field_values = snapshot.field_state.values + proposed_field_delta

    # Apply repair if safety proposed one
    if field_repair is not None:
        new_field_values = new_field_values + field_repair.delta

    # Clamp field to safe range
    new_field_values = np.clip(new_field_values, -cfg.field_max_abs, cfg.field_max_abs)
    new_field_values = np.nan_to_num(new_field_values, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 3. Apply memory delta
    # ------------------------------------------------------------------
    new_traces = np.clip(snapshot.memory.traces + delta_traces, -100.0, 100.0)
    new_salience = np.clip(snapshot.memory.salience + delta_salience, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 4. Apply uncertainty delta
    # ------------------------------------------------------------------
    new_epistemic = np.clip(snapshot.uncertainty.epistemic + delta_ep, 0.0, 1.0)
    new_aleatoric = np.clip(snapshot.uncertainty.aleatoric + delta_al, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 5. Apply resource delta (with floor enforcement)
    # ------------------------------------------------------------------
    new_compute = np.clip(
        snapshot.resources.compute + delta_comp,
        cfg.resource_floor,
        None,
    )
    new_memory_alloc = np.clip(
        snapshot.resources.memory_alloc + delta_mem,
        cfg.resource_floor,
        None,
    )
    # Re-normalise so they sum to 1
    if new_compute.sum() > 0:
        new_compute = new_compute / new_compute.sum()
    if new_memory_alloc.sum() > 0:
        new_memory_alloc = new_memory_alloc / new_memory_alloc.sum()

    # ------------------------------------------------------------------
    # 6. Apply cognition delta
    # ------------------------------------------------------------------
    new_attention = np.clip(snapshot.cognition.attention + delta_attn, 0.0, None)
    attn_sum = new_attention.sum()
    if attn_sum > 0:
        new_attention = new_attention / attn_sum
    new_predictions = np.clip(
        snapshot.cognition.predictions + delta_pred, -cfg.field_max_abs, cfg.field_max_abs
    )

    # ------------------------------------------------------------------
    # 7. Apply stress delta
    # ------------------------------------------------------------------
    new_stress_level = float(np.clip(snapshot.stress.level + delta_level, 0.0, 1.0))
    new_stress_regions = np.clip(snapshot.stress.regions + delta_regions, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 8. Assemble intermediate state before structural events
    # ------------------------------------------------------------------
    intermediate = WorldState(
        field_state=FieldState(new_field_values),
        geometry=snapshot.geometry,
        topology=snapshot.topology,
        memory=MemoryState(new_traces, new_salience, snapshot.memory.timestamps),
        cognition=CognitionState(new_attention, new_predictions),
        hierarchy=snapshot.hierarchy,
        guidance=snapshot.guidance,
        uncertainty=UncertaintyState(new_epistemic, new_aleatoric),
        stress=StressState(new_stress_level, new_stress_regions),
        resources=ResourceState(new_compute, new_memory_alloc, snapshot.resources.budget),
        goals=snapshot.goals,
        metadata=snapshot.metadata,
    )

    # ------------------------------------------------------------------
    # 9. Apply structural events (with explicit transport)
    # ------------------------------------------------------------------
    current = intermediate
    transport_diags: list[TransportDiagnostics] = []
    for event in structural_events:
        current, diag = apply_event(current, event)
        transport_diags.append(diag)

    # ------------------------------------------------------------------
    # 10. Bump metadata
    # ------------------------------------------------------------------
    import time as _time
    from hrm_ecology.state import MetadataState

    new_metadata = MetadataState(
        step=snapshot.metadata.step + 1,
        timestamp=_time.time(),
        seed=snapshot.metadata.seed,
        version=snapshot.metadata.version,
    )
    next_state = WorldState(
        field_state=current.field_state,
        geometry=current.geometry,
        topology=current.topology,
        memory=current.memory,
        cognition=current.cognition,
        hierarchy=current.hierarchy,
        guidance=current.guidance,
        uncertainty=current.uncertainty,
        stress=current.stress,
        resources=current.resources,
        goals=current.goals,
        metadata=new_metadata,
    )

    # ------------------------------------------------------------------
    # 11. Compute residual (causal completeness)
    # ------------------------------------------------------------------
    actual_field_change = (
        next_state.field_state.values - snapshot.field_state.values
    )
    residual = actual_field_change - proposed_field_delta
    residual_norm = float(np.linalg.norm(residual))

    return next_state, residual_norm


# ---------------------------------------------------------------------------
# Top-level step function
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of one complete transition step."""
    next_state: WorldState
    attribution: AttributionEntry
    rollback: bool = False


class Engine:
    """The HRM Ecology transition engine.

    Orchestrates:
    * Common snapshot distribution to all mechanisms.
    * Safety constitution application.
    * Singular commit.
    * Ledger recording.
    """

    def __init__(
        self,
        mechanisms: list[Mechanism],
        safety_config: SafetyConfig | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.mechanisms = list(mechanisms)
        self.safety = SafetyConstitution(safety_config or SafetyConfig())
        self.ledger = ledger or Ledger()
        self._safety_config = safety_config or SafetyConfig()

    def step(
        self,
        state: WorldState,
        observations: Observations,
        structural_events: list[StructuralEvent] | None = None,
        seed_override: int | None = None,
    ) -> StepResult:
        """Execute one transition step.

        Parameters
        ----------
        state:
            Current authoritative world state ``Xₜ``.
        observations:
            Exogenous sensor bundle (same for all mechanisms).
        structural_events:
            Proposed structural events (validated before application).
        seed_override:
            If given, used instead of ``state.metadata.seed`` for the RNG.
        """
        prior_hash = state.content_hash()

        # Deterministic RNG seeded by (global_seed XOR step)
        seed = seed_override if seed_override is not None else state.metadata.seed
        rng_seed = seed ^ state.metadata.step
        rng = np.random.default_rng(rng_seed)

        events = list(structural_events or [])

        # ------------------------------------------------------------------
        # 1. Common Snapshot: all mechanisms receive the same immutable state
        # ------------------------------------------------------------------
        snapshot = state  # Already immutable (frozen dataclass, read-only arrays)

        # ------------------------------------------------------------------
        # 2. Gather proposals (can be parallelised; each sees only snapshot)
        # ------------------------------------------------------------------
        raw_contributions: list[Contribution] = []
        for mechanism in self.mechanisms:
            # Each mechanism gets its own sub-RNG derived from the main seed
            sub_rng = np.random.default_rng(rng.integers(0, 2**31))
            try:
                contrib = mechanism.propose(snapshot, observations, sub_rng)
            except Exception as exc:
                # Mechanism failed; record a null contribution and continue
                contrib = Contribution(
                    mechanism_id=mechanism.info.mechanism_id,
                    confidence=0.0,
                    activation=0.0,
                    estimated_cost=0.0,
                    rationale=f"mechanism raised exception: {exc}",
                    diagnostics={"exception": str(exc)},
                )
            raw_contributions.append(contrib)

        # ------------------------------------------------------------------
        # 3. Validate structural events
        # ------------------------------------------------------------------
        valid_events: list[StructuralEvent] = []
        struct_records: list[StructuralEventRecord] = []
        sorted_events = sort_events(events)

        for event in sorted_events:
            error = validate_event(
                event, snapshot.n_nodes, snapshot.topology.channel_ids
            )
            accepted = error is None
            struct_records.append(StructuralEventRecord(
                event_type=event.kind.name,
                description=event.rationale,
                affected_nodes=tuple(),
                accepted=accepted,
                rejection_reason=error,
            ))
            if accepted:
                valid_events.append(event)

        # ------------------------------------------------------------------
        # 4. Safety constitution
        # ------------------------------------------------------------------
        safety_result = self.safety.apply(snapshot, raw_contributions, valid_events)

        # ------------------------------------------------------------------
        # 5. Singular commit
        # ------------------------------------------------------------------
        if safety_result.rollback:
            # Keep prior state; record the failed transition
            next_state = state
            residual_norm = 0.0
        else:
            next_state, residual_norm = commit(
                snapshot=snapshot,
                contributions=safety_result.contributions,
                structural_events=valid_events,
                safety_interventions=safety_result.interventions,
                field_repair=safety_result.proposed_field_repair,
                safety_config=self._safety_config,
                rng=rng,
            )

        # ------------------------------------------------------------------
        # 6. Build contribution records for the ledger
        # ------------------------------------------------------------------
        accepted_ids = {c.mechanism_id for c in safety_result.contributions}
        contrib_records = []
        for contrib in raw_contributions:
            accepted = contrib.mechanism_id in accepted_ids
            rejection_reason = None if accepted else "rejected by safety constitution"
            contrib_records.append(
                ContributionRecord.from_contribution(
                    contrib, accepted=accepted, rejection_reason=rejection_reason
                )
            )

        # ------------------------------------------------------------------
        # 7. Determine dominant cause kind
        # ------------------------------------------------------------------
        if safety_result.rollback:
            cause = CauseKind.SAFETY_REPAIR
        elif safety_result.proposed_field_repair is not None:
            cause = CauseKind.SAFETY_REPAIR
        elif valid_events:
            cause = CauseKind.STRUCTURAL_EVENT
        elif any(c.accepted for c in contrib_records):
            cause = CauseKind.MECHANISM_PROPOSAL
        else:
            cause = CauseKind.RESIDUAL

        # ------------------------------------------------------------------
        # 8. Append to ledger
        # ------------------------------------------------------------------
        entry = AttributionEntry(
            step=state.metadata.step,
            prior_state_hash=prior_hash,
            next_state_hash=next_state.content_hash(),
            timestamp=time.time(),
            seed=seed,
            contributions=tuple(contrib_records),
            safety_interventions=tuple(safety_result.interventions),
            structural_events=tuple(struct_records),
            cause_kind=cause,
            residual_norm=residual_norm,
            residual_tolerance=self._safety_config.residual_tolerance,
        )
        self.ledger.append(entry)

        return StepResult(
            next_state=next_state,
            attribution=entry,
            rollback=safety_result.rollback,
        )

    def run(
        self,
        initial_state: WorldState,
        observation_sequence: list[Observations],
        structural_event_sequence: list[list[StructuralEvent]] | None = None,
    ) -> tuple[WorldState, list[StepResult]]:
        """Run the engine for ``len(observation_sequence)`` steps.

        Returns the final state and a list of per-step results.
        """
        state = initial_state
        results: list[StepResult] = []
        events_seq = structural_event_sequence or [[] for _ in observation_sequence]

        for obs, events in zip(observation_sequence, events_seq):
            result = self.step(state, obs, events)
            results.append(result)
            state = result.next_state

        return state, results
