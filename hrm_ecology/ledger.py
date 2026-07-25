"""Append-only attribution ledger.

Every committed transition is recorded in the ledger.  The ledger is part
of computation, not merely a side-channel log.  Every accepted state change
must be attributable to exactly one of the recognised cause categories.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from hrm_ecology.contribution import Contribution


class CauseKind(Enum):
    """Recognised categories of transition causation."""
    MECHANISM_PROPOSAL = auto()
    INTERACTION = auto()
    STRUCTURAL_EVENT = auto()
    TRANSPORT = auto()
    SAFETY_REPAIR = auto()
    STOCHASTIC = auto()
    RESIDUAL = auto()


@dataclass(frozen=True)
class ContributionRecord:
    """Summary of one mechanism's accepted contribution for a transition."""
    mechanism_id: str
    confidence: float
    activation: float
    estimated_cost: float
    rationale: str
    accepted: bool
    rejection_reason: str | None
    diagnostics: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_contribution(
        cls,
        contrib: Contribution,
        *,
        accepted: bool,
        rejection_reason: str | None = None,
    ) -> "ContributionRecord":
        return cls(
            mechanism_id=contrib.mechanism_id,
            confidence=contrib.confidence,
            activation=contrib.activation,
            estimated_cost=contrib.estimated_cost,
            rationale=contrib.rationale,
            accepted=accepted,
            rejection_reason=rejection_reason,
            diagnostics=dict(contrib.diagnostics),
        )


@dataclass(frozen=True)
class SafetyIntervention:
    """Record of a safety constitution action during a transition."""
    rule_name: str
    description: str
    severity: str  # "info" | "warning" | "error" | "critical"


@dataclass(frozen=True)
class StructuralEventRecord:
    """Record of a structural event that occurred during a transition."""
    event_type: str
    description: str
    affected_nodes: tuple[int, ...]
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class AttributionEntry:
    """Full causal attribution for one transition.

    ``residual_norm`` must be ≤ ``residual_tolerance``; otherwise the
    transition is considered unexplained and should be rejected.
    """
    step: int
    prior_state_hash: str
    next_state_hash: str
    timestamp: float
    seed: int

    contributions: tuple[ContributionRecord, ...]
    safety_interventions: tuple[SafetyIntervention, ...]
    structural_events: tuple[StructuralEventRecord, ...]

    cause_kind: CauseKind
    residual_norm: float
    residual_tolerance: float

    diagnostics: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def explained(self) -> bool:
        return self.residual_norm <= self.residual_tolerance

    def entry_hash(self) -> str:
        payload = (
            f"{self.step}:{self.prior_state_hash}:{self.next_state_hash}"
            f":{self.seed}:{self.residual_norm:.8f}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Ledger:
    """Append-only sequence of :class:`AttributionEntry` records.

    The ledger is the persistent computational record of the system's
    history.  It can be used to verify causal completeness and to replay
    any past transition.

    Invariants:
    - Entries are appended in strictly increasing step order.
    - No entry may be modified after insertion.
    """

    def __init__(self) -> None:
        self._entries: list[AttributionEntry] = []

    # ------------------------------------------------------------------
    # Append (the only mutation operation)
    # ------------------------------------------------------------------

    def append(self, entry: AttributionEntry) -> None:
        """Append *entry* to the ledger.  Steps must be monotonically increasing."""
        if self._entries:
            last_step = self._entries[-1].step
            if entry.step <= last_step:
                raise ValueError(
                    f"Ledger steps must increase monotonically; "
                    f"got step {entry.step} after {last_step}"
                )
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._entries)

    def get(self, step: int) -> AttributionEntry | None:
        """Return the entry for *step*, or ``None`` if not found."""
        for e in self._entries:
            if e.step == step:
                return e
        return None

    def entries(self) -> tuple[AttributionEntry, ...]:
        """Return an immutable view of all entries."""
        return tuple(self._entries)

    def last(self) -> AttributionEntry | None:
        return self._entries[-1] if self._entries else None

    # ------------------------------------------------------------------
    # Causal completeness check
    # ------------------------------------------------------------------

    def audit(self) -> list[str]:
        """Return a list of audit failures (empty if all transitions explained)."""
        failures: list[str] = []
        for entry in self._entries:
            if not entry.explained:
                failures.append(
                    f"Step {entry.step}: residual {entry.residual_norm:.4g} "
                    f"> tolerance {entry.residual_tolerance:.4g}"
                )
        return failures

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Serialise the ledger to a plain Python structure for persistence."""
        out = []
        for e in self._entries:
            out.append({
                "step": e.step,
                "prior_state_hash": e.prior_state_hash,
                "next_state_hash": e.next_state_hash,
                "timestamp": e.timestamp,
                "seed": e.seed,
                "cause_kind": e.cause_kind.name,
                "residual_norm": e.residual_norm,
                "residual_tolerance": e.residual_tolerance,
                "explained": e.explained,
                "n_contributions": len(e.contributions),
                "n_accepted": sum(1 for c in e.contributions if c.accepted),
                "n_safety_interventions": len(e.safety_interventions),
                "n_structural_events": len(e.structural_events),
                "entry_hash": e.entry_hash(),
            })
        return out
