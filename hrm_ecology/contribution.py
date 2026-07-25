"""Contribution (proposal) types produced by mechanisms.

A :class:`Contribution` is the *only* thing a mechanism may return.
It proposes changes; it never mutates state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hrm_ecology.state import _ro


@dataclass(frozen=True)
class FieldDelta:
    """Proposed additive change to the field activations.

    Shape must match ``(n_nodes, n_features)`` of the current world state.
    ``weight`` scales how strongly this delta is applied during arbitration.
    """
    delta: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta", _ro(self.delta))


@dataclass(frozen=True)
class MemoryDelta:
    """Proposed updates to memory traces and salience."""
    delta_traces: np.ndarray = field(hash=False, compare=False)
    delta_salience: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_traces", _ro(self.delta_traces))
        object.__setattr__(self, "delta_salience", _ro(self.delta_salience))


@dataclass(frozen=True)
class UncertaintyDelta:
    """Proposed updates to epistemic and aleatoric uncertainty."""
    delta_epistemic: np.ndarray = field(hash=False, compare=False)
    delta_aleatoric: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_epistemic", _ro(self.delta_epistemic))
        object.__setattr__(self, "delta_aleatoric", _ro(self.delta_aleatoric))


@dataclass(frozen=True)
class ResourceDelta:
    """Proposed reallocation of per-node compute and memory."""
    delta_compute: np.ndarray = field(hash=False, compare=False)
    delta_memory: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_compute", _ro(self.delta_compute))
        object.__setattr__(self, "delta_memory", _ro(self.delta_memory))


@dataclass(frozen=True)
class CognitionDelta:
    """Proposed updates to attention and predictions."""
    delta_attention: np.ndarray = field(hash=False, compare=False)
    delta_predictions: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_attention", _ro(self.delta_attention))
        object.__setattr__(self, "delta_predictions", _ro(self.delta_predictions))


@dataclass(frozen=True)
class StressDelta:
    """Proposed change to global and per-node stress."""
    delta_level: float
    delta_regions: np.ndarray = field(hash=False, compare=False)
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delta_regions", _ro(self.delta_regions))


@dataclass(frozen=True)
class Contribution:
    """The *only* output a mechanism may produce.

    Every field is optional; a mechanism sets only the fields it wishes
    to influence.  The transition engine arbitrates all contributions and
    constructs the next :class:`~hrm_ecology.state.WorldState`.
    """
    mechanism_id: str
    confidence: float
    activation: float
    estimated_cost: float
    rationale: str

    # Optional sub-proposals
    field_delta: FieldDelta | None = None
    memory_delta: MemoryDelta | None = None
    uncertainty_delta: UncertaintyDelta | None = None
    resource_delta: ResourceDelta | None = None
    cognition_delta: CognitionDelta | None = None
    stress_delta: StressDelta | None = None

    # Diagnostics: arbitrary key/value observability data
    diagnostics: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1]; got {self.confidence}")
        if not (0.0 <= self.activation <= 1.0):
            raise ValueError(f"activation must be in [0,1]; got {self.activation}")
        if self.estimated_cost < 0.0:
            raise ValueError(f"estimated_cost must be ≥ 0; got {self.estimated_cost}")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def is_null(self) -> bool:
        """True when this contribution proposes no actual changes."""
        return all(
            v is None
            for v in (
                self.field_delta,
                self.memory_delta,
                self.uncertainty_delta,
                self.resource_delta,
                self.cognition_delta,
                self.stress_delta,
            )
        )
