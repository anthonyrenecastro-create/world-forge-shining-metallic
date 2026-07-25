"""Mechanism protocol and base class.

A *mechanism* is the fundamental unit of the computational ecology.
Mechanisms are independent: no mechanism may imperatively invoke another,
observe another's proposal, or mutate state.  They only propose changes
via :class:`~hrm_ecology.contribution.Contribution`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from hrm_ecology.contribution import Contribution
from hrm_ecology.state import WorldState


# ---------------------------------------------------------------------------
# Observation bundle passed to every mechanism
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observations:
    """Exogenous sensor data delivered to all mechanisms at each step.

    The observations bundle is read-only and the same instance is passed
    to every mechanism (Common Snapshot Principle extended to inputs).
    """
    sensor_data: np.ndarray   # shape: (n_sensors,) or (n_sensors, n_features)
    sensor_mask: np.ndarray   # shape: (n_sensors,) – True = valid reading
    step: int
    extra: dict = None        # type: ignore[assignment]

    def __post_init__(self) -> None:
        from hrm_ecology.state import _ro, _ro_int
        object.__setattr__(self, "sensor_data", _ro(self.sensor_data))
        object.__setattr__(self, "sensor_mask",
                           np.array(self.sensor_mask, dtype=bool))
        self.sensor_mask.flags.writeable = False
        if self.extra is None:
            object.__setattr__(self, "extra", {})


# ---------------------------------------------------------------------------
# Mechanism version metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MechanismInfo:
    """Static metadata for a mechanism (used in deterministic replay)."""
    mechanism_id: str
    version: str
    description: str


# ---------------------------------------------------------------------------
# Protocol (structural subtyping)
# ---------------------------------------------------------------------------

@runtime_checkable
class Mechanism(Protocol):
    """Protocol every mechanism must satisfy."""

    @property
    def info(self) -> MechanismInfo:
        """Static identity and version information."""
        ...

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        """Propose a change based solely on *snapshot* and *observations*.

        The mechanism MUST NOT:
        - Mutate *snapshot* or any field within it.
        - Store mutable references to *snapshot* across calls.
        - Communicate with other mechanisms.
        - Observe its own output from previous calls via shared mutable state.

        The mechanism MAY:
        - Maintain internal read-only caches derived from ``info``.
        - Use *rng* for stochastic computations.
        """
        ...


# ---------------------------------------------------------------------------
# Convenience ABC
# ---------------------------------------------------------------------------

class BaseMechanism(ABC):
    """Convenience ABC implementing the :class:`Mechanism` protocol."""

    def __init__(self, mechanism_id: str, version: str = "0.1.0",
                 description: str = "") -> None:
        self._info = MechanismInfo(
            mechanism_id=mechanism_id,
            version=version,
            description=description,
        )

    @property
    def info(self) -> MechanismInfo:
        return self._info

    @abstractmethod
    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        ...

    def _null_contribution(self, rationale: str = "inactive") -> Contribution:
        """Helper to return a no-op contribution."""
        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.0,
            activation=0.0,
            estimated_cost=0.0,
            rationale=rationale,
        )
