"""Deterministic replay module.

Every transition can be replayed identically given:
- the prior state
- the observations
- the seed
- the registered mechanism versions

Replay should reproduce identical state hashes, verifying that the
system is fully deterministic and causally attributable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hrm_ecology.engine import Engine, StepResult
from hrm_ecology.ledger import AttributionEntry, Ledger
from hrm_ecology.mechanisms.base import Mechanism, Observations
from hrm_ecology.safety import SafetyConfig
from hrm_ecology.state import WorldState
from hrm_ecology.structural.events import StructuralEvent


@dataclass
class ReplayRecord:
    """A single step to be replayed."""
    step: int
    observations: Observations
    structural_events: list[StructuralEvent]
    seed: int
    expected_next_hash: str
    expected_prior_hash: str


@dataclass
class ReplayReport:
    """Result of a full replay verification run."""
    total_steps: int
    matching_steps: int
    mismatched_steps: list[int]
    hash_errors: list[str]

    @property
    def success(self) -> bool:
        return len(self.mismatched_steps) == 0


class ReplayEngine:
    """Replays a sequence of transitions and verifies state hashes.

    Parameters
    ----------
    mechanisms:
        The same mechanism instances (same IDs and versions) used originally.
    safety_config:
        The same safety configuration used originally.
    """

    def __init__(
        self,
        mechanisms: list[Mechanism],
        safety_config: SafetyConfig | None = None,
    ) -> None:
        self.mechanisms = mechanisms
        self.safety_config = safety_config

    def build_replay_records(
        self,
        ledger: Ledger,
        observations_history: list[Observations],
        events_history: list[list[StructuralEvent]] | None = None,
    ) -> list[ReplayRecord]:
        """Construct replay records from a ledger and observation history."""
        records: list[ReplayRecord] = []
        entries = ledger.entries()
        events_seq = events_history or [[] for _ in entries]

        for entry, obs, events in zip(entries, observations_history, events_seq):
            records.append(ReplayRecord(
                step=entry.step,
                observations=obs,
                structural_events=list(events),
                seed=entry.seed,
                expected_next_hash=entry.next_state_hash,
                expected_prior_hash=entry.prior_state_hash,
            ))
        return records

    def replay(
        self,
        initial_state: WorldState,
        records: list[ReplayRecord],
    ) -> ReplayReport:
        """Replay all records from *initial_state* and verify hashes.

        Parameters
        ----------
        initial_state:
            The world state at ``step=0`` (must match the first record's
            ``expected_prior_hash``).
        records:
            Ordered list of :class:`ReplayRecord` objects.

        Returns
        -------
        ReplayReport
            Summary of matching and mismatching steps.
        """
        mismatched: list[int] = []
        errors: list[str] = []
        state = initial_state
        replay_ledger = Ledger()

        engine = Engine(
            mechanisms=self.mechanisms,
            safety_config=self.safety_config,
            ledger=replay_ledger,
        )

        for record in records:
            # Verify prior hash
            actual_prior = state.content_hash()
            if actual_prior != record.expected_prior_hash:
                errors.append(
                    f"Step {record.step}: prior hash mismatch – "
                    f"expected {record.expected_prior_hash!r}, got {actual_prior!r}"
                )
                mismatched.append(record.step)
                continue

            # Run the step with the recorded seed
            result = engine.step(
                state=state,
                observations=record.observations,
                structural_events=record.structural_events,
                seed_override=record.seed,
            )

            # Verify next hash
            actual_next = result.next_state.content_hash()
            if actual_next != record.expected_next_hash:
                errors.append(
                    f"Step {record.step}: next hash mismatch – "
                    f"expected {record.expected_next_hash!r}, got {actual_next!r}"
                )
                mismatched.append(record.step)

            state = result.next_state

        return ReplayReport(
            total_steps=len(records),
            matching_steps=len(records) - len(set(mismatched)),
            mismatched_steps=sorted(set(mismatched)),
            hash_errors=errors,
        )
