"""Tests for the safety constitution."""
import numpy as np
import pytest

from hrm_ecology.contribution import Contribution, FieldDelta
from hrm_ecology.safety import SafetyConstitution, SafetyConfig
from hrm_ecology.state import WorldState, StressState


def make_state(n: int = 4, f: int = 3, seed: int = 0) -> WorldState:
    return WorldState.create_initial(n, f, seed=seed)


def make_valid_contribution(mech_id: str = "test", n: int = 4, f: int = 3) -> Contribution:
    return Contribution(
        mechanism_id=mech_id,
        confidence=0.8,
        activation=0.5,
        estimated_cost=1.0,
        rationale="test contribution",
        field_delta=FieldDelta(delta=np.ones((n, f)) * 0.01, weight=1.0),
    )


class TestSafetyConstitution:
    def test_valid_contribution_passes(self):
        state = make_state()
        safety = SafetyConstitution()
        contrib = make_valid_contribution()
        result = safety.apply(state, [contrib], [])
        assert len(result.contributions) == 1
        assert not result.rollback

    def test_nan_field_delta_rejected(self):
        state = make_state()
        safety = SafetyConstitution()
        delta = np.ones((4, 3)) * 0.01
        delta[0, 0] = np.nan
        contrib = Contribution(
            mechanism_id="bad_mech",
            confidence=0.8,
            activation=0.5,
            estimated_cost=1.0,
            rationale="bad delta",
            field_delta=FieldDelta(delta=delta, weight=1.0),
        )
        result = safety.apply(state, [contrib], [])
        assert len(result.contributions) == 0
        assert any(i.rule_name == "finite_value" for i in result.interventions)

    def test_high_energy_delta_rejected(self):
        n, f = 4, 3
        state = make_state(n, f)
        safety = SafetyConstitution(SafetyConfig(max_field_update_energy=0.1))
        delta = np.ones((n, f)) * 100.0
        contrib = Contribution(
            mechanism_id="high_energy",
            confidence=0.8,
            activation=0.5,
            estimated_cost=1.0,
            rationale="big delta",
            field_delta=FieldDelta(delta=delta, weight=1.0),
        )
        result = safety.apply(state, [contrib], [])
        assert len(result.contributions) == 0
        assert any(i.rule_name == "bounded_update_energy" for i in result.interventions)

    def test_stress_lockout_blocks_structural_events(self):
        """With stress >= threshold, structural events should be blocked."""
        from hrm_ecology.structural.events import AddEdgeEvent, EventKind

        base = make_state()
        # Create high-stress state
        from hrm_ecology.state import WorldState
        state = WorldState(
            field_state=base.field_state,
            geometry=base.geometry,
            topology=base.topology,
            memory=base.memory,
            cognition=base.cognition,
            hierarchy=base.hierarchy,
            guidance=base.guidance,
            uncertainty=base.uncertainty,
            stress=StressState(0.95, base.stress.regions),
            resources=base.resources,
            goals=base.goals,
            metadata=base.metadata,
        )
        safety = SafetyConstitution(SafetyConfig(stress_lockout_threshold=0.9))
        event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add edge under stress",
            src=0,
            dst=1,
            weight=1.0,
        )
        pending = [event]
        result = safety.apply(state, [], pending)
        # Events should have been cleared
        assert len(pending) == 0
        assert any(i.rule_name == "stress_lockout" for i in result.interventions)

    def test_nan_state_triggers_repair(self):
        """State with NaN field should trigger a repair delta."""
        from hrm_ecology.state import WorldState, FieldState
        base = make_state()
        bad_vals = np.zeros((4, 3))
        bad_vals[0, 0] = np.nan
        state = WorldState(
            field_state=FieldState(bad_vals),
            geometry=base.geometry,
            topology=base.topology,
            memory=base.memory,
            cognition=base.cognition,
            hierarchy=base.hierarchy,
            guidance=base.guidance,
            uncertainty=base.uncertainty,
            stress=base.stress,
            resources=base.resources,
            goals=base.goals,
            metadata=base.metadata,
        )
        safety = SafetyConstitution()
        result = safety.apply(state, [], [])
        assert result.proposed_field_repair is not None
        assert any(i.rule_name == "field_integrity" for i in result.interventions)

    def test_multiple_contributions_partially_filtered(self):
        """Bad contributions rejected; good ones passed through."""
        n, f = 4, 3
        state = make_state(n, f)
        safety = SafetyConstitution(SafetyConfig(max_field_update_energy=5.0))

        good = make_valid_contribution("good", n, f)
        bad_delta = np.ones((n, f)) * 1000.0
        bad = Contribution(
            mechanism_id="bad",
            confidence=0.9,
            activation=0.9,
            estimated_cost=1.0,
            rationale="too large",
            field_delta=FieldDelta(delta=bad_delta, weight=1.0),
        )
        result = safety.apply(state, [good, bad], [])
        accepted_ids = {c.mechanism_id for c in result.contributions}
        assert "good" in accepted_ids
        assert "bad" not in accepted_ids
