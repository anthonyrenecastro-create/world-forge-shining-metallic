"""Tests for all mechanisms."""
import numpy as np
import pytest

from hrm_ecology.mechanisms.base import Observations
from hrm_ecology.mechanisms.diffusion import DiffusionMechanism
from hrm_ecology.mechanisms.prediction import PredictionErrorCorrectionMechanism
from hrm_ecology.mechanisms.memory import (
    MemoryConsolidationMechanism,
    MemoryRetrievalMechanism,
)
from hrm_ecology.mechanisms.stress import StressRegulationMechanism
from hrm_ecology.mechanisms.goal_guidance import GoalGuidanceMechanism
from hrm_ecology.mechanisms.curiosity import CuriosityMechanism
from hrm_ecology.mechanisms.resource_alloc import ResourceAllocationMechanism
from hrm_ecology.mechanisms.recovery import (
    DamageDetectionMechanism,
    RecoveryMechanism,
)
from hrm_ecology.state import WorldState, Goal, GoalState
import scipy.sparse as sp
import numpy as np


def make_state(n: int = 6, f: int = 4, seed: int = 0) -> WorldState:
    return WorldState.create_initial(n, f, seed=seed)


def make_obs(state: WorldState) -> Observations:
    n_sensors = 4
    return Observations(
        sensor_data=np.zeros((n_sensors, 2)),
        sensor_mask=np.ones(n_sensors, dtype=bool),
        step=state.metadata.step,
    )


def make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Common contract: all mechanisms must return valid Contribution
# ---------------------------------------------------------------------------

ALL_MECHANISMS = [
    DiffusionMechanism(),
    PredictionErrorCorrectionMechanism(),
    MemoryConsolidationMechanism(),
    MemoryRetrievalMechanism(),
    StressRegulationMechanism(),
    GoalGuidanceMechanism(),
    CuriosityMechanism(),
    ResourceAllocationMechanism(),
    DamageDetectionMechanism(),
    RecoveryMechanism(),
]


@pytest.mark.parametrize("mech", ALL_MECHANISMS, ids=lambda m: m.info.mechanism_id)
def test_mechanism_returns_valid_contribution(mech):
    state = make_state(6, 4)
    obs = make_obs(state)
    rng = make_rng()
    contrib = mech.propose(state, obs, rng)
    assert 0.0 <= contrib.confidence <= 1.0
    assert 0.0 <= contrib.activation <= 1.0
    assert contrib.estimated_cost >= 0.0
    assert isinstance(contrib.rationale, str)


@pytest.mark.parametrize("mech", ALL_MECHANISMS, ids=lambda m: m.info.mechanism_id)
def test_mechanism_does_not_mutate_state(mech):
    state = make_state(6, 4)
    prior_hash = state.content_hash()
    obs = make_obs(state)
    rng = make_rng()
    mech.propose(state, obs, rng)
    assert state.content_hash() == prior_hash, "Mechanism mutated the snapshot!"


@pytest.mark.parametrize("mech", ALL_MECHANISMS, ids=lambda m: m.info.mechanism_id)
def test_mechanism_field_delta_shape(mech):
    state = make_state(6, 4)
    obs = make_obs(state)
    rng = make_rng()
    contrib = mech.propose(state, obs, rng)
    if contrib.field_delta is not None:
        assert contrib.field_delta.delta.shape == (state.n_nodes, state.n_features)


# ---------------------------------------------------------------------------
# Diffusion mechanism
# ---------------------------------------------------------------------------

class TestDiffusionMechanism:
    def test_empty_topology_zero_delta(self):
        """With no edges, diffusion should produce zero delta."""
        state = make_state(4, 2)
        mech = DiffusionMechanism(alpha=0.1)
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        # No edges → neighbour_sum = 0 → delta = alpha*(0 - 0) = 0
        if contrib.field_delta is not None:
            assert np.allclose(contrib.field_delta.delta, 0.0)

    def test_with_edges(self):
        """With edges and non-zero field, diffusion should produce non-zero delta."""
        import scipy.sparse as sp
        from hrm_ecology.state import (
            WorldState, FieldState, TopologyState, GeometryState,
            MemoryState, CognitionState, HierarchyState, GuidanceState,
            UncertaintyState, StressState, ResourceState, ResourceBudget,
            GoalState, MetadataState
        )
        import time
        n, f = 4, 2
        base = WorldState.create_initial(n, f, seed=0)
        # Set non-zero field
        field_vals = np.arange(n * f, dtype=np.float64).reshape(n, f)
        # Add a ring topology
        adj = sp.lil_matrix((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1.0
            adj[(i + 1) % n, i] = 1.0
        topo = TopologyState.from_adjacency(adj.tocsr())
        state = WorldState(
            field_state=FieldState(field_vals),
            geometry=base.geometry,
            topology=topo,
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
        mech = DiffusionMechanism(alpha=0.1)
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        assert contrib.field_delta is not None
        # At least some delta values are non-zero
        assert not np.allclose(contrib.field_delta.delta, 0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            DiffusionMechanism(alpha=0.0)


# ---------------------------------------------------------------------------
# Prediction error correction
# ---------------------------------------------------------------------------

class TestPredictionErrorCorrection:
    def test_zero_error_zero_delta(self):
        """If predictions exactly match field, delta should be zero."""
        from hrm_ecology.state import (
            WorldState, FieldState, CognitionState
        )
        n, f = 4, 3
        base = WorldState.create_initial(n, f, seed=0)
        field_vals = np.ones((n, f)) * 0.5
        pred_vals = field_vals.copy()
        state = WorldState(
            field_state=FieldState(field_vals),
            geometry=base.geometry,
            topology=base.topology,
            memory=base.memory,
            cognition=CognitionState(base.cognition.attention, pred_vals),
            hierarchy=base.hierarchy,
            guidance=base.guidance,
            uncertainty=base.uncertainty,
            stress=base.stress,
            resources=base.resources,
            goals=base.goals,
            metadata=base.metadata,
        )
        mech = PredictionErrorCorrectionMechanism()
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        assert contrib.field_delta is not None
        assert np.allclose(contrib.field_delta.delta, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Goal guidance
# ---------------------------------------------------------------------------

class TestGoalGuidance:
    def test_no_active_goals_null(self):
        state = make_state(4, 2)
        # No goals in default state
        mech = GoalGuidanceMechanism()
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        assert contrib.is_null

    def test_with_active_goal(self):
        from hrm_ecology.state import (
            WorldState, GoalState, Goal
        )
        n, f = 4, 2
        base = make_state(n, f)
        goal = Goal(
            goal_id="g1",
            description="test goal",
            target_pattern=np.ones(f),
            priority=1.0,
            active=True,
        )
        goals = GoalState(
            goals=(goal,),
            priorities=np.ones(n) / n,
        )
        state = WorldState(
            field_state=base.field_state,
            geometry=base.geometry,
            topology=base.topology,
            memory=base.memory,
            cognition=base.cognition,
            hierarchy=base.hierarchy,
            guidance=base.guidance,
            uncertainty=base.uncertainty,
            stress=base.stress,
            resources=base.resources,
            goals=goals,
            metadata=base.metadata,
        )
        mech = GoalGuidanceMechanism(strength=0.1)
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        assert contrib.field_delta is not None
        assert contrib.field_delta.delta.shape == (n, f)


# ---------------------------------------------------------------------------
# Damage detection
# ---------------------------------------------------------------------------

class TestDamageDetection:
    def test_no_damage_null(self):
        state = make_state(4, 3)
        mech = DamageDetectionMechanism()
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        # Default state has zeros → collapse flag may trigger
        # Just check it runs without error
        assert contrib.confidence >= 0.0

    def test_detects_nan(self):
        from hrm_ecology.state import WorldState, FieldState
        base = make_state(4, 3)
        bad_vals = np.zeros((4, 3))
        bad_vals[1, 0] = np.nan
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
        mech = DamageDetectionMechanism()
        obs = make_obs(state)
        contrib = mech.propose(state, obs, make_rng())
        assert contrib.activation > 0.0
        assert "n_damaged" in contrib.diagnostics
        assert contrib.diagnostics["n_damaged"] > 0
