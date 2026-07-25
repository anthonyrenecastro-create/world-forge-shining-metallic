"""Tests for immutable world state."""
import numpy as np
import pytest
import scipy.sparse as sp

from hrm_ecology.state import (
    FieldState,
    GeometryState,
    TopologyState,
    MemoryState,
    CognitionState,
    HierarchyState,
    GuidanceState,
    UncertaintyState,
    StressState,
    ResourceState,
    ResourceBudget,
    GoalState,
    Goal,
    MetadataState,
    WorldState,
    _ro,
)


# ---------------------------------------------------------------------------
# FieldState
# ---------------------------------------------------------------------------

class TestFieldState:
    def test_values_are_readonly(self):
        fs = FieldState(np.ones((4, 3)))
        with pytest.raises((ValueError, TypeError)):
            fs.values[0, 0] = 99.0

    def test_shape_properties(self):
        fs = FieldState(np.zeros((5, 7)))
        assert fs.n_nodes == 5
        assert fs.n_features == 7

    def test_content_hash_deterministic(self):
        arr = np.arange(12, dtype=np.float64).reshape(4, 3)
        h1 = FieldState(arr).content_hash()
        h2 = FieldState(arr.copy()).content_hash()
        assert h1 == h2

    def test_different_arrays_different_hash(self):
        a = FieldState(np.zeros((4, 3)))
        b = FieldState(np.ones((4, 3)))
        assert a.content_hash() != b.content_hash()

    def test_equality(self):
        arr = np.ones((3, 2))
        assert FieldState(arr) == FieldState(arr.copy())
        assert FieldState(arr) != FieldState(arr * 2)

    def test_float64_cast(self):
        fs = FieldState(np.array([[1, 2], [3, 4]], dtype=np.int32))
        assert fs.values.dtype == np.float64

    def test_frozen(self):
        fs = FieldState(np.zeros((2, 2)))
        with pytest.raises((TypeError, AttributeError)):
            fs.values = np.ones((2, 2))  # type: ignore


# ---------------------------------------------------------------------------
# TopologyState
# ---------------------------------------------------------------------------

class TestTopologyState:
    def test_from_adjacency_round_trip(self):
        adj = sp.eye(5, format="csr")
        topo = TopologyState.from_adjacency(adj)
        reconstructed = topo.adjacency
        diff = (reconstructed - adj).nnz
        assert diff == 0

    def test_channel_ids_immutable(self):
        topo = TopologyState.from_adjacency(
            sp.eye(3), channel_ids=frozenset(["a", "b"])
        )
        assert "a" in topo.channel_ids
        assert "b" in topo.channel_ids

    def test_adj_arrays_readonly(self):
        adj = sp.eye(4, format="csr")
        topo = TopologyState.from_adjacency(adj)
        with pytest.raises((ValueError, TypeError)):
            topo._adj_data[0] = 99.0


# ---------------------------------------------------------------------------
# StressState
# ---------------------------------------------------------------------------

class TestStressState:
    def test_level_out_of_range_raises(self):
        with pytest.raises(ValueError):
            StressState(level=1.5, regions=np.zeros(3))

    def test_valid_construction(self):
        ss = StressState(0.5, np.zeros(4))
        assert ss.level == 0.5


# ---------------------------------------------------------------------------
# ResourceBudget
# ---------------------------------------------------------------------------

class TestResourceBudget:
    def test_invalid_compute(self):
        with pytest.raises(ValueError):
            ResourceBudget(max_compute_units=-1.0)

    def test_defaults(self):
        b = ResourceBudget()
        assert b.max_compute_units > 0


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

class TestWorldState:
    def _make_state(self, n: int = 4, f: int = 3, seed: int = 0) -> WorldState:
        return WorldState.create_initial(n, f, seed=seed)

    def test_create_initial(self):
        state = self._make_state(4, 3)
        assert state.n_nodes == 4
        assert state.n_features == 3

    def test_content_hash_deterministic(self):
        s1 = self._make_state(4, 3, seed=7)
        s2 = self._make_state(4, 3, seed=7)
        assert s1.content_hash() == s2.content_hash()

    def test_different_seeds_different_hash(self):
        s1 = self._make_state(4, 3, seed=1)
        s2 = self._make_state(4, 3, seed=2)
        # Positions differ → hashes differ
        assert s1.content_hash() != s2.content_hash()

    def test_frozen_prevents_mutation(self):
        state = self._make_state(4, 3)
        with pytest.raises((TypeError, AttributeError)):
            state.metadata = None  # type: ignore

    def test_field_values_readonly(self):
        state = self._make_state(4, 3)
        with pytest.raises((ValueError, TypeError)):
            state.field_state.values[0, 0] = 999.0

    def test_n_nodes_matches(self):
        state = self._make_state(6, 2)
        assert state.n_nodes == 6
        assert state.geometry.n_nodes == 6
        assert len(state.uncertainty.epistemic) == 6

    def test_metadata_step_zero(self):
        state = self._make_state()
        assert state.metadata.step == 0
