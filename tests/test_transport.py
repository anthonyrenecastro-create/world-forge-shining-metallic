"""Tests for structural events and state transport."""
import numpy as np
import pytest
import scipy.sparse as sp

from hrm_ecology.state import WorldState, TopologyState
from hrm_ecology.structural.events import (
    AddEdgeEvent,
    AddNodeEvent,
    RemoveEdgeEvent,
    RemoveNodeEvent,
    ActivateChannelEvent,
    DeactivateChannelEvent,
    EventKind,
    sort_events,
    validate_event,
)
from hrm_ecology.structural.transport import (
    TransportDiagnostics,
    apply_add_edge,
    apply_add_node,
    apply_remove_edge,
    apply_remove_node,
    apply_activate_channel,
    apply_deactivate_channel,
    apply_event,
)


def make_state(n: int = 5, f: int = 3, seed: int = 0) -> WorldState:
    return WorldState.create_initial(n, f, seed=seed)


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------

class TestEventValidation:
    def test_valid_add_edge(self):
        event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="test",
            src=0,
            dst=1,
            weight=1.0,
        )
        assert validate_event(event, 5, frozenset()) is None

    def test_invalid_src_out_of_range(self):
        event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="test",
            src=99,
            dst=1,
            weight=1.0,
        )
        error = validate_event(event, 5, frozenset())
        assert error is not None

    def test_self_loop_rejected(self):
        event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="self loop",
            src=2,
            dst=2,
            weight=1.0,
        )
        error = validate_event(event, 5, frozenset())
        assert error is not None

    def test_remove_node_out_of_range(self):
        event = RemoveNodeEvent(
            kind=EventKind.REMOVE_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="remove out of range",
            node_index=99,
        )
        error = validate_event(event, 5, frozenset())
        assert error is not None

    def test_deactivate_missing_channel(self):
        event = DeactivateChannelEvent(
            kind=EventKind.DEACTIVATE_CHANNEL,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="deactivate missing",
            channel_id="missing",
        )
        error = validate_event(event, 5, frozenset(["a"]))
        assert error is not None


# ---------------------------------------------------------------------------
# Event ordering
# ---------------------------------------------------------------------------

class TestEventOrdering:
    def test_sort_is_deterministic(self):
        events = [
            AddEdgeEvent(kind=EventKind.ADD_EDGE, proposer_id="b", priority=1.0,
                         cost=1.0, rationale="b", src=0, dst=1),
            AddEdgeEvent(kind=EventKind.ADD_EDGE, proposer_id="a", priority=1.0,
                         cost=1.0, rationale="a", src=1, dst=2),
        ]
        sorted1 = sort_events(events)
        sorted2 = sort_events(events[::-1])
        assert [e.proposer_id for e in sorted1] == [e.proposer_id for e in sorted2]

    def test_higher_priority_first(self):
        low = AddEdgeEvent(kind=EventKind.ADD_EDGE, proposer_id="low", priority=0.5,
                           cost=1.0, rationale="low", src=0, dst=1)
        high = AddEdgeEvent(kind=EventKind.ADD_EDGE, proposer_id="high", priority=2.0,
                            cost=1.0, rationale="high", src=1, dst=2)
        sorted_events = sort_events([low, high])
        assert sorted_events[0].proposer_id == "high"


# ---------------------------------------------------------------------------
# Transport: add/remove nodes and edges
# ---------------------------------------------------------------------------

class TestAddNode:
    def test_node_count_increases(self):
        state = make_state(4, 3)
        event = AddNodeEvent(
            kind=EventKind.ADD_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add node",
            connect_to=(0, 1),
        )
        next_state, diag = apply_add_node(state, event)
        assert next_state.n_nodes == 5

    def test_field_shape_extended(self):
        state = make_state(4, 3)
        event = AddNodeEvent(
            kind=EventKind.ADD_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add node",
        )
        next_state, diag = apply_add_node(state, event)
        assert next_state.field_state.values.shape == (5, 3)

    def test_transport_diagnostics_returned(self):
        state = make_state(4, 3)
        event = AddNodeEvent(
            kind=EventKind.ADD_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add node",
        )
        _, diag = apply_add_node(state, event)
        assert isinstance(diag, TransportDiagnostics)
        assert diag.event_kind == "add_node"

    def test_original_state_unchanged(self):
        state = make_state(4, 3)
        prior_hash = state.content_hash()
        event = AddNodeEvent(
            kind=EventKind.ADD_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add node",
        )
        apply_add_node(state, event)
        assert state.content_hash() == prior_hash


class TestRemoveNode:
    def test_node_count_decreases(self):
        state = make_state(5, 3)
        event = RemoveNodeEvent(
            kind=EventKind.REMOVE_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="remove node",
            node_index=2,
        )
        next_state, diag = apply_remove_node(state, event)
        assert next_state.n_nodes == 4

    def test_field_shape_contracted(self):
        state = make_state(5, 3)
        event = RemoveNodeEvent(
            kind=EventKind.REMOVE_NODE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="remove node",
            node_index=0,
        )
        next_state, diag = apply_remove_node(state, event)
        assert next_state.field_state.values.shape == (4, 3)


class TestAddRemoveEdge:
    def test_add_edge_increases_nnz(self):
        state = make_state(5, 3)
        # Initially no edges
        assert state.topology.adjacency.nnz == 0
        event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add edge",
            src=0,
            dst=1,
            weight=1.0,
        )
        next_state, _ = apply_add_edge(state, event)
        assert next_state.topology.adjacency.nnz > 0

    def test_remove_edge_decreases_nnz(self):
        state = make_state(5, 3)
        # First add an edge
        add_event = AddEdgeEvent(
            kind=EventKind.ADD_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="add edge",
            src=0,
            dst=1,
            weight=1.0,
        )
        state_with_edge, _ = apply_add_edge(state, add_event)
        nnz_with_edge = state_with_edge.topology.adjacency.nnz

        remove_event = RemoveEdgeEvent(
            kind=EventKind.REMOVE_EDGE,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="remove edge",
            src=0,
            dst=1,
        )
        state_without_edge, _ = apply_remove_edge(state_with_edge, remove_event)
        assert state_without_edge.topology.adjacency.nnz < nnz_with_edge


class TestChannelEvents:
    def test_activate_channel_adds_to_set(self):
        state = make_state(4, 3)
        assert "test_channel" not in state.topology.channel_ids
        event = ActivateChannelEvent(
            kind=EventKind.ACTIVATE_CHANNEL,
            proposer_id="test",
            priority=1.0,
            cost=0.0,
            rationale="activate",
            channel_id="test_channel",
        )
        next_state, _ = apply_activate_channel(state, event)
        assert "test_channel" in next_state.topology.channel_ids

    def test_deactivate_channel_removes_from_set(self):
        state = make_state(4, 3)
        # First activate
        act_event = ActivateChannelEvent(
            kind=EventKind.ACTIVATE_CHANNEL,
            proposer_id="test",
            priority=1.0,
            cost=0.0,
            rationale="activate",
            channel_id="ch",
        )
        state_with_ch, _ = apply_activate_channel(state, act_event)
        assert "ch" in state_with_ch.topology.channel_ids

        deact_event = DeactivateChannelEvent(
            kind=EventKind.DEACTIVATE_CHANNEL,
            proposer_id="test",
            priority=1.0,
            cost=0.0,
            rationale="deactivate",
            channel_id="ch",
        )
        state_without_ch, _ = apply_deactivate_channel(state_with_ch, deact_event)
        assert "ch" not in state_without_ch.topology.channel_ids


# ---------------------------------------------------------------------------
# apply_event dispatcher
# ---------------------------------------------------------------------------

class TestApplyEventDispatcher:
    def test_unknown_event_is_noop(self):
        state = make_state(4, 3)
        from hrm_ecology.structural.events import SplitRegionEvent
        event = SplitRegionEvent(
            kind=EventKind.SPLIT_REGION,
            proposer_id="test",
            priority=1.0,
            cost=1.0,
            rationale="split",
            region_nodes=(0, 1, 2),
        )
        next_state, diag = apply_event(state, event)
        # Not implemented → state unchanged
        assert next_state.content_hash() == state.content_hash()
