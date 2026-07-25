"""State transport: explicit reshaping of state when structure changes.

Whenever the computational structure changes (nodes added/removed, regions
split/merged), state must be transported explicitly with full diagnostics.
Silent array reshaping is forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    WorldState,
    _ro,
    _ro_int,
)
from hrm_ecology.structural.events import (
    AddEdgeEvent,
    AddNodeEvent,
    RemoveEdgeEvent,
    RemoveNodeEvent,
    ActivateChannelEvent,
    DeactivateChannelEvent,
    StructuralEvent,
    EventKind,
)


# ---------------------------------------------------------------------------
# Transport diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransportDiagnostics:
    """Report of mass conservation during state transport."""
    event_kind: str
    transported_mass: float
    introduced_mass: float
    lost_mass: float
    continuity_error: float
    conservation_residual: float
    unmapped_references: int
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def is_conservative(self) -> bool:
        return self.conservation_residual < 1e-6


# ---------------------------------------------------------------------------
# Individual event application
# ---------------------------------------------------------------------------

def apply_add_node(
    state: WorldState,
    event: AddNodeEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Add one node; interpolate state from *event.connect_to* neighbours."""
    n = state.n_nodes
    nf = state.n_features
    nd = state.geometry.n_dims

    # New position
    if event.position and len(event.position) == nd:
        new_pos = np.array(event.position, dtype=np.float64)
    elif event.connect_to:
        new_pos = state.geometry.positions[list(event.connect_to)].mean(axis=0)
    else:
        new_pos = np.zeros(nd)

    # New field: interpolate from neighbours if available
    if event.initial_features and len(event.initial_features) == nf:
        new_field = np.array(event.initial_features, dtype=np.float64)
        introduced = float(np.abs(new_field).sum())
    elif event.connect_to:
        new_field = state.field_state.values[list(event.connect_to)].mean(axis=0)
        introduced = 0.0
    else:
        new_field = np.zeros(nf)
        introduced = 0.0

    prior_mass = float(np.abs(state.field_state.values).sum())

    # Extended field
    new_field_vals = np.vstack([state.field_state.values, new_field[None, :]])

    # Extended positions and distances
    all_pos = np.vstack([state.geometry.positions, new_pos[None, :]])
    diff = all_pos[:, None, :] - all_pos[None, :, :]
    new_distances = np.sqrt((diff ** 2).sum(axis=-1))

    # Extended topology: add row/col of zeros, then add edges to neighbours
    adj_old = state.topology.adjacency
    new_n = n + 1
    adj_new = sp.lil_matrix((new_n, new_n), dtype=np.float64)
    adj_new[:n, :n] = adj_old
    for nb in event.connect_to:
        if 0 <= nb < n:
            adj_new[n, nb] = 1.0
            adj_new[nb, n] = 1.0  # symmetric by default

    # Extended per-node vectors
    def _extend(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        return np.append(arr, fill)

    def _extend2(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
        return np.vstack([arr, np.full((1, arr.shape[1]), fill)])

    # Build new sub-states
    new_field_state = FieldState(new_field_vals)
    new_geometry = GeometryState(all_pos, new_distances)
    new_topology = TopologyState.from_adjacency(
        adj_new.tocsr(),
        channel_ids=state.topology.channel_ids,
    )

    new_attention = _extend(state.cognition.attention, 1.0 / new_n)
    new_attention /= new_attention.sum()
    new_predictions = _extend2(state.cognition.predictions, 0.0)

    new_parent = dict(state.hierarchy.parent)
    new_parent[n] = -1
    new_children = dict(state.hierarchy.children)
    new_children[n] = ()
    new_levels = np.append(state.hierarchy.levels, 0)

    new_guidance_bias = _extend2(state.guidance.bias, 0.0)
    new_guidance_priority = _extend(state.guidance.priority, 1.0 / new_n)

    new_epistemic = _extend(state.uncertainty.epistemic, 1.0)  # high uncertainty for new node
    new_aleatoric = _extend(state.uncertainty.aleatoric, 0.1)
    new_stress_regions = _extend(state.stress.regions, 0.0)

    new_compute = _extend(state.resources.compute, 1.0 / new_n)
    new_mem = _extend(state.resources.memory_alloc, 1.0 / new_n)

    next_state = WorldState(
        field_state=new_field_state,
        geometry=new_geometry,
        topology=new_topology,
        memory=state.memory,
        cognition=CognitionState(new_attention, new_predictions),
        hierarchy=HierarchyState(new_parent, new_children, new_levels),
        guidance=GuidanceState(new_guidance_bias, new_guidance_priority),
        uncertainty=UncertaintyState(new_epistemic, new_aleatoric),
        stress=StressState(state.stress.level, new_stress_regions),
        resources=ResourceState(new_compute, new_mem, state.resources.budget),
        goals=state.goals,
        metadata=state.metadata,
    )

    post_mass = float(np.abs(next_state.field_state.values).sum())
    diag = TransportDiagnostics(
        event_kind="add_node",
        transported_mass=prior_mass,
        introduced_mass=introduced,
        lost_mass=0.0,
        continuity_error=0.0,
        conservation_residual=abs(post_mass - prior_mass - introduced),
        unmapped_references=0,
    )
    return next_state, diag


def apply_remove_node(
    state: WorldState,
    event: RemoveNodeEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Remove node *event.node_index*, redistributing its field mass."""
    idx = event.node_index
    n = state.n_nodes
    nf = state.n_features

    removed_field = state.field_state.values[idx].copy()
    removed_mass = float(np.abs(removed_field).sum())

    # Keep all other nodes
    keep = [i for i in range(n) if i != idx]
    new_n = len(keep)

    # Re-index mapping old→new
    old_to_new: dict[int, int] = {old: new for new, old in enumerate(keep)}

    # Field: redistribute removed mass to neighbours
    adj = state.topology.adjacency
    neighbours = adj[idx].nonzero()[1]
    new_field = state.field_state.values[keep].copy()
    if len(neighbours) > 0:
        nb_keep = [nb for nb in neighbours if nb != idx]
        for nb in nb_keep:
            if nb in old_to_new:
                new_idx = old_to_new[nb]
                new_field[new_idx] += removed_field / max(len(nb_keep), 1)

    prior_total = float(np.abs(state.field_state.values).sum())

    # Positions and distances
    new_pos = state.geometry.positions[keep]
    diff = new_pos[:, None, :] - new_pos[None, :, :]
    new_dist = np.sqrt((diff ** 2).sum(axis=-1))

    # Topology: remove row/col, re-index
    adj_lil = adj.tolil()
    adj_arr = np.array(adj_lil.toarray())[np.ix_(keep, keep)]
    new_adj = sp.csr_matrix(adj_arr)

    # Per-node arrays
    def _shrink(arr: np.ndarray) -> np.ndarray:
        return arr[keep]

    def _shrink2(arr: np.ndarray) -> np.ndarray:
        return arr[keep]

    new_attention = _shrink(state.cognition.attention)
    s = new_attention.sum()
    if s > 0:
        new_attention = new_attention / s

    new_predictions = _shrink2(state.cognition.predictions)
    new_guidance_bias = _shrink2(state.guidance.bias)
    new_guidance_priority = _shrink(state.guidance.priority)
    new_epistemic = _shrink(state.uncertainty.epistemic)
    new_aleatoric = _shrink(state.uncertainty.aleatoric)
    new_stress_regions = _shrink(state.stress.regions)
    new_compute = _shrink(state.resources.compute)
    new_mem = _shrink(state.resources.memory_alloc)
    new_levels = state.hierarchy.levels[keep]

    # Hierarchy: re-map parent/children
    new_parent: dict[int, int] = {}
    new_children: dict[int, tuple[int, ...]] = {}
    unmapped = 0
    for old, new in old_to_new.items():
        p = state.hierarchy.parent.get(old, -1)
        if p == idx or p not in old_to_new:
            new_parent[new] = -1
            if p == idx:
                unmapped += 1
        else:
            new_parent[new] = old_to_new[p]
        ch = state.hierarchy.children.get(old, ())
        new_children[new] = tuple(old_to_new[c] for c in ch if c in old_to_new)

    next_state = WorldState(
        field_state=FieldState(new_field),
        geometry=GeometryState(new_pos, new_dist),
        topology=TopologyState.from_adjacency(new_adj, channel_ids=state.topology.channel_ids),
        memory=state.memory,
        cognition=CognitionState(new_attention, new_predictions),
        hierarchy=HierarchyState(new_parent, new_children, new_levels),
        guidance=GuidanceState(new_guidance_bias, new_guidance_priority),
        uncertainty=UncertaintyState(new_epistemic, new_aleatoric),
        stress=StressState(state.stress.level, new_stress_regions),
        resources=ResourceState(new_compute, new_mem, state.resources.budget),
        goals=state.goals,
        metadata=state.metadata,
    )

    post_total = float(np.abs(next_state.field_state.values).sum())
    diag = TransportDiagnostics(
        event_kind="remove_node",
        transported_mass=prior_total - removed_mass,
        introduced_mass=0.0,
        lost_mass=removed_mass,
        continuity_error=0.0,
        conservation_residual=abs(post_total - (prior_total - removed_mass)),
        unmapped_references=unmapped,
    )
    return next_state, diag


def apply_add_edge(
    state: WorldState,
    event: AddEdgeEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Add a directed (or symmetric) edge to the topology."""
    adj = state.topology.adjacency.tolil()
    adj[event.src, event.dst] = event.weight
    adj[event.dst, event.src] = event.weight  # symmetric by default
    new_topology = TopologyState.from_adjacency(
        adj.tocsr(), channel_ids=state.topology.channel_ids
    )
    next_state = WorldState(
        field_state=state.field_state,
        geometry=state.geometry,
        topology=new_topology,
        memory=state.memory,
        cognition=state.cognition,
        hierarchy=state.hierarchy,
        guidance=state.guidance,
        uncertainty=state.uncertainty,
        stress=state.stress,
        resources=state.resources,
        goals=state.goals,
        metadata=state.metadata,
    )
    diag = TransportDiagnostics(
        event_kind="add_edge",
        transported_mass=0.0,
        introduced_mass=0.0,
        lost_mass=0.0,
        continuity_error=0.0,
        conservation_residual=0.0,
        unmapped_references=0,
        extra={"src": event.src, "dst": event.dst, "weight": event.weight},
    )
    return next_state, diag


def apply_remove_edge(
    state: WorldState,
    event: RemoveEdgeEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Remove a directed edge from the topology."""
    adj = state.topology.adjacency.tolil()
    adj[event.src, event.dst] = 0.0
    adj[event.dst, event.src] = 0.0
    adj_csr = adj.tocsr()
    adj_csr.eliminate_zeros()  # in-place; returns None in scipy
    new_topology = TopologyState.from_adjacency(
        adj_csr, channel_ids=state.topology.channel_ids
    )
    next_state = WorldState(
        field_state=state.field_state,
        geometry=state.geometry,
        topology=new_topology,
        memory=state.memory,
        cognition=state.cognition,
        hierarchy=state.hierarchy,
        guidance=state.guidance,
        uncertainty=state.uncertainty,
        stress=state.stress,
        resources=state.resources,
        goals=state.goals,
        metadata=state.metadata,
    )
    diag = TransportDiagnostics(
        event_kind="remove_edge",
        transported_mass=0.0,
        introduced_mass=0.0,
        lost_mass=0.0,
        continuity_error=0.0,
        conservation_residual=0.0,
        unmapped_references=0,
        extra={"src": event.src, "dst": event.dst},
    )
    return next_state, diag


def apply_activate_channel(
    state: WorldState,
    event: ActivateChannelEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Add a channel to the topology's active channel set."""
    new_channels = state.topology.channel_ids | frozenset([event.channel_id])
    adj_data, adj_idx, adj_iptr = (
        state.topology._adj_data,
        state.topology._adj_indices,
        state.topology._adj_indptr,
    )
    new_topology = TopologyState(
        n_nodes=state.topology.n_nodes,
        _adj_data=adj_data,
        _adj_indices=adj_idx,
        _adj_indptr=adj_iptr,
        channel_ids=new_channels,
    )
    next_state = WorldState(
        field_state=state.field_state,
        geometry=state.geometry,
        topology=new_topology,
        memory=state.memory,
        cognition=state.cognition,
        hierarchy=state.hierarchy,
        guidance=state.guidance,
        uncertainty=state.uncertainty,
        stress=state.stress,
        resources=state.resources,
        goals=state.goals,
        metadata=state.metadata,
    )
    diag = TransportDiagnostics(
        event_kind="activate_channel",
        transported_mass=0.0, introduced_mass=0.0, lost_mass=0.0,
        continuity_error=0.0, conservation_residual=0.0, unmapped_references=0,
        extra={"channel_id": event.channel_id},
    )
    return next_state, diag


def apply_deactivate_channel(
    state: WorldState,
    event: DeactivateChannelEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Remove a channel from the topology's active channel set."""
    new_channels = state.topology.channel_ids - frozenset([event.channel_id])
    adj_data, adj_idx, adj_iptr = (
        state.topology._adj_data,
        state.topology._adj_indices,
        state.topology._adj_indptr,
    )
    new_topology = TopologyState(
        n_nodes=state.topology.n_nodes,
        _adj_data=adj_data,
        _adj_indices=adj_idx,
        _adj_indptr=adj_iptr,
        channel_ids=new_channels,
    )
    next_state = WorldState(
        field_state=state.field_state,
        geometry=state.geometry,
        topology=new_topology,
        memory=state.memory,
        cognition=state.cognition,
        hierarchy=state.hierarchy,
        guidance=state.guidance,
        uncertainty=state.uncertainty,
        stress=state.stress,
        resources=state.resources,
        goals=state.goals,
        metadata=state.metadata,
    )
    diag = TransportDiagnostics(
        event_kind="deactivate_channel",
        transported_mass=0.0, introduced_mass=0.0, lost_mass=0.0,
        continuity_error=0.0, conservation_residual=0.0, unmapped_references=0,
        extra={"channel_id": event.channel_id},
    )
    return next_state, diag


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_event(
    state: WorldState,
    event: StructuralEvent,
) -> tuple[WorldState, TransportDiagnostics]:
    """Dispatch *event* to the appropriate apply function."""
    if isinstance(event, AddNodeEvent):
        return apply_add_node(state, event)
    if isinstance(event, RemoveNodeEvent):
        return apply_remove_node(state, event)
    if isinstance(event, AddEdgeEvent):
        return apply_add_edge(state, event)
    if isinstance(event, RemoveEdgeEvent):
        return apply_remove_edge(state, event)
    if isinstance(event, ActivateChannelEvent):
        return apply_activate_channel(state, event)
    if isinstance(event, DeactivateChannelEvent):
        return apply_deactivate_channel(state, event)
    # Unknown or unimplemented event: no-op with zero transport
    diag = TransportDiagnostics(
        event_kind=event.kind.name,
        transported_mass=0.0, introduced_mass=0.0, lost_mass=0.0,
        continuity_error=0.0, conservation_residual=0.0, unmapped_references=0,
        extra={"note": "event type not implemented; treated as no-op"},
    )
    return state, diag
