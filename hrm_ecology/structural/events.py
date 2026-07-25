"""Structural events: explicit representations of topological adaptation.

Structural events alter the computational substrate itself—adding or
removing nodes and edges, splitting or merging regions, etc.  Every event
requires validation, arbitration, and deterministic ordering before it
may be applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventKind(Enum):
    ADD_NODE = auto()
    REMOVE_NODE = auto()
    ADD_EDGE = auto()
    REMOVE_EDGE = auto()
    SPLIT_REGION = auto()
    MERGE_REGION = auto()
    ACTIVATE_CHANNEL = auto()
    DEACTIVATE_CHANNEL = auto()
    CREATE_CHILD_MANIFOLD = auto()
    PRUNE_CHILD_MANIFOLD = auto()
    HIERARCHY_REASSIGNMENT = auto()
    EDGE_HEALING = auto()
    TOPOLOGY_REPAIR = auto()


@dataclass(frozen=True)
class StructuralEvent:
    """Base representation of a structural event proposal.

    All structural events are proposals until accepted by the arbitrator.
    Only deterministically-ordered, validated events are applied.

    Attributes
    ----------
    kind:
        The event category.
    proposer_id:
        Mechanism that proposed this event.
    priority:
        Higher priority events are applied first (ties broken by kind order).
    cost:
        Estimated structural budget cost.
    rationale:
        Human-readable explanation.
    metadata:
        Arbitrary event-specific payload.
    """
    kind: EventKind
    proposer_id: str
    priority: float
    cost: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class AddNodeEvent(StructuralEvent):
    """Add a new node to the graph."""
    position: tuple[float, ...] = field(default_factory=tuple, hash=False, compare=False)
    initial_features: tuple[float, ...] = field(default_factory=tuple, hash=False, compare=False)
    connect_to: tuple[int, ...] = field(default_factory=tuple, hash=False, compare=False)

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass(frozen=True)
class RemoveNodeEvent(StructuralEvent):
    """Remove an existing node from the graph."""
    node_index: int = 0

    def validate(self, n_nodes: int) -> str | None:
        if not (0 <= self.node_index < n_nodes):
            return f"node_index {self.node_index} out of range [0, {n_nodes})"
        return None


@dataclass(frozen=True)
class AddEdgeEvent(StructuralEvent):
    """Add a directed edge from src to dst with given weight."""
    src: int = 0
    dst: int = 0
    weight: float = 1.0

    def validate(self, n_nodes: int) -> str | None:
        if not (0 <= self.src < n_nodes):
            return f"src {self.src} out of range"
        if not (0 <= self.dst < n_nodes):
            return f"dst {self.dst} out of range"
        if self.src == self.dst:
            return "self-loops not permitted"
        return None


@dataclass(frozen=True)
class RemoveEdgeEvent(StructuralEvent):
    """Remove a directed edge from src to dst."""
    src: int = 0
    dst: int = 0

    def validate(self, n_nodes: int) -> str | None:
        if not (0 <= self.src < n_nodes):
            return f"src {self.src} out of range"
        if not (0 <= self.dst < n_nodes):
            return f"dst {self.dst} out of range"
        return None


@dataclass(frozen=True)
class SplitRegionEvent(StructuralEvent):
    """Split the specified region (subset of nodes) into two sub-regions."""
    region_nodes: tuple[int, ...] = field(default_factory=tuple, hash=False, compare=False)
    split_ratio: float = 0.5  # fraction of nodes assigned to first sub-region

    def validate(self, n_nodes: int) -> str | None:
        if len(self.region_nodes) < 2:
            return "cannot split a region with fewer than 2 nodes"
        if any(not (0 <= n < n_nodes) for n in self.region_nodes):
            return "region contains out-of-range node indices"
        return None


@dataclass(frozen=True)
class MergeRegionEvent(StructuralEvent):
    """Merge two regions (each a set of nodes) into one."""
    region_a: tuple[int, ...] = field(default_factory=tuple, hash=False, compare=False)
    region_b: tuple[int, ...] = field(default_factory=tuple, hash=False, compare=False)

    def validate(self, n_nodes: int) -> str | None:
        all_nodes = set(self.region_a) | set(self.region_b)
        if any(not (0 <= n < n_nodes) for n in all_nodes):
            return "regions contain out-of-range node indices"
        if set(self.region_a) & set(self.region_b):
            return "regions must be disjoint"
        return None


@dataclass(frozen=True)
class ActivateChannelEvent(StructuralEvent):
    """Activate a named communication channel."""
    channel_id: str = ""


@dataclass(frozen=True)
class DeactivateChannelEvent(StructuralEvent):
    """Deactivate a named communication channel."""
    channel_id: str = ""

    def validate(self, active_channels: frozenset[str]) -> str | None:
        if self.channel_id not in active_channels:
            return f"channel '{self.channel_id}' is not active"
        return None


# ---------------------------------------------------------------------------
# Event ordering and validation
# ---------------------------------------------------------------------------

# Canonical priority order for resolving ties (lower = applied first)
_KIND_ORDER: dict[EventKind, int] = {
    EventKind.TOPOLOGY_REPAIR: 0,
    EventKind.EDGE_HEALING: 1,
    EventKind.REMOVE_NODE: 2,
    EventKind.REMOVE_EDGE: 3,
    EventKind.ADD_NODE: 4,
    EventKind.ADD_EDGE: 5,
    EventKind.SPLIT_REGION: 6,
    EventKind.MERGE_REGION: 7,
    EventKind.HIERARCHY_REASSIGNMENT: 8,
    EventKind.ACTIVATE_CHANNEL: 9,
    EventKind.DEACTIVATE_CHANNEL: 10,
    EventKind.CREATE_CHILD_MANIFOLD: 11,
    EventKind.PRUNE_CHILD_MANIFOLD: 12,
}


def sort_events(events: list[StructuralEvent]) -> list[StructuralEvent]:
    """Sort events into a deterministic application order.

    Primary key: descending priority (higher = first).
    Secondary key: canonical kind order (lower kind value = first).
    Tertiary key: proposer_id lexicographic order.
    """
    return sorted(
        events,
        key=lambda e: (
            -e.priority,
            _KIND_ORDER.get(e.kind, 99),
            e.proposer_id,
        ),
    )


def validate_event(event: StructuralEvent, n_nodes: int, channel_ids: frozenset[str]) -> str | None:
    """Return an error string if *event* is invalid, else ``None``."""
    if isinstance(event, RemoveNodeEvent):
        return event.validate(n_nodes)
    if isinstance(event, AddEdgeEvent):
        return event.validate(n_nodes)
    if isinstance(event, RemoveEdgeEvent):
        return event.validate(n_nodes)
    if isinstance(event, SplitRegionEvent):
        return event.validate(n_nodes)
    if isinstance(event, MergeRegionEvent):
        return event.validate(n_nodes)
    if isinstance(event, DeactivateChannelEvent):
        return event.validate(channel_ids)
    return None  # Unknown event subtypes pass through (treated as no-op)
