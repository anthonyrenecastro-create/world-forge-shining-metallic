"""Immutable HRM world state definitions.

Every field in every sub-state is immutable: numpy arrays are read-only,
dataclasses are frozen, and nested objects follow the same contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ro(arr: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Return an immutable (read-only) copy of *arr* cast to *dtype*."""
    out = np.array(arr, dtype=dtype)
    out.flags.writeable = False
    return out


def _ro_int(arr: np.ndarray) -> np.ndarray:
    return _ro(arr, dtype=np.int64)


def _freeze_csr(m: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Serialise a CSR matrix into immutable components."""
    m = m.tocsr()
    return (
        _ro(m.data),
        _ro_int(m.indices),
        _ro_int(m.indptr),
        m.shape,
    )


def _thaw_csr(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: tuple[int, int],
) -> sp.csr_matrix:
    """Reconstruct a CSR matrix from frozen components."""
    return sp.csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=shape)


# ---------------------------------------------------------------------------
# Sub-state objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldState:
    """Continuous field activations across computational nodes.

    Shape: ``(n_nodes, n_features)``
    """
    values: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _ro(self.values))

    @property
    def n_nodes(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1]) if self.values.ndim > 1 else 1

    def content_hash(self) -> str:
        return hashlib.sha256(self.values.tobytes()).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldState):
            return NotImplemented
        return np.array_equal(self.values, other.values)


@dataclass(frozen=True)
class GeometryState:
    """Spatial embedding of computational nodes.

    ``positions``: ``(n_nodes, n_dims)`` – coordinates in embedding space.
    ``distances``: ``(n_nodes, n_nodes)`` – pairwise distance matrix.
    """
    positions: np.ndarray = field(hash=False, compare=False)
    distances: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _ro(self.positions))
        object.__setattr__(self, "distances", _ro(self.distances))

    @property
    def n_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_dims(self) -> int:
        return int(self.positions.shape[1]) if self.positions.ndim > 1 else 1

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.positions.tobytes() + self.distances.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeometryState):
            return NotImplemented
        return np.array_equal(self.positions, other.positions) and np.array_equal(
            self.distances, other.distances
        )


@dataclass(frozen=True)
class TopologyState:
    """Connectivity structure stored as an immutable CSR decomposition.

    ``n_nodes``: number of nodes.
    ``channel_ids``: frozenset of active channel names.
    The adjacency matrix is reconstructed on demand via :meth:`adjacency`.
    """
    n_nodes: int
    _adj_data: np.ndarray = field(hash=False, compare=False, repr=False)
    _adj_indices: np.ndarray = field(hash=False, compare=False, repr=False)
    _adj_indptr: np.ndarray = field(hash=False, compare=False, repr=False)
    channel_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_adj_data", _ro(self._adj_data))
        object.__setattr__(self, "_adj_indices", _ro_int(self._adj_indices))
        object.__setattr__(self, "_adj_indptr", _ro_int(self._adj_indptr))

    @classmethod
    def from_adjacency(
        cls,
        adj: sp.spmatrix,
        channel_ids: frozenset[str] | None = None,
    ) -> "TopologyState":
        data, indices, indptr, shape = _freeze_csr(adj.tocsr())
        return cls(
            n_nodes=shape[0],
            _adj_data=data,
            _adj_indices=indices,
            _adj_indptr=indptr,
            channel_ids=channel_ids or frozenset(),
        )

    @property
    def adjacency(self) -> sp.csr_matrix:
        return _thaw_csr(
            self._adj_data, self._adj_indices, self._adj_indptr,
            (self.n_nodes, self.n_nodes),
        )

    def content_hash(self) -> str:
        return hashlib.sha256(
            self._adj_data.tobytes()
            + self._adj_indices.tobytes()
            + self._adj_indptr.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopologyState):
            return NotImplemented
        return (
            self.n_nodes == other.n_nodes
            and self.channel_ids == other.channel_ids
            and np.array_equal(self._adj_data, other._adj_data)
            and np.array_equal(self._adj_indices, other._adj_indices)
            and np.array_equal(self._adj_indptr, other._adj_indptr)
        )


@dataclass(frozen=True)
class MemoryState:
    """Long-term memory traces and salience.

    ``traces``: ``(n_traces, n_features)`` – stored memory patterns.
    ``salience``: ``(n_traces,)`` – importance weight of each trace.
    ``timestamps``: ``(n_traces,)`` – step at which each trace was written.
    """
    traces: np.ndarray = field(hash=False, compare=False)
    salience: np.ndarray = field(hash=False, compare=False)
    timestamps: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", _ro(self.traces))
        object.__setattr__(self, "salience", _ro(self.salience))
        object.__setattr__(self, "timestamps", _ro(self.timestamps))

    @property
    def n_traces(self) -> int:
        return int(self.traces.shape[0])

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.traces.tobytes() + self.salience.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryState):
            return NotImplemented
        return (
            np.array_equal(self.traces, other.traces)
            and np.array_equal(self.salience, other.salience)
            and np.array_equal(self.timestamps, other.timestamps)
        )


@dataclass(frozen=True)
class CognitionState:
    """Current attentional focus and predictive beliefs.

    ``attention``: ``(n_nodes,)`` – per-node attention weight in [0, 1].
    ``predictions``: ``(n_nodes, n_features)`` – predicted next field values.
    """
    attention: np.ndarray = field(hash=False, compare=False)
    predictions: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention", _ro(self.attention))
        object.__setattr__(self, "predictions", _ro(self.predictions))

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.attention.tobytes() + self.predictions.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CognitionState):
            return NotImplemented
        return np.array_equal(self.attention, other.attention) and np.array_equal(
            self.predictions, other.predictions
        )


@dataclass(frozen=True)
class HierarchyState:
    """Multi-level hierarchy of computational regions.

    ``parent``: node index → parent index (``-1`` means root).
    ``children``: node index → tuple of child indices.
    ``levels``: ``(n_nodes,)`` – hierarchy depth of each node.
    """
    parent: dict[int, int]
    children: dict[int, tuple[int, ...]]
    levels: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", _ro_int(self.levels))
        # Make dicts immutable (values already immutable for int / tuple)
        object.__setattr__(self, "parent", dict(self.parent))
        object.__setattr__(self, "children", {k: tuple(v) for k, v in self.children.items()})

    def content_hash(self) -> str:
        parent_bytes = str(sorted(self.parent.items())).encode()
        return hashlib.sha256(
            parent_bytes + self.levels.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HierarchyState):
            return NotImplemented
        return (
            self.parent == other.parent
            and self.children == other.children
            and np.array_equal(self.levels, other.levels)
        )


@dataclass(frozen=True)
class GuidanceState:
    """External top-down guidance signals.

    ``bias``: ``(n_nodes, n_features)`` – additive guidance bias per node.
    ``priority``: ``(n_nodes,)`` – relative priority weight per node.
    """
    bias: np.ndarray = field(hash=False, compare=False)
    priority: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bias", _ro(self.bias))
        object.__setattr__(self, "priority", _ro(self.priority))

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.bias.tobytes() + self.priority.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GuidanceState):
            return NotImplemented
        return np.array_equal(self.bias, other.bias) and np.array_equal(
            self.priority, other.priority
        )


@dataclass(frozen=True)
class UncertaintyState:
    """Two-component uncertainty.

    ``epistemic``: ``(n_nodes,)`` – model uncertainty (reducible with data).
    ``aleatoric``: ``(n_nodes,)`` – irreducible observational noise.
    """
    epistemic: np.ndarray = field(hash=False, compare=False)
    aleatoric: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "epistemic", _ro(self.epistemic))
        object.__setattr__(self, "aleatoric", _ro(self.aleatoric))

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.epistemic.tobytes() + self.aleatoric.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UncertaintyState):
            return NotImplemented
        return np.array_equal(self.epistemic, other.epistemic) and np.array_equal(
            self.aleatoric, other.aleatoric
        )


@dataclass(frozen=True)
class StressState:
    """Scalar and per-region stress indicators.

    ``level``: global stress in [0, 1].
    ``regions``: ``(n_nodes,)`` – per-node stress level.
    """
    level: float
    regions: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", _ro(self.regions))
        if not (0.0 <= self.level <= 1.0):
            raise ValueError(f"Stress level must be in [0, 1]; got {self.level}")

    def content_hash(self) -> str:
        return hashlib.sha256(
            str(self.level).encode() + self.regions.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StressState):
            return NotImplemented
        return self.level == other.level and np.array_equal(self.regions, other.regions)


@dataclass(frozen=True)
class ResourceBudget:
    """Hard limits on computational resources per step."""
    max_compute_units: float = 1000.0
    max_memory_units: float = 1000.0
    max_structural_events: int = 10
    structural_event_budget: int = 10

    def __post_init__(self) -> None:
        if self.max_compute_units <= 0:
            raise ValueError("max_compute_units must be positive")
        if self.max_memory_units <= 0:
            raise ValueError("max_memory_units must be positive")


@dataclass(frozen=True)
class ResourceState:
    """Endogenous resource allocations.

    ``compute``: ``(n_nodes,)`` – fractional compute allocation per node.
    ``memory_alloc``: ``(n_nodes,)`` – fractional memory allocation per node.
    ``budget``: hard resource limits for this step.
    """
    compute: np.ndarray = field(hash=False, compare=False)
    memory_alloc: np.ndarray = field(hash=False, compare=False)
    budget: ResourceBudget = field(default_factory=ResourceBudget)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute", _ro(self.compute))
        object.__setattr__(self, "memory_alloc", _ro(self.memory_alloc))

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.compute.tobytes() + self.memory_alloc.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceState):
            return NotImplemented
        return (
            np.array_equal(self.compute, other.compute)
            and np.array_equal(self.memory_alloc, other.memory_alloc)
            and self.budget == other.budget
        )


@dataclass(frozen=True)
class Goal:
    """A single goal specification."""
    goal_id: str
    description: str
    target_pattern: np.ndarray = field(hash=False, compare=False)
    priority: float = 1.0
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_pattern", _ro(self.target_pattern))

    def content_hash(self) -> str:
        return hashlib.sha256(
            self.goal_id.encode() + self.target_pattern.tobytes()
        ).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.goal_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Goal):
            return NotImplemented
        return self.goal_id == other.goal_id and np.array_equal(
            self.target_pattern, other.target_pattern
        )


@dataclass(frozen=True)
class GoalState:
    """Active goals and their per-node priority projections.

    ``goals``: immutable tuple of :class:`Goal` objects.
    ``priorities``: ``(n_nodes,)`` – aggregated priority weight per node.
    """
    goals: tuple[Goal, ...]
    priorities: np.ndarray = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "goals", tuple(self.goals))
        object.__setattr__(self, "priorities", _ro(self.priorities))

    def content_hash(self) -> str:
        goal_bytes = b"".join(g.content_hash().encode() for g in self.goals)
        return hashlib.sha256(goal_bytes + self.priorities.tobytes()).hexdigest()[:16]

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoalState):
            return NotImplemented
        return self.goals == other.goals and np.array_equal(
            self.priorities, other.priorities
        )


@dataclass(frozen=True)
class MetadataState:
    """Transition metadata for traceability and replay."""
    step: int
    timestamp: float
    seed: int
    version: str = "0.1.0"
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", dict(self.extra))


# ---------------------------------------------------------------------------
# Top-level world state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldState:
    """The single authoritative, immutable world state ``Xₜ``.

    No mechanism may mutate this object.  Mechanisms only propose changes
    via :class:`~hrm_ecology.contribution.Contribution`; the engine's
    ``commit`` function assembles ``X_{t+1}``.
    """
    field_state: FieldState
    geometry: GeometryState
    topology: TopologyState
    memory: MemoryState
    cognition: CognitionState
    hierarchy: HierarchyState
    guidance: GuidanceState
    uncertainty: UncertaintyState
    stress: StressState
    resources: ResourceState
    goals: GoalState
    metadata: MetadataState

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.field_state.n_nodes

    @property
    def n_features(self) -> int:
        return self.field_state.n_features

    # ------------------------------------------------------------------
    # Content identity
    # ------------------------------------------------------------------

    def content_hash(self) -> str:
        """Deterministic hash of the entire world state."""
        components = "".join([
            self.field_state.content_hash(),
            self.geometry.content_hash(),
            self.topology.content_hash(),
            self.memory.content_hash(),
            self.cognition.content_hash(),
            self.hierarchy.content_hash(),
            self.guidance.content_hash(),
            self.uncertainty.content_hash(),
            self.stress.content_hash(),
            self.resources.content_hash(),
            self.goals.content_hash(),
            str(self.metadata.step),
            str(self.metadata.seed),
        ])
        return hashlib.sha256(components.encode()).hexdigest()

    def __hash__(self) -> int:
        return hash(self.content_hash())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldState):
            return NotImplemented
        return self.content_hash() == other.content_hash()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def create_initial(
        cls,
        n_nodes: int,
        n_features: int,
        n_dims: int = 2,
        n_traces: int = 16,
        seed: int = 0,
        budget: ResourceBudget | None = None,
    ) -> "WorldState":
        """Create a default zero-initialised world state."""
        rng = np.random.default_rng(seed)

        field_vals = np.zeros((n_nodes, n_features))
        positions = rng.standard_normal((n_nodes, n_dims))
        # Pairwise Euclidean distances
        diff = positions[:, None, :] - positions[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=-1))

        # Default topology: empty (no edges)
        adj = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

        traces = np.zeros((n_traces, n_features))
        salience = np.zeros(n_traces)
        timestamps = np.zeros(n_traces)

        attention = np.ones(n_nodes) / n_nodes
        predictions = np.zeros((n_nodes, n_features))

        parent = {i: -1 for i in range(n_nodes)}
        children: dict[int, tuple[int, ...]] = {i: () for i in range(n_nodes)}
        levels = np.zeros(n_nodes, dtype=np.int64)

        bias = np.zeros((n_nodes, n_features))
        priority = np.ones(n_nodes) / n_nodes

        epistemic = np.ones(n_nodes) * 0.5
        aleatoric = np.ones(n_nodes) * 0.1

        stress_regions = np.zeros(n_nodes)

        if budget is None:
            budget = ResourceBudget()

        compute = np.ones(n_nodes) / n_nodes
        mem_alloc = np.ones(n_nodes) / n_nodes

        default_goal_pattern = np.zeros(n_features)
        goals_tuple: tuple[Goal, ...] = ()
        goal_priorities = np.zeros(n_nodes)

        import time as _time

        return cls(
            field_state=FieldState(field_vals),
            geometry=GeometryState(positions, distances),
            topology=TopologyState.from_adjacency(adj),
            memory=MemoryState(traces, salience, timestamps),
            cognition=CognitionState(attention, predictions),
            hierarchy=HierarchyState(parent, children, levels),
            guidance=GuidanceState(bias, priority),
            uncertainty=UncertaintyState(epistemic, aleatoric),
            stress=StressState(0.0, stress_regions),
            resources=ResourceState(compute, mem_alloc, budget),
            goals=GoalState(goals_tuple, goal_priorities),
            metadata=MetadataState(
                step=0,
                timestamp=_time.time(),
                seed=seed,
            ),
        )
