"""Memory mechanisms: Consolidation and Retrieval.

Memory Consolidation strengthens traces for recently active nodes.
Memory Retrieval biases the field toward the most-similar stored trace.
"""
from __future__ import annotations

import numpy as np

from hrm_ecology.contribution import Contribution, FieldDelta, MemoryDelta
from hrm_ecology.mechanisms.base import BaseMechanism, Observations
from hrm_ecology.state import WorldState


class MemoryConsolidationMechanism(BaseMechanism):
    """Strengthen memory traces for nodes whose current field is active.

    Parameters
    ----------
    gamma:
        Consolidation rate in (0, 1].
    decay:
        Salience decay per step (multiplicative).
    """

    def __init__(self, gamma: float = 0.1, decay: float = 0.99) -> None:
        super().__init__(
            mechanism_id="memory_consolidation",
            version="0.1.0",
            description="Writes current field patterns into memory traces.",
        )
        self.gamma = gamma
        self.decay = decay

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        memory = snapshot.memory
        field_vals = snapshot.field_state.values  # (n_nodes, n_features)
        n_traces = memory.n_traces
        n_features = snapshot.n_features

        # Compute mean field activation (one scalar per node)
        node_activity = np.abs(field_vals).mean(axis=1)  # (n_nodes,)

        # Find the least-salient trace slot for potential overwrite
        least_salient_idx = int(np.argmin(memory.salience))

        # Mean field over all nodes as the pattern to consolidate
        mean_pattern = field_vals.mean(axis=0)  # (n_features,)
        mean_activity = float(node_activity.mean())

        # Propose trace update: move the chosen slot toward mean_pattern
        delta_traces = np.zeros_like(memory.traces)  # (n_traces, n_features)
        delta_traces[least_salient_idx] = self.gamma * (
            mean_pattern - memory.traces[least_salient_idx]
        )

        # Salience: decay all, boost the consolidated one
        delta_salience = np.zeros(n_traces)
        delta_salience += (self.decay - 1.0) * memory.salience  # decay
        delta_salience[least_salient_idx] += self.gamma * mean_activity

        activation = min(mean_activity, 1.0)

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=0.7,
            activation=activation,
            estimated_cost=float(np.abs(delta_traces).sum()),
            rationale=(
                f"Consolidating pattern into trace {least_salient_idx}; "
                f"activity={mean_activity:.4g}"
            ),
            memory_delta=MemoryDelta(
                delta_traces=delta_traces,
                delta_salience=delta_salience,
                weight=1.0,
            ),
            diagnostics={
                "consolidated_slot": least_salient_idx,
                "mean_activity": mean_activity,
                "gamma": self.gamma,
            },
        )


class MemoryRetrievalMechanism(BaseMechanism):
    """Bias the field toward the most-similar stored memory trace.

    Parameters
    ----------
    beta:
        Retrieval strength in [0, 1].
    threshold:
        Minimum similarity to trigger retrieval.
    """

    def __init__(self, beta: float = 0.15, threshold: float = 0.1) -> None:
        super().__init__(
            mechanism_id="memory_retrieval",
            version="0.1.0",
            description="Biases field toward best-matching memory trace.",
        )
        self.beta = beta
        self.threshold = threshold

    def propose(
        self,
        snapshot: WorldState,
        observations: Observations,
        rng: np.random.Generator,
    ) -> Contribution:
        memory = snapshot.memory
        field_vals = snapshot.field_state.values  # (n_nodes, n_features)

        if memory.n_traces == 0 or float(memory.salience.max()) < 1e-9:
            return self._null_contribution("no salient memory traces")

        # Global mean field pattern to compare against traces
        query = field_vals.mean(axis=0)  # (n_features,)

        # Cosine-like similarity between query and each trace
        trace_norms = np.linalg.norm(memory.traces, axis=1) + 1e-9
        query_norm = float(np.linalg.norm(query)) + 1e-9
        similarities = (memory.traces @ query) / (trace_norms * query_norm)

        # Weight by salience
        weighted_sim = similarities * memory.salience
        best_idx = int(np.argmax(weighted_sim))
        best_sim = float(weighted_sim[best_idx])

        if best_sim < self.threshold:
            return self._null_contribution(
                f"best similarity {best_sim:.4g} < threshold {self.threshold:.4g}"
            )

        # Broadcast best trace back as a field bias
        retrieved = memory.traces[best_idx]  # (n_features,)
        field_delta = self.beta * best_sim * (
            retrieved[None, :] - field_vals
        )  # (n_nodes, n_features)

        activation = min(best_sim, 1.0)

        return Contribution(
            mechanism_id=self.info.mechanism_id,
            confidence=float(similarities[best_idx]),
            activation=activation,
            estimated_cost=float(np.abs(field_delta).sum()),
            rationale=(
                f"Retrieving trace {best_idx}; similarity={best_sim:.4g}; "
                f"salience={memory.salience[best_idx]:.4g}"
            ),
            field_delta=FieldDelta(delta=field_delta, weight=best_sim),
            diagnostics={
                "best_trace_idx": best_idx,
                "best_similarity": best_sim,
                "beta": self.beta,
            },
        )
