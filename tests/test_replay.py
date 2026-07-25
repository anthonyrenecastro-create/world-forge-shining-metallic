"""Tests for deterministic replay."""
import numpy as np
import pytest

from hrm_ecology.engine import Engine
from hrm_ecology.mechanisms.base import Observations
from hrm_ecology.mechanisms.diffusion import DiffusionMechanism
from hrm_ecology.mechanisms.memory import MemoryConsolidationMechanism
from hrm_ecology.mechanisms.prediction import PredictionErrorCorrectionMechanism
from hrm_ecology.replay import ReplayEngine, ReplayReport
from hrm_ecology.safety import SafetyConfig
from hrm_ecology.state import WorldState


def make_state(n: int = 5, f: int = 3, seed: int = 42) -> WorldState:
    return WorldState.create_initial(n, f, seed=seed)


def make_obs(step: int = 0) -> Observations:
    return Observations(
        sensor_data=np.zeros((4, 2)),
        sensor_mask=np.ones(4, dtype=bool),
        step=step,
    )


def make_mechanisms():
    return [
        DiffusionMechanism(alpha=0.05),
        PredictionErrorCorrectionMechanism(eta=0.1),
        MemoryConsolidationMechanism(gamma=0.05),
    ]


class TestDeterministicReplay:
    def _run_original(self, n_steps: int = 5, seed: int = 42):
        """Run original and return (initial_state, observations, final_state, engine)."""
        state = make_state(seed=seed)
        initial_state = state
        mechanisms = make_mechanisms()
        config = SafetyConfig()
        engine = Engine(mechanisms=mechanisms, safety_config=config)
        observations = []
        for i in range(n_steps):
            obs = make_obs(i)
            observations.append(obs)
            result = engine.step(state, obs, seed_override=seed)
            state = result.next_state
        return initial_state, observations, state, engine

    def test_replay_matches_original(self):
        """Replaying with the same seed should produce identical state hashes."""
        n_steps = 5
        initial_state, observations, final_state, engine = self._run_original(n_steps)

        replay_engine = ReplayEngine(
            mechanisms=make_mechanisms(),
            safety_config=SafetyConfig(),
        )
        records = replay_engine.build_replay_records(engine.ledger, observations)
        report = replay_engine.replay(initial_state, records)

        assert report.success, (
            f"Replay failed at steps: {report.mismatched_steps}\n"
            f"Errors: {report.hash_errors}"
        )
        assert report.total_steps == n_steps
        assert report.matching_steps == n_steps

    def test_replay_report_structure(self):
        initial_state, observations, _, engine = self._run_original(3)
        replay_engine = ReplayEngine(mechanisms=make_mechanisms())
        records = replay_engine.build_replay_records(engine.ledger, observations)
        report = replay_engine.replay(initial_state, records)
        assert isinstance(report.total_steps, int)
        assert isinstance(report.mismatched_steps, list)
        assert isinstance(report.hash_errors, list)
        assert isinstance(report.success, bool)

    def test_wrong_initial_state_fails(self):
        """Providing wrong initial state should cause hash mismatch."""
        n_steps = 3
        initial_state, observations, _, engine = self._run_original(n_steps)

        wrong_initial = make_state(seed=999)  # Different seed → different state

        replay_engine = ReplayEngine(mechanisms=make_mechanisms())
        records = replay_engine.build_replay_records(engine.ledger, observations)
        report = replay_engine.replay(wrong_initial, records)

        # The very first step should fail (prior hash mismatch)
        assert not report.success

    def test_same_seed_reproducible(self):
        """Two independent runs with the same seed must produce identical hashes."""
        n_steps = 4
        seed = 77

        state1 = make_state(seed=seed)
        state2 = make_state(seed=seed)
        assert state1.content_hash() == state2.content_hash()

        mechanisms1 = make_mechanisms()
        mechanisms2 = make_mechanisms()
        engine1 = Engine(mechanisms=mechanisms1, safety_config=SafetyConfig())
        engine2 = Engine(mechanisms=mechanisms2, safety_config=SafetyConfig())

        for i in range(n_steps):
            obs = make_obs(i)
            r1 = engine1.step(state1, obs, seed_override=seed)
            r2 = engine2.step(state2, obs, seed_override=seed)
            assert r1.next_state.content_hash() == r2.next_state.content_hash(), (
                f"Diverged at step {i}"
            )
            state1 = r1.next_state
            state2 = r2.next_state
