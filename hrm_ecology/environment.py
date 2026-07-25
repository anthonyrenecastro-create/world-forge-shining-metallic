"""Simulated adaptive environment for HRM Ecology.

Provides a non-stationary world with moving objects, sensor noise,
communication failures, resource scarcity, and structural damage.
The environment generates :class:`~hrm_ecology.mechanisms.base.Observations`
at each step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hrm_ecology.mechanisms.base import Observations


@dataclass
class EnvironmentConfig:
    """Configuration for the simulated environment."""
    n_sensors: int = 8
    n_objects: int = 4
    sensor_noise_std: float = 0.1
    failure_probability: float = 0.05     # per-sensor failure rate
    resource_scarcity: float = 0.0        # fraction of nodes with reduced resources
    topology_change_rate: float = 0.02    # prob. of a topology change event per step
    goal_change_interval: int = 50        # steps between goal changes
    damage_probability: float = 0.01      # prob. of structural damage per step


class SimulatedEnvironment:
    """A non-stationary simulated world.

    Objects move on sinusoidal trajectories.  Sensors observe object
    positions with Gaussian noise and occasional dropout.  The
    environment periodically changes goals and introduces damage.

    Parameters
    ----------
    config:
        Environment configuration.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        self._step = 0

        # Object state: position (n_objects, 2), velocity (n_objects, 2)
        self._positions = self.rng.uniform(-1.0, 1.0, (self.config.n_objects, 2))
        self._velocities = self.rng.uniform(-0.05, 0.05, (self.config.n_objects, 2))
        self._phases = self.rng.uniform(0, 2 * math.pi, self.config.n_objects)

        # Sensor positions fixed in a ring
        angles = np.linspace(0, 2 * math.pi, self.config.n_sensors, endpoint=False)
        self._sensor_positions = np.column_stack([np.cos(angles), np.sin(angles)])

    @property
    def step(self) -> int:
        return self._step

    def _update_objects(self) -> None:
        """Move objects on sinusoidal trajectories."""
        t = self._step
        for i in range(self.config.n_objects):
            # Sinusoidal motion superimposed on linear drift
            self._positions[i, 0] = (
                0.5 * math.sin(self._phases[i] + t * 0.07 * (i + 1))
                + 0.1 * self._velocities[i, 0] * t
            )
            self._positions[i, 1] = (
                0.5 * math.cos(self._phases[i] + t * 0.05 * (i + 1))
                + 0.1 * self._velocities[i, 1] * t
            )
            # Wrap positions
            self._positions[i] = np.clip(self._positions[i], -2.0, 2.0)

    def observe(self) -> Observations:
        """Generate one step of sensor observations."""
        self._update_objects()

        n_sensors = self.config.n_sensors
        n_objects = self.config.n_objects

        # Each sensor reads the distance to each object
        sensor_data = np.zeros((n_sensors, n_objects))
        for s in range(n_sensors):
            for o in range(n_objects):
                dist = float(np.linalg.norm(
                    self._sensor_positions[s] - self._positions[o]
                ))
                sensor_data[s, o] = dist

        # Add Gaussian noise
        noise = self.rng.normal(0.0, self.config.sensor_noise_std, sensor_data.shape)
        sensor_data = sensor_data + noise

        # Random sensor failures
        failure_mask = self.rng.random(n_sensors) < self.config.failure_probability
        sensor_mask = ~failure_mask  # True = valid

        obs = Observations(
            sensor_data=sensor_data,
            sensor_mask=sensor_mask,
            step=self._step,
            extra={
                "object_positions": self._positions.copy().tolist(),
                "n_failures": int(failure_mask.sum()),
            },
        )
        self._step += 1
        return obs

    def generate_sequence(self, n_steps: int) -> list[Observations]:
        """Generate a sequence of *n_steps* observation bundles."""
        return [self.observe() for _ in range(n_steps)]

    def introduce_damage(self, rng: np.random.Generator | None = None) -> bool:
        """Stochastically introduce structural damage.

        Returns True if damage was introduced this step.
        """
        r = rng or self.rng
        return bool(r.random() < self.config.damage_probability)
