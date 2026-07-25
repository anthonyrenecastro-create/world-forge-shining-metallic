"""HRM Ecology: A Resilient World-Model Runtime.

A persistent adaptive world-model substrate for machines that must
continuously exist inside changing environments.

Constitutional principles enforced by this package:
1. One authoritative immutable world state (``WorldState``).
2. Common Snapshot – all mechanisms see the same ``Xₜ``.
3. Proposal, Never Mutation – mechanisms return ``Contribution`` only.
4. Singular Commit – ``engine.commit()`` produces ``X_{t+1}``.
5. Structure is independent from state (explicit transport).
6. Computational Conservation of Explanation (append-only ledger).
"""
from hrm_ecology.state import WorldState, ResourceBudget
from hrm_ecology.contribution import Contribution
from hrm_ecology.ledger import Ledger, AttributionEntry
from hrm_ecology.engine import Engine, StepResult
from hrm_ecology.safety import SafetyConstitution, SafetyConfig
from hrm_ecology.replay import ReplayEngine, ReplayReport
from hrm_ecology.registry import MechanismRegistry
from hrm_ecology.environment import SimulatedEnvironment, EnvironmentConfig
from hrm_ecology.mechanisms.base import Observations

__all__ = [
    "WorldState",
    "ResourceBudget",
    "Contribution",
    "Ledger",
    "AttributionEntry",
    "Engine",
    "StepResult",
    "SafetyConstitution",
    "SafetyConfig",
    "ReplayEngine",
    "ReplayReport",
    "MechanismRegistry",
    "SimulatedEnvironment",
    "EnvironmentConfig",
    "Observations",
]

__version__ = "0.1.0"
