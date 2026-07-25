"""Mechanism registry: tracks mechanism IDs and versions for replay.

The registry maps mechanism IDs to :class:`~hrm_ecology.mechanisms.base.MechanismInfo`
and ensures that replay uses the correct mechanism versions.
"""
from __future__ import annotations

from hrm_ecology.mechanisms.base import Mechanism, MechanismInfo


class MechanismRegistry:
    """Maintains a versioned record of all registered mechanisms.

    Used to verify version consistency during deterministic replay.
    """

    def __init__(self) -> None:
        self._registry: dict[str, MechanismInfo] = {}

    def register(self, mechanism: Mechanism) -> None:
        """Register a mechanism (overwrites if already registered)."""
        info = mechanism.info
        self._registry[info.mechanism_id] = info

    def register_all(self, mechanisms: list[Mechanism]) -> None:
        for m in mechanisms:
            self.register(m)

    def get(self, mechanism_id: str) -> MechanismInfo | None:
        return self._registry.get(mechanism_id)

    def version_snapshot(self) -> dict[str, str]:
        """Return a mapping of mechanism_id → version string."""
        return {mid: info.version for mid, info in self._registry.items()}

    def verify_versions(self, expected: dict[str, str]) -> list[str]:
        """Return a list of mismatches between expected and registered versions."""
        mismatches: list[str] = []
        for mid, expected_version in expected.items():
            info = self._registry.get(mid)
            if info is None:
                mismatches.append(f"Missing mechanism: {mid!r}")
            elif info.version != expected_version:
                mismatches.append(
                    f"Version mismatch for {mid!r}: "
                    f"expected {expected_version!r}, got {info.version!r}"
                )
        return mismatches

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        ids = list(self._registry)
        return f"MechanismRegistry(n={len(ids)}, ids={ids})"
