"""
jarvis/computer/__init__.py
───────────────────────────
v0.28 Part 4/20 — restricted computer-control abstraction.

The restricted abstraction defines narrow capabilities (get_screen,
click_target, type_text, press_key, scroll, move_pointer) with validated
targets (label / bbox / observation_id — NOT raw coordinates as first-class
input) and risk metadata that maps into the existing permission system.

v0.28 ships EXACTLY ONE backend:

  DisabledHostBackend — refuses everything, with an honest explanation.
  No host mouse/keyboard library (pyautogui etc.) exists in the process;
  the abstraction exists so a future isolated backend (Docker-hosted
  automation, explicit enablement ladder like code execution) can be added
  without re-shaping the tool surface or the permission layer.

Host boundary (Part 20) — never granted by this layer, by construction:
  shell/PowerShell execution, arbitrary filesystem writes, registry
  manipulation, process creation, unrestricted clipboard access.
  Untrusted code execution remains the Docker sandbox's job only.

The module keeps the same safety contract as the browser layer: every
capability would require a fresh observation ID and validated targets; raw
coordinates (if ever enabled) would require screen bounds + observation
freshness + confirmation for risky operations.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenTarget:
    """
    Validated target abstraction. Coordinates are optional and SECONDARY:
    the primary target is a label + the observation the model saw. Bbox
    coordinates are validated against screen bounds by the backend before
    any use.
    """

    label: str | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    observation_id: str | None = None
    confidence: float | None = None


class HostBackend(abc.ABC):
    """Backend contract for host computer control (all fail closed)."""

    provides_isolation: bool = False

    @abc.abstractmethod
    def get_screen(self) -> str: ...

    @abc.abstractmethod
    def click_target(self, target: ScreenTarget) -> str: ...

    @abc.abstractmethod
    def type_text(self, text: str, target: ScreenTarget | None = None) -> str: ...

    @abc.abstractmethod
    def press_key(self, key: str) -> str: ...

    @abc.abstractmethod
    def scroll(self, amount: int) -> str: ...


_DISABLED_MSG = (
    "ERROR: host computer control is disabled by design in this deployment. "
    "JARVIS v0.28 provides browser interaction only (safe, policy-controlled, "
    "confirmation-gated). Desktop mouse/keyboard control requires an isolated "
    "execution backend which is not enabled here. No host input library is "
    "installed or reachable from the agent."
)


class DisabledHostBackend(HostBackend):
    """
    Refuses every capability with the same honest message. The ONLY
    backend registered in v0.28; ``provides_isolation`` is False so the
    enablement ladder (mirroring code execution) can never activate it by
    accident.
    """

    provides_isolation = False

    def get_screen(self) -> str:
        return _DISABLED_MSG

    def click_target(self, target: ScreenTarget) -> str:
        return _DISABLED_MSG

    def type_text(self, text: str, target: ScreenTarget | None = None) -> str:
        return _DISABLED_MSG

    def press_key(self, key: str) -> str:
        return _DISABLED_MSG

    def scroll(self, amount: int) -> str:
        return _DISABLED_MSG


def build_host_backend() -> HostBackend:
    """
    Backend selection ladder (fail closed): the only configured backend in
    v0.28 is the disabled one. A future isolated backend would require
    BOTH explicit config AND verified isolation — same ladder as
    ENABLE_CODE_EXECUTION.
    """
    return DisabledHostBackend()


__all__ = [
    "ScreenTarget",
    "HostBackend",
    "DisabledHostBackend",
    "build_host_backend",
]
