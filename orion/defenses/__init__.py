"""Pluggable defense / hardening mechanisms.

Orion never automatically claims a model is secure. A defense is a candidate
control that must be *evaluated by retesting* — replaying the same attack and
comparing metrics before and after.

    A defense is not validated until the attack is replayed.

Defenses register themselves in :data:`DEFENSE_REGISTRY` and are referenced by
name from scenarios (``hardening.defenses``).
"""
from __future__ import annotations

from orion.defenses.base import Defense, DefenseResult, DEFENSE_REGISTRY, get_defense, register_defense
from orion.defenses.builtin import (
    InputPreprocessingDefense,
    ConfidenceThresholdDefense,
    RateLimitDefense,
    InputValidationDefense,
    MonitoringHookDefense,
    AdversarialTrainingDefense,
)

__all__ = [
    "Defense",
    "DefenseResult",
    "DEFENSE_REGISTRY",
    "get_defense",
    "register_defense",
    "InputPreprocessingDefense",
    "ConfidenceThresholdDefense",
    "RateLimitDefense",
    "InputValidationDefense",
    "MonitoringHookDefense",
    "AdversarialTrainingDefense",
]
