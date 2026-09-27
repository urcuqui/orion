"""Explicit, LLM-independent threat modeling for AI systems.

The threat model answers *who* the adversary is, *what* they want, and *where*
they can act — before any attack algorithm is selected. This package is pure
Python (dataclasses + enums) and never depends on an LLM.

    An attack algorithm without a threat model is only an experiment.
"""
from __future__ import annotations

from orion.threat_model.assets import Asset, ASSET_CATALOG
from orion.threat_model.adversary import (
    Adversary,
    AccessLevel,
    KnowledgeLevel,
    Budget,
    Goal,
)
from orion.threat_model.boundaries import AttackSurface, ATTACK_SURFACES, Target
from orion.threat_model.scenario import ThreatModel

__all__ = [
    "Asset",
    "ASSET_CATALOG",
    "Adversary",
    "AccessLevel",
    "KnowledgeLevel",
    "Budget",
    "Goal",
    "AttackSurface",
    "ATTACK_SURFACES",
    "Target",
    "ThreatModel",
]
