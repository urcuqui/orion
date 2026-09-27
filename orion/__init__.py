"""Orion: an AI security experimentation framework.

Orion combines adversarial machine learning, threat modeling, MITRE ATLAS,
agent-assisted workflows, security tooling, and reproducible evidence.

The methodology-layer package (this ``orion`` package) is intentionally kept
free of heavy or optional dependencies (torch, ART, langgraph, flask) so that
threat models, metrics, scenarios, evidence and policies can be imported and
tested anywhere. Modules that need those dependencies import them lazily.

Core methodology:

    Understand -> Threat Model -> Attack -> Measure -> Harden -> Retest

Guiding principle:

    An attack algorithm without a threat model is only an experiment.
"""
from __future__ import annotations

__version__ = "2.0.0"

from orion.methodology import Phase, PHASES, describe_phase  # noqa: E402

__all__ = ["Phase", "PHASES", "describe_phase", "__version__"]
