"""Experiment execution: scenario -> attack -> measure -> evidence.

An experiment is executed in one of three modes:

    baseline  : run the model on clean inputs only (NO_ATTACK).
    attack    : run the attack and measure degradation.
    hardened  : apply the scenario's defenses, then replay the attack (retest).

The runner is dependency-light by default (a deterministic synthetic backend)
so the full methodology is reproducible without a GPU. The real adversarial
image path (torch + ART, wrapping the existing ``libs.adversarial``) is used
when those dependencies and inputs are available.
"""
from __future__ import annotations

from orion.experiments.base import ExperimentMode, ExperimentOutcome
from orion.experiments.runner import run_scenario, replay, compare

__all__ = ["ExperimentMode", "ExperimentOutcome", "run_scenario", "replay", "compare"]
