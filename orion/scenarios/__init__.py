"""Reusable, declarative scenario definitions.

A scenario is a YAML file that binds together the methodology: target,
adversary/threat model, the attack to run, the metrics to compute, and the
MITRE ATLAS mappings. Scenarios are loaded and validated here.
"""
from __future__ import annotations

from orion.scenarios.model import Scenario, AttackSpec, HardeningSpec, ScenarioValidationError
from orion.scenarios.loader import load_scenario, load_scenario_dict, list_scenarios, validate_scenario

__all__ = [
    "Scenario",
    "AttackSpec",
    "HardeningSpec",
    "ScenarioValidationError",
    "load_scenario",
    "load_scenario_dict",
    "list_scenarios",
    "validate_scenario",
]
