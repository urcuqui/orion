"""Orion's unified catalogs: one source of truth for attacks, metrics, controls.

These registries are referenced by applicability analysis, threat modelling,
experiment planning, attack execution, metrics rendering and remediation — so an
attack, metric or control is defined once and reused everywhere, across the
Traditional ML, Generative AI and Agentic AI families.
"""
from __future__ import annotations

from orion.catalog import attacks, controls, metrics
from orion.catalog.attacks import (
    AGENTIC_AI, FAMILIES, FAMILY_LABELS, GENERATIVE_AI, TRADITIONAL_ML,
    AttackDefinition,
)
from orion.catalog.controls import ControlDefinition
from orion.catalog.metrics import MetricDefinition

__all__ = [
    "attacks", "metrics", "controls",
    "AttackDefinition", "MetricDefinition", "ControlDefinition",
    "FAMILIES", "FAMILY_LABELS", "TRADITIONAL_ML", "GENERATIVE_AI", "AGENTIC_AI",
]
