"""Reusable metric interfaces for measuring adversarial degradation.

Metrics are deliberately dependency-light (pure Python, numpy optional) and are
never computed inside Flask routes. Each function works on plain Python numbers
or sequences so it can be unit-tested without a model.

For a single-image experiment, report per-example metrics
(:func:`perturbation_linf`, :func:`confidence_shift`, ...). For a
dataset/batch experiment, use the aggregate helpers
(:func:`calculate_clean_accuracy`, :func:`calculate_robust_accuracy`,
:func:`calculate_attack_success_rate`, :func:`class_level_degradation`).
"""
from __future__ import annotations

from orion.metrics.core import (
    calculate_clean_accuracy,
    calculate_robust_accuracy,
    calculate_attack_success_rate,
    calculate_perturbation,
    perturbation_linf,
    perturbation_l2,
    calculate_confidence_shift,
    confidence_shift,
    class_level_degradation,
    attack_cost,
    MetricResult,
)

__all__ = [
    "calculate_clean_accuracy",
    "calculate_robust_accuracy",
    "calculate_attack_success_rate",
    "calculate_perturbation",
    "perturbation_linf",
    "perturbation_l2",
    "calculate_confidence_shift",
    "confidence_shift",
    "class_level_degradation",
    "attack_cost",
    "MetricResult",
]
