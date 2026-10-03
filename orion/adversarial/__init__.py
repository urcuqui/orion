"""Adversarial ML experiments.

This package preserves the existing image-generation behavior (re-exported from
``libs.adversarial``) and adds a structured wrapper that returns measurable
evidence alongside the visual artifact.

    The experiment should produce both a visual artifact and measurable evidence.
"""
from __future__ import annotations

# Backward-compatible re-export: existing callers keep working.
from libs.adversarial import generate_advimage  # noqa: F401

from orion.adversarial.image import (
    generate_adversarial_evidence,
    infer_num_outputs,
    AdversarialImageResult,
    TORCH_AVAILABLE,
)
from orion.adversarial.experiment import run_adversarial_experiment
from orion.adversarial.blackbox import run_blackbox_evasion
from orion.adversarial import catalog

__all__ = [
    "generate_advimage",
    "generate_adversarial_evidence",
    "infer_num_outputs",
    "run_adversarial_experiment",
    "run_blackbox_evasion",
    "catalog",
    "AdversarialImageResult",
    "TORCH_AVAILABLE",
]
