"""Experiment mode and outcome types."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentMode(str, Enum):
    BASELINE = "baseline"
    ATTACK = "attack"
    HARDENED = "hardened"


@dataclass
class ExperimentOutcome:
    """Backend-agnostic result of running one experiment, before evidence I/O."""

    mode: ExperimentMode
    baseline_result: Dict[str, Any] = field(default_factory=dict)
    adversarial_result: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    controls_tested: List[Dict[str, Any]] = field(default_factory=list)
    images: Dict[str, str] = field(default_factory=dict)  # label -> file path
    attack_success: bool = False
    limitations: List[str] = field(default_factory=list)
    model_version: str = "unknown"
    notes: str = ""
