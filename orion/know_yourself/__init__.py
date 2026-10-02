"""Know Yourself: structured AI security profiling.

    What exactly do I have, how does it behave, what assumptions does it make,
    how fragile is it, and which security experiments are actually applicable?

Two branches that do not share security assumptions:
  - Traditional ML  (model-centric: model, features, predictions, robustness)
  - Generative AI   (system-centric: prompts, context, memory, tools, identity)

Know Yourself characterizes and measures; it never launches attacks. Attack
execution belongs to the Attack phase.
"""
from __future__ import annotations

from orion.know_yourself.common import (
    TRADITIONAL_ML, GENERATIVE_AI, HYBRID, UNKNOWN_SYSTEM,
    detect_system_type, derive_capabilities, build_posture, recommended_experiments,
)
from orion.know_yourself.service import analyze

__all__ = [
    "TRADITIONAL_ML", "GENERATIVE_AI", "HYBRID", "UNKNOWN_SYSTEM",
    "detect_system_type", "derive_capabilities", "build_posture",
    "recommended_experiments", "analyze",
]
