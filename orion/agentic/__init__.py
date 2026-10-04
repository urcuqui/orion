"""Controlled GenAI / Agentic experiments: injection, tool & privilege abuse.

A deterministic lab that makes agent security behaviour observable, so Orion's
methodology (Context → Threat → Plan → Attack → Measure → Finding → Defend →
Retest → Evidence) applies to agentic systems exactly as it does to adversarial
ML. Verdicts come from the recorded trace, never from a narrative.
"""
from __future__ import annotations

from orion.agentic import lab
from orion.agentic.experiment import run_agentic_experiment
from orion.agentic.live import run_live_prompt_injection

__all__ = ["lab", "run_agentic_experiment", "run_live_prompt_injection"]
