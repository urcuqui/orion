"""Experiment plans: the approve-and-handoff bridge from analysis to Attack.

    Analysis proposes. Humans approve. Attack prepares. Humans execute.
    Approve Plan != Run Attack.
"""
from __future__ import annotations

from orion.plans.plan import (
    ExperimentPlan, ExperimentProposal,
    PROPOSED, UNDER_REVIEW, APPROVED, READY, NEEDS_INPUT, EXCLUDED, RUNNING, COMPLETED, FAILED,
    new_plan_id, save_plan, load_plan, load_handoff, approve_plan, apply_plan_edits,
    build_plan_from_know_yourself, build_plan_from_target_analysis, build_plan_from_context,
)
from orion.plans.runner import run_experiment, ExperimentNotRunnable

__all__ = [
    "ExperimentPlan", "ExperimentProposal",
    "PROPOSED", "UNDER_REVIEW", "APPROVED", "READY", "NEEDS_INPUT", "EXCLUDED",
    "RUNNING", "COMPLETED", "FAILED",
    "new_plan_id", "save_plan", "load_plan", "load_handoff", "approve_plan", "apply_plan_edits",
    "build_plan_from_know_yourself", "build_plan_from_target_analysis", "build_plan_from_context",
    "run_experiment", "ExperimentNotRunnable",
]
