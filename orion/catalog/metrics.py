"""The unified Metric Catalog.

Metrics are not image-adversarial-only. Each metric declares the family it
belongs to and whether a higher or lower value is *better* for security, so the
Measure screen can render per experiment type and so before/after retests can be
scored consistently across Traditional ML, Generative AI and Agentic AI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orion.catalog.attacks import AGENTIC_AI, GENERATIVE_AI, TRADITIONAL_ML

HIGHER_BETTER = "higher_better"   # security improves as the value rises (accuracy, refusals)
LOWER_BETTER = "lower_better"     # security improves as the value falls (attack success, leaks)
INFORMATIONAL = "informational"   # context only, no intrinsic direction


@dataclass
class MetricDefinition:
    id: str
    name: str
    families: List[str]
    direction: str = INFORMATIONAL
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "families": list(self.families),
                "direction": self.direction, "unit": self.unit, "description": self.description}


_METRICS: List[MetricDefinition] = [
    # ----------------------------- Traditional ML ----------------------------- #
    MetricDefinition("clean_accuracy", "Clean accuracy", [TRADITIONAL_ML], HIGHER_BETTER, "ratio",
                     "Accuracy on unperturbed inputs."),
    MetricDefinition("robust_accuracy", "Robust accuracy", [TRADITIONAL_ML], HIGHER_BETTER, "ratio",
                     "Accuracy under attack; higher means more robust."),
    MetricDefinition("perturbation_linf", "L∞ distance", [TRADITIONAL_ML], LOWER_BETTER, "",
                     "Max per-pixel change; smaller means a stealthier perturbation."),
    MetricDefinition("perturbation_l2", "L2 distance", [TRADITIONAL_ML], LOWER_BETTER, "",
                     "Euclidean perturbation magnitude."),
    MetricDefinition("confidence_shift", "Confidence shift", [TRADITIONAL_ML], INFORMATIONAL, "",
                     "Change in prediction confidence between clean and adversarial."),
    MetricDefinition("query_count", "Query count", [TRADITIONAL_ML], LOWER_BETTER, "queries",
                     "Queries spent by a black-box attack; fewer means cheaper to attack."),
    # ----------------------------- Generative AI ------------------------------ #
    MetricDefinition("refusal_rate", "Refusal rate", [GENERATIVE_AI], HIGHER_BETTER, "ratio",
                     "Share of malicious prompts the system refused."),
    MetricDefinition("policy_violation_rate", "Policy violation rate",
                     [GENERATIVE_AI, AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials that violated a stated policy."),
    MetricDefinition("secret_leakage_rate", "Secret leakage rate", [GENERATIVE_AI], LOWER_BETTER, "ratio",
                     "Share of trials that leaked a secret/system prompt."),
    MetricDefinition("instruction_following_rate", "Injected-instruction following rate",
                     [GENERATIVE_AI], LOWER_BETTER, "ratio",
                     "Share of trials where the injected instruction was followed."),
    # ------------------------------- Agentic AI ------------------------------- #
    MetricDefinition("unauthorized_tool_call_rate", "Unauthorized tool-call rate", [AGENTIC_AI],
                     LOWER_BETTER, "ratio", "Share of trials with a tool call that was not authorised."),
    MetricDefinition("privilege_boundary_violation_rate", "Privilege boundary violation rate",
                     [AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials that crossed a privilege boundary."),
    MetricDefinition("unsafe_action_rate", "Unsafe action rate", [AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials that executed an unsafe action."),
    MetricDefinition("goal_hijack_rate", "Goal hijack rate", [AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials where the agent's goal was redirected."),
    MetricDefinition("approval_bypass_rate", "Approval bypass rate", [AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials that bypassed a required human approval."),
    # -------------------------- Cross-family -------------------------------- #
    MetricDefinition("attack_success_rate", "Attack success rate",
                     [TRADITIONAL_ML, GENERATIVE_AI, AGENTIC_AI], LOWER_BETTER, "ratio",
                     "Share of trials in which the attack met its success criteria."),
]

METRICS: Dict[str, MetricDefinition] = {m.id: m for m in _METRICS}


def get(metric_id: str) -> Optional[MetricDefinition]:
    return METRICS.get(metric_id)


def for_family(family: str) -> List[MetricDefinition]:
    return [m for m in _METRICS if family in m.families]


def for_attack(attack_id: str) -> List[MetricDefinition]:
    from orion.catalog.attacks import get as get_attack
    a = get_attack(attack_id)
    if not a:
        return []
    return [METRICS[mid] for mid in a.supported_metrics if mid in METRICS]


def direction_of(metric_id: str) -> str:
    m = METRICS.get(metric_id)
    return m.direction if m else INFORMATIONAL


def is_improvement(metric_id: str, before: float, after: float) -> Optional[bool]:
    """True if `after` is better than `before` for this metric's direction."""
    d = direction_of(metric_id)
    try:
        before, after = float(before), float(after)
    except (TypeError, ValueError):
        return None
    if d == HIGHER_BETTER:
        return after > before
    if d == LOWER_BETTER:
        return after < before
    return None
