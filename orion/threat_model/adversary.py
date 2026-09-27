"""Adversary: knowledge, access, goal, budget and constraints."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class KnowledgeLevel(str, Enum):
    """How much the adversary knows about the target."""

    NONE = "none"          # no knowledge
    LIMITED = "limited"    # partial / query-only observations
    FULL = "full"          # architecture, weights and data known (white box)


class AccessLevel(str, Enum):
    """How the adversary can interact with the target."""

    BLACK_BOX = "black_box"    # query the deployed API only
    GRAY_BOX = "gray_box"      # some internals (e.g. logits, architecture)
    WHITE_BOX = "white_box"    # full access to weights and gradients


class Budget(str, Enum):
    """Rough resource budget available to the adversary."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Goal(str, Enum):
    """The adversary's objective."""

    UNTARGETED_MISCLASSIFICATION = "untargeted_misclassification"
    TARGETED_MISCLASSIFICATION = "targeted_misclassification"
    MODEL_EXTRACTION = "model_extraction"
    MEMBERSHIP_INFERENCE = "membership_inference"
    DATA_POISONING = "data_poisoning"
    EVASION = "evasion"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class Adversary:
    """A structured description of the adversary."""

    goal: Goal
    knowledge: KnowledgeLevel = KnowledgeLevel.LIMITED
    access: AccessLevel = AccessLevel.BLACK_BOX
    budget: Budget = Budget.LOW
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "goal": self.goal.value,
            "knowledge": self.knowledge.value,
            "access": self.access.value,
            "budget": self.budget.value,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Adversary":
        return cls(
            goal=Goal(_norm(data.get("goal", Goal.EVASION.value))),
            knowledge=KnowledgeLevel(_norm(data.get("knowledge", KnowledgeLevel.LIMITED.value))),
            access=AccessLevel(_norm(data.get("access", AccessLevel.BLACK_BOX.value))),
            budget=Budget(_norm(data.get("budget", Budget.LOW.value))),
            constraints=list(data.get("constraints", []) or []),
        )


def _norm(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
