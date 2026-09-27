"""The Orion methodology as first-class code.

Every Orion experiment or scenario can identify which phase it belongs to.
See ``docs/methodology.md`` for the narrative explanation.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List


class Phase(str, Enum):
    """The six phases of the Orion methodology.

    The value is a stable, lowercase identifier suitable for use in scenario
    files, evidence records and CLI arguments.
    """

    UNDERSTAND = "understand"
    THREAT_MODEL = "threat_model"
    ATTACK = "attack"
    MEASURE = "measure"
    HARDEN = "harden"
    RETEST = "retest"

    @property
    def order(self) -> int:
        return PHASE_ORDER[self]

    def describe(self) -> str:
        return describe_phase(self)


# Canonical ordering of the methodology.
PHASES: List[Phase] = [
    Phase.UNDERSTAND,
    Phase.THREAT_MODEL,
    Phase.ATTACK,
    Phase.MEASURE,
    Phase.HARDEN,
    Phase.RETEST,
]

PHASE_ORDER: Dict[Phase, int] = {phase: idx for idx, phase in enumerate(PHASES)}

_DESCRIPTIONS: Dict[Phase, str] = {
    Phase.UNDERSTAND: (
        "Know the target and the environment: the ML task, the data, the "
        "model, the inference surface and the surrounding system. Orion's "
        "reconnaissance workflow is the 'Know the Environment' expression of "
        "this phase."
    ),
    Phase.THREAT_MODEL: (
        "Define the adversary and what is at stake: assets, adversary "
        "knowledge/access/goal/budget, and attack surfaces. An attack "
        "algorithm without a threat model is only an experiment."
    ),
    Phase.ATTACK: (
        "Execute a controlled attack that is justified by the threat model "
        "(e.g. an evasion attack against an image classifier)."
    ),
    Phase.MEASURE: (
        "Quantify degradation with metrics (attack success rate, robust "
        "accuracy, confidence shift, perturbation magnitude, query cost). "
        "Metrics over screenshots."
    ),
    Phase.HARDEN: (
        "Apply a candidate defense (adversarial training, input "
        "preprocessing, anomaly/confidence checks, rate limiting, input "
        "validation, monitoring). A defense is not validated until the attack "
        "is replayed."
    ),
    Phase.RETEST: (
        "Replay the same attack against the hardened system and compare "
        "before/after. State the trade-offs and the limitations; never claim "
        "universal robustness."
    ),
}


def describe_phase(phase: Phase) -> str:
    """Return the human-readable description of a methodology phase."""
    return _DESCRIPTIONS[phase]


def coerce_phase(value: "Phase | str") -> Phase:
    """Coerce a string (as found in scenario files) into a :class:`Phase`."""
    if isinstance(value, Phase):
        return value
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return Phase(normalized)
    except ValueError as exc:  # pragma: no cover - defensive
        valid = ", ".join(p.value for p in PHASES)
        raise ValueError(f"Unknown Orion phase {value!r}; expected one of: {valid}") from exc
