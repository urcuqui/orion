"""Experiment status vocabulary."""
from __future__ import annotations

from enum import Enum


class ExperimentStatus(str, Enum):
    """Final status of an experiment.

    NO_ATTACK           No attack was executed (e.g. baseline-only run).
    ATTACK_SUCCESS      The attack met its goal (model degraded / evaded).
    ATTACK_BLOCKED      A defense blocked or fully neutralized the attack.
    PARTIALLY_MITIGATED The defense reduced but did not eliminate the impact.
    """

    NO_ATTACK = "NO_ATTACK"
    ATTACK_SUCCESS = "ATTACK_SUCCESS"
    ATTACK_BLOCKED = "ATTACK_BLOCKED"
    PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
