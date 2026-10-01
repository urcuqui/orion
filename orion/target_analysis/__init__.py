"""Know Your Target: turn reconnaissance evidence into an interpretation.

    Recon collects.  Orion interprets.  Humans decide.

The interpretation here is deterministic and evidence-driven: the proposed
threat model and candidate experiments are derived from observed recon evidence
with explicit rules, and are always labelled PROPOSED until a human approves
them. An LLM agent may add narrative on top, but it never decides security.
"""
from __future__ import annotations

from orion.target_analysis.analysis import (
    display_run_id,
    summarize_recon,
    propose_threat_model,
    propose_experiments,
    threat_hypotheses,
    build_assessment,
    build_assessment_from_context,
    ANALYSES,
    approve_threat_model,
)

__all__ = [
    "display_run_id",
    "summarize_recon",
    "propose_threat_model",
    "propose_experiments",
    "threat_hypotheses",
    "build_assessment",
    "build_assessment_from_context",
    "ANALYSES",
    "approve_threat_model",
]
