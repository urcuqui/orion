"""Know Your Target: evidence-grounded interpretation of reconnaissance.

    Recon collects.  Orion interprets.  Humans decide.
    Evidence -> applicability -> hypothesis -> test -> finding.

AI-specific threats are surfaced only when recon actually observed an AI/ML
surface. Orion knows when it does not know.
"""
from __future__ import annotations

from orion.target_analysis.analysis import (
    display_run_id,
    summarize_recon,
    build_evidence,
    classify_target_types,
    detect_ai_surface,
    check_applicability,
    propose_threat_model,
    propose_experiments,
    threat_hypotheses,
    build_assessment,
    build_assessment_from_context,
    build_assessment_from_summary,
    validate_analysis,
    approve_threat_model,
    ATTACK_CATALOG,
    ANALYSES,
)
from orion.target_analysis.probe import probe_url, build_probe_summary  # noqa: E402
from orion.target_analysis.analysis import (  # noqa: E402  (re-exported enums)
    OBSERVED, INFERRED, HYPOTHESIS, NOT_APPLICABLE,
    HIGH, MEDIUM, LOW, NONE,
)

__all__ = [
    "display_run_id", "summarize_recon", "build_evidence", "classify_target_types",
    "detect_ai_surface", "check_applicability", "propose_threat_model",
    "propose_experiments", "threat_hypotheses", "build_assessment",
    "build_assessment_from_context", "build_assessment_from_summary",
    "validate_analysis", "approve_threat_model", "probe_url", "build_probe_summary",
    "ATTACK_CATALOG", "ANALYSES",
    "OBSERVED", "INFERRED", "HYPOTHESIS", "NOT_APPLICABLE",
    "HIGH", "MEDIUM", "LOW", "NONE",
]
