"""The Finding: Orion's security interpretation of what the evidence shows.

    Evidence = what happened.
    Finding  = the security meaning of what happened.

A Finding is never auto-confirmed from a single hypothesis: one successful run is
OBSERVED; CONFIRMED requires corroborating evidence. Every Finding points back to
the runs that justify it, and its retest_status records whether a control
actually improved security.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---- lifecycle statuses --------------------------------------------------- #
HYPOTHESIS = "HYPOTHESIS"            # proposed, not yet observed
OBSERVED = "OBSERVED"               # seen once in evidence
CONFIRMED = "CONFIRMED"             # reproduced / corroborated
NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
MITIGATED = "MITIGATED"
PARTIALLY_MITIGATED = "PARTIALLY_MITIGATED"
STATUSES = (HYPOTHESIS, OBSERVED, CONFIRMED, NOT_REPRODUCIBLE, MITIGATED, PARTIALLY_MITIGATED)

# ---- control effectiveness (set by a retest) ------------------------------ #
INEFFECTIVE = "INEFFECTIVE"
PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
EFFECTIVE = "EFFECTIVE"


def new_finding_id() -> str:
    return "ORN-FND-" + uuid.uuid4().hex[:8].upper()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Finding:
    id: str = field(default_factory=new_finding_id)
    title: str = ""
    description: str = ""
    status: str = HYPOTHESIS
    severity: str = "MEDIUM"           # LOW | MEDIUM | HIGH | CRITICAL
    confidence: str = "LOW"            # LOW | MEDIUM | HIGH
    attack_id: str = ""
    family: str = ""
    affected_target: str = ""
    affected_component: str = ""
    boundary_crossed: str = ""
    success_condition: str = ""
    observations: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)
    recommended_controls: List[str] = field(default_factory=list)
    framework_mappings: Dict[str, List[str]] = field(default_factory=dict)
    # Defense / retest linkage.
    applied_control: Optional[str] = None
    retest_status: Optional[str] = None       # INEFFECTIVE | PARTIALLY_EFFECTIVE | EFFECTIVE
    retest_refs: List[str] = field(default_factory=list)
    # Provenance back to the whole chain.
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "status": self.status, "severity": self.severity, "confidence": self.confidence,
            "attack_id": self.attack_id, "family": self.family,
            "affected_target": self.affected_target, "affected_component": self.affected_component,
            "boundary_crossed": self.boundary_crossed, "success_condition": self.success_condition,
            "observations": self.observations, "metrics": self.metrics,
            "evidence_refs": self.evidence_refs, "recommended_controls": self.recommended_controls,
            "framework_mappings": self.framework_mappings, "applied_control": self.applied_control,
            "retest_status": self.retest_status, "retest_refs": self.retest_refs,
            "provenance": self.provenance,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        f = cls(id=data.get("id", new_finding_id()))
        for k, v in data.items():
            if hasattr(f, k):
                setattr(f, k, v)
        return f


# --------------------------- derivation helpers ---------------------------- #
_BOUNDARY = {
    "traditional_ml": "Input → Model → Prediction",
    "generative_ai": "Untrusted Context → LLM → Output",
    "agentic_ai": "Untrusted Context → Agent → Tool",
}


def _severity(record, asr: float) -> str:
    risk = (record.target or {}).get("access", "")
    if asr >= 0.99:
        return "HIGH" if record.family != "traditional_ml" else "MEDIUM"
    if asr > 0:
        return "MEDIUM"
    return "LOW"


def build_finding_from_record(record, attack=None, provenance: Optional[Dict[str, Any]] = None) -> Finding:
    """Interpret an ExperimentRecord as a (single-run) OBSERVED finding.

    A single successful run is OBSERVED, not CONFIRMED — confirmation needs
    corroboration (:func:`corroborate`).
    """
    from orion.catalog import attacks as CAT
    attack = attack or CAT.get(record.parameters.get("attack_id")) or CAT.get(record.attack_technique)
    asr = (record.metrics.get("attack_success_rate", {}) or {}).get("value")
    if asr is None:
        asr = 1.0 if record.status == "ATTACK_SUCCESS" else 0.0

    family = record.family or (attack.family if attack else "")
    succeeded = record.status == "ATTACK_SUCCESS" or (asr or 0) > 0
    status = OBSERVED if succeeded else NOT_REPRODUCIBLE

    boundary = _BOUNDARY.get(family, "")
    name = attack.name if attack else (record.attack_technique or "Attack")
    title = f"{name} on {(record.target or {}).get('model_name') or (record.target or {}).get('task','target')}"
    succ_crit = ", ".join(attack.success_criteria) if attack else ""

    return Finding(
        title=title,
        description=(attack.description if attack else record.notes or ""),
        status=status,
        severity=_severity(record, asr or 0),
        confidence="MEDIUM" if succeeded else "LOW",
        attack_id=(attack.id if attack else record.attack_technique),
        family=family,
        affected_target=(record.target or {}).get("model_name") or (record.target or {}).get("task", ""),
        affected_component=(record.target or {}).get("task", ""),
        boundary_crossed=boundary,
        success_condition=succ_crit,
        observations={"attack_success_rate": asr, "successes": (record.metrics.get("successes", {}) or {}).get("value")},
        metrics=record.metrics,
        evidence_refs=[record.trace_id],
        recommended_controls=(attack.recommended_controls if attack else []),
        framework_mappings=(attack.framework_mappings if attack else {}),
        provenance=provenance or record.provenance or {},
    )


def corroborate(finding: Finding, record) -> Finding:
    """Add a second (or later) successful run; promote OBSERVED → CONFIRMED."""
    if record.trace_id not in finding.evidence_refs:
        finding.evidence_refs.append(record.trace_id)
    successful = [r for r in finding.evidence_refs]
    if finding.status in (HYPOTHESIS, OBSERVED) and len(successful) >= 2:
        finding.status = CONFIRMED
        finding.confidence = "HIGH"
    finding.updated_at = _now()
    return finding


def classify_control(before_rate: float, after_rate: float, eps: float = 1e-9) -> str:
    """Classify a control from measured before/after attack-success rates."""
    try:
        before, after = float(before_rate), float(after_rate)
    except (TypeError, ValueError):
        return INEFFECTIVE
    if after <= eps:
        return EFFECTIVE
    if after < before - eps:
        return PARTIALLY_EFFECTIVE
    return INEFFECTIVE


def apply_retest(finding: Finding, before_rate: float, after_rate: float,
                 control_id: str, retest_run_id: str) -> Finding:
    """Record a retest outcome and update the finding's status honestly."""
    effectiveness = classify_control(before_rate, after_rate)
    finding.applied_control = control_id
    finding.retest_status = effectiveness
    if retest_run_id and retest_run_id not in finding.retest_refs:
        finding.retest_refs.append(retest_run_id)
    if effectiveness == EFFECTIVE:
        finding.status = MITIGATED
    elif effectiveness == PARTIALLY_EFFECTIVE:
        finding.status = PARTIALLY_MITIGATED
    # INEFFECTIVE leaves the finding CONFIRMED/OBSERVED — not mitigated.
    finding.updated_at = _now()
    return finding
