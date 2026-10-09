"""SuccessSignature: a deterministic, normalized identity for a reproduced
security condition.

It is NOT a copy of the experiment configuration — it captures just enough
context to distinguish one security failure from another (attack, target,
criterion, boundary, capability, security effect). Fields are attack-family
aware; absent dimensions are simply omitted rather than invented.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from orion.catalog import attacks as CAT

# criterion / family -> a stable, human-readable security effect
_EFFECT_BY_CRITERION = {
    "unauthorized_tool_action": "unauthorized_external_action",
    "privilege_boundary_crossed": "privilege_escalation",
    "secret_leakage": "sensitive_data_disclosure",
    "policy_violation": "policy_bypass",
    "injected_instruction_followed": "instruction_override",
    "prediction_changed": "adversarial_evasion",
    "decision_changed": "adversarial_evasion",
    "unauthorized_privileged_tool_execution": "unauthorized_external_action",
    "restricted_instruction_followed": "policy_bypass",
    "prediction_changed_under_allowed_budget": "adversarial_evasion",
}
_EFFECT_BY_FAMILY = {
    "traditional_ml": "adversarial_evasion",
    "generative_ai": "policy_bypass",
    "agentic_ai": "unauthorized_external_action",
}


def build_success_signature(attack_id: str, target: str, criterion: str,
                            record=None) -> Dict[str, Any]:
    """Normalize a reproduced criterion into a SuccessSignature.

    ``record`` (a retest/attack ExperimentRecord) supplies the agentic boundary /
    capability when available. Only present dimensions are included.
    """
    attack = CAT.get(attack_id)
    family = attack.family if attack else ""
    sig: Dict[str, Any] = {
        "attack_id": attack.id if attack else attack_id,
        "target": target or "unknown",
        "criterion": criterion,
        "security_effect": _EFFECT_BY_CRITERION.get(criterion) or _EFFECT_BY_FAMILY.get(family, "security_impact"),
    }
    # Agentic / GenAI add boundary + capability when the evidence has them.
    adv = (getattr(record, "adversarial_result", None) or {}) if record is not None else {}
    params = (getattr(record, "parameters", None) or {}) if record is not None else {}
    if family in ("agentic_ai", "generative_ai"):
        boundary = params.get("boundary_crossing") or ""
        # Prefer the crossed boundary's gate (e.g. "Tool Authorization").
        if "→" in boundary:
            sig["boundary"] = boundary.split("→")[0].strip().lower().replace(" ", "_")
        tool = adv.get("privileged_tool_requested")
        if tool:
            sig["capability"] = tool
            sig["tool"] = tool
    return {k: v for k, v in sig.items() if v not in (None, "")}


def signature_key(sig: Dict[str, Any]) -> str:
    """A stable key for comparing signatures (ignores field order, drops empties)."""
    relevant = {k: sig[k] for k in sorted(sig) if sig.get(k) not in (None, "")}
    return json.dumps(relevant, sort_keys=True, separators=(",", ":"))


def signature_hash(sig: Dict[str, Any]) -> str:
    return hashlib.sha256(signature_key(sig).encode("utf-8")).hexdigest()[:16]
