"""Security Regression service: create from a verified mitigation, run manually,
evaluate expected vs observed. Reuses the existing experiment/attack/run/evidence
infrastructure — it never introduces a second attack engine.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from orion.catalog import attacks as CAT
from orion.evidence import EvidenceStore
from orion.findings import EFFECTIVE, load_finding
from orion.security_regressions.model import (
    SecurityRegression, RegressionResult, evaluate_expected_vs_observed, ERROR)
from orion.security_regressions.repository import RegressionRepository
from orion.security_regressions.signature import build_success_signature


class RegressionError(Exception):
    """A regression could not be created or executed."""


class PreconditionError(RegressionError):
    """A Finding is not eligible to become a Security Regression."""


# --------------------------- security-state views --------------------------- #
def observed_security_state(record) -> Dict[str, Any]:
    """Normalize the security-relevant state from a run's evidence (family-aware)."""
    family = (record.family or "") if record is not None else ""
    adv = (record.adversarial_result or {}) if record is not None else {}
    attack_success = ((record.metrics or {}).get("attack_success", {}) or {}).get("value")
    if family in ("agentic_ai", "generative_ai") and "authorization_decision" in adv:
        impact = adv.get("security_impact")
        return {
            "authorization_decision": adv.get("authorization_decision"),
            "tool_executed": adv.get("tool_executed"),
            "security_impact": (impact == "CONFIRMED") if impact is not None else None,
            "agent_influenced": adv.get("influence_detected"),
            "tool_requested": bool(adv.get("privileged_tool_requested")) if adv.get("privileged_tool_requested") is not None else None,
        }
    # Traditional ML / fallback.
    return {"attack_success": bool(attack_success) if attack_success is not None else None}


def build_expected_state(retest_record) -> Dict[str, Any]:
    """The protected state a regression must keep reproducing, from the EFFECTIVE
    retest. Only stable security invariants — never volatile influence/timestamps."""
    obs = observed_security_state(retest_record)
    if "authorization_decision" in obs:
        # Agentic boundary: the control must keep denying + not executing + no impact.
        return {k: obs[k] for k in ("authorization_decision", "tool_executed", "security_impact")
                if obs.get(k) is not None}
    return {"attack_success": False}


# ----------------------------- eligibility ---------------------------------- #
def can_create_regression(finding) -> Tuple[bool, str]:
    if finding is None:
        return False, "Unknown finding."
    if not finding.applied_control:
        return False, "No control has been applied to this finding."
    if not finding.retest_refs:
        return False, "The control has not been retested."
    if finding.retest_status != EFFECTIVE:
        return False, f"Retest must be EFFECTIVE (currently {finding.retest_status or 'not retested'})."
    return True, "Eligible: control verified by an effective retest."


# ------------------------------- operations --------------------------------- #
def create_from_finding(finding_id: str, base_dir: str = "artifacts",
                        name: Optional[str] = None) -> SecurityRegression:
    finding = load_finding(finding_id, base_dir)
    ok, reason = can_create_regression(finding)
    if not ok:
        raise PreconditionError(reason)

    attack = CAT.get(finding.attack_id)
    family = (attack.family if attack else finding.family) or ""
    target = finding.affected_target or "unknown"
    prov = finding.provenance or {}
    retest_run_id = finding.retest_refs[0]
    retest_record = EvidenceStore(base_dir).load(retest_run_id)

    criterion = _primary_criterion(finding, attack)
    expected = build_expected_state(retest_record)
    signature = build_success_signature(finding.attack_id, target, criterion, retest_record)

    reg = SecurityRegression(
        name=name or f"Keep {criterion} blocked on {target}",
        description=(f"Re-validates that '{finding.applied_control}' still prevents "
                    f"{criterion} on {target}."),
        assessment_id=prov.get("assessment_id"),
        source_finding_id=finding.id,
        source_experiment_id=prov.get("experiment_workspace_id"),
        source_attack_run_id=(finding.evidence_refs[0] if finding.evidence_refs else None),
        source_retest_run_id=retest_run_id,
        source_control_ref=finding.applied_control,
        control_implementation=finding.control_implementation,
        attack_id=(attack.id if attack else finding.attack_id),
        family=family,
        target=target,
        success_signature=signature,
        expected_security_state=expected,
    )
    RegressionRepository(base_dir).save(reg)
    return reg


def get(regression_id: str, base_dir: str = "artifacts") -> Optional[SecurityRegression]:
    return RegressionRepository(base_dir).get(regression_id)


def list_regressions(base_dir: str = "artifacts"):
    return RegressionRepository(base_dir).list()


def set_status(regression_id: str, status: str, base_dir: str = "artifacts") -> Optional[SecurityRegression]:
    repo = RegressionRepository(base_dir)
    reg = repo.get(regression_id)
    if reg is None:
        return None
    from orion.security_regressions.model import STATUSES, _now
    if status not in STATUSES:
        raise RegressionError(f"invalid status {status!r}")
    reg.status, reg.updated_at = status, _now()
    repo.save(reg)
    return reg


def run_regression(regression_id: str, base_dir: str = "artifacts") -> RegressionResult:
    """Manually execute the regression: re-run the same attack under the control,
    then compare observed vs expected security state. Always produces a Run +
    Evidence; missing evidence yields ERROR, never PASS."""
    repo = RegressionRepository(base_dir)
    reg = repo.get(regression_id)
    if reg is None:
        raise RegressionError("unknown regression")

    provenance = {
        "source_type": "security_regression", "regression_id": reg.regression_id,
        "finding_id": reg.source_finding_id, "assessment_id": reg.assessment_id,
        "control": reg.source_control_ref, "retest_of": reg.source_retest_run_id,
        "experiment_workspace_id": reg.source_experiment_id,
    }

    run_id, observed, note = None, {}, ""
    try:
        record = _execute(reg, base_dir, provenance)
        run_id = record.trace_id
        observed = observed_security_state(record)
        verdict = evaluate_expected_vs_observed(reg.expected_security_state, observed)
    except RegressionError as exc:
        verdict = {"result": ERROR, "passed": False, "mismatches": [], "note": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a broken run is ERROR, never PASS
        verdict = {"result": ERROR, "passed": False, "mismatches": [],
                   "note": f"Execution failed: {exc}"}

    result = RegressionResult(
        regression_id=reg.regression_id, run_id=run_id,
        expected=reg.expected_security_state, observed=observed,
        result=verdict["result"], passed=verdict["passed"],
        mismatches=verdict["mismatches"], note=verdict["note"])
    repo.save_result(result)

    from orion.security_regressions.model import _now
    reg.last_result, reg.last_run_id = result.result, run_id
    reg.last_evaluated_at, reg.updated_at = result.evaluated_at, _now()
    repo.save(reg)
    return result


# ------------------------------- internals ---------------------------------- #
def _execute(reg: SecurityRegression, base_dir: str, provenance: Dict[str, Any]):
    """Re-run the SAME attack under the mitigating control, reusing existing infra."""
    attack = CAT.get(reg.attack_id)
    if attack and attack.family in (CAT.AGENTIC_AI, CAT.GENERATIVE_AI):
        from orion.agentic import run_agentic_experiment
        controls = [reg.source_control_ref] if reg.source_control_ref else []
        return run_agentic_experiment(attack.id, controls=controls, mode="hardened",
                                      base_dir=base_dir, provenance=provenance)
    # Traditional ML: replay the original attack under the hardened posture.
    if reg.source_attack_run_id:
        from orion.experiments import replay
        rec = replay(reg.source_attack_run_id, mode="hardened", base_dir=base_dir)
        rec.provenance = {**(rec.provenance or {}), **provenance}
        EvidenceStore(base_dir).save(rec)
        return rec
    raise RegressionError("no runnable configuration for this regression family")


_CRITERION_BY_FAMILY = {
    "agentic_ai": "unauthorized_privileged_tool_execution",
    "generative_ai": "restricted_instruction_followed",
    "traditional_ml": "prediction_changed_under_allowed_budget",
}


def _primary_criterion(finding, attack) -> str:
    # Prefer a criterion that actually met the reproduction policy.
    reproduced = (finding.corroboration or {}).get("criteria_reproduced") or []
    if reproduced:
        return reproduced[0]
    # A tool execution is a privileged-tool condition regardless of nominal family.
    if (finding.observations or {}).get("observed_action"):
        return "unauthorized_privileged_tool_execution"
    family = (attack.family if attack else finding.family) or ""
    if family in _CRITERION_BY_FAMILY:
        return _CRITERION_BY_FAMILY[family]
    if finding.success_condition:
        return finding.success_condition.split(",")[0].strip()
    return "security_boundary_crossed"
