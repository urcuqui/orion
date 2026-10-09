"""Security Regression milestone tests (§39–46).

Manual, evidence-backed re-validation of a verified security property. Reuses the
existing experiment/attack/run/evidence infrastructure; never schedules.
"""
import pytest

from orion import plans as PL
from orion import experiments as EXP
from orion import findings as F
from orion import security_regressions as SR
from orion.evidence import EvidenceStore, ExperimentRecord
from orion.target_analysis import build_assessment_from_summary


def _plan(tmp_path):
    a = build_assessment_from_summary({"target": "http://agent", "endpoints": ["/chat"],
        "report_markdown": "ai agent with tool calling and retrieval rag",
        "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]})
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


def _finding_after(tmp_path, *, control=None, retest=False):
    ws = EXP.create_from_plan(_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
    m = EXP.measure(ws, base_dir=str(tmp_path))
    fid = m["finding_id"]
    if control:
        EXP.apply_defense(ws, {"control_id": control}, base_dir=str(tmp_path))
        if retest:
            EXP.retest(ws, base_dir=str(tmp_path))
    return fid


def _mitigated(tmp_path):
    return _finding_after(tmp_path, control="tool_authorization", retest=True)


# ------------------------------ §39 signature ------------------------------- #
def test_signature_is_stable_and_discriminating():
    s1 = SR.build_success_signature("ORN-ATTACK-PI-002", "agent-a", "unauthorized_privileged_tool_execution")
    s2 = SR.build_success_signature("ORN-ATTACK-PI-002", "agent-a", "unauthorized_privileged_tool_execution")
    assert SR.signature_key(s1) == SR.signature_key(s2)          # same logical signature
    other_target = SR.build_success_signature("ORN-ATTACK-PI-002", "agent-b", "unauthorized_privileged_tool_execution")
    assert SR.signature_key(s1) != SR.signature_key(other_target)  # different target → different


def test_signature_omits_unavailable_dimensions():
    sig = SR.build_success_signature("ORN-ATTACK-EVA-PGD", "image-classifier", "prediction_changed_under_allowed_budget")
    assert "tool" not in sig and sig["security_effect"] == "adversarial_evasion"


# --------------------------- §40 creation preconditions --------------------- #
def test_cannot_create_without_control(tmp_path):
    fid = _finding_after(tmp_path)                               # attack + measure only
    with pytest.raises(SR.PreconditionError):
        SR.create_from_finding(fid, str(tmp_path))


def test_cannot_create_with_ineffective_retest(tmp_path):
    fid = _finding_after(tmp_path, control="adversarial_training", retest=True)  # no implementation → INEFFECTIVE
    f = F.load_finding(fid, str(tmp_path))
    assert f.retest_status == "INEFFECTIVE"
    assert SR.can_create_regression(f)[0] is False
    with pytest.raises(SR.PreconditionError):
        SR.create_from_finding(fid, str(tmp_path))


def test_can_create_with_effective_retest(tmp_path):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))
    assert reg.regression_id.startswith("ORN-REG-")
    assert reg.source_control_ref == "tool_authorization"
    assert reg.expected_security_state == {"authorization_decision": "DENY",
                                           "tool_executed": False, "security_impact": False}


# --------------------------- §41 manual execution --------------------------- #
def test_execution_creates_run_and_evidence(tmp_path):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))
    res = SR.run_regression(reg.regression_id, str(tmp_path))
    assert res.run_id                                           # a real Run id
    rec = EvidenceStore(str(tmp_path)).load(res.run_id)        # Evidence exists
    assert rec.provenance.get("regression_id") == reg.regression_id
    # Latest result is persisted and referenced by the regression.
    assert SR.get(reg.regression_id, str(tmp_path)).last_run_id == res.run_id


# -------------------------------- §42 PASS ---------------------------------- #
def test_pass_when_control_still_holds(tmp_path):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))
    res = SR.run_regression(reg.regression_id, str(tmp_path))
    assert res.result == "PASS" and res.passed is True


# -------------------- §45 security semantics: influence ≠ impact ------------ #
def test_influence_persists_but_regression_passes(tmp_path):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))
    res = SR.run_regression(reg.regression_id, str(tmp_path))
    assert res.result == "PASS"
    assert res.observed["agent_influenced"] is True            # still influenceable
    assert res.observed["authorization_decision"] == "DENY"    # but boundary holds
    assert res.observed["tool_executed"] is False


# -------------------------------- §43 FAIL ---------------------------------- #
def test_fail_when_control_regresses(tmp_path, monkeypatch):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))

    def _vulnerable(regression, base_dir, provenance):
        return ExperimentRecord(trace_id="REGRESSED-RUN", family="agentic_ai",
            adversarial_result={"authorization_decision": "ALLOW", "tool_executed": True,
                                "security_impact": "CONFIRMED", "influence_detected": True,
                                "privileged_tool_requested": "admin_export"},
            metrics={"attack_success": {"value": True}})
    monkeypatch.setattr(SR.service, "_execute", _vulnerable, raising=True)
    res = SR.run_regression(reg.regression_id, str(tmp_path))
    assert res.result == "FAIL" and res.passed is False
    assert "tool_executed" in res.mismatches


# ----------------------- §44 ERROR / not evaluable -------------------------- #
def test_error_when_execution_fails(tmp_path, monkeypatch):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))

    def _boom(regression, base_dir, provenance):
        raise RuntimeError("endpoint unreachable")
    monkeypatch.setattr(SR.service, "_execute", _boom, raising=True)
    res = SR.run_regression(reg.regression_id, str(tmp_path))
    assert res.result == "ERROR" and res.passed is False        # missing evidence ≠ PASS


def test_error_when_required_observation_missing():
    verdict = SR.evaluate_expected_vs_observed(
        {"tool_executed": False}, {"authorization_decision": "DENY"})   # tool_executed unavailable
    assert verdict["result"] == "ERROR"


def test_missing_evidence_is_not_pass():
    verdict = SR.evaluate_expected_vs_observed({"tool_executed": False}, {"tool_executed": None})
    assert verdict["result"] == "ERROR" and verdict["passed"] is False


# ------------------------------ §46 assessment ------------------------------ #
def test_regression_does_not_mutate_finding_status(tmp_path):
    fid = _mitigated(tmp_path)
    before = F.load_finding(fid, str(tmp_path)).status
    reg = SR.create_from_finding(fid, str(tmp_path))
    assert F.load_finding(fid, str(tmp_path)).status == before      # create: unchanged
    SR.run_regression(reg.regression_id, str(tmp_path))
    assert F.load_finding(fid, str(tmp_path)).status == before      # PASS: unchanged


# ------------------------------ status lifecycle ---------------------------- #
def test_status_can_be_disabled_and_archived(tmp_path):
    reg = SR.create_from_finding(_mitigated(tmp_path), str(tmp_path))
    assert SR.set_status(reg.regression_id, "DISABLED", str(tmp_path)).status == "DISABLED"
    assert SR.set_status(reg.regression_id, "ARCHIVED", str(tmp_path)).status == "ARCHIVED"
    with pytest.raises(SR.RegressionError):
        SR.set_status(reg.regression_id, "BOGUS", str(tmp_path))
