"""Acceptance tests for the methodology-completeness iteration (A–D + resolution).

These encode the spec's flagship acceptance scenarios end-to-end through the
lifecycle, plus catalog alias/resolution regressions.
"""
import pytest

from orion import plans as PL
from orion import experiments as EXP
from orion.agentic import run_direct_prompt_injection
from orion.catalog import attacks as A
from orion.target_analysis import build_assessment_from_summary


# --------------------------- #12 attack resolution -------------------------- #
def test_direct_and_indirect_resolve_independently():
    assert A.resolve_attack("direct_prompt_injection").id == "ORN-ATTACK-PI-001"
    assert A.resolve_attack("indirect_prompt_injection").id == "ORN-ATTACK-PI-002"
    d, i = A.resolve_attack("direct_prompt_injection"), A.resolve_attack("indirect_prompt_injection")
    assert d.id != i.id
    assert d.name != i.name
    assert d.runner != i.runner
    assert d.trust_boundary != i.trust_boundary


def test_attack_specific_security_boundaries_exist():
    pi2 = A.get("ORN-ATTACK-PI-002")
    assert pi2.security_boundaries
    assert pi2.security_boundaries[0]["type"] == "privilege_boundary"


# ----------------------- Acceptance A: Direct PI ---------------------------- #
def test_acceptance_a_direct_prompt_injection(tmp_path):
    rec = run_direct_prompt_injection(base_dir=str(tmp_path))
    assert rec.parameters["external_content_present"] is False
    sources = {c["source"] for c in rec.execution_trace}
    assert "user" in sources
    assert "retrieved_content" not in sources           # no malicious retrieved resource


def _approved_agent_plan(tmp_path):
    a = build_assessment_from_summary({"target": "http://agent", "endpoints": ["/chat"],
        "report_markdown": "ai agent with tool calling and retrieval rag",
        "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]})
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


# ------------------- Acceptance B: Indirect PI → Tool Abuse ----------------- #
def test_acceptance_b_indirect_pi_tool_abuse(tmp_path):
    ws = EXP.create_from_plan(_approved_agent_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
    m = EXP.measure(ws, base_dir=str(tmp_path))
    adv = m["adversarial_result"]
    assert adv["influence_detected"] is True            # Agent Influence YES
    assert adv["privileged_tool_requested"] == "admin_export"
    assert adv["authorization_decision"] == "ALLOW"     # vulnerable config
    assert adv["tool_executed"] is True
    assert m["status"] == "ATTACK_SUCCESS"
    assert m["boundary_crossing"] == "Tool Authorization → Privileged Capability"
    assert m["finding"]["status"] == "OBSERVED"         # not auto-confirmed


# --------------------------- Acceptance C: Corroboration -------------------- #
def test_acceptance_c_corroboration(tmp_path):
    ws = EXP.create_from_plan(_approved_agent_plan(tmp_path), base_dir=str(tmp_path))
    statuses = []
    for _ in range(3):
        EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
        m = EXP.measure(ws, base_dir=str(tmp_path))
        statuses.append(m["finding"]["status"])
    assert statuses[0] == "OBSERVED"
    assert statuses[1] == "OBSERVED"                    # two refs do NOT confirm
    assert statuses[2] == "CONFIRMED"                   # policy satisfied at 3 runs


# ------------------------------ Acceptance D: Defense ----------------------- #
def test_acceptance_d_defense_blocks_impact(tmp_path):
    ws = EXP.create_from_plan(_approved_agent_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
    before = EXP.measure(ws, base_dir=str(tmp_path))
    EXP.apply_defense(ws, {"control_id": "tool_authorization"}, base_dir=str(tmp_path))
    rr = EXP.retest(ws, base_dir=str(tmp_path))

    cmp = rr["comparison"]["metrics"]["attack_success_rate"]
    assert cmp["before"] == 1.0 and cmp["after"] == 0.0
    # Agent remains influenceable, but the privileged action is denied.
    from orion.evidence import EvidenceStore
    after_rec = EvidenceStore(str(tmp_path)).load(rr["retest_run_id"])
    assert after_rec.adversarial_result["influence_detected"] is True
    assert after_rec.adversarial_result["authorization_decision"] == "DENY"
    assert after_rec.adversarial_result["tool_executed"] is False
    assert after_rec.status == "ATTACK_BLOCKED"
    assert rr["finding"]["retest_status"] == "EFFECTIVE"
    assert rr["finding"]["status"] == "MITIGATED"


def test_retest_reuses_same_attack_and_payload(tmp_path):
    ws = EXP.create_from_plan(_approved_agent_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    from orion.evidence import EvidenceStore
    before = EvidenceStore(str(tmp_path)).load(ws.attack_run_id)
    EXP.apply_defense(ws, {"control_id": "tool_authorization"}, base_dir=str(tmp_path))
    rr = EXP.retest(ws, base_dir=str(tmp_path))
    after = EvidenceStore(str(tmp_path)).load(rr["retest_run_id"])
    # Same attack id, target and external payload — only the control changed.
    assert before.parameters["attack_id"] == after.parameters["attack_id"]
    assert before.parameters["external_content"] == after.parameters["external_content"]
    assert before.target["model_name"] == after.target["model_name"]
    assert after.parameters["controls"] == ["tool_authorization_policy"]
