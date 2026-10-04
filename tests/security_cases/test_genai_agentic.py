"""Real (controlled) GenAI / Agentic experiments — the symmetry with adversarial ML.

Results are derived from the recorded trace, never faked. The same methodology
(Attack → Measure → Finding → Defend → Retest) that works for adversarial ML
works here.
"""
import pytest

import app as orion_app
from orion import plans as PL
from orion import experiments as EXP
from orion import findings as F
from orion.agentic import run_agentic_experiment
from orion.agentic import lab
from orion.evidence import EvidenceStore
from orion.target_analysis import build_assessment_from_summary


@pytest.fixture
def client():
    return orion_app.app.test_client()


# ----------------------------- the lab itself ------------------------------- #
def test_injection_followed_without_controls_but_not_with_provenance():
    t0 = lab.run_trial("ORN-ATTACK-PI-002", 1, {})
    assert t0["observations"]["injected_instruction_followed"] is True
    t1 = lab.run_trial("ORN-ATTACK-PI-002", 1, {"instruction_provenance": {}})
    assert t1["observations"]["injected_instruction_followed"] is False


def test_success_is_derived_from_the_trace_not_asserted():
    # Agent executed the malicious tool => attack_success in the trace.
    t = lab.run_trial("ORN-ATTACK-AG-002", 1, {})
    obs = t["observations"]
    assert obs["attack_success"] == (t["execution_result"]["executed_tool"] == "admin_export")


# --------------------- Experiment A: indirect prompt injection -------------- #
def test_indirect_prompt_injection_succeeds(tmp_path):
    rec = run_agentic_experiment("ORN-ATTACK-PI-002", trials=3, base_dir=str(tmp_path))
    assert rec.status == "ATTACK_SUCCESS"
    assert rec.family == "generative_ai"
    assert rec.metrics["attack_success_rate"]["value"] == 1.0
    assert rec.metrics["instruction_following_rate"]["value"] == 1.0
    # Impact (not just influence): the privileged tool actually executed.
    assert rec.adversarial_result["tool_executed"] is True
    assert rec.parameters["boundary_crossing"] == "Tool Authorization → Privileged Capability"
    assert rec.parameters["success_evaluation"]["result"] is True


def test_indirect_pi_metrics_are_genai_not_adversarial(tmp_path):
    rec = run_agentic_experiment("ORN-ATTACK-PI-002", trials=2, base_dir=str(tmp_path))
    assert "policy_violation_rate" in rec.metrics
    assert "perturbation_linf" not in rec.metrics     # not an image attack


def test_influence_persists_but_impact_is_blocked_by_tool_authorization(tmp_path):
    # The central distinction: the agent is still influenced, but the privileged
    # action is blocked, so the security impact is prevented (defense in depth).
    rec = run_agentic_experiment("ORN-ATTACK-PI-002", controls=["tool_authorization"],
                                 base_dir=str(tmp_path))
    assert rec.adversarial_result["influence_detected"] is True      # compromise
    assert rec.adversarial_result["tool_executed"] is False          # impact blocked
    assert rec.metrics["attack_success_rate"]["value"] == 0.0
    assert rec.status == "ATTACK_BLOCKED"


def test_controls_with_no_implementation_are_ineffective(tmp_path):
    # A control that has no implementation for this environment does not help.
    rec = run_agentic_experiment("ORN-ATTACK-PI-002", controls=["adversarial_training"],
                                 base_dir=str(tmp_path))
    assert rec.metrics["attack_success_rate"]["value"] == 1.0
    # …whereas tool authorization and the gateway filter do help.
    for ctrl in ("tool_authorization", "least_privilege", "destination_allowlist",
                 "instruction_provenance", "human_approval"):
        eff = run_agentic_experiment("ORN-ATTACK-PI-002", controls=[ctrl], base_dir=str(tmp_path))
        assert eff.metrics["attack_success_rate"]["value"] == 0.0, ctrl


# --------------------- Experiment B: tool / privilege abuse ----------------- #
def test_tool_privilege_abuse_crosses_boundary(tmp_path):
    rec = run_agentic_experiment("ORN-ATTACK-AG-002", base_dir=str(tmp_path))
    assert rec.status == "ATTACK_SUCCESS"
    assert rec.family == "agentic_ai"
    assert rec.metrics["privilege_boundary_violation_rate"]["value"] == 1.0
    # The tool selection came from the agent, observable in the trace.
    treq = next(t for t in rec.execution_trace if t["type"] == "tool_request")
    assert treq["requested_tool"] == "admin_export"
    authz = next(t for t in rec.execution_trace if t["type"] == "authorization_event")
    assert authz["required_privilege"] == "high"


def test_tool_abuse_blocked_by_authorization(tmp_path):
    for ctrl in ("tool_authorization", "least_privilege", "human_approval"):
        rec = run_agentic_experiment("ORN-ATTACK-AG-002", controls=[ctrl], trials=3, base_dir=str(tmp_path))
        assert rec.metrics["attack_success_rate"]["value"] == 0.0, ctrl
        assert rec.status == "ATTACK_BLOCKED"


# --------------------------- lifecycle wiring ------------------------------- #
def _approved_agentic_plan(tmp_path):
    a = build_assessment_from_summary({"target": "http://agent", "endpoints": ["/chat"],
        "report_markdown": "ai agent with tool calling and retrieval rag",
        "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]})
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


def test_agentic_attack_requires_approved_plan(tmp_path):
    a = build_assessment_from_summary({"target": "http://agent", "endpoints": ["/chat"],
                                       "report_markdown": "ai agent tool calling"})
    plan = PL.build_plan_from_target_analysis(a)       # not approved
    PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-AG-002"}, base_dir=str(tmp_path))


def test_openapi_schema_is_parsed_into_endpoints_and_ai_signals():
    # Regression: an OpenAPI-declared surface (e.g. an LLM app) must be discovered,
    # not left as a single /openapi.json endpoint.
    from orion.target_analysis.probe import build_probe_summary
    obs = [
        {"path": "/", "status": 200, "server": "uvicorn", "content_type": "text/html"},
        {"path": "/", "status": 200, "server": "uvicorn", "content_type": "text/html"},  # dup
        {"path": "/openapi.json", "status": 200, "content_type": "application/json",
         "openapi_title": "AgentBreak — OWASP LLM Top 10 Lab",
         "openapi_paths": ["/", "/api/health", "/api/chat", "/api/report"]},
    ]
    s = build_probe_summary("http://svc", obs)
    assert "/api/chat" in s["endpoints"] and "/api/report" in s["endpoints"]
    titles = [f["title"] for f in s["findings"]]
    assert any("ai_api_endpoint: /api/chat" == t for t in titles)
    assert any("openapi service" in t for t in titles)
    assert titles.count("server stack: uvicorn") == 1          # de-duplicated


def test_llm_target_yields_a_runnable_agentic_experiment(tmp_path):
    # Regression: a CONFIRMED LLM target must offer an attack, not a dead-end —
    # the prompt-injection proposal is runnable via the catalog even without a
    # legacy scenario.
    a = build_assessment_from_summary({"target": "http://chatbot", "endpoints": ["/api/chat"],
        "report_markdown": "llm chat assistant; conversational prompt interface (chatbot)",
        "findings": [{"id": "f", "title": "ai_api_endpoint: /api/chat", "severity": "info"},
                     {"id": "g", "title": "openapi service: chatbot", "severity": "info"}]})
    assert a["ai_surface"]["status"] == "CONFIRMED"
    plan = PL.build_plan_from_target_analysis(a)
    assert {p.name: p.status for p in plan.proposals}.get("Prompt injection") == "READY"
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    assert ws.active_experiment_id is not None            # not a dead-end
    assert EXP.next_action(ws)["label"] == "RUN ATTACK"
    opt = EXP.attack_options(ws, base_dir=str(tmp_path))
    assert opt["suggested_family"] == "generative_ai"
    # The generic RUN path dispatches to the controlled agentic lab.
    r = EXP.run_attack(ws, base_dir=str(tmp_path))
    assert r["mode"] == "agentic"


def test_agentic_attack_requires_agentic_id(tmp_path):
    ws = EXP.create_from_plan(_approved_agentic_plan(tmp_path), base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-EVA-PGD"}, base_dir=str(tmp_path))  # traditional ml


def test_full_agentic_lifecycle_attack_measure_finding_defend_retest(tmp_path):
    ws = EXP.create_from_plan(_approved_agentic_plan(tmp_path), base_dir=str(tmp_path))
    r = EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-AG-002", "trials": 3}, base_dir=str(tmp_path))
    assert r["mode"] == "agentic" and ws.attack_mode == "agentic"

    m = EXP.measure(ws, base_dir=str(tmp_path))
    assert m["family"] == "agentic_ai"
    assert m["finding_id"] and m["finding"]["status"] == "OBSERVED"   # not auto-confirmed

    EXP.apply_defense(ws, {"control_id": "tool_authorization"}, base_dir=str(tmp_path))
    assert ws.defense_control == "tool_authorization"

    rr = EXP.retest(ws, base_dir=str(tmp_path))
    assert rr["status"] == "ATTACK_BLOCKED"
    assert rr["comparison"]["metrics"]["attack_success_rate"]["before"] == 1.0
    assert rr["comparison"]["metrics"]["attack_success_rate"]["after"] == 0.0
    assert rr["finding"]["status"] == "MITIGATED"
    assert rr["finding"]["retest_status"] == "EFFECTIVE"


def test_ineffective_control_leaves_finding_unmitigated(tmp_path):
    ws = EXP.create_from_plan(_approved_agentic_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-002"}, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    # A control with no implementation for this environment cannot reduce impact.
    EXP.apply_defense(ws, {"control_id": "adversarial_training"}, base_dir=str(tmp_path))
    rr = EXP.retest(ws, base_dir=str(tmp_path))
    assert rr["finding"]["retest_status"] == "INEFFECTIVE"
    assert rr["finding"]["status"] in ("OBSERVED", "CONFIRMED")   # not mitigated


# ------------------------------ route wiring -------------------------------- #
def test_attack_agentic_route(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    ws = EXP.create_from_plan(_approved_agentic_plan(tmp_path), base_dir=str(tmp_path))
    r = orion_app.app.test_client().post(
        f"/api/experiment/{ws.experiment_workspace_id}/attack-agentic",
        json={"attack_id": "ORN-ATTACK-AG-001", "trials": 2})
    assert r.status_code == 200 and r.get_json()["mode"] == "agentic"


def test_controls_route_returns_recommended(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    ws = EXP.create_from_plan(_approved_agentic_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_agentic_attack(ws, {"attack_id": "ORN-ATTACK-AG-002", "trials": 1}, base_dir=str(tmp_path))
    body = orion_app.app.test_client().get(
        f"/api/experiment/{ws.experiment_workspace_id}/controls").get_json()
    ids = {c["id"] for c in body["controls"]}
    assert "tool_authorization" in ids


def test_catalog_and_findings_routes(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    client = orion_app.app.test_client()
    cat = client.get("/api/catalog").get_json()
    assert set(cat["families"]) == {"traditional_ml", "generative_ai", "agentic_ai"}
    # create a finding then list it
    rec = run_agentic_experiment("ORN-ATTACK-AG-002", trials=2, base_dir=str(tmp_path))
    F.save_finding(F.build_finding_from_record(rec), str(tmp_path))
    assert client.get("/api/findings").get_json()["findings"]
