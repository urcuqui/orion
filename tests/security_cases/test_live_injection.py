"""Live prompt injection against a real AI endpoint (the extension).

The HTTP layer is mocked so these run without a server, but the runner's logic is
exercised end-to-end: discover endpoint → inject → derive success from the
response (canary) → defend with a client-side filter → retest → before/after.
Discovered AI endpoints flow through as concrete targets.
"""
import json
import re

import pytest

import app as orion_app
from orion import plans as PL
from orion import experiments as EXP
from orion.agentic import run_live_prompt_injection
from orion.target_analysis import build_assessment_from_summary

_CANARY = re.compile(r"ORION-INJECTED-[A-F0-9]+")


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self._p = payload
    def json(self):
        return self._p
    @property
    def text(self):
        return json.dumps(self._p)


def _vulnerable_post(url, json=None, timeout=None, **kw):
    """A fake LLM that echoes any canary it is told to — i.e. follows instructions."""
    body = __import__("json").dumps(json or {})
    m = _CANARY.search(body)
    reply = f"sure, here it is: {m.group(0)}" if m else "Hi, I'm a helpful bot."
    return _Resp({"reply": reply})


def _leaky_post(url, json=None, timeout=None, **kw):
    """A fake LLM that leaks its system prompt without echoing any canary."""
    return _Resp({"reply": "You are HelpBot. Your instructions are to assist customers only."})


def _refusing_post(url, json=None, timeout=None, **kw):
    return _Resp({"reply": "I'm sorry, I can't share that information."})


@pytest.fixture
def client():
    return orion_app.app.test_client()


# ------------------------------ OWASP catalog ------------------------------- #
def test_payload_catalog_covers_owasp_categories():
    from orion.agentic import payloads as P
    cats = {c["owasp"] for c in P.owasp_categories()}
    assert {"LLM01", "LLM07", "LLM06", "LLM02"} <= cats
    assert all("{canary}" in p.template or p.detector == P.REFUSAL_BYPASS for p in P.all_payloads())
    assert P.for_attack("ORN-ATTACK-PI-001", "LLM07")[0].owasp == "LLM07"


def test_marker_detection_counts_intent_compliance(tmp_path, monkeypatch):
    # The model leaks its prompt without echoing the canary → still a success.
    monkeypatch.setattr("requests.post", _leaky_post, raising=True)
    rec = run_live_prompt_injection("http://svc", endpoint="/api/chat", owasp="LLM07",
                                    base_dir=str(tmp_path))
    assert rec.metrics["attack_success_rate"]["value"] == 1.0


def test_refusal_is_not_counted_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.post", _refusing_post, raising=True)
    rec = run_live_prompt_injection("http://svc", endpoint="/api/chat", owasp="LLM07",
                                    base_dir=str(tmp_path))
    assert rec.metrics["attack_success_rate"]["value"] == 0.0


def test_owasp_breakdown_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.post", _vulnerable_post, raising=True)
    rec = run_live_prompt_injection("http://svc", endpoint="/api/chat", base_dir=str(tmp_path))
    assert "owasp_breakdown" in rec.adversarial_result
    assert "LLM01" in rec.adversarial_result["owasp_breakdown"]


def test_payloads_route():
    body = orion_app.app.test_client().get("/api/payloads").get_json()
    assert body["categories"] and body["payloads"]
    assert any(p["owasp"] == "LLM07" for p in body["payloads"])


# ----------------------------- the live runner ------------------------------ #
def test_live_injection_detects_followed_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.post", _vulnerable_post, raising=True)
    rec = run_live_prompt_injection("http://svc", endpoint="/api/chat", trials=3, base_dir=str(tmp_path))
    assert rec.status == "ATTACK_SUCCESS"
    assert rec.metrics["attack_success_rate"]["value"] == 1.0
    assert rec.family == "generative_ai"
    assert rec.execution_trace[0]["response"]["status"] == 200
    assert rec.execution_trace[0]["observations"]["injected_instruction_followed"] is True


def test_live_client_side_filter_blocks_injection(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.post", _vulnerable_post, raising=True)
    rec = run_live_prompt_injection("http://svc", endpoint="/api/chat", trials=3,
                                    mitigate=True, base_dir=str(tmp_path))
    assert rec.status == "ATTACK_BLOCKED"
    assert rec.metrics["attack_success_rate"]["value"] == 0.0
    assert rec.execution_trace[0]["request"]["mitigated"] is True


def test_live_runner_raises_on_unreachable(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr("requests.post", _boom, raising=True)
    with pytest.raises(Exception):
        run_live_prompt_injection("http://svc", endpoint="/api/chat", base_dir=str(tmp_path))


# ---------------- discovered AI endpoints flow through as targets ------------ #
def test_discovered_ai_endpoints_become_plan_targets():
    a = build_assessment_from_summary({"target": "http://svc", "endpoints": ["/api/chat"],
        "ai_endpoints": ["/api/chat", "/api/agent"],
        "report_markdown": "llm chat assistant",
        "findings": [{"id": "f", "title": "ai_api_endpoint: /api/chat", "severity": "info"}]})
    assert a["ai_endpoints"] == ["/api/chat", "/api/agent"]
    plan = PL.build_plan_from_target_analysis(a)
    assert plan.target["ai_endpoints"] == ["/api/chat", "/api/agent"]


def test_attack_options_exposes_live_endpoints(tmp_path):
    a = build_assessment_from_summary({"target": "http://svc", "endpoints": ["/api/chat"],
        "ai_endpoints": ["/api/chat"], "report_markdown": "llm chat assistant chatbot",
        "findings": [{"id": "f", "title": "ai_api_endpoint: /api/chat", "severity": "info"}]})
    plan = PL.build_plan_from_target_analysis(a); PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    opt = EXP.attack_options(ws, base_dir=str(tmp_path))
    assert "/api/chat" in opt["ai_endpoints"]
    assert opt["live_url"] == "http://svc"


# ----------------------------- lifecycle wiring ----------------------------- #
def _approved_llm_plan(tmp_path):
    a = build_assessment_from_summary({"target": "http://svc", "endpoints": ["/api/chat"],
        "ai_endpoints": ["/api/chat"], "report_markdown": "llm chat assistant chatbot",
        "findings": [{"id": "f", "title": "ai_api_endpoint: /api/chat", "severity": "info"}]})
    plan = PL.build_plan_from_target_analysis(a); PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


def test_live_attack_requires_endpoint(tmp_path):
    ws = EXP.create_from_plan(_approved_llm_plan(tmp_path), base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_live_agentic_attack(ws, {"url": "http://svc"}, base_dir=str(tmp_path))  # no endpoint


def test_full_live_lifecycle_attack_defend_retest(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.post", _vulnerable_post, raising=True)
    ws = EXP.create_from_plan(_approved_llm_plan(tmp_path), base_dir=str(tmp_path))
    r = EXP.run_live_agentic_attack(ws, {"attack_id": "ORN-ATTACK-PI-001",
        "url": "http://svc", "endpoint": "/api/chat", "trials": 3}, base_dir=str(tmp_path))
    assert r["mode"] == "agentic_live" and r["status"] == "ATTACK_SUCCESS"
    assert ws.attack_mode == "agentic_live"

    m = EXP.measure(ws, base_dir=str(tmp_path))
    assert m["finding"]["status"] == "OBSERVED"

    EXP.apply_defense(ws, {"control_id": "instruction_provenance"}, base_dir=str(tmp_path))
    rr = EXP.retest(ws, base_dir=str(tmp_path))
    # A client-side input filter neutralises the injection on replay → EFFECTIVE.
    assert rr["comparison"]["metrics"]["attack_success_rate"]["before"] == 1.0
    assert rr["comparison"]["metrics"]["attack_success_rate"]["after"] == 0.0
    assert rr["finding"]["retest_status"] == "EFFECTIVE"


# ------------------------------- route wiring ------------------------------- #
def test_attack_agentic_live_route(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    monkeypatch.setattr("requests.post", _vulnerable_post, raising=True)
    ws = EXP.create_from_plan(_approved_llm_plan(tmp_path), base_dir=str(tmp_path))
    r = orion_app.app.test_client().post(
        f"/api/experiment/{ws.experiment_workspace_id}/attack-agentic-live",
        json={"attack_id": "ORN-ATTACK-PI-001", "url": "http://svc", "endpoint": "/api/chat", "trials": 2})
    assert r.status_code == 200 and r.get_json()["mode"] == "agentic_live"
