"""Integration coverage for the remaining /api flows: probe, plans and the full
experiment lifecycle driven over HTTP. Deterministic (synthetic adversarial
scenario, no torch / no network), tmp artifact dir, security semantics unchanged.
"""
import json

import pytest

import app as orion_app
import orion.integrations.flask_blueprint as bp
from orion.target_analysis import build_assessment_from_summary


_SPEC = {"openapi": "3.1.0", "info": {"title": "Svc"}, "paths": {"/": {}, "/api/chat": {}}}


class _Resp:
    def __init__(self, status=200, headers=None, text="", payload=None):
        self.status_code, self.headers, self.text, self._payload = status, headers or {}, text, payload

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _fake_get(url, timeout=None, allow_redirects=True):
    from urllib.parse import urlparse
    path = urlparse(url).path or "/"
    if path.endswith("openapi.json"):
        return _Resp(200, {"Content-Type": "application/json"}, text=json.dumps(_SPEC), payload=_SPEC)
    if path == "/":
        return _Resp(200, {"Server": "uvicorn", "Content-Type": "text/html"}, text="<html>api</html>")
    return _Resp(404, {"Content-Type": "text/html"}, text="")


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    monkeypatch.setattr("requests.get", _fake_get, raising=True)
    monkeypatch.setattr("requests.post", lambda *a, **k: _Resp(404, {"Content-Type": "text/html"}), raising=True)
    return orion_app.app.test_client(), str(tmp_path)


def _ml_assessment():
    return build_assessment_from_summary({
        "target": "svc", "endpoints": ["/v1/predict"],
        "report_markdown": "pytorch model inference classifier",
        "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]})


# ------------------------- analysis entry points ---------------------------- #
def test_probe_route_builds_assessment(api):
    client, _ = api
    r = client.post("/api/target-analysis/probe", json={"url": "http://svc", "active": False})
    assert r.status_code == 200
    assert r.get_json()["ai_surface"]["status"] in ("CONFIRMED", "POSSIBLE", "NOT_OBSERVED")
    assert client.post("/api/target-analysis/probe", json={}).status_code == 400


def test_agent_analyze_context(api):
    client, _ = api
    ok = client.post("/api/agent/analyze-context",
                     json={"context": "an llm chat assistant with tool calling", "target": "svc"})
    assert ok.status_code == 200
    assert client.post("/api/agent/analyze-context", json={}).status_code == 400


# ------------------------------- plans flow --------------------------------- #
def test_plans_draft_then_approve_by_id(api):
    client, _ = api
    draft = client.post("/api/plans/draft",
                        json={"source_type": "know_your_target", "analysis": _ml_assessment()})
    assert draft.status_code == 200
    plan_id = draft.get_json()["plan"]["plan_id"]
    assert client.get(f"/api/plans/{plan_id}").status_code == 200
    approved = client.post(f"/api/plans/{plan_id}/approve", json={})
    assert approved.status_code == 200 and approved.get_json()["plan"]["approved_by_human"] is True
    assert client.get(f"/api/plans/{plan_id}/handoff").status_code == 200


def test_plans_approve_requires_valid_source(api):
    client, _ = api
    assert client.post("/api/plans/approve", json={}).status_code == 400
    assert client.post("/api/plans/draft", json={"source_type": "bogus"}).status_code == 400


# ----------------------- full experiment lifecycle -------------------------- #
def test_experiment_lifecycle_over_http(api):
    client, _ = api
    plan = client.post("/api/plans/approve",
                       json={"source_type": "know_your_target", "analysis": _ml_assessment()})
    plan_id = plan.get_json()["plan"]["plan_id"]

    ws = client.post(f"/api/experiment/from-plan/{plan_id}").get_json()
    ws_id = ws["experiment_workspace_id"]
    assert ws.get("active_experiment_id")                       # adversarial scenario is runnable
    # Reuse path: a second from-plan returns the same workspace.
    assert client.post(f"/api/experiment/from-plan/{plan_id}").get_json()["experiment_workspace_id"] == ws_id

    assert client.get(f"/api/experiment/{ws_id}").status_code == 200
    assert client.post(f"/api/experiment/{ws_id}/attack").status_code == 200
    assert client.post(f"/api/experiment/{ws_id}/measure").status_code == 200
    assert client.get(f"/api/experiment/{ws_id}/controls").status_code == 200
    assert client.post(f"/api/experiment/{ws_id}/defend",
                       json={"control_id": "input_preprocessing"}).status_code == 200
    retest = client.post(f"/api/experiment/{ws_id}/retest")
    assert retest.status_code == 200 and "comparison" in retest.get_json()


def test_plan_run_single_experiment_route(api):
    client, _ = api
    plan_id = client.post("/api/plans/approve",
                          json={"source_type": "know_your_target", "analysis": _ml_assessment()}
                          ).get_json()["plan"]["plan_id"]
    r = client.post(f"/api/plans/{plan_id}/experiments/EXP-01/run")
    assert r.status_code in (200, 400)                          # runs or NOT_RUNNABLE, both exercised


# -------------------------- compare skip branch ----------------------------- #
def test_compare_skips_unloadable_trace(api):
    client, _ = api
    body = client.post("/api/compare", json={"traces": {"BASELINE": "MISSING-TRACE"}}).get_json()
    assert body["columns"] == ["BASELINE"] and body["matrix"] == []


# --------------------- adversarial/run input validation --------------------- #
def test_adversarial_run_validation_branches(api):
    client, _ = api
    r = client.get("/api/catalog")  # warm app
    assert r.status_code == 200
    from orion.adversarial import TORCH_AVAILABLE
    if not TORCH_AVAILABLE:
        assert client.post("/api/adversarial/run", json={}).status_code == 503
        return
    # No files supplied → 400 before any model is loaded.
    assert client.post("/api/adversarial/run", json={}).status_code == 400
    # Weights path that does not exist under weights/ → 400.
    assert client.post("/api/adversarial/run",
                       json={"weights_path": "weights/does-not-exist.pth"}).status_code == 400
    # Existing weights but a missing image under static/ → 400 (no attack runs).
    assert client.post("/api/adversarial/run", json={
        "weights_path": "weights/vit_teacher.pth", "image": "static/missing.jpg"}).status_code == 400


# ----------------------- select + unknown-id 404s --------------------------- #
def test_select_experiment_route(api):
    client, _ = api
    plan_id = client.post("/api/plans/approve",
                          json={"source_type": "know_your_target", "analysis": _ml_assessment()}
                          ).get_json()["plan"]["plan_id"]
    ws_id = client.post(f"/api/experiment/from-plan/{plan_id}").get_json()["experiment_workspace_id"]
    r = client.post(f"/api/experiment/{ws_id}/select/EXP-01")
    assert r.status_code == 200


def test_unknown_target_analysis_and_threat_model_are_404(api):
    client, _ = api
    assert client.get("/api/target-analysis/UNKNOWN-RUN").status_code == 404
    assert client.post("/api/threat-model/UNKNOWN-RUN/approve", json={"approved": True}).status_code == 404
