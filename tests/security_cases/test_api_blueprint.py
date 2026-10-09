"""Integration coverage for the /api blueprint: list endpoints, error branches,
and a few safe happy paths. All state writes target a tmp artifact dir so the
real store is never touched. Security behaviour is unchanged — these only
exercise HTTP wiring and error handling.
"""
import pytest

import app as orion_app
import orion.integrations.flask_blueprint as bp
from orion.evidence import EvidenceStore, ExperimentRecord


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    return orion_app.app.test_client(), str(tmp_path)


# ------------------------------- GET lists ---------------------------------- #
@pytest.mark.parametrize("path,key", [
    ("/api/scenarios", "scenarios"),
    ("/api/runs", "runs"),
    ("/api/profiles", "environment"),
    ("/api/recon/runs", None),
    ("/api/catalog", "families"),
    ("/api/payloads", "payloads"),
    ("/api/findings", "findings"),
    ("/api/experiment", "workspaces"),
    ("/api/mcp_tools", "tools"),
])
def test_get_list_endpoints_return_200_json(api, path, key):
    client, _ = api
    r = client.get(path)
    assert r.status_code == 200
    body = r.get_json()
    assert isinstance(body, dict)
    if key:
        assert key in body


# --------------------------- unknown-id 404 paths --------------------------- #
@pytest.mark.parametrize("path", [
    "/api/runs/UNKNOWN",
    "/api/environment/from-recon/UNKNOWN",
    "/api/environment/UNKNOWN",
    "/api/environment/UNKNOWN/analyze",
    "/api/context/UNKNOWN",
    "/api/plans/UNKNOWN",
    "/api/plans/UNKNOWN/handoff",
    "/api/experiment/UNKNOWN",
    "/api/findings/UNKNOWN",
])
def test_unknown_id_returns_404(api, path):
    client, _ = api
    r = client.post(path) if path.endswith(("from-recon/UNKNOWN", "/analyze")) else client.get(path)
    # POST-only routes answered with GET will 405; the read routes 404.
    assert r.status_code in (404, 405)


def test_replay_unknown_trace_is_404(api):
    client, _ = api
    assert client.post("/api/replay/UNKNOWN", json={"mode": "hardened"}).status_code == 404


# ------------------------------ 400 bad input ------------------------------- #
def test_env_from_url_requires_url(api):
    client, _ = api
    r = client.post("/api/environment/from-url", json={})
    assert r.status_code == 400 and "url" in r.get_json()["error"]


def test_compare_requires_traces_mapping(api):
    client, _ = api
    assert client.post("/api/compare", json={}).status_code == 400


def test_know_yourself_requires_some_input(api):
    client, _ = api
    assert client.post("/api/know-yourself/analyze", json={}).status_code == 400


def test_scenario_run_bad_path_is_400(api):
    client, _ = api
    assert client.post("/api/scenario-run", json={"scenario": "no/such.yaml"}).status_code == 400


# --------------------------- artifact path guards --------------------------- #
@pytest.mark.parametrize("path,code", [
    ("/api/artifacts/RUN/evil.txt", 400),        # unsupported type
    ("/api/artifacts/RUN/..%2fescape.png", 400),  # traversal rejected by filename guard
    ("/api/artifacts/RUN/missing.png", 404),     # valid type, absent
])
def test_artifact_path_validation(api, path, code):
    client, _ = api
    assert client.get(path).status_code == code


# ------------------------------ happy paths --------------------------------- #
def test_scenario_run_baseline_ok(api):
    client, _ = api
    r = client.post("/api/scenario-run", json={"mode": "baseline"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "NO_ATTACK"


def test_know_yourself_descriptor_ok(api):
    client, _ = api
    r = client.post("/api/know-yourself/analyze", json={"descriptor": {
        "task": "image_classification", "input_type": "image", "framework": "pytorch"}})
    assert r.status_code == 200 and isinstance(r.get_json(), dict)


def test_compare_builds_matrix_from_saved_runs(api):
    client, base = api
    store = EvidenceStore(base)
    a = ExperimentRecord(scenario_name="b", status="NO_ATTACK",
                         metrics={"robust_accuracy": {"value": 0.9}})
    b = ExperimentRecord(scenario_name="a", status="ATTACK_SUCCESS",
                         metrics={"robust_accuracy": {"value": 0.5}})
    store.save(a); store.save(b)
    r = client.post("/api/compare", json={"traces": {"BASELINE": a.trace_id, "ATTACK": b.trace_id}})
    assert r.status_code == 200
    body = r.get_json()
    assert body["columns"] == ["BASELINE", "ATTACK"]
    row = next(m for m in body["matrix"] if m["metric"] == "robust_accuracy")
    assert row["BASELINE"] == 0.9 and row["ATTACK"] == 0.5
