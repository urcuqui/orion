"""Route-level integration: legacy compatibility + plan handoff endpoints."""
import io

import pytest

import app as orion_app
import orion.adversarial as adv


@pytest.fixture
def client():
    return orion_app.app.test_client()


def test_legacy_red_pill_redirects(client):
    # Backward-compatible URL serves the current Red Team / Attack experience.
    r = client.get("/red-pill.html")
    assert r.status_code in (200, 301, 302)
    body = r.data.decode().lower()
    # Renders the Red Team landing (not the old monolith).
    assert "know your enemy" in body or "red team" in body


def test_legacy_adversarial_endpoint_uses_shared_runner(client, monkeypatch):
    """/adverimage must delegate to the shared experiment service, not its own
    attack logic."""
    calls = {}

    class _Rec:
        trace_id = "TRACE-SHARED-1"
        status = "ATTACK_SUCCESS"

    def _stub(**kwargs):
        calls["used"] = True
        calls["weights_path"] = kwargs.get("weights_path")
        return _Rec()

    # The handler imports run_adversarial_experiment from orion.adversarial at call time.
    monkeypatch.setattr(adv, "run_adversarial_experiment", _stub, raising=True)

    data = {
        "weights": (io.BytesIO(b"dummy-weights"), "model.pth"),
        "file": (io.BytesIO(b"dummy-image"), "img.png"),
        "numberoutputs": "2",
    }
    r = client.post("/adverimage", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert calls.get("used") is True                 # shared runner was invoked
    assert body.get("trace_id") == "TRACE-SHARED-1"  # structured result surfaced


def test_plan_handoff_endpoint(client):
    ky = client.post("/api/know-yourself/analyze", json={"descriptor": {
        "task": "image_classification", "input_type": "image",
        "access": "white_box", "framework": "pytorch"}}).get_json()
    ap = client.post("/api/plans/approve", json={"source_type": "know_yourself", "analysis": ky}).get_json()
    pid = ap["plan"]["plan_id"]
    hf = client.get(f"/api/plans/{pid}/handoff").get_json()
    assert hf["handoff_id"].startswith("ORN-HANDOFF-")
    assert hf["status"] == "READY_FOR_ATTACK_WORKSPACE"
