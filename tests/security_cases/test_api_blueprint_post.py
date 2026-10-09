"""Integration coverage for the /api POST flows (environment, profiles, context).

Network is mocked; all writes target a tmp artifact dir. Security behaviour is
unchanged — these exercise the HTTP wiring and persistence of the context pillars.
"""
import json

import pytest

import app as orion_app
import orion.integrations.flask_blueprint as bp


_SPEC = {"openapi": "3.1.0", "info": {"title": "LLM Lab"},
         "paths": {"/": {}, "/api/chat": {}}}


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
        return _Resp(200, {"Server": "uvicorn", "Content-Type": "text/html"},
                     text="<html>LLM chat assistant</html>")
    return _Resp(404, {"Content-Type": "text/html"}, text="")


def _fake_post(url, timeout=None, allow_redirects=True, **kw):
    return _Resp(404, {"Content-Type": "text/html"}, text="")


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    monkeypatch.setattr("requests.get", _fake_get, raising=True)
    monkeypatch.setattr("requests.post", _fake_post, raising=True)
    return orion_app.app.test_client(), str(tmp_path)


# --------------------------- environment pillar ----------------------------- #
def test_environment_from_url_build_get_analyze(api):
    client, _ = api
    built = client.post("/api/environment/from-url", json={"url": "http://svc", "active": False})
    assert built.status_code == 200
    env_id = built.get_json()["environment_profile_id"]

    assert client.get(f"/api/environment/{env_id}").status_code == 200
    analyzed = client.post(f"/api/environment/{env_id}/analyze")
    assert analyzed.status_code == 200
    assert analyzed.get_json()["environment_profile_id"] == env_id


# ------------------------------ target profile ------------------------------ #
def test_target_profile_persisted_and_listed(api):
    client, _ = api
    r = client.post("/api/target-profile", json={
        "name": "demo-lab", "objective": "map the attack surface",
        "target_type": "api_service", "scope": {"authorization": "authorized lab"}})
    assert r.status_code == 200
    assert r.get_json().get("target_profile_id")
    profiles = client.get("/api/profiles").get_json()
    assert any(p["id"] for p in profiles["target"])


# ----------------------------- analysis context ----------------------------- #
def test_context_build_and_get(api):
    client, _ = api
    built = client.post("/api/context", json={})
    assert built.status_code == 200
    ctx_id = built.get_json()["analysis_context_id"]
    assert client.get(f"/api/context/{ctx_id}").status_code == 200


def test_context_analyze_requires_a_profile(api):
    client, _ = api
    assert client.post("/api/context/analyze", json={}).status_code == 400


def test_context_analyze_with_environment_profile(api):
    client, _ = api
    env_id = client.post("/api/environment/from-url",
                         json={"url": "http://svc"}).get_json()["environment_profile_id"]
    r = client.post("/api/context/analyze", json={"environment_profile_id": env_id})
    assert r.status_code == 200
    body = r.get_json()
    assert body["environment_profile_id"] == env_id and body["analysis_context_id"]
