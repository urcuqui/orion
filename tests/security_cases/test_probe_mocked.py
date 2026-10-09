"""Coverage for orion.target_analysis.probe with HTTP mocked (no real network).

Exercises the active-probe loop and OpenAPI-schema parsing deterministically.
"""
import json

import pytest

from orion.target_analysis import probe as P

_SPEC = {
    "openapi": "3.1.0",
    "info": {"title": "AgentBreak LLM Lab"},
    "paths": {"/": {}, "/api/chat": {}, "/api/scan": {}, "/api/report": {}},
}


class _Resp:
    def __init__(self, status=200, headers=None, text="", payload=None):
        self.status_code = status
        self.headers = headers or {}
        self.text = text
        self._payload = payload

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
                     text="<html>LLM chat assistant chatbot for support</html>")
    return _Resp(404, {"Content-Type": "text/html"}, text="")


def _fake_post(url, timeout=None, allow_redirects=True, **kw):
    # No active inference endpoint present → the active loop records nothing.
    return _Resp(404, {"Content-Type": "text/html"}, text="")


@pytest.fixture
def mock_http(monkeypatch):
    monkeypatch.setattr("requests.get", _fake_get, raising=True)
    monkeypatch.setattr("requests.post", _fake_post, raising=True)


def test_probe_parses_openapi_endpoints(mock_http):
    summary = P.probe_url("http://svc", active=True, timeout=1)
    assert "/api/chat" in summary["endpoints"] and "/api/scan" in summary["endpoints"]
    assert "/api/chat" in summary.get("ai_endpoints", [])
    titles = [f["title"] for f in summary["findings"]]
    assert any("openapi service" in t for t in titles)
    assert any("ai_api_endpoint: /api/chat" == t for t in titles)


def test_probe_get_only_mode(mock_http):
    # active=False skips the POST loop but still discovers the schema.
    summary = P.probe_url("http://svc", active=False, timeout=1)
    assert "/api/chat" in summary["endpoints"]


def test_probe_rejects_non_http_url():
    with pytest.raises(ValueError):
        P.probe_url("ftp://nope", active=False)


def test_probe_dedupes_server_stack_finding(mock_http):
    summary = P.probe_url("http://svc", active=False, timeout=1)
    titles = [f["title"] for f in summary["findings"]]
    assert titles.count("server stack: uvicorn") <= 1
