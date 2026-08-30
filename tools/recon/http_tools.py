"""Mock HTTP tools: GET, POST and header inspection.

These simulate HTTP interactions using the deterministic site profile. They
make no real network requests and return structured Pydantic payloads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool
from tools.recon.fixtures import build_profile, normalise_target


def _response_headers(
    profile: Any, content_type: str = "text/html; charset=utf-8"
) -> dict[str, str]:
    """Build a synthetic response header set from a site profile."""
    headers = {
        "Server": profile.server,
        "X-Powered-By": profile.powered_by,
        "Content-Type": content_type,
    }
    for name in profile.present_security_headers:
        headers[name] = "<configured>"
    return headers


class HttpResponse(BaseModel):
    """Structured representation of a simulated HTTP response."""

    url: str
    method: str
    status_code: int
    headers: dict[str, str] = Field(default_factory=dict)
    body_preview: str = ""
    body_bytes: int = 0


class HttpGetTool(BaseTool):
    """Simulate an HTTP GET request to a path on the target."""

    name = "http_get"
    description = "Perform a (mock) HTTP GET request and return status, headers and a body preview"
    argument_help = {"target": "base host or URL", "path": "request path, default '/'"}

    def _run(self, target: str, path: str = "/", **_: Any) -> ToolResult:
        base_url, _host = normalise_target(target)
        profile = build_profile(target)
        url = f"{base_url}{path if path.startswith('/') else '/' + path}"
        status = 200 if path in profile.paths or path == "/" else 404
        body = (
            f"<html><head><title>{profile.host}</title></head>"
            f"<body>Simulated content for {path}</body></html>"
            if status == 200
            else "<html><body>404 Not Found</body></html>"
        )
        resp = HttpResponse(
            url=url,
            method="GET",
            status_code=status,
            headers=_response_headers(profile),
            body_preview=body[:120],
            body_bytes=len(body),
        )
        return ToolResult.ok(self.name, f"GET {url} -> {status}", resp)


class HttpPostTool(BaseTool):
    """Simulate an HTTP POST request (e.g. a form submission) to the target."""

    name = "http_post"
    description = "Perform a (mock) HTTP POST request and return the simulated response"
    sensitive = True
    argument_help = {
        "target": "base host or URL",
        "path": "request path, default '/login'",
        "data": "optional dict of form fields",
    }

    def _run(
        self, target: str, path: str = "/login", data: dict | None = None, **_: Any
    ) -> ToolResult:
        base_url, _host = normalise_target(target)
        profile = build_profile(target)
        url = f"{base_url}{path if path.startswith('/') else '/' + path}"
        # Deterministic, non-exploitative behaviour: reflect that a form exists.
        status = 302 if path in profile.paths else 404
        resp = HttpResponse(
            url=url,
            method="POST",
            status_code=status,
            headers=_response_headers(profile),
            body_preview=f"Submitted {len(data or {})} field(s) (simulated)",
            body_bytes=0,
        )
        return ToolResult.ok(self.name, f"POST {url} -> {status}", resp)


class HeaderInspectionTool(BaseTool):
    """Inspect and categorise the response headers of the target's root."""

    name = "http_header_inspect"
    description = "Retrieve response headers and separate informational from security headers"
    argument_help = {"target": "base host or URL", "path": "path to inspect, default '/'"}

    def _run(self, target: str, path: str = "/", **_: Any) -> ToolResult:
        base_url, _host = normalise_target(target)
        profile = build_profile(target)
        headers = _response_headers(profile)
        data = {
            "url": f"{base_url}{path}",
            "headers": headers,
            "disclosed_software": [headers.get("Server", ""), headers.get("X-Powered-By", "")],
            "present_security_headers": profile.present_security_headers,
            "missing_security_headers": profile.missing_security_headers,
        }
        summary = (
            f"{len(headers)} headers; discloses '{headers.get('Server')}' / "
            f"'{headers.get('X-Powered-By')}'"
        )
        return ToolResult.ok(self.name, summary, data)
