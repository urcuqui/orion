"""Direct URL probe: gather *real* evidence from a running service.

This bridges "I have an app at a URL" and evidence-grounded analysis without a
full recon run. It performs bounded, **GET-only** requests to the target and a
small set of well-known endpoints, then builds a recon-style summary that
``build_assessment_from_summary`` can interpret.

Authorized use only: point it at systems you own or are allowed to test. It is
read-only (GET), follows a capped number of requests, and never sends payloads.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

# Well-known endpoints worth checking on an AI/ML or web service. GET-only.
PROBE_PATHS = [
    "/", "/mcp_tools", "/chat", "/chat_stream", "/v1/models", "/v1/chat/completions",
    "/v1/completions", "/v1/embeddings", "/openapi.json", "/docs", "/api", "/agent",
    "/adverimage", "/orion/methodology", "/health", "/predict", "/embeddings",
]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _text_from_html(body: str, limit: int = 4000) -> str:
    """Crude HTML→text: title + stripped tags, capped."""
    title = ""
    m = _TITLE_RE.search(body or "")
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    stripped = re.sub(r"<script[\s\S]*?</script>", " ", body or "", flags=re.IGNORECASE)
    stripped = re.sub(r"<style[\s\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return (title + " " + stripped)[:limit]


def build_probe_summary(url: str, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a recon-style summary from probe observations (pure/testable).

    ``observations``: list of {path, status, server, content_type, snippet}.
    An endpoint is considered to *exist* when it responds with any status other
    than 404/501/502/503 (405/401/403/400 still prove the route is handled).
    """
    endpoints: List[str] = []
    report_parts: List[str] = [f"# Direct URL probe of {url}"]
    for o in observations:
        status = o.get("status")
        path = o.get("path", "")
        exists = status is not None and status not in (404, 501, 502, 503)
        # IMPORTANT: only record paths that ACTUALLY exist. Never log attempted
        # (404) probe paths into the report text — otherwise the probe's own
        # path list would leak into the AI-signal haystack and self-confirm.
        if not exists:
            continue
        if path and path != "/" and path not in endpoints:
            endpoints.append(path)
        server = o.get("server") or ""
        ctype = o.get("content_type") or ""
        report_parts.append(f"- {path} -> {status} {server} {ctype}".rstrip())
        if o.get("snippet"):
            report_parts.append(o["snippet"])

    return {
        "target": url,
        "objective": "direct URL probe",
        "endpoints": endpoints,
        "auth_indicators": [p for p in endpoints if any(k in p for k in ("login", "auth", "signin"))],
        "screenshots": [],
        "findings": [],
        "report_markdown": "\n".join(report_parts),
        "counts": {"endpoints": len(endpoints), "findings": 0, "screenshots": 0,
                   "auth_flows": 0},
    }


def probe_url(url: str, timeout: float = 3.0, max_requests: int = 18) -> Dict[str, Any]:
    """GET-only probe of ``url`` and well-known endpoints; returns a summary.

    Raises ValueError for an obviously invalid URL. Network errors per request
    are swallowed (recorded as status None) so one dead route never aborts the
    probe.
    """
    parsed = urlparse(url if "://" in url else "http://" + url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Provide an http(s) URL, e.g. http://127.0.0.1:5001")
    base = f"{parsed.scheme}://{parsed.netloc}"

    import requests  # declared dependency

    observations: List[Dict[str, Any]] = []
    for i, path in enumerate(PROBE_PATHS[:max_requests]):
        target = urljoin(base + "/", path.lstrip("/"))
        obs: Dict[str, Any] = {"path": path if path != "/" else "/"}
        try:
            r = requests.get(target, timeout=timeout, allow_redirects=True)
            obs["status"] = r.status_code
            obs["server"] = r.headers.get("Server", "")
            obs["content_type"] = r.headers.get("Content-Type", "")
            # Capture page text only for HTML-ish base responses (signal source).
            if path in ("/", "/docs") and "html" in obs["content_type"].lower():
                obs["snippet"] = _text_from_html(r.text)
            elif "json" in obs["content_type"].lower():
                obs["snippet"] = (r.text or "")[:1500]
        except Exception as exc:  # noqa: BLE001 - dead route must not abort
            obs["status"] = None
            obs["error"] = str(exc)
        observations.append(obs)

    return build_probe_summary(base, observations)
