"""Real (non-mock) HTTP and reconnaissance tools.

These tools make actual network requests using ``httpx``. They are registered
instead of the simulated equivalents when ``mock_mode=False``. Every class
carries the same ``name`` as its mock counterpart so the rest of the system
(planner, evaluator) is unchanged.

Design constraints:
- Passive only: GET requests, no credential stuffing, no brute-force.
- Scope-bounded: crawled links are filtered to the authorised target domain.
- Resilient: every network failure is caught and surfaced as a ToolResult with
  the error detail rather than an uncaught exception.
- Lightweight: no third-party HTML parsers beyond the stdlib ``html.parser``.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import urllib3
import httpx

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool
from tools.recon.fixtures import normalise_target

import logging

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_UA = "Orion/1.0 (educational security scanner; contact: lab-use-only)"
_SECURITY_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
]
_INTERESTING_KEYWORDS = ("admin", "api", "login", "private", "auth", "user", "dashboard", "panel")
_TECH_PATTERNS: list[tuple[str, str]] = [
    (r"WordPress", "WordPress"),
    (r"wp-content", "WordPress"),
    (r"Drupal", "Drupal"),
    (r"Joomla", "Joomla"),
    (r"Django", "Django"),
    (r"Laravel", "Laravel"),
    (r"Rails", "Ruby on Rails"),
    (r"ASP\.NET", "ASP.NET"),
    (r"Express", "Express.js"),
    (r"Next\.?[Jj][Ss]", "Next.js"),
    (r"React", "React"),
]


def _client() -> httpx.Client:
    return httpx.Client(
        timeout=_TIMEOUT,
        headers={"User-Agent": _UA},
        follow_redirects=True,
        verify=False,  # educational lab tool — cert errors are informative, not blockers
    )


def _same_domain(link: str, base_host: str) -> bool:
    parsed = urlparse(link)
    return (not parsed.netloc) or parsed.netloc == base_host or parsed.netloc.endswith(f".{base_host}")


class _LinkExtractor(HTMLParser):
    """Minimal HTML parser that extracts href values from anchor tags."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for attr, value in attrs:
                if attr == "href" and value:
                    self.links.append(value)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def _detect_tech(headers: dict[str, str], body: str) -> list[str]:
    techs: list[str] = []
    combined = " ".join(headers.values()) + " " + body[:4000]
    for pattern, name in _TECH_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE) and name not in techs:
            techs.append(name)
    server = headers.get("server", "")
    powered = headers.get("x-powered-by", "")
    for val in (server, powered):
        if val and val not in techs:
            techs.append(val)
    return techs


class HttpGetTool(BaseTool):
    """Perform a real HTTP GET request and return status, headers and body preview."""

    name = "http_get"
    description = "Perform an HTTP GET request and return status, headers and a body preview"
    argument_help = {"target": "base host or URL", "path": "request path, default '/'"}

    def _run(self, target: str, path: str = "/", **_: Any) -> ToolResult:
        base_url, _ = normalise_target(target)
        url = f"{base_url}{path if path.startswith('/') else '/' + path}"
        try:
            with _client() as client:
                resp = client.get(url)
            data = {
                "url": str(resp.url),
                "method": "GET",
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body_preview": resp.text[:300],
                "body_bytes": len(resp.content),
            }
            return ToolResult.ok(self.name, f"GET {url} -> {resp.status_code}", data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"GET {url} failed: {exc}")


class HeaderInspectionTool(BaseTool):
    """Retrieve and categorise real HTTP response headers from the target."""

    name = "http_header_inspect"
    description = "Retrieve response headers and separate informational from security headers"
    argument_help = {"target": "base host or URL", "path": "path to inspect, default '/'"}

    def _run(self, target: str, path: str = "/", **_: Any) -> ToolResult:
        base_url, _ = normalise_target(target)
        url = f"{base_url}{path if path.startswith('/') else '/' + path}"
        try:
            with _client() as client:
                resp = client.head(url)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            headers_display = dict(resp.headers)
            present = [h for h in _SECURITY_HEADERS if h.lower() in headers]
            missing = [h for h in _SECURITY_HEADERS if h.lower() not in headers]
            server = headers.get("server", "")
            powered = headers.get("x-powered-by", "")
            data = {
                "url": str(resp.url),
                "headers": headers_display,
                "disclosed_software": [v for v in (server, powered) if v],
                "present_security_headers": present,
                "missing_security_headers": missing,
            }
            summary = (
                f"{len(headers_display)} headers; discloses "
                f"'{server or '(none)'}' / '{powered or '(none)'}'"
            )
            return ToolResult.ok(self.name, summary, data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"Header inspection of {url} failed: {exc}")


class RobotsTxtTool(BaseTool):
    """Fetch and parse the real robots.txt from the target."""

    name = "robots_txt"
    description = "Fetch and parse robots.txt, returning disallowed paths and sitemap hints"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        base_url, _ = normalise_target(target)
        url = f"{base_url}/robots.txt"
        try:
            with _client() as client:
                resp = client.get(url)
            if resp.status_code == 404:
                return ToolResult.ok(
                    self.name,
                    "robots.txt not found (404)",
                    {"url": url, "raw": "", "disallowed_paths": [], "sitemaps": []},
                )
            raw = resp.text
            disallowed: list[str] = []
            sitemaps: list[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if line.lower().startswith("disallow:"):
                    path = line[len("disallow:"):].strip()
                    if path and path not in disallowed:
                        disallowed.append(path)
                elif line.lower().startswith("sitemap:"):
                    sm = line[len("sitemap:"):].strip()
                    if sm:
                        sitemaps.append(sm)
            data = {"url": url, "raw": raw[:2000], "disallowed_paths": disallowed, "sitemaps": sitemaps}
            return ToolResult.ok(self.name, f"robots.txt lists {len(disallowed)} disallowed path(s)", data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"robots.txt fetch from {url} failed: {exc}")


class SecurityHeadersTool(BaseTool):
    """Check real HTTP security headers from the target's root path."""

    name = "security_headers"
    description = "Check for common security headers (HSTS, CSP, X-Frame-Options, ...)"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        base_url, _ = normalise_target(target)
        url = f"{base_url}/"
        try:
            with _client() as client:
                resp = client.head(url)
            headers_lower = {k.lower(): v for k, v in resp.headers.items()}
            present = [h for h in _SECURITY_HEADERS if h.lower() in headers_lower]
            missing = [h for h in _SECURITY_HEADERS if h.lower() not in headers_lower]
            ratio = len(present) / (len(_SECURITY_HEADERS) or 1)
            grade = "A" if ratio > 0.8 else "B" if ratio > 0.6 else "C" if ratio > 0.4 else "D"
            data = {"present": present, "missing": missing, "grade": grade, "coverage": round(ratio, 2)}
            return ToolResult.ok(self.name, f"Security header grade {grade}; {len(missing)} missing", data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"Security header check of {url} failed: {exc}")


class TechFingerprintTool(BaseTool):
    """Identify server software and frameworks from real HTTP headers and page content."""

    name = "tech_fingerprint"
    description = "Fingerprint server software, frameworks and notable technologies"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        base_url, _ = normalise_target(target)
        url = f"{base_url}/"
        try:
            with _client() as client:
                resp = client.get(url)
            headers_lower = {k.lower(): v for k, v in resp.headers.items()}
            techs = _detect_tech(headers_lower, resp.text)
            server = headers_lower.get("server", "unknown")
            data = {
                "server": server,
                "powered_by": headers_lower.get("x-powered-by", ""),
                "technologies": techs,
            }
            return ToolResult.ok(
                self.name,
                f"Detected: {', '.join(techs) or 'nothing conclusive'} (server {server})",
                data,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"Tech fingerprint of {url} failed: {exc}")


class CrawlerTool(BaseTool):
    """Discover endpoints by fetching the homepage and extracting links (real)."""

    name = "url_crawler"
    description = "Crawl the site and enumerate discoverable endpoints"
    argument_help = {"target": "base host or URL", "max_pages": "max links to return, default 30"}

    def _run(self, target: str, max_pages: int = 30, **_: Any) -> ToolResult:
        base_url, host = normalise_target(target)
        try:
            with _client() as client:
                resp = client.get(f"{base_url}/")
            parser = _LinkExtractor()
            parser.feed(resp.text)

            seen: set[str] = set()
            discovered: list[str] = []
            for href in parser.links:
                absolute = urljoin(base_url, href)
                parsed = urlparse(absolute)
                # Keep only http/https links on the same domain, no fragments.
                if parsed.scheme not in ("http", "https"):
                    continue
                if not _same_domain(absolute, host):
                    continue
                clean = parsed._replace(fragment="").geturl()
                if clean not in seen:
                    seen.add(clean)
                    discovered.append(clean)
                if len(discovered) >= int(max_pages):
                    break

            interesting = [
                u for u in discovered
                if any(k in u.lower() for k in _INTERESTING_KEYWORDS)
            ]
            data = {
                "discovered_urls": discovered,
                "count": len(discovered),
                "interesting_urls": interesting,
            }
            return ToolResult.ok(
                self.name,
                f"Discovered {len(discovered)} endpoint(s), {len(interesting)} of interest",
                data,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.fail(self.name, f"Crawl of {base_url} failed: {exc}")
