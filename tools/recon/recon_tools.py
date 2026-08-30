"""Mock reconnaissance tools: robots.txt, security headers, fingerprinting, crawl.

All results are derived from the deterministic site profile and perform no real
network activity.
"""

from __future__ import annotations

from typing import Any

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool
from tools.recon.fixtures import build_profile, normalise_target


class RobotsTxtTool(BaseTool):
    """Retrieve and parse the target's ``robots.txt`` (simulated)."""

    name = "robots_txt"
    description = "Fetch and parse robots.txt, returning disallowed paths and sitemap hints"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        base_url, _host = normalise_target(target)
        profile = build_profile(target)
        lines = ["User-agent: *", *[f"Disallow: {p}" for p in profile.disallowed]]
        sitemap = f"{base_url}/sitemap.xml"
        lines.append(f"Sitemap: {sitemap}")
        data = {
            "url": f"{base_url}/robots.txt",
            "raw": "\n".join(lines),
            "disallowed_paths": profile.disallowed,
            "sitemaps": [sitemap],
        }
        return ToolResult.ok(
            self.name, f"robots.txt lists {len(profile.disallowed)} disallowed path(s)", data
        )


class SecurityHeadersTool(BaseTool):
    """Assess presence/absence of common HTTP security headers (simulated)."""

    name = "security_headers"
    description = "Check for common security headers (HSTS, CSP, X-Frame-Options, ...)"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        profile = build_profile(target)
        present = profile.present_security_headers
        missing = profile.missing_security_headers
        ratio = len(present) / (len(present) + len(missing) or 1)
        grade = "A" if ratio > 0.8 else "B" if ratio > 0.6 else "C" if ratio > 0.4 else "D"
        data = {
            "present": present,
            "missing": missing,
            "grade": grade,
            "coverage": round(ratio, 2),
        }
        return ToolResult.ok(
            self.name, f"Security header grade {grade}; {len(missing)} missing", data
        )


class TechFingerprintTool(BaseTool):
    """Identify the technology stack of the target (simulated)."""

    name = "tech_fingerprint"
    description = "Fingerprint server software, frameworks and notable technologies"
    argument_help = {"target": "base host or URL"}

    def _run(self, target: str, **_: Any) -> ToolResult:
        profile = build_profile(target)
        data = {
            "server": profile.server,
            "powered_by": profile.powered_by,
            "technologies": profile.technologies,
        }
        return ToolResult.ok(
            self.name,
            f"Detected: {', '.join(profile.technologies)} (server {profile.server})",
            data,
        )


class CrawlerTool(BaseTool):
    """Discover endpoints by crawling the target up to a depth (simulated)."""

    name = "url_crawler"
    description = "Crawl the site and enumerate discoverable endpoints"
    argument_help = {"target": "base host or URL", "max_pages": "max pages to crawl, default 10"}

    def _run(self, target: str, max_pages: int = 10, **_: Any) -> ToolResult:
        base_url, _host = normalise_target(target)
        profile = build_profile(target)
        discovered = [f"{base_url}{p}" for p in profile.paths][: max(1, int(max_pages))]
        interesting = [
            u for u in discovered if any(k in u for k in ("admin", "api", "login", "private"))
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
