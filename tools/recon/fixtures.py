"""Deterministic mock data for the simulated tools.

All tools draw from a single per-target "site profile" derived deterministically
from the target host (via a stable hash). This means the fingerprint, headers,
robots.txt and crawl results all describe the *same* imaginary site for a given
target, every run. No network traffic ever occurs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from urllib.parse import urlparse

_STACKS = [
    {"server": "nginx/1.24.0", "tech": ["nginx", "React", "Node.js"], "powered_by": "Express"},
    {
        "server": "Apache/2.4.58",
        "tech": ["Apache", "PHP/8.2", "WordPress 6.5"],
        "powered_by": "PHP/8.2.4",
    },
    {"server": "Microsoft-IIS/10.0", "tech": ["IIS", "ASP.NET", "MSSQL"], "powered_by": "ASP.NET"},
    {
        "server": "gunicorn/21.2.0",
        "tech": ["nginx", "Django 5.0", "Python/3.12"],
        "powered_by": "Werkzeug",
    },
]

_COMMON_PATHS = ["/", "/login", "/about", "/contact", "/api", "/admin", "/static/app.js"]
_ROBOTS_DISALLOW = [
    ["/admin", "/private"],
    ["/wp-admin", "/cgi-bin"],
    ["/internal"],
    ["/api/v1/debug"],
]
_OPEN_PORTS = [[80, 443], [80, 443, 22], [80, 443, 8080], [80, 443, 3306]]

_SECURITY_HEADERS = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
    "Permissions-Policy",
]


@dataclass(frozen=True)
class SiteProfile:
    """A coherent, deterministic description of a simulated target site."""

    host: str
    server: str
    technologies: list[str]
    powered_by: str
    paths: list[str]
    disallowed: list[str]
    open_ports: list[int]
    present_security_headers: list[str] = field(default_factory=list)

    @property
    def missing_security_headers(self) -> list[str]:
        """Security headers absent from the simulated responses."""
        return [h for h in _SECURITY_HEADERS if h not in self.present_security_headers]


def normalise_target(target: str) -> tuple[str, str]:
    """Return ``(base_url, host)`` for a user-supplied target string.

    Accepts bare hosts or full URLs; defaults to https when no scheme is given.
    """
    candidate = target if "://" in target else f"https://{target}"
    parsed = urlparse(candidate)
    host = parsed.netloc or parsed.path
    base_url = f"{parsed.scheme or 'https'}://{host}".rstrip("/")
    return base_url, host


def _seed(host: str) -> int:
    """Stable integer seed derived from the host name."""
    digest = hashlib.sha256(host.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_profile(target: str) -> SiteProfile:
    """Deterministically construct a :class:`SiteProfile` for ``target``."""
    _, host = normalise_target(target)
    seed = _seed(host)
    stack = _STACKS[seed % len(_STACKS)]
    disallow = _ROBOTS_DISALLOW[seed % len(_ROBOTS_DISALLOW)]
    ports = _OPEN_PORTS[seed % len(_OPEN_PORTS)]
    present = [h for i, h in enumerate(_SECURITY_HEADERS) if (seed >> i) & 1]
    paths = list(dict.fromkeys(_COMMON_PATHS + disallow))
    return SiteProfile(
        host=host,
        server=stack["server"],
        technologies=list(stack["tech"]),
        powered_by=stack["powered_by"],
        paths=paths,
        disallowed=disallow,
        open_ports=ports,
        present_security_headers=present,
    )
