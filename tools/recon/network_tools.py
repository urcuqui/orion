"""Mock network tools: a deliberately minimal port-scan wrapper.

Returns a deterministic set of "open" ports from the site profile and performs
no actual scanning.
"""

from __future__ import annotations

from typing import Any

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool
from tools.recon.fixtures import build_profile, normalise_target

_SERVICE_NAMES = {
    22: "ssh",
    80: "http",
    443: "https",
    3306: "mysql",
    8080: "http-alt",
}


class PortScanTool(BaseTool):
    """Simulate a lightweight TCP port scan of the target host."""

    name = "port_scan"
    description = "Mock port scan returning open ports and guessed services (no real scanning)"
    argument_help = {"target": "base host or URL", "top_ports": "number of top ports, default 100"}

    def _run(self, target: str, top_ports: int = 100, **_: Any) -> ToolResult:
        _base_url, host = normalise_target(target)
        profile = build_profile(target)
        open_ports = [
            {"port": p, "service": _SERVICE_NAMES.get(p, "unknown"), "state": "open"}
            for p in profile.open_ports
        ]
        data = {
            "host": host,
            "scanned_top_ports": int(top_ports),
            "open_ports": open_ports,
            "note": "Simulated result — no real scanning is performed.",
        }
        ports_str = ", ".join(str(p) for p in profile.open_ports)
        return ToolResult.ok(self.name, f"{host}: open ports {ports_str}", data)
