"""Tool registry: a small dependency-injection container mapping names to tools.

Agents receive a registry rather than importing tools directly, so the
available toolset can be customised per run.
"""

from __future__ import annotations

import logging

from libs.recon.models import ToolResult
from tools.recon.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds the set of tools available to a workflow run."""

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Add a tool, raising on duplicate names."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return the tool registered under ``name`` (or ``None``)."""
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools)

    def all(self) -> list[BaseTool]:
        """Return all registered tool instances."""
        return list(self._tools.values())

    def catalogue(self) -> str:
        """Render the human-readable catalogue of all tools."""
        return "\n".join(tool.catalogue_entry() for tool in self._tools.values())

    def run(self, name: str, **kwargs: object) -> ToolResult:
        """Look up and execute a tool by name.

        Returns an error :class:`ToolResult` if the tool is unknown.
        """
        tool = self.get(name)
        if tool is None:
            logger.warning("Requested unknown tool '%s'", name)
            return ToolResult.fail(name, f"Unknown tool '{name}'")
        return tool.run(**kwargs)


def default_registry(
    mock_mode: bool = True,
    enable_playwright: bool = False,
    enable_nuclei: bool = False,
    playwright_options: dict | None = None,
    nuclei_options: dict | None = None,
) -> ToolRegistry:
    """Build a registry pre-populated with HTTP and reconnaissance tools.

    Args:
        mock_mode: When ``True`` (default), use simulated/deterministic tools
            that make no network requests. When ``False``, use real tools that
            make actual HTTP requests -- suitable only for authorised lab
            targets.
        enable_playwright: When ``True``, register the six Playwright browser
            tools. Playwright must be installed (``pip install playwright &&
            playwright install chromium``). Off by default.
        enable_nuclei: When ``True``, also register the real
            ``nuclei_scan_url``/``nuclei_scan_urls``/``nuclei_check_installation``
            tools, which shell out to a local ``nuclei`` binary. Off by
            default.
        playwright_options: Optional overrides for :func:`build_browser_tools`
            (``headless``, ``timeout_ms``, ``max_requests``,
            ``max_body_preview_bytes``, ``screenshot_dir``). Ignored unless
            ``enable_playwright`` is set.
        nuclei_options: Optional overrides for the Nuclei tool constructors
            (``binary``, ``templates_dir``, ``default_tags``,
            ``default_severity``, ``allow_high``, ``allow_critical``,
            ``rate_limit``, ``timeout``, ``max_results``, ``max_targets``).
            Ignored unless ``enable_nuclei`` is set.
    """
    if mock_mode:
        from tools.recon.http_tools import HeaderInspectionTool, HttpGetTool, HttpPostTool
        from tools.recon.recon_tools import (
            CrawlerTool,
            RobotsTxtTool,
            SecurityHeadersTool,
            TechFingerprintTool,
        )
    else:
        from tools.recon.live_http_tools import (  # type: ignore[assignment]
            CrawlerTool,
            HeaderInspectionTool,
            HttpGetTool,
            RobotsTxtTool,
            SecurityHeadersTool,
            TechFingerprintTool,
        )
        from tools.recon.http_tools import HttpPostTool  # POST stays mock (non-exploitative)

        logger.info("Using real HTTP tools (mock_mode=False)")

    from tools.recon.network_tools import PortScanTool

    tools: list[BaseTool] = [
        HttpGetTool(),
        HttpPostTool(),
        HeaderInspectionTool(),
        RobotsTxtTool(),
        SecurityHeadersTool(),
        TechFingerprintTool(),
        PortScanTool(),
        CrawlerTool(),
    ]

    if enable_nuclei:
        from tools.recon.nuclei_tool import (
            NucleiCheckInstallationTool,
            NucleiScanUrlsTool,
            NucleiScanUrlTool,
        )

        opts = dict(nuclei_options or {})
        max_targets = opts.pop("max_targets", 20)
        binary = opts.get("binary", "nuclei")

        tools.append(NucleiScanUrlTool(**opts))
        tools.append(NucleiScanUrlsTool(**opts, max_targets=max_targets))
        tools.append(NucleiCheckInstallationTool(binary=binary))

    if enable_playwright:
        from tools.recon.browser_tools import build_browser_tools

        tools.extend(build_browser_tools(**(playwright_options or {})))

    return ToolRegistry(tools)
