"""Executor agent: selects a tool for a step and dispatches it to a specialist.

Mock-only, so tool selection is always heuristic: the tool named on the step,
or a sensible default. Running the chosen tool is delegated to the domain
specialist that owns it (recon vs. HTTP analysis), which also enforces the
human-approval gate for sensitive tools.
"""

from __future__ import annotations

import logging

from libs.recon.models import PlanStep, ToolCall, ToolResult
from libs.recon.specialists import (
    ApprovalCallback,
    BrowserAgent,
    HttpAnalysisAgent,
    ReconAgent,
    SpecialistAgent,
)
from tools.recon.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Chooses a tool for a step and dispatches it to the owning specialist."""

    def __init__(
        self,
        registry: ToolRegistry,
        approval_callback: ApprovalCallback | None = None,
        require_sensitive_approval: bool = False,
    ) -> None:
        self._registry = registry
        self._specialists: list[SpecialistAgent] = [
            ReconAgent(registry, approval_callback, require_sensitive_approval),
            HttpAnalysisAgent(registry, approval_callback, require_sensitive_approval),
            BrowserAgent(registry, approval_callback, require_sensitive_approval),
        ]

    def select(self, step: PlanStep, target: str) -> ToolCall:
        """Decide which tool to run and with what arguments for ``step``."""
        tool_name = step.tool_name or "http_header_inspect"
        args = dict(step.arguments)
        args.setdefault("target", target)
        return ToolCall(tool_name=tool_name, arguments=args, rationale="Step-declared tool")

    def execute(self, step: PlanStep, target: str) -> ToolResult:
        """Select a tool for ``step`` and run it via its owning specialist."""
        call = self.select(step, target)
        logger.info("Executing step %s via tool '%s'", step.id, call.tool_name)
        specialist = next((s for s in self._specialists if s.owns(call.tool_name)), None)
        if specialist is not None:
            return specialist.run(call)
        # Fallback for any tool not yet assigned to a specialist.
        return self._registry.run(call.tool_name, **call.arguments)
