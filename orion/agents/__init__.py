"""AI agents as copilots — not security oracles.

This package re-exports the existing supervisor/agent workflow (``libs.agent``)
and adds explicit, code-level boundaries around what agents may and may not do.

    The agent can plan and explain. Metrics and experiments determine the result.

See ``docs/agent-role.md``.
"""
from __future__ import annotations

from orion.agents.boundaries import (
    ALLOWED_AGENT_ROLES,
    FORBIDDEN_AGENT_CLAIMS,
    AgentBoundaryError,
    assert_not_security_verdict,
)

# Backward-compatible re-exports of the existing agent workflow.
try:  # langgraph is an optional heavy dependency.
    from libs.agent import (  # noqa: F401
        run_supervisor,
        run_supervisor_state,
        run_supervisor_stream,
        format_report,
        build_api_state,
    )
except Exception:  # pragma: no cover - environment dependent
    run_supervisor = run_supervisor_state = run_supervisor_stream = None  # type: ignore
    format_report = build_api_state = None  # type: ignore

__all__ = [
    "ALLOWED_AGENT_ROLES",
    "FORBIDDEN_AGENT_CLAIMS",
    "AgentBoundaryError",
    "assert_not_security_verdict",
    "run_supervisor",
    "run_supervisor_state",
    "run_supervisor_stream",
    "format_report",
    "build_api_state",
]
