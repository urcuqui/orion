"""Web-facing orchestration for the recon workflow: sessions, SSE, approvals.

Ported from HexAgent's ``app/web.py``. Runs the compiled LangGraph in a
background thread per session and streams one event per graph-node update over
Server-Sent Events, pausing on a real HTTP round trip (Approve/Deny) instead of
blocking on a terminal ``input()`` prompt.

Sessions are kept in an in-memory registry (``RUNS``) — fine for this
single-process, no-auth educational tool.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import markdown
import nh3

from libs.recon.graph import AgentState, build_nodes, build_workflow
from libs.recon.models import ToolCall
from tools.recon.registry import default_registry

logger = logging.getLogger(__name__)

RUNS: dict[str, "RunSession"] = {}

_HEARTBEAT_SECONDS = 1.0
_MAX_IDLE_SECONDS = 300.0
# Emitted into session.events while a run is active so a single tool call that
# blocks longer than _MAX_IDLE_SECONDS (e.g. a real Nuclei scan) still produces
# new SSE events and never trips the idle-timeout below.
_PROGRESS_HEARTBEAT_SECONDS = 20.0

# The report is built from Report model fields, but objective/target and tool
# arguments flow into it verbatim from user input — so its rendered HTML is
# sanitised rather than trusted. Only the structural tags the reporter's
# markdown actually produces are allow-listed.
_REPORT_HTML_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td",
    "pre", "code", "strong", "em", "ul", "ol", "li", "hr", "br", "blockquote",
}
_REPORT_MD_EXTENSIONS = ["tables", "fenced_code"]


def _render_report_html(report_markdown: str) -> str:
    """Render report markdown to sanitised HTML, safe for user-influenced content."""
    raw_html = markdown.markdown(report_markdown, extensions=_REPORT_MD_EXTENSIONS)
    return nh3.clean(raw_html, tags=_REPORT_HTML_TAGS, attributes={"code": {"class"}})


@dataclass
class RunSession:
    """Live state for one triggered workflow run.

    ``lock`` guards every mutable field below so the approve/deny HTTP handler
    and the background worker thread never race on
    ``status``/``pending_approval``/``approval_decision``.
    """

    run_id: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"  # running | awaiting_approval | completed | error
    pending_approval: dict[str, Any] | None = None
    approval_gate: threading.Event = field(default_factory=threading.Event)
    approval_decision: bool | None = None
    final_markdown: str | None = None
    error: str | None = None
    # SSE idle-timeout for this run. Widened at creation time when a tool with
    # a longer worst-case timeout (e.g. Nuclei) is enabled, so a genuinely slow
    # single tool call doesn't outlive the stream's idle window.
    idle_timeout_seconds: float = _MAX_IDLE_SECONDS

    def __post_init__(self) -> None:
        self.cv = threading.Condition(self.lock)


def _emit(session: RunSession, event: dict[str, Any]) -> None:
    with session.lock:
        session.events.append(event)
        session.cv.notify_all()


def _make_approval_callback(session: RunSession):
    """Build an approval callback that pauses the worker thread on a real gate.

    A fresh :class:`threading.Event` is created for *each* approval request
    (never reused via ``.clear()``) so a stale ``.set()`` from a duplicate/late
    approval POST can never resolve a later, unrelated approval.
    """

    def _callback(call: ToolCall) -> bool:
        gate = threading.Event()
        with session.lock:
            session.pending_approval = {"tool_name": call.tool_name, "arguments": call.arguments}
            session.status = "awaiting_approval"
            session.approval_gate = gate
            session.events.append(
                {
                    "type": "approval_requested",
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                }
            )
            session.cv.notify_all()

        gate.wait()  # blocks the worker thread only

        with session.lock:
            decision = bool(session.approval_decision)
            session.pending_approval = None
            session.status = "running"
            session.events.append({"type": "approval_resolved", "approved": decision})
            session.cv.notify_all()
        return decision

    return _callback


def _heartbeat_ticker(session: RunSession, stop: threading.Event) -> None:
    """Emit a keep-alive event every _PROGRESS_HEARTBEAT_SECONDS while the run
    is active, so a long blocking tool call (real Nuclei scan) still produces
    SSE traffic instead of sitting silent until it returns.
    """
    while not stop.wait(_PROGRESS_HEARTBEAT_SECONDS):
        with session.lock:
            status = session.status
        if status not in ("running", "awaiting_approval"):
            break
        _emit(session, {"type": "heartbeat"})


def _translate_event(node_name: str, update: dict[str, Any]) -> dict[str, Any]:
    """Turn one LangGraph ``{node: partial_state}`` update into a UI event."""
    event: dict[str, Any] = {"type": node_name}

    if node_name == "plan" or node_name == "replan":
        plan = update.get("plan")
        if plan is not None:
            event["step_count"] = len(plan.steps)
            event["rationale"] = plan.rationale
            event["steps"] = [
                {"id": s.id, "description": s.description, "tool_name": s.tool_name}
                for s in plan.steps
            ]
    elif node_name == "execute":
        tool_results = update.get("tool_results")
        if tool_results:
            result = tool_results[-1]
            event.update(
                {
                    "tool_name": result.tool_name,
                    "status": result.status.value,
                    "summary": result.summary,
                    "duration_ms": result.duration_ms,
                }
            )
            # ToolResult.fail() puts the actual reason in .error and leaves
            # .summary as a generic "<tool> failed" -- surface it here so a
            # failure (missing binary, blocked tag, out-of-scope target, ...)
            # is visible in the live log instead of only in server-side logs.
            if result.error:
                event["error"] = result.error
            # Attach browser-specific fields when a browser tool ran.
            if result.tool_name.startswith("browser_"):
                data = result.data or {}
                event["browser_tool"] = True
                event["current_url"] = data.get("current_url")
                event["page_title"] = data.get("title")
                screenshot_path = data.get("screenshot_path")
                if screenshot_path:
                    event["screenshot_filename"] = Path(screenshot_path).name
                event["forms_count"] = len(data.get("forms") or [])
                event["links_count"] = len(data.get("links") or [])
                event["network_count"] = len(data.get("network_requests") or [])
                event["api_endpoints"] = (data.get("potential_api_endpoints") or [])[:10]
                event["auth_indicators"] = data.get("auth_indicators") or []
                event["browser_errors"] = data.get("errors") or ([result.error] if result.error else [])
                event["browser_success"] = bool(
                    data.get("success", result.status.value == "success")
                )
        else:
            history = update.get("reasoning_history") or []
            event["message"] = history[-1] if history else "no runnable step"
    elif node_name == "evaluate":
        if update:
            findings = update.get("findings") or []
            event["findings"] = [f.model_dump(mode="json") for f in findings]
            event["needs_replan"] = bool(update.get("needs_replan"))
            event["replan_reason"] = update.get("replan_reason", "")
        else:
            event["message"] = "nothing to evaluate (synthesis step)"
    elif node_name == "human_checkpoint":
        event["message"] = "Awaiting human approval checkpoint."
    elif node_name == "report":
        report_markdown = update.get("report_markdown") or ""
        event["stopped_reason"] = update.get("stopped_reason")
        event["next_actions"] = update.get("next_actions", [])
        event["report_markdown"] = report_markdown
        event["report_html"] = _render_report_html(report_markdown)

    return event


def _run_graph(session: RunSession, graph: Any, initial_state: AgentState, config: dict) -> None:
    stop_heartbeat = threading.Event()
    ticker = threading.Thread(
        target=_heartbeat_ticker, args=(session, stop_heartbeat), daemon=True
    )
    ticker.start()
    try:
        _emit(
            session,
            {"type": "start", "objective": initial_state.objective, "target": initial_state.target},
        )
        for update in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, partial in update.items():
                _emit(session, _translate_event(node_name, partial))
                if node_name == "report":
                    with session.lock:
                        session.final_markdown = partial.get("report_markdown")
        with session.lock:
            session.status = "completed"
        _emit(session, {"type": "done"})
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of hanging it
        logger.exception("Workflow run %s failed", session.run_id)
        with session.lock:
            session.status = "error"
            session.error = str(exc)
        _emit(session, {"type": "error", "message": str(exc)})
    finally:
        stop_heartbeat.set()


def start_run(
    objective: str,
    target: str,
    max_iterations: int = 12,
    require_human_approval: bool = False,
    require_sensitive_approval: bool = False,
    mock_mode: bool = True,
    enable_playwright: bool = False,
    enable_nuclei: bool = False,
    browser_username: str = "",
    browser_password: str = "",
    playwright_options: dict[str, Any] | None = None,
    nuclei_options: dict[str, Any] | None = None,
) -> str:
    """Launch a new recon run in a background thread; return its run_id.

    Args:
        mock_mode: Use simulated HTTP/recon tools (default) instead of real
            ``httpx`` requests.
        enable_playwright: Register the real Playwright browser tools.
            Requires ``pip install playwright && playwright install chromium``.
        enable_nuclei: Register the real Nuclei scan tools. Requires the
            ``nuclei`` binary on PATH.
        browser_username, browser_password: Optional lab credentials, wired
            directly into the shared ``BrowserManager`` so they never appear
            in plan step arguments or reach the report.
        playwright_options, nuclei_options: Passed straight through to
            :func:`tools.recon.registry.default_registry`.
    """
    session = RunSession(run_id=uuid.uuid4().hex)
    if enable_nuclei:
        # A safe-default Nuclei scan can legitimately run for the full
        # configured timeout with no intermediate tool_result events; give
        # the stream enough idle headroom to outlast it even if the
        # heartbeat ticker were ever disabled.
        nuclei_timeout = float((nuclei_options or {}).get("timeout", 600.0))
        session.idle_timeout_seconds = max(_MAX_IDLE_SECONDS, nuclei_timeout + 60.0)
    RUNS[session.run_id] = session

    registry = default_registry(
        mock_mode=mock_mode,
        enable_playwright=enable_playwright,
        enable_nuclei=enable_nuclei,
        playwright_options=playwright_options,
        nuclei_options=nuclei_options,
    )
    # Inject lab credentials into the shared BrowserManager so they never
    # appear in plan step arguments or report/report_html content.
    if enable_playwright and browser_username:
        login_tool = registry.get("browser_login")
        if login_tool is not None and hasattr(login_tool, "_mgr"):
            login_tool._mgr.set_credentials(browser_username, browser_password)  # type: ignore[union-attr]

    nodes = build_nodes(
        registry=registry,
        approval_callback=_make_approval_callback(session),
        require_sensitive_approval=require_sensitive_approval,
    )
    graph = build_workflow(nodes)

    initial_state = AgentState(
        objective=objective,
        target=target,
        max_iterations=max_iterations,
        require_human_approval=require_human_approval,
    )
    config = {"recursion_limit": max_iterations * 4 + 20}

    logger.info("Starting recon run %s (target=%r)", session.run_id, target)
    thread = threading.Thread(
        target=_run_graph, args=(session, graph, initial_state, config), daemon=True
    )
    thread.start()
    return session.run_id


def get_run(run_id: str) -> RunSession | None:
    return RUNS.get(run_id)


def stream_events(run_id: str):
    """Yield SSE-formatted ``data: ...\\n\\n`` chunks for ``run_id``."""
    session = RUNS.get(run_id)
    if session is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'unknown run_id'})}\n\n"
        return

    idx = 0
    idle_timeout = session.idle_timeout_seconds
    deadline = time.monotonic() + idle_timeout
    while True:
        with session.lock:
            while idx == len(session.events) and session.status in ("running", "awaiting_approval"):
                if time.monotonic() > deadline:
                    break
                session.cv.wait(timeout=_HEARTBEAT_SECONDS)
            new_events = session.events[idx:]
            idx = len(session.events)
            terminal = session.status in ("completed", "error") and idx == len(session.events)

        for event in new_events:
            deadline = time.monotonic() + idle_timeout
            yield f"data: {json.dumps(event, default=str)}\n\n"

        if terminal:
            break
        if time.monotonic() > deadline:
            yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
            break


def submit_approval(run_id: str, approved: bool) -> dict[str, Any]:
    """Resolve the pending approval gate for ``run_id``. Returns a status dict."""
    session = RUNS.get(run_id)
    if session is None:
        return {"error": "unknown run_id", "status_code": 404}

    with session.lock:
        if session.status != "awaiting_approval":
            return {"error": "no pending approval", "status_code": 409}
        session.approval_decision = approved
        gate = session.approval_gate

    gate.set()
    return {"ok": True, "approved": approved, "status_code": 200}
