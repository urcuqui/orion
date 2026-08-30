"""Planner: a deterministic heuristic planner, wrapped by a thin agent facade.

Ported from HexAgent's ``HeuristicPlanner`` — the offline path it falls back to
whenever no LLM is configured. This integration is mock-only, so that is the
only planner: it starts with a port scan, then grows the plan reactively based
on what that (and later steps) reveal.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from libs.recon.models import Plan, PlanStep, ReplanReason, ToolResult
from tools.recon.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Ports that make the HTTP-layer phase worth queueing.
_WEB_PORTS = {80, 443}

# Steps queued once a port scan reveals a listening web service — the
# "if the scan finds 80/443 -> use HTTP tools" decision.
_HTTP_PHASE_STEPS: list[tuple[str, str]] = [
    ("Identify technologies and server software", "tech_fingerprint"),
    ("Inspect HTTP response headers", "http_header_inspect"),
    ("Evaluate security headers", "security_headers"),
    ("Review robots.txt for disallowed paths", "robots_txt"),
    ("Discover endpoints by crawling", "url_crawler"),
]

# Steps appended after the HTTP phase when Playwright tools are registered.
_BROWSER_PHASE_STEPS: list[tuple[str, str]] = [
    ("Explore web application with browser", "browser_open"),
    ("Analyse page content and forms", "browser_analyze_page"),
    ("Close browser session", "browser_close"),
]

# Nuclei step queued once the HTTP phase is confirmed useful (open web port).
# Uses the tool's own safe-default profile -- listed explicitly here only so
# the rendered plan/report show what will run.
_NUCLEI_PHASE_STEP = (
    "Run safe-default Nuclei template scan for candidate misconfigurations",
    "nuclei_scan_url",
)


class HeuristicPlanner:
    """Deterministic planner that reacts to results instead of front-loading
    a fixed recipe: it starts with a port scan, then grows the plan based on
    what that (and later steps) reveal.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def create_plan(self, objective: str, target: str) -> Plan:
        scan_step = PlanStep(
            id="s1",
            description="Scan for open ports",
            tool_name="port_scan",
            arguments={"target": target},
        )
        summary_step = PlanStep(
            id="s2", description="Summarise findings", tool_name=None, depends_on=[scan_step.id]
        )
        return Plan(
            objective=objective,
            steps=[scan_step, summary_step],
            rationale="Start with a port scan; grow the plan based on what it reveals.",
        )

    def replan(
        self,
        plan: Plan,
        reason: str,
        observations: list[str],
        last_result: ToolResult | None = None,
    ) -> Plan:
        if last_result is None:
            return plan
        if reason == ReplanReason.OPEN_WEB_PORTS_FOUND:
            return self._on_open_web_ports(plan, last_result)
        if reason == ReplanReason.ROBOTS_PATHS_FOUND:
            return self._on_robots_paths(plan, last_result)
        if reason == ReplanReason.LOGIN_ENDPOINT_FOUND:
            return self._on_login_endpoint(plan, last_result)
        if reason == ReplanReason.BROWSER_LOGIN_FORM_FOUND:
            return self._on_browser_login_form(plan, last_result)
        if reason == ReplanReason.NUCLEI_CANDIDATE_FOUND:
            return self._on_nuclei_candidate(plan, last_result)
        return plan

    def _on_open_web_ports(self, plan: Plan, result: ToolResult) -> Plan:
        # Re-verify the decision here rather than trusting the caller: this
        # handler is the actual "if 80/443 open -> use HTTP tools" branch.
        open_ports = {p.get("port") for p in result.data.get("open_ports", [])}
        if not (open_ports & _WEB_PORTS):
            plan.rationale = "No web ports (80/443) open; skipping HTTP-layer analysis."
            return plan
        if any(s.tool_name == "tech_fingerprint" for s in plan.steps):
            return plan  # already queued
        target = self._target_from_plan(plan)
        new_steps = [
            PlanStep(
                id=f"s{len(plan.steps) + i + 1}",
                description=desc,
                tool_name=tool,
                arguments={"target": target},
            )
            for i, (desc, tool) in enumerate(_HTTP_PHASE_STEPS)
        ]
        # Add browser exploration phase when Playwright tools are registered.
        browser_steps = [
            PlanStep(
                id=f"s{len(plan.steps) + len(new_steps) + i + 1}",
                description=desc,
                tool_name=tool,
                arguments={"target": target},
            )
            for i, (desc, tool) in enumerate(_BROWSER_PHASE_STEPS)
            if self._registry.get(tool) is not None
        ]
        # Add a Nuclei candidate-discovery step when it's registered.
        nuclei_desc, nuclei_tool = _NUCLEI_PHASE_STEP
        nuclei_steps = (
            [
                PlanStep(
                    id=f"s{len(plan.steps) + len(new_steps) + len(browser_steps) + 1}",
                    description=nuclei_desc,
                    tool_name=nuclei_tool,
                    arguments={"target": target},
                )
            ]
            if self._registry.get(nuclei_tool) is not None
            else []
        )
        all_new = new_steps + browser_steps + nuclei_steps
        extras = []
        if browser_steps:
            extras.append("browser exploration")
        if nuclei_steps:
            extras.append("a Nuclei candidate scan")
        plan.rationale = "Open web port(s) found; queued HTTP-layer analysis" + (
            f" and {', '.join(extras)}." if extras else "."
        )
        return self._insert_before_summary(plan, all_new)

    def _on_robots_paths(self, plan: Plan, result: ToolResult) -> Plan:
        disallowed = result.data.get("disallowed_paths") or []
        if not disallowed:
            return plan
        path = disallowed[0]
        if any(s.tool_name == "http_get" and s.arguments.get("path") == path for s in plan.steps):
            return plan  # already queued
        target = self._target_from_plan(plan)
        step = PlanStep(
            id=f"s{len(plan.steps) + 1}",
            description=f"Inspect disallowed path {path}",
            tool_name="http_get",
            arguments={"target": target, "path": path},
        )
        plan.rationale = f"robots.txt disallowed {path!r}; inspecting it directly."
        return self._insert_before_summary(plan, [step])

    def _on_login_endpoint(self, plan: Plan, result: ToolResult) -> Plan:
        if any(s.tool_name == "http_post" for s in plan.steps):
            return plan  # already queued
        login_url = next((u for u in result.data.get("interesting_urls", []) if "login" in u), None)
        if login_url is None:
            return plan
        path = urlparse(login_url).path or "/login"
        target = self._target_from_plan(plan)
        step = PlanStep(
            id=f"s{len(plan.steps) + 1}",
            description=f"Submit a controlled POST to {path}",
            tool_name="http_post",
            arguments={"target": target, "path": path, "data": {"probe": "orion"}},
        )
        plan.rationale = f"Login endpoint discovered ({path}); queued a controlled POST."
        return self._insert_before_summary(plan, [step])

    def _on_browser_login_form(self, plan: Plan, result: ToolResult) -> Plan:
        """Queue browser_login when a login form is detected by browser_open."""
        if any(s.tool_name == "browser_login" for s in plan.steps):
            return plan  # already queued
        if self._registry.get("browser_login") is None:
            return plan  # browser tools not registered
        target = self._target_from_plan(plan)
        login_url = result.data.get("current_url") or ""
        step = PlanStep(
            id=f"s{len(plan.steps) + 1}",
            description="Authenticate through the discovered login form",
            tool_name="browser_login",
            arguments={"target": target, "url": login_url},
        )
        plan.rationale = "Login form detected in browser; queued browser_login."
        return self._insert_before_summary(plan, [step])

    def _on_nuclei_candidate(self, plan: Plan, result: ToolResult) -> Plan:
        """Queue an http_get validation step for the top Nuclei candidate.

        Nuclei findings are never auto-confirmed (see EvaluatorAgent): the
        planner is what decides to hand the matched URL to the existing HTTP
        tool for confirmation, exactly as the candidate -> validate -> confirm
        loop requires.
        """
        findings = result.data.get("findings") or []
        if not findings:
            return plan
        matched_at = findings[0].get("matched_at")
        if not matched_at:
            return plan
        path = urlparse(matched_at).path or "/"
        if any(s.tool_name == "http_get" and s.arguments.get("path") == path for s in plan.steps):
            return plan  # already queued
        target = self._target_from_plan(plan)
        template_id = findings[0].get("template_id") or "nuclei finding"
        step = PlanStep(
            id=f"s{len(plan.steps) + 1}",
            description=f"Validate Nuclei candidate ({template_id}) at {path}",
            tool_name="http_get",
            arguments={"target": target, "path": path},
        )
        plan.rationale = f"Nuclei candidate found ({template_id} at {path}); queued validation."
        return self._insert_before_summary(plan, [step])

    @staticmethod
    def _insert_before_summary(plan: Plan, new_steps: list[PlanStep]) -> Plan:
        summary = next((s for s in plan.steps if s.tool_name is None), None)
        insert_at = plan.steps.index(summary) if summary is not None else len(plan.steps)
        plan.steps[insert_at:insert_at] = new_steps
        if summary is not None:
            summary.depends_on = list({*summary.depends_on, *(s.id for s in new_steps)})
        return plan

    @staticmethod
    def _target_from_plan(plan: Plan) -> str:
        return next((s.arguments["target"] for s in plan.steps if s.arguments.get("target")), "")


class PlannerAgent:
    """Creates and revises plans on behalf of the graph."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._planner = HeuristicPlanner(registry)

    def plan(self, objective: str, target: str) -> Plan:
        """Generate an initial plan."""
        logger.info("Planning for objective=%r target=%r", objective, target)
        plan = self._planner.create_plan(objective, target)
        logger.info("Plan created with %d step(s)", len(plan.steps))
        return plan

    def replan(
        self,
        plan: Plan,
        reason: str,
        observations: list[str],
        last_result: ToolResult | None = None,
    ) -> Plan:
        """Revise an existing plan given new information."""
        logger.info("Replanning due to: %s", reason)
        return self._planner.replan(plan, reason, observations, last_result)
