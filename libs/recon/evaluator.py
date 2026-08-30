"""Evaluator agent: interprets a tool result into observations and findings.

Mock-only, so this always applies the deterministic rule set keyed on the tool
name — the same offline path HexAgent uses when no LLM is configured.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from libs.recon.models import (
    Finding,
    Observation,
    PlanStep,
    ReplanReason,
    Severity,
    ToolResult,
    ToolStatus,
)

# Ports that, if open, mean the target is worth analysing at the HTTP layer.
_WEB_PORTS = {80, 443}
# Ports the evaluator flags as sensitive regardless of the HTTP-layer decision.
_SENSITIVE_PORTS = {22, 3306}

# Matches the step description HeuristicPlanner._on_nuclei_candidate() creates,
# so the http_get result it produces can be linked back to the Nuclei
# template that triggered it -- see the "candidate -> validate -> confirm"
# handling of the nuclei_scan_url/nuclei_scan_urls branch below.
_NUCLEI_VALIDATION_RE = re.compile(r"^Validate Nuclei candidate \(([^)]+)\)")


def _classify_nuclei_validation(
    status_code: int | None, url: str
) -> tuple[str, Severity, str, bool, str]:
    """Map an http_get validation response to (status, severity, verb, human, detail)."""
    if status_code == 200:
        return "validated", Severity.MEDIUM, "Validated", True, f"confirmed reachable at {url}"
    if status_code in (401, 403):
        detail = f"requires authentication at {url} (HTTP {status_code}); not exploitable"
        return "false_positive", Severity.INFO, "False positive", False, detail
    if status_code == 404:
        return "false_positive", Severity.INFO, "False positive", False, f"not present at {url}"
    detail = f"returned HTTP {status_code} at {url}; manual review needed"
    return "needs_validation", Severity.INFO, "Inconclusive", True, detail


class EvaluationResult(BaseModel):
    """Structured output of the evaluator for a single tool result."""

    observations: list[Observation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    needs_replan: bool = False
    replan_reason: str = ""
    # Browser-specific state carried back to the graph node.
    new_screenshots: list[str] = Field(default_factory=list)
    browser_session_active: bool | None = None  # None = no change


class EvaluatorAgent:
    """Turns raw tool results into observations and security findings."""

    def evaluate(self, step: PlanStep, result: ToolResult) -> EvaluationResult:
        """Evaluate ``result`` for ``step``."""
        data = result.data
        obs = [Observation(source_tool=result.tool_name, step_id=step.id, content=result.summary)]
        findings: list[Finding] = []
        needs_replan = False
        reason = ""

        def add(title: str, sev: Severity, desc: str, rec: str, human: bool = False) -> None:
            findings.append(
                Finding(
                    title=title,
                    severity=sev,
                    description=desc,
                    evidence=[result.summary],
                    recommendation=rec,
                    requires_human_validation=human,
                )
            )

        if result.status is ToolStatus.SKIPPED:
            add(
                "Sensitive action skipped",
                Severity.INFO,
                f"'{result.tool_name}' was not executed: {result.summary}",
                "Review whether this action should be approved and re-run manually.",
                human=True,
            )

        if result.tool_name == "http_get" and step is not None:
            match = _NUCLEI_VALIDATION_RE.match(step.description)
            if match:
                template_id = match.group(1)
                status_code = data.get("status_code")
                url = data.get("url") or step.arguments.get("path", "")
                outcome, sev, verb, human, detail = _classify_nuclei_validation(status_code, url)
                add(
                    f"{verb}: Nuclei candidate {template_id}",
                    sev,
                    f"Nuclei candidate '{template_id}' {detail}.",
                    "Review the validation evidence and remediate if applicable.",
                    human=human,
                )
                findings[-1].evidence = [url, f"status_code={status_code}", template_id]
                findings[-1].validation_status = outcome

        if result.tool_name == "security_headers" and data.get("missing"):
            missing = ", ".join(data["missing"])
            add(
                "Missing security headers",
                Severity.MEDIUM,
                f"Response is missing: {missing} (grade {data.get('grade')}).",
                "Add the missing headers to harden the application.",
            )
        if result.tool_name == "tech_fingerprint":
            add(
                "Technology disclosure",
                Severity.LOW,
                f"Stack disclosed: {', '.join(data.get('technologies', []))}.",
                "Suppress version banners where possible.",
            )
        if result.tool_name == "robots_txt" and data.get("disallowed_paths"):
            add(
                "Sensitive paths in robots.txt",
                Severity.INFO,
                f"Disallowed paths hint at: {', '.join(data['disallowed_paths'])}.",
                "Review whether these paths require authentication.",
                human=True,
            )
            needs_replan = True
            reason = ReplanReason.ROBOTS_PATHS_FOUND
        if result.tool_name == "url_crawler" and data.get("interesting_urls"):
            add(
                "Interesting endpoints discovered",
                Severity.LOW,
                f"Endpoints of interest: {', '.join(data['interesting_urls'])}.",
                "Confirm access controls on these endpoints.",
                human=True,
            )
            if any("login" in u for u in data["interesting_urls"]):
                needs_replan = True
                reason = ReplanReason.LOGIN_ENDPOINT_FOUND
        if result.tool_name == "port_scan":
            open_ports = {p.get("port") for p in data.get("open_ports", [])}
            risky = open_ports & _SENSITIVE_PORTS
            if risky:
                add(
                    "Sensitive service exposure",
                    Severity.MEDIUM,
                    f"Potentially sensitive ports open: {sorted(risky)}.",
                    "Restrict management/database ports via firewall or VPN.",
                    human=True,
                )
            # Decision logic: only bother with HTTP-layer analysis if there's
            # actually a web service listening.
            if open_ports & _WEB_PORTS:
                needs_replan = True
                reason = ReplanReason.OPEN_WEB_PORTS_FOUND

        if result.tool_name in ("nuclei_scan_url", "nuclei_scan_urls"):
            skipped = data.get("targets_skipped") or []
            if skipped:
                obs.append(
                    Observation(
                        source_tool=result.tool_name,
                        step_id=step.id if step else None,
                        content=f"Nuclei skipped {len(skipped)} out-of-scope/oversized target(s).",
                    )
                )
            nuclei_findings = data.get("findings") or []
            for nf in nuclei_findings:
                template_id = nf.get("template_id") or "unknown-template"
                matched_at = nf.get("matched_at") or "unknown URL"
                try:
                    sev = Severity(str(nf.get("severity") or "info").lower())
                except ValueError:
                    sev = Severity.INFO
                obs.append(
                    Observation(
                        source_tool=result.tool_name,
                        step_id=step.id if step else None,
                        content=f"Nuclei candidate [{template_id}] at {matched_at}.",
                    )
                )
                # Nuclei output is always a candidate observation, never an
                # auto-confirmed vulnerability: validation_status="candidate"
                # and requires_human_validation for anything above medium.
                add(
                    f"Candidate: {nf.get('template_name') or template_id}",
                    sev,
                    (
                        f"Nuclei template '{template_id}' matched at {matched_at}. "
                        f"{nf.get('description') or ''} This is an unverified candidate; "
                        "validate with http_get before treating it as confirmed."
                    ).strip(),
                    f"Validate {matched_at} with the HTTP tool to confirm or rule out.",
                    human=sev in (Severity.HIGH, Severity.CRITICAL),
                )
                findings[-1].evidence = [matched_at, template_id]
                findings[-1].validation_status = "candidate"
            if nuclei_findings:
                needs_replan = True
                reason = ReplanReason.NUCLEI_CANDIDATE_FOUND

        # -- Browser tool heuristics ------------------------------------------
        new_screenshots: list[str] = []
        browser_session_active: bool | None = None

        if result.tool_name == "browser_open":
            browser_session_active = data.get("success", False)
            title = data.get("title") or ""
            url = data.get("current_url") or ""
            api_hints = data.get("potential_api_endpoints") or []
            auth_indicators = data.get("auth_indicators") or []
            forms = data.get("forms") or []
            screenshot = data.get("screenshot_path")
            if screenshot:
                new_screenshots.append(screenshot)

            obs.append(
                Observation(
                    source_tool=result.tool_name,
                    step_id=step.id if step else None,
                    content=f"Browser opened {url!r} (title={title!r}). "
                    f"Auth indicators: {auth_indicators}. "
                    f"API hints: {api_hints}. "
                    f"Forms found: {len(forms)}.",
                )
            )
            # Surface API endpoints as a finding.
            if api_hints:
                add(
                    "Browser-discovered API endpoints",
                    Severity.INFO,
                    f"JavaScript-rendered links hint at API endpoints: {', '.join(api_hints[:10])}.",
                    "Replay these requests with the HTTP validation tool to confirm access controls.",
                )
            # Trigger a replan to queue browser_login when a login form is present.
            has_password_field = any(
                f.get("type") == "password"
                for form in forms
                for f in form.get("fields", [])
            )
            if has_password_field or any(
                "password" in str(ind).lower() for ind in auth_indicators
            ):
                needs_replan = True
                reason = ReplanReason.BROWSER_LOGIN_FORM_FOUND

        if result.tool_name == "browser_analyze_page":
            browser_session_active = True
            url = data.get("current_url") or ""
            api_hints = data.get("potential_api_endpoints") or []
            screenshot = data.get("screenshot_path")
            if screenshot:
                new_screenshots.append(screenshot)
            obs.append(
                Observation(
                    source_tool=result.tool_name,
                    step_id=step.id if step else None,
                    content=f"Browser page analysis at {url!r}. "
                    f"API hints: {api_hints}. "
                    f"Network requests captured: "
                    f"{len(data.get('network_requests') or [])}.",
                )
            )
            if api_hints:
                add(
                    "Browser-observed API endpoints",
                    Severity.INFO,
                    f"API-like endpoints observed in browser session: {', '.join(api_hints[:10])}.",
                    "Replay these requests with the HTTP validation tool.",
                )

        if result.tool_name == "browser_login":
            browser_session_active = data.get("success", False)
            url = data.get("current_url") or ""
            screenshot = data.get("screenshot_path")
            if screenshot:
                new_screenshots.append(screenshot)
            outcome = "succeeded" if data.get("success") else "failed"
            obs.append(
                Observation(
                    source_tool=result.tool_name,
                    step_id=step.id if step else None,
                    content=f"Browser login {outcome}; now at {url!r}. "
                    f"Network requests captured: "
                    f"{len(data.get('network_requests') or [])}.",
                )
            )
            # Surface API calls captured post-login as findings.
            api_reqs = [
                r["url"]
                for r in (data.get("network_requests") or [])
                if any(k in r.get("url", "") for k in ("/api/", "/graphql", "/v1/", "/v2/"))
            ]
            if api_reqs:
                add(
                    "Post-login API requests captured",
                    Severity.INFO,
                    f"API endpoints observed after authentication: {', '.join(api_reqs[:10])}.",
                    "Replay these authenticated requests with the HTTP validation tool to test "
                    "authorisation controls.",
                    human=True,
                )

        if result.tool_name == "browser_screenshot":
            browser_session_active = True
            path = data.get("screenshot_path")
            if path:
                new_screenshots.append(path)
                obs.append(
                    Observation(
                        source_tool=result.tool_name,
                        step_id=step.id if step else None,
                        content=f"Evidence screenshot saved: {path}.",
                    )
                )

        if result.tool_name == "browser_close":
            browser_session_active = False
            obs.append(
                Observation(
                    source_tool=result.tool_name,
                    step_id=step.id if step else None,
                    content="Browser session closed.",
                )
            )

        return EvaluationResult(
            observations=obs,
            findings=findings,
            needs_replan=needs_replan,
            replan_reason=reason,
            new_screenshots=new_screenshots,
            browser_session_active=browser_session_active,
        )
