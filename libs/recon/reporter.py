"""Reporter agent: renders a :class:`~libs.recon.models.Report` to markdown.

The deterministic renderer guarantees the required section structure and
always works offline (mock-only; no LLM narrative summary).
"""

from __future__ import annotations

import json

from libs.recon.models import Finding, Report, Severity

_SEVERITY_ORDER = [
    Severity.CRITICAL,
    Severity.HIGH,
    Severity.MEDIUM,
    Severity.LOW,
    Severity.INFO,
]


class ReporterAgent:
    """Produces the final markdown report for a workflow run."""

    def render(self, report: Report) -> str:
        """Return the full markdown report as a string."""
        return self._structured(report).strip() + "\n"

    def _structured(self, report: Report) -> str:
        parts = [
            f"# Reconnaissance Report\n\n**Objective:** {report.objective}\n\n"
            f"**Generated:** {report.generated_at}  |  **Iterations:** {report.iterations}  |  "
            f"**Stopped:** {report.stopped_reason}",
            self._plan_section(report),
            self._executed_section(report),
            self._tool_outputs_section(report),
            self._findings_section(report),
            self._list_section("Suggested Next Actions", report.next_actions),
            self._list_section("Human Validation Points", report.human_validation_points),
            "## Disclaimer\n\nThis is an educational proof-of-concept using simulated tools. "
            "It is not a substitute for authorised, professional penetration testing.",
        ]
        return "\n\n".join(parts)

    def _plan_section(self, report: Report) -> str:
        rows = [
            f"| {i} | {s.description} | `{s.tool_name or 'n/a'}` | {s.status.value} |"
            for i, s in enumerate(report.plan.steps, start=1)
        ]
        header = "## Plan\n\n| # | Step | Tool | Status |\n|---|------|------|--------|"
        rationale = f"\n\n_Rationale: {report.plan.rationale}_" if report.plan.rationale else ""
        return f"{header}\n" + "\n".join(rows) + rationale

    def _executed_section(self, report: Report) -> str:
        if not report.executed_steps:
            return "## Executed Steps\n\n_No steps were executed._"
        rows = [
            f"| {e.step_id} | {e.description} | `{e.tool_name}` | {e.status} | {e.result_summary} |"
            for e in report.executed_steps
        ]
        header = (
            "## Executed Steps\n\n| ID | Description | Tool | Status | Result |\n"
            "|----|-------------|------|--------|--------|"
        )
        return f"{header}\n" + "\n".join(rows)

    def _tool_outputs_section(self, report: Report) -> str:
        if not report.tool_results:
            return "## Tool Outputs\n\n_No tool outputs recorded._"
        blocks = []
        for r in report.tool_results:
            blocks.append(
                f"### `{r.tool_name}` ({r.status.value}, {r.duration_ms} ms)\n\n"
                f"{r.summary}\n\n```json\n{self._compact_json(r.data)}\n```"
            )
        return "## Tool Outputs\n\n" + "\n\n".join(blocks)

    def _findings_section(self, report: Report) -> str:
        if not report.findings:
            return "## Findings\n\n_No findings recorded._"
        ordered = sorted(report.findings, key=lambda f: _SEVERITY_ORDER.index(f.severity))
        blocks = [self._finding_block(f) for f in ordered]
        return "## Findings\n\n" + "\n\n".join(blocks)

    @staticmethod
    def _finding_block(f: Finding) -> str:
        flag = " ⚠️ _human validation required_" if f.requires_human_validation else ""
        rec = f"\n- **Recommendation:** {f.recommendation}" if f.recommendation else ""
        return f"### [{f.severity.value.upper()}] {f.title}{flag}\n\n{f.description}{rec}"

    @staticmethod
    def _list_section(title: str, items: list[str]) -> str:
        if not items:
            return f"## {title}\n\n_None._"
        bullets = "\n".join(f"- {item}" for item in items)
        return f"## {title}\n\n{bullets}"

    @staticmethod
    def _compact_json(data: dict) -> str:
        return json.dumps(data, indent=2, default=str)[:1500]
