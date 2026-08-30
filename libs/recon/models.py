"""Pydantic models shared across the recon workflow.

Ported from HexAgent's ``app/models/`` package. Kept in one module here since
this integration has far fewer moving parts (no LLM, no CLI, no report I/O).
"""

from __future__ import annotations

import datetime as _dt
import time
import uuid
from enum import Enum

from pydantic import BaseModel, Field


# -- plan ---------------------------------------------------------------------


class StepStatus(str, Enum):
    """Lifecycle state of a single plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReplanReason(str, Enum):
    """Stable, machine-readable codes the evaluator uses to request a replan."""

    OPEN_WEB_PORTS_FOUND = "open_web_ports_found"
    ROBOTS_PATHS_FOUND = "robots_paths_found"
    LOGIN_ENDPOINT_FOUND = "login_endpoint_found"
    BROWSER_LOGIN_FORM_FOUND = "browser_login_form_found"
    NUCLEI_CANDIDATE_FOUND = "nuclei_candidate_found"


class PlanStep(BaseModel):
    """A single actionable unit of the plan."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = Field(..., description="Human-readable goal of this step.")
    tool_name: str | None = Field(
        default=None, description="Preferred tool; may be chosen at execution time if None."
    )
    arguments: dict = Field(
        default_factory=dict, description="Static arguments known at planning time."
    )
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of steps whose output this step consumes."
    )
    status: StepStatus = StepStatus.PENDING

    def is_runnable(self, completed_ids: set[str]) -> bool:
        """Return True if the step is pending and all dependencies are complete."""
        return self.status == StepStatus.PENDING and set(self.depends_on).issubset(completed_ids)


class Plan(BaseModel):
    """An ordered, mutable collection of plan steps."""

    objective: str
    steps: list[PlanStep] = Field(default_factory=list)
    rationale: str | None = Field(default=None, description="Planner's reasoning for the plan.")

    def pending_steps(self) -> list[PlanStep]:
        """Steps that have not yet completed, failed or been skipped."""
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    def completed_ids(self) -> set[str]:
        """IDs of steps whose dependencies are considered satisfied.

        A dependent step becomes runnable once its prerequisite reaches any
        terminal state (done, failed or skipped) — not only on success —
        otherwise a single failed/skipped step would permanently block every
        step that depends on it (e.g. the final synthesis step).
        """
        terminal = {StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED}
        return {s.id for s in self.steps if s.status in terminal}

    def next_runnable(self) -> PlanStep | None:
        """Return the first runnable step honouring declared dependencies."""
        done = self.completed_ids()
        for step in self.steps:
            if step.is_runnable(done):
                return step
        return None

    def get(self, step_id: str) -> PlanStep | None:
        """Look up a step by id."""
        return next((s for s in self.steps if s.id == step_id), None)

    def is_complete(self) -> bool:
        """True when no pending steps remain."""
        return len(self.pending_steps()) == 0


# -- findings -------------------------------------------------------------------


class Severity(str, Enum):
    """Qualitative severity rating for a finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Observation(BaseModel):
    """A neutral fact gathered from a tool result, prior to interpretation."""

    source_tool: str = Field(..., description="Tool that produced the observation.")
    step_id: str | None = Field(default=None, description="Plan step that triggered the tool.")
    content: str = Field(..., description="What was observed.")


class Finding(BaseModel):
    """An interpreted, security-relevant conclusion drawn from observations."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    severity: Severity = Severity.INFO
    description: str
    evidence: list[str] = Field(
        default_factory=list, description="Observation snippets supporting the finding."
    )
    recommendation: str | None = Field(
        default=None, description="Suggested remediation or next investigative action."
    )
    requires_human_validation: bool = Field(
        default=False,
        description="True when a human should confirm before any follow-up action.",
    )
    validation_status: str | None = Field(
        default=None,
        description=(
            "Lifecycle of a scanner-sourced candidate finding: candidate, "
            "needs_validation, validated, false_positive, out_of_scope or "
            "informational. None for findings that don't originate from a "
            "candidate-producing tool (e.g. Nuclei) and are reported as-is."
        ),
    )


# -- tool I/O -------------------------------------------------------------------


class ToolStatus(str, Enum):
    """Outcome of a tool invocation."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class ToolCall(BaseModel):
    """A request to execute a named tool with arbitrary keyword arguments."""

    tool_name: str = Field(..., description="Registered name of the tool to run.")
    arguments: dict = Field(
        default_factory=dict, description="Keyword arguments forwarded to the tool."
    )
    rationale: str | None = Field(
        default=None, description="Why the agent selected this tool (for the report)."
    )


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool."""

    tool_name: str
    status: ToolStatus = ToolStatus.SUCCESS
    summary: str = Field(..., description="Human-readable one-line summary of the result.")
    data: dict = Field(default_factory=dict, description="Structured tool-specific payload.")
    error: str | None = Field(default=None, description="Error message when status is ERROR.")
    duration_ms: float = Field(default=0.0, description="Execution time in ms.")
    timestamp: float = Field(default_factory=time.time)

    @classmethod
    def ok(
        cls,
        tool_name: str,
        summary: str,
        data: BaseModel | dict | None = None,
        duration_ms: float = 0.0,
    ) -> "ToolResult":
        """Convenience constructor for a successful result."""
        payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else (data or {})
        return cls(
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            summary=summary,
            data=payload,
            duration_ms=duration_ms,
        )

    @classmethod
    def fail(cls, tool_name: str, error: str) -> "ToolResult":
        """Convenience constructor for a failed result."""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.ERROR,
            summary=f"{tool_name} failed",
            error=error,
        )

    @classmethod
    def skipped(cls, tool_name: str, reason: str) -> "ToolResult":
        """Convenience constructor for a tool that was not run (e.g. denied approval)."""
        return cls(
            tool_name=tool_name,
            status=ToolStatus.SKIPPED,
            summary=f"{tool_name} skipped: {reason}",
        )


# -- report ---------------------------------------------------------------------


class ExecutedStep(BaseModel):
    """Record of a step that was executed, pairing the step with its tool result."""

    step_id: str
    description: str
    tool_name: str
    status: str
    result_summary: str


class Report(BaseModel):
    """Aggregated outcome of a workflow run, rendered to markdown by the reporter."""

    objective: str
    generated_at: str = Field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    )
    plan: Plan
    executed_steps: list[ExecutedStep] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    human_validation_points: list[str] = Field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""


# -- nuclei ---------------------------------------------------------------------


class NucleiFinding(BaseModel):
    """A single matched template, treated as an unverified candidate."""

    template_id: str | None = None
    template_name: str | None = None
    severity: str | None = None
    matched_at: str | None = None
    matcher_name: str | None = None
    extracted_results: list[str] = Field(default_factory=list)
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    curl_command: str | None = None
    confidence: str = "candidate"
    validation_required: bool = True


class NucleiScanResult(BaseModel):
    """Uniform payload for both ``nuclei_scan_url`` and ``nuclei_scan_urls``."""

    success: bool
    action: str
    targets_scanned: list[str] = Field(default_factory=list)
    targets_skipped: list[dict] = Field(default_factory=list)
    command_summary: str = ""
    findings: list[NucleiFinding] = Field(default_factory=list)
    result_count: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
