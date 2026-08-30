"""LangGraph workflow: state, nodes, routing and assembly.

The state machine: intake -> plan -> execute -> evaluate -> (execute again |
replan | human checkpoint | report). Nodes are methods on :class:`WorkflowNodes`,
which receives its collaborating agents via constructor injection. Each node
takes the current :class:`AgentState` and returns a partial ``dict`` of updates
that LangGraph merges into the state.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from libs.recon.evaluator import EvaluatorAgent
from libs.recon.executor import ExecutorAgent
from libs.recon.models import ExecutedStep, Finding, Observation, Plan, Report, StepStatus, ToolResult, ToolStatus
from libs.recon.planner import PlannerAgent
from libs.recon.reporter import ReporterAgent
from libs.recon.specialists import ApprovalCallback
from tools.recon.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)

# A single reactive run can legitimately replan multiple times (open web ports
# -> HTTP phase, robots.txt -> targeted GET, login endpoint -> controlled POST).
MAX_REPLANS = 5

_STEP_STATUS_BY_TOOL_STATUS = {
    ToolStatus.SUCCESS: StepStatus.DONE,
    ToolStatus.SKIPPED: StepStatus.SKIPPED,
    ToolStatus.ERROR: StepStatus.FAILED,
}


class AgentState(BaseModel):
    """Mutable state threaded through every node of the workflow."""

    # Inputs
    objective: str
    target: str
    max_iterations: int = 12
    require_human_approval: bool = False

    # Plan & progress
    plan: Plan | None = None
    completed_step_ids: list[str] = Field(default_factory=list)

    # Accumulated knowledge
    observations: list[Observation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    executed_steps: list[ExecutedStep] = Field(default_factory=list)
    reasoning_history: list[str] = Field(default_factory=list)

    # Control flow
    iterations: int = 0
    needs_replan: bool = False
    replan_reason: str = ""
    replans: int = 0
    awaiting_human: bool = False
    stopped_reason: str = ""

    # Outputs
    human_validation_points: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    report: Report | None = None
    report_markdown: str | None = None

    # Browser session (populated by evaluator when Playwright tools are active)
    browser_session_active: bool = False
    browser_screenshots: list[str] = Field(default_factory=list)

    def observation_texts(self) -> list[str]:
        """Return observations as plain strings (handy for prompts)."""
        return [o.content for o in self.observations]


# -- routing --------------------------------------------------------------------

EXECUTE = "execute"
REPLAN = "replan"
HUMAN = "human"
REPORT = "report"


def route_after_evaluate(state: AgentState) -> str:
    """Decide the next node after an evaluation step.

    Priority order:
      1. Stop if the iteration budget is exhausted.
      2. Replan when the evaluator requested it.
      3. Continue executing while runnable steps remain.
      4. Pause for human approval if configured.
      5. Otherwise produce the report.
    """
    if state.iterations >= state.max_iterations:
        return REPORT
    if state.needs_replan:
        return REPLAN
    if state.plan is not None and state.plan.next_runnable() is not None:
        return EXECUTE
    if state.require_human_approval:
        return HUMAN
    return REPORT


# -- nodes ------------------------------------------------------------------


class WorkflowNodes:
    """Bundles the agents and exposes them as LangGraph node callables."""

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        evaluator: EvaluatorAgent,
        reporter: ReporterAgent,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._evaluator = evaluator
        self._reporter = reporter

    def intake(self, state: AgentState) -> dict:
        """Record the user objective and seed the reasoning history."""
        msg = f"Objective received: {state.objective} (target: {state.target})"
        logger.info(msg)
        return {"reasoning_history": [*state.reasoning_history, msg]}

    def plan(self, state: AgentState) -> dict:
        """Create the initial plan."""
        plan = self._planner.plan(state.objective, state.target)
        note = f"Planned {len(plan.steps)} step(s): {plan.rationale}"
        return {"plan": plan, "reasoning_history": [*state.reasoning_history, note]}

    def execute(self, state: AgentState) -> dict:
        """Execute the next runnable plan step and record its tool result."""
        assert state.plan is not None
        step = state.plan.next_runnable()
        if step is None:
            return {}
        step.status = StepStatus.IN_PROGRESS
        if step.tool_name is None:
            # Synthesis/summary step: no tool is run; just mark it complete.
            step.status = StepStatus.DONE
            note = f"Completed synthesis step {step.id}: {step.description}"
            return {
                "plan": state.plan,
                "completed_step_ids": [*state.completed_step_ids, step.id],
                "iterations": state.iterations + 1,
                "reasoning_history": [*state.reasoning_history, note],
            }
        result = self._executor.execute(step, state.target)
        step.status = _STEP_STATUS_BY_TOOL_STATUS.get(result.status, StepStatus.FAILED)
        executed = ExecutedStep(
            step_id=step.id,
            description=step.description,
            tool_name=result.tool_name,
            status=result.status.value,
            result_summary=result.summary,
        )
        note = f"Executed {step.id} -> {result.tool_name}: {result.summary}"
        return {
            "plan": state.plan,
            "completed_step_ids": [*state.completed_step_ids, step.id],
            "tool_results": [*state.tool_results, result],
            "executed_steps": [*state.executed_steps, executed],
            "iterations": state.iterations + 1,
            "reasoning_history": [*state.reasoning_history, note],
        }

    def evaluate(self, state: AgentState) -> dict:
        """Interpret the most recent tool result into observations/findings."""
        if not state.tool_results or not state.completed_step_ids:
            return {}
        step = state.plan.get(state.completed_step_ids[-1]) if state.plan else None
        # Synthesis steps (no tool) produce no new result to evaluate.
        if step is not None and step.tool_name is None:
            return {}
        result = state.tool_results[-1]
        evaluation = self._evaluator.evaluate(step, result)  # type: ignore[arg-type]
        human_points = [
            f"{f.title}: {f.description}"
            for f in evaluation.findings
            if f.requires_human_validation
        ]
        note = f"Evaluated {result.tool_name}: {len(evaluation.findings)} finding(s)"
        updates: dict = {
            "observations": [*state.observations, *evaluation.observations],
            "findings": [*state.findings, *evaluation.findings],
            "human_validation_points": [*state.human_validation_points, *human_points],
            "needs_replan": evaluation.needs_replan and state.replans < MAX_REPLANS,
            "replan_reason": evaluation.replan_reason,
            "reasoning_history": [*state.reasoning_history, note],
        }
        # Merge browser-specific state when the evaluator produced it.
        if evaluation.new_screenshots:
            updates["browser_screenshots"] = [
                *state.browser_screenshots,
                *evaluation.new_screenshots,
            ]
        if evaluation.browser_session_active is not None:
            updates["browser_session_active"] = evaluation.browser_session_active
        return updates

    def replan(self, state: AgentState) -> dict:
        """Revise the plan in response to new information."""
        assert state.plan is not None
        last_result = state.tool_results[-1] if state.tool_results else None
        plan = self._planner.replan(
            state.plan, state.replan_reason, state.observation_texts(), last_result
        )
        note = f"Replanned ({state.replan_reason}); now {len(plan.steps)} step(s)"
        return {
            "plan": plan,
            "replans": state.replans + 1,
            "needs_replan": False,
            "reasoning_history": [*state.reasoning_history, note],
        }

    def human_checkpoint(self, state: AgentState) -> dict:
        """Mark the run as awaiting human approval (a terminal pause)."""
        note = "Human approval checkpoint reached; report generated for review."
        logger.info(note)
        return {"awaiting_human": True, "reasoning_history": [*state.reasoning_history, note]}

    def report(self, state: AgentState) -> dict:
        """Assemble and render the final markdown report."""
        next_actions = self._derive_next_actions(state)
        plan_complete = state.plan.is_complete() if state.plan else False
        if state.awaiting_human:
            stopped = "awaiting human approval"
        elif state.iterations >= state.max_iterations and not plan_complete:
            stopped = "maximum iterations reached"
        else:
            stopped = "objective completed"
        report = Report(
            objective=state.objective,
            plan=state.plan,  # type: ignore[arg-type]
            executed_steps=state.executed_steps,
            tool_results=state.tool_results,
            findings=state.findings,
            next_actions=next_actions,
            human_validation_points=state.human_validation_points,
            iterations=state.iterations,
            stopped_reason=stopped,
        )
        markdown = self._reporter.render(report)
        return {
            "report": report,
            "report_markdown": markdown,
            "next_actions": next_actions,
            "stopped_reason": stopped,
        }

    @staticmethod
    def _derive_next_actions(state: AgentState) -> list[str]:
        actions = [f.recommendation for f in state.findings if f.recommendation]
        seen: set[str] = set()
        unique = [a for a in actions if not (a in seen or seen.add(a))]
        if not unique:
            unique = ["Review the gathered reconnaissance data and define follow-up tests."]
        return unique


# -- assembly -----------------------------------------------------------------


def build_workflow(nodes: WorkflowNodes):
    """Construct and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", nodes.intake)
    graph.add_node("plan", nodes.plan)
    graph.add_node("execute", nodes.execute)
    graph.add_node("evaluate", nodes.evaluate)
    graph.add_node("replan", nodes.replan)
    graph.add_node("human", nodes.human_checkpoint)
    graph.add_node("report", nodes.report)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {EXECUTE: "execute", REPLAN: "replan", HUMAN: "human", REPORT: "report"},
    )
    graph.add_edge("replan", "execute")
    graph.add_edge("human", "report")
    graph.add_edge("report", END)

    return graph.compile()


def build_nodes(
    registry: ToolRegistry | None = None,
    approval_callback: ApprovalCallback | None = None,
    require_sensitive_approval: bool = False,
) -> WorkflowNodes:
    """Create the agent node bundle.

    Args:
        approval_callback: Consulted before any tool marked ``sensitive`` runs
            (e.g. ``http_post``) when ``require_sensitive_approval`` is set.
            Without one, such actions are denied by default (fail-closed).
    """
    registry = registry or default_registry()
    return WorkflowNodes(
        planner=PlannerAgent(registry),
        executor=ExecutorAgent(
            registry,
            approval_callback=approval_callback,
            require_sensitive_approval=require_sensitive_approval,
        ),
        evaluator=EvaluatorAgent(),
        reporter=ReporterAgent(),
    )
