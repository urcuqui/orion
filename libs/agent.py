from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, TypedDict

from langgraph.graph import END, StateGraph

from tools.execute_code import execute_code
from tools.file_tools import edit_file, read_file, write_file
from tools.mcp_client import call_tool, list_tools
from libs.utils import clean_response_deepseek


class AgentState(TypedDict):
    objective: str
    todo: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    subagents: List[Dict[str, Any]]
    messages: List[str]
    iteration: int
    max_iterations: int
    done: bool


@dataclass
class ToolSpec:
    name: str
    description: str
    func: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def list(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)


def heuristic_generate(prompt: str) -> str:
    """LLM-backed generator using DeepSeek."""
    system_prompt = (
        "You are a precise task planning and execution assistant. "
        "Follow the user's request and respond concisely."
    )
    response = ""
    try:
        import ollama

        model = os.getenv("LLM_GENERATE_MODEL", "deepseek-r1:7b")
        result = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        response = result.get("message", {}).get("content", "")
        print("planning was successful created")
    except Exception:
        try:
            from libs.agent_wrap import generate_text

            response = generate_text(f"{system_prompt}\n\n{prompt}")
        except Exception:
            response = ""
    return clean_response_deepseek(response)


def plan_objective(objective: str) -> List[str]:
    plan_prompt = (
        "Create a TODO list for the objective. Return each step on a new line.\n"
        f"Objective: {objective}"
    )
    response = heuristic_generate(plan_prompt)
    lines = [line.strip("- ") for line in response.splitlines() if line.strip()]
    steps: List[str] = []
    for line in lines:
        match = re.match(r"\d+\.\s*(.*)", line)
        steps.append(match.group(1) if match else line)
    if len(steps) < 3:
        steps.extend(
            [
                "Analyze the objective and constraints",
                "Design the supervisor workflow and subagents",
                "Implement tools and iterate to completion",
            ]
        )
    return steps[:8]


LOCAL_TOOLS = {"read_file", "write_file", "edit_file", "execute_code"}


def _extract_json_list(text: str) -> List[str]:
    code_block = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    candidate = code_block.group(1) if code_block else None
    if candidate is None:
        match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
        candidate = match.group(0) if match else None
    if not candidate:
        return []
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def _llm_route_tools(objective: str, allowed_tools: List[str]) -> List[str]:
    system_prompt = (
        "You are a routing agent. Choose the minimal set of tools needed to fulfill the task. "
        "Return ONLY a JSON array of tool names."
    )
    user_prompt = (
        "Objective:\n"
        f"{objective}\n\n"
        "Available tools:\n"
        f"{', '.join(sorted(allowed_tools))}"
    )
    response = ""
    try:
        import ollama

        model = os.getenv("LLM_ROUTER_MODEL", "deepseek-r1:7b")
        result = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = result.get("message", {}).get("content", "")
    except Exception:
        try:
            from libs.agent_wrap import generate_text

            response = generate_text(f"{system_prompt}\n\n{user_prompt}")
        except Exception:
            response = ""
    return _extract_json_list(response)


def select_tools_for_objective(objective: str, available_mcp_tools: List[str]) -> List[str]:
    allowed_tools = sorted(LOCAL_TOOLS | set(available_mcp_tools))
    selected = _llm_route_tools(objective, allowed_tools)
    if selected:
        filtered = [tool for tool in selected if tool in allowed_tools]
        if filtered:
            return sorted(set(filtered))
    objective_lower = objective.lower()
    tools: Set[str] = {"read_file", "write_file", "edit_file"}
    if any(word in objective_lower for word in ["code", "implement", "execute"]):
        tools.add("execute_code")
    if "search_internet" in available_mcp_tools and any(
        word in objective_lower for word in ["search", "research", "web", "internet"]
    ):
        tools.add("search_internet")
    if "web_scrape" in available_mcp_tools and any(
        word in objective_lower for word in ["scrape", "crawl", "extract"]
    ):
        tools.add("web_scrape")
    return sorted(tools)


def build_subagent(task: str, tools: List[str]) -> Dict[str, Any]:
    return {
        "name": f"Subagent-{abs(hash(task)) % 1000}",
        "prompt": (
            "You are a specialized subagent. Complete the task using the provided tools if needed.\n"
            f"Task: {task}\n"
            f"Tools: {', '.join(tools)}\n"
            "If a tool is needed, reply with a JSON object containing tool and input."
        ),
        "tools": tools,
    }


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{\s*\"tool\".*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_subagent(subagent: Dict[str, Any], registry: ToolRegistry) -> Dict[str, Any]:
    response = heuristic_generate(subagent["prompt"])
    tool_call = parse_tool_call(response)
    tool_result = None
    if tool_call:
        tool_name = tool_call.get("tool")
        tool_input = tool_call.get("input")
        if tool_name:
            try:
                if tool_name in LOCAL_TOOLS:
                    tool_spec = registry.get(tool_name)
                    if tool_spec:
                        tool_result = tool_spec.func(tool_input)
                else:
                    tool_result = call_tool(tool_name, tool_input or {})
            except Exception as exc:
                tool_result = f"Tool error: {exc}"
    return {
        "response": response,
        "tool_result": tool_result,
    }


def supervisor_node(state: AgentState) -> AgentState:
    if not state["todo"]:
        available_mcp_tools = list_tools()
        steps = plan_objective(state["objective"])
        selected_tools = select_tools_for_objective(
            state["objective"], available_mcp_tools
        )
        state["todo"] = [
            {
                "id": idx + 1,
                "task": step,
                "status": "pending",
                "tools": selected_tools,
            }
            for idx, step in enumerate(steps)
        ]
        tools_summary = ", ".join(selected_tools) if selected_tools else "(none)"
        state["messages"].append(
            f"Supervisor: Selected tools based on input: {tools_summary}"
        )
        state["messages"].append("Supervisor: TODO list created.")
    state = refine_plan(state)
    if state["iteration"] >= state["max_iterations"]:
        state["done"] = True
        state["messages"].append("Supervisor: Iteration limit reached.")
    if all(item["status"] == "done" for item in state["todo"]):
        state["done"] = True
        state["messages"].append("Supervisor: Objective completed.")
    return state


def refine_plan(state: AgentState) -> AgentState:
    """Workflow-only tool to update the plan after each task result."""
    if state["results"]:
        state["messages"].append("Supervisor: Plan refined with latest results.")
    return state


def task_node(state: AgentState) -> AgentState:
    if state["iteration"] >= state["max_iterations"]:
        state["done"] = True
        state["messages"].append("Supervisor: Iteration limit reached before task.")
        return state

    pending = next((item for item in state["todo"] if item["status"] != "done"), None)
    if not pending:
        return state
   
    registry = ToolRegistry()
    allowed_tools = set(pending.get("tools", []))
    if not allowed_tools:
        allowed_tools = {"read_file", "write_file", "edit_file"}
    if "read_file" in allowed_tools:
        registry.register(ToolSpec("read_file", "Read a file from the workspace", read_file))
    if "write_file" in allowed_tools:
        registry.register(
            ToolSpec("write_file", "Write to a file in the workspace", write_file)
        )
    if "edit_file" in allowed_tools:
        registry.register(ToolSpec("edit_file", "Edit a file in the workspace", edit_file))
    if "execute_code" in allowed_tools:
        registry.register(
            ToolSpec("execute_code", "Write and execute Python code", execute_code)
        )
    for tool_name in sorted(allowed_tools):
        if tool_name in LOCAL_TOOLS:
            continue
        registry.register(
            ToolSpec(
                tool_name,
                f"MCP tool: {tool_name}",
                lambda payload, name=tool_name: call_tool(name, payload or {}),
            )
        )
    print("building subagent for tool: ", pending["tools"])
    subagent = build_subagent(pending["task"], pending["tools"])
    state["subagents"].append(subagent)

    result = run_subagent(subagent, registry)
    pending["status"] = "done"
    state["results"].append(
        {
            "task": pending["task"],
            "response": result["response"],
            "tool_result": result["tool_result"],
        }
    )
    state["messages"].append(
        f"Supervisor: Collected result for task {pending['id']}"
    )
    state["iteration"] += 1
    return state


def route_from_supervisor(state: AgentState) -> str:
    return "end" if state.get("done") else "task"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("task", task_node)
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", route_from_supervisor, {"task": "task", "end": END}
    )
    graph.add_edge("task", "supervisor")
    return graph.compile()


def format_report(state: AgentState) -> str:
    todo_lines = "\n".join(
        f"- [{'x' if item['status'] == 'done' else ' '}] {item['task']}"
        for item in state["todo"]
    )
    results_lines = "\n".join(
        f"- {item['task']}: {item['response']}"
        for item in state["results"]
    )
    return (
        "## Supervisor Report\n"
        f"**Objective:** {state['objective']}\n\n"
        "### TODO List\n"
        f"{todo_lines}\n\n"
        "### Results\n"
        f"{results_lines}\n"
    )


def build_api_state(state: AgentState) -> Dict[str, Any]:
    todo = [
        {
            "id": item["id"],
            "task": item["task"],
            "status": item["status"],
            "tools": item.get("tools", []),
        }
        for item in state["todo"]
    ]
    tools = sorted({tool for item in todo for tool in item.get("tools", [])})
    subagents = [
        {"name": agent.get("name", "Subagent"), "tools": agent.get("tools", [])}
        for agent in state["subagents"]
    ]
    results = [
        {"task": result["task"], "response": result["response"]}
        for result in state["results"]
    ]
    return {
        "objective": state["objective"],
        "todo": todo,
        "tools": tools,
        "subagents": subagents,
        "results": results,
        "messages": list(state["messages"]),
        "iteration": state["iteration"],
        "done": state["done"],
    }


def run_supervisor_state(objective: str, max_iterations: int = 6) -> AgentState:
    app = build_graph()
    state: AgentState = {
        "objective": objective,
        "todo": [],
        "results": [],
        "subagents": [],
        "messages": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "done": False,
    }
    return app.invoke(state, {"recursion_limit": max_iterations * 4 + 4})


def run_supervisor(objective: str, max_iterations: int = 6) -> str:
    final_state = run_supervisor_state(objective, max_iterations=max_iterations)
    return format_report(final_state)


def run_supervisor_stream(objective: str, max_iterations: int = 6):
    app = build_graph()
    state: AgentState = {
        "objective": objective,
        "todo": [],
        "results": [],
        "subagents": [],
        "messages": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "done": False,
    }
    latest_state: AgentState = state
    for event in app.stream(state, {"recursion_limit": max_iterations * 4 + 4}):
        if isinstance(event, dict):
            for value in event.values():
                if isinstance(value, dict) and "objective" in value:
                    latest_state = value
        yield latest_state
