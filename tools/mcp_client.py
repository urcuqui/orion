import asyncio
import os
from typing import Any, Dict, List, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    MultiServerMCPClient = None
    _IMPORT_ERROR = exc

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "sse")
_cached_tools: Optional[List[str]] = None


def _run_async(coro):
    return asyncio.run(coro)


def _normalize_tools(tools: Any) -> List[str]:
    if not tools:
        return []
    names: List[str] = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name"):
            names.append(tool["name"])
        elif hasattr(tool, "name"):
            names.append(getattr(tool, "name"))
    return names


async def _list_tools_async() -> List[str]:
    if MultiServerMCPClient is None:
        raise RuntimeError(
            "langchain_mcp_adapters import failed. "
            "Install/upgrade langchain_core and langchain-mcp-adapters. "
            f"Original error: {_IMPORT_ERROR}"
        )
    client = MultiServerMCPClient(
        {"default": {"url": MCP_SERVER_URL, "transport": MCP_TRANSPORT}}
    )
    if hasattr(client, "get_tools"):
        tools = await client.get_tools()
        print(f"tools: {[i.name for i in tools]}")
    elif hasattr(client, "session"):
        async with client.session("default") as session:
            if hasattr(session, "list_tools"):
                tools = await session.list_tools()
            else:
                raise RuntimeError("MCP session has no tools listing method.")
    elif hasattr(client, "list_tools"):
        tools = await client.list_tools("default")
    else:
        raise RuntimeError("MultiServerMCPClient has no tools listing method.")
    return _normalize_tools(tools)


def list_tools(force_refresh: bool = False) -> List[str]:
    global _cached_tools
    if _cached_tools is None or _cached_tools == [] or force_refresh:
        try:
            _cached_tools = _run_async(_list_tools_async())
        except Exception as exc:
            _cached_tools = []
            print(f"MCP tools/list exception: {exc}")
    return _cached_tools


def _log_available_tools() -> None:
    global _cached_tools
    if _cached_tools is None:
        _cached_tools = list_tools()
    if _cached_tools:
        print(f"MCP tools available: {', '.join(_cached_tools)}")
    else:
        print("MCP tools available: (none reported)")


async def _call_tool_async(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if MultiServerMCPClient is None:
        raise RuntimeError(
            "langchain_mcp_adapters import failed. "
            "Install/upgrade langchain_core and langchain-mcp-adapters. "
            f"Original error: {_IMPORT_ERROR}"
        )
    client = MultiServerMCPClient(
        {"default": {"url": MCP_SERVER_URL, "transport": MCP_TRANSPORT}}
    )
    if hasattr(client, "call_tool"):
        try:
            result = await client.call_tool(tool_name, arguments)
        except TypeError:
            result = await client.call_tool("default", tool_name, arguments)
    elif hasattr(client, "session"):
        async with client.session("default") as session:
            if hasattr(session, "call_tool"):
                result = await session.call_tool(tool_name, arguments)
            else:
                raise RuntimeError("MCP session has no call_tool method.")
    else:
        raise RuntimeError("MultiServerMCPClient has no call_tool method.")
    return {"ok": True, "result": result}


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP server tool using the MCP adapter client."""
    _log_available_tools()
    print(
        f"MCP tool call: {tool_name} | args keys: {', '.join(arguments.keys()) if arguments else 'none'}"
    )
    try:
        return _run_async(_call_tool_async(tool_name, arguments))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
