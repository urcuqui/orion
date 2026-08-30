import asyncio
import os
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    MultiServerMCPClient = None
    _IMPORT_ERROR = exc

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "sse")
_cached_tools: Optional[List[str]] = None
_cached_tools_detailed: Optional[List[Dict[str, Any]]] = None


class ToolDetail(TypedDict, total=False):
    name: str
    description: str
    input_schema: Any


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


def _normalize_tool_details(tools: Any) -> List[ToolDetail]:
    if not tools:
        return []
    details: List[ToolDetail] = []
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name")
            description = tool.get("description")
            input_schema = tool.get("input_schema") or tool.get("schema")
        else:
            name = getattr(tool, "name", None)
            description = getattr(tool, "description", None)
            input_schema = getattr(tool, "input_schema", None)
        if name:
            detail: ToolDetail = {"name": name}
            if description:
                detail["description"] = description
            if input_schema is not None:
                detail["input_schema"] = input_schema
            details.append(detail)
    return details


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


async def _list_tools_detailed_async() -> List[ToolDetail]:
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
    return _normalize_tool_details(tools)


def list_tools(force_refresh: bool = False) -> List[str]:
    global _cached_tools
    if _cached_tools is None or _cached_tools == [] or force_refresh:
        try:
            _cached_tools = _run_async(_list_tools_async())
        except Exception as exc:
            _cached_tools = []
            print(f"MCP tools/list exception: {exc}")
    return _cached_tools


def list_tools_detailed(force_refresh: bool = False) -> List[ToolDetail]:
    global _cached_tools_detailed
    if _cached_tools_detailed is None or _cached_tools_detailed == [] or force_refresh:
        try:
            _cached_tools_detailed = _run_async(_list_tools_detailed_async())
        except Exception as exc:
            _cached_tools_detailed = []
            print(f"MCP tools/list detailed exception: {exc}")
    return _cached_tools_detailed


def _log_available_tools() -> None:
    global _cached_tools
    if _cached_tools is None:
        _cached_tools = list_tools()
    if _cached_tools:
        print(f"MCP tools available: {', '.join(_cached_tools)}")
    else:
        print("MCP tools available: (none reported)")


def _extract_tool_text(result: Any) -> Optional[str]:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        text_value = result.get("text")
        if isinstance(text_value, str):
            return text_value
        content = result.get("content")
        if isinstance(content, list):
            texts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_text = item.get("text")
                    if isinstance(item_text, str):
                        texts.append(item_text)
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                return "\n".join(texts)
        return None
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    content_attr = getattr(result, "content", None)
    if isinstance(content_attr, list):
        texts = []
        for item in content_attr:
            if hasattr(item, "text") and isinstance(item.text, str):
                texts.append(item.text)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    return None


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
            print("MCP call_tool {tool_name}")
            result = await client.call_tool(tool_name, arguments)
        except TypeError:
            print("MCP call_tool (with server) {tool_name}")
            result = await client.call_tool("default", tool_name, arguments)
    elif hasattr(client, "session"):
        async with client.session("default") as session:
            if hasattr(session, "call_tool"):
                print(f"MCP session call_tool {tool_name}")
                result = await session.call_tool(tool_name, arguments)
                print(f"MCP result was successful")
            else:
                print("MCP session has no call_tool method. {tool_name}")
                raise RuntimeError("MCP session has no call_tool method.")
    else:
        raise RuntimeError("MultiServerMCPClient has no call_tool method.")
    result_payload = result.json() if hasattr(result, "json") else result
    return {
        "ok": True,
        "result": result_payload,
        "text": _extract_tool_text(result) or _extract_tool_text(result_payload),
    }


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call an MCP server tool using the MCP adapter client."""
    _log_available_tools()
    print(
        f"MCP tool call: {tool_name} | args keys: {', '.join(arguments.keys()) if arguments else 'none'}"
    )
    try:
        return _run_async(_call_tool_async(tool_name, arguments))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "result": None, "text": None}
