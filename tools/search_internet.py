from typing import Any, Dict

from tools.mcp_client import call_tool


def search_internet(payload: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        query = payload.get("query", "")
        max_results = payload.get("max_results", 5)
    else:
        query = payload
        max_results = 5

    if not query:
        raise ValueError("search_internet requires a query.")

    return call_tool(
        "search_internet", {"query": query, "max_results": max_results}
    )
