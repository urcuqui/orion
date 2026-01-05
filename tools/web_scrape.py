from typing import Any, Dict

from tools.mcp_client import call_tool


def web_scrape(payload: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        url = payload.get("url", "")
        options = payload.get("options", {})
    else:
        url = payload
        options = {}

    if not url:
        raise ValueError("web_scrape requires a url.")

    return call_tool("web_scrape", {"url": url, "options": options})
