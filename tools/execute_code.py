import contextlib
import io
from typing import Any


def execute_code(payload: str | dict) -> dict:
    if isinstance(payload, dict):
        code = payload.get("code", "")
    else:
        code = payload
    if not isinstance(code, str) or not code.strip():
        raise ValueError("execute_code expects non-empty code.")

    stdout = io.StringIO()
    local_vars: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": stdout.getvalue()}

    return {"ok": True, "stdout": stdout.getvalue(), "locals": local_vars}
