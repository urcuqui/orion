import difflib
from typing import Any, Dict, List

_VFS: Dict[str, str] = {}


def _get_path(payload: Any) -> str:
    if isinstance(payload, dict):
        path = payload.get("path")
    else:
        path = payload
    if not path or not isinstance(path, str):
        raise ValueError("file tool requires a path string.")
    return path


def read_file(path: str | Dict[str, Any]) -> str:
    file_path = _get_path(path)
    return _VFS.get(file_path, "")


def write_file(path: str | Dict[str, Any], content: str | None = None) -> Dict[str, Any]:
    if isinstance(path, dict):
        payload = path
        file_path = _get_path(payload)
        file_content = payload.get("content", "")
    else:
        file_path = _get_path(path)
        file_content = content or ""
    _VFS[file_path] = file_content
    return {"path": file_path, "bytes_written": len(file_content.encode("utf-8"))}


def edit_file(path: str | Dict[str, Any], edits: List[Dict[str, str]] | str | None = None) -> Dict[str, Any]:
    if isinstance(path, dict):
        payload = path
        file_path = _get_path(payload)
        edit_instructions = payload.get("edits", payload.get("replace"))
    else:
        file_path = _get_path(path)
        edit_instructions = edits

    original = _VFS.get(file_path, "")
    updated = original
    diff_note = ""

    if isinstance(edit_instructions, list):
        for change in edit_instructions:
            find_text = change.get("find", "")
            replace_text = change.get("replace", "")
            if not find_text:
                continue
            updated = updated.replace(find_text, replace_text, 1)
        diff_note = "Applied list edits."
    elif isinstance(edit_instructions, str):
        updated = edit_instructions
        diff_note = "Replaced full content."
    else:
        raise ValueError("edit_file requires edits as list or string.")

    _VFS[file_path] = updated
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=file_path,
            tofile=file_path,
        )
    )
    return {"path": file_path, "diff": diff or diff_note}
