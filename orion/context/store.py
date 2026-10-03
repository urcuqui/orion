"""Persistence for context profiles (reuses the artifact model)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DIR = "artifacts"
# profile_id prefix -> filename
_FILENAME = {"ORN-ENV": "environment.json", "ORN-TARGET": "target.json",
             "ORN-CTX": "context.json", "ORN-SELF": "self.json"}


def _file_for(profile_id: str) -> str:
    for pfx, fname in _FILENAME.items():
        if profile_id.startswith(pfx):
            return fname
    return "profile.json"


def save_profile(profile_id: str, data: Dict[str, Any], base_dir: str = DEFAULT_DIR) -> Path:
    pdir = Path(base_dir) / profile_id
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / _file_for(profile_id)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_profile(profile_id: str, base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    path = Path(base_dir) / profile_id / _file_for(profile_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
