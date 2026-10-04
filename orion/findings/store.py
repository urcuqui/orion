"""Persist findings on disk (no database), under ``artifacts/findings/``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from orion.findings.model import Finding

DEFAULT_DIR = "artifacts"


class FindingStore:
    def __init__(self, base_dir: "str | Path" = DEFAULT_DIR) -> None:
        self.dir = Path(base_dir) / "findings"

    def _path(self, finding_id: str) -> Path:
        return self.dir / f"{finding_id}.json"

    def save(self, finding: Finding) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(finding.id)
        p.write_text(json.dumps(finding.to_dict(), indent=2, default=str), encoding="utf-8")
        return p

    def load(self, finding_id: str) -> Optional[Finding]:
        p = self._path(finding_id)
        if not p.exists():
            return None
        return Finding.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self) -> List[Finding]:
        if not self.dir.exists():
            return []
        out: List[Finding] = []
        for f in sorted(self.dir.glob("*.json")):
            try:
                out.append(Finding.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001 - a bad file must not break the listing
                continue
        out.sort(key=lambda x: x.updated_at, reverse=True)
        return out

    def find_by_evidence(self, trace_id: str) -> Optional[Finding]:
        for f in self.list():
            if trace_id in f.evidence_refs:
                return f
        return None


def save_finding(finding: Finding, base_dir: str = DEFAULT_DIR) -> Path:
    return FindingStore(base_dir).save(finding)


def load_finding(finding_id: str, base_dir: str = DEFAULT_DIR) -> Optional[Finding]:
    return FindingStore(base_dir).load(finding_id)


def list_findings(base_dir: str = DEFAULT_DIR) -> List[Finding]:
    return FindingStore(base_dir).list()
