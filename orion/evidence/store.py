"""Persist and load experiment evidence on disk."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from orion.evidence.record import ExperimentRecord
from orion.evidence.report import render_report

DEFAULT_ARTIFACT_DIR = Path("artifacts")


class EvidenceStore:
    """Reads/writes ``artifacts/<trace_id>/`` evidence directories."""

    def __init__(self, base_dir: "str | Path" = DEFAULT_ARTIFACT_DIR) -> None:
        self.base_dir = Path(base_dir)

    def trace_dir(self, trace_id: str) -> Path:
        return self.base_dir / trace_id

    def save(self, record: ExperimentRecord, images: Optional[Dict[str, str]] = None) -> Path:
        """Persist a record (+ optional images) and return the trace directory.

        ``images`` maps a label (``original`` / ``adversarial`` / ``difference``)
        to a source file path to copy in.
        """
        tdir = self.trace_dir(record.trace_id)
        tdir.mkdir(parents=True, exist_ok=True)

        # Copy image artifacts in and record their relative paths.
        for label, src in (images or {}).items():
            src_path = Path(src)
            if src_path.exists():
                dest = tdir / f"{label}.png"
                if src_path.resolve() != dest.resolve():
                    shutil.copyfile(src_path, dest)
                record.artifacts[label] = dest.name

        (tdir / "experiment.json").write_text(
            json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        (tdir / "report.md").write_text(render_report(record), encoding="utf-8")
        return tdir

    def load(self, trace_id: str) -> ExperimentRecord:
        path = self.trace_dir(trace_id) / "experiment.json"
        if not path.exists():
            raise FileNotFoundError(f"No evidence found for trace_id {trace_id!r} at {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentRecord.from_dict(data)

    def list_traces(self) -> list:
        if not self.base_dir.exists():
            return []
        traces = []
        for d in sorted(self.base_dir.iterdir()):
            if (d / "experiment.json").exists():
                traces.append(d.name)
        return traces


def save_record(record: ExperimentRecord, images: Optional[Dict[str, str]] = None,
                base_dir: "str | Path" = DEFAULT_ARTIFACT_DIR) -> Path:
    return EvidenceStore(base_dir).save(record, images)


def load_record(trace_id: str, base_dir: "str | Path" = DEFAULT_ARTIFACT_DIR) -> ExperimentRecord:
    return EvidenceStore(base_dir).load(trace_id)
