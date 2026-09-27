"""Structured, reproducible evidence for experiments.

Every experiment produces a trace directory under ``artifacts/<trace_id>/``
containing machine-readable (``experiment.json``) and human-readable
(``report.md``) evidence, plus any visual artifacts (original/adversarial/
difference images).

    Evidence over intuition. Metrics over screenshots.
"""
from __future__ import annotations

from orion.evidence.status import ExperimentStatus
from orion.evidence.record import ExperimentRecord, new_trace_id
from orion.evidence.store import EvidenceStore, save_record, load_record

__all__ = [
    "ExperimentStatus",
    "ExperimentRecord",
    "new_trace_id",
    "EvidenceStore",
    "save_record",
    "load_record",
]
