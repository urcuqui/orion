"""The ExperimentRecord: the canonical evidence object."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orion.evidence.status import ExperimentStatus


def new_trace_id() -> str:
    """A short, sortable-ish, unique trace id."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRecord:
    """All evidence for a single experiment run.

    This is serialized verbatim to ``experiment.json`` and rendered to
    ``report.md``. Fields mirror the report structure required by the spec.
    """

    trace_id: str = field(default_factory=new_trace_id)
    timestamp: str = field(default_factory=_now_iso)
    scenario_name: str = ""
    phase: str = "attack"
    mode: str = "attack"  # baseline | attack | hardened

    target: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    threat_model: Dict[str, Any] = field(default_factory=dict)

    attack_technique: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    baseline_result: Dict[str, Any] = field(default_factory=dict)
    adversarial_result: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    mitre_atlas: List[Dict[str, Any]] = field(default_factory=list)
    # Declared defenses from the scenario (recorded on every run so a replay can
    # re-apply them); controls_tested are the ones actually exercised this run.
    hardening: List[Dict[str, Any]] = field(default_factory=list)
    controls_tested: List[Dict[str, Any]] = field(default_factory=list)

    status: str = ExperimentStatus.NO_ATTACK.value
    limitations: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # label -> relative path
    # Per-trial execution trace for GenAI / agentic experiments (what was observed).
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)
    # Family of the attack (traditional_ml | generative_ai | agentic_ai) so the
    # Measure screen renders the right metrics instead of assuming adversarial ML.
    family: str = ""
    notes: str = ""
    # Traceability: analysis_id -> threat_model_id -> plan_id -> experiment_id -> run_id.
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "scenario_name": self.scenario_name,
            "phase": self.phase,
            "mode": self.mode,
            "provenance": self.provenance,
            "target": self.target,
            "model_version": self.model_version,
            "threat_model": self.threat_model,
            "attack_technique": self.attack_technique,
            "parameters": self.parameters,
            "baseline_result": self.baseline_result,
            "adversarial_result": self.adversarial_result,
            "metrics": self.metrics,
            "mitre_atlas": self.mitre_atlas,
            "hardening": self.hardening,
            "controls_tested": self.controls_tested,
            "status": self.status,
            "limitations": self.limitations,
            "artifacts": self.artifacts,
            "execution_trace": self.execution_trace,
            "family": self.family,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRecord":
        rec = cls(trace_id=data.get("trace_id", new_trace_id()))
        for key, value in data.items():
            if hasattr(rec, key):
                setattr(rec, key, value)
        return rec
