"""Shared experiment-plan schema + handoff from analysis into the Attack phase.

    Analysis proposes. Humans approve. Attack prepares. Humans execute.
    Approve Plan != Run Attack.

A plan is built from Know Yourself or Know Your Target, approved by a human
(which prepares execution but never runs anything), handed off to the Attack
workspace with full context, and each experiment is executed explicitly there.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Experiment status vocabulary.
PROPOSED = "PROPOSED"
UNDER_REVIEW = "UNDER_REVIEW"
APPROVED = "APPROVED"
READY = "READY"
NEEDS_INPUT = "NEEDS_INPUT"
EXCLUDED = "EXCLUDED"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"

DEFAULT_DIR = "artifacts"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_plan_id() -> str:
    return "ORN-PLAN-" + uuid.uuid4().hex[:8].upper()


# Scenario mapping: which attack names map to an executable Orion scenario.
_EVASION = ("evasion", "pgd", "fgsm", "c&w", "deepfool")


def _attack_to_scenario(name: str) -> Optional[str]:
    low = name.lower()
    if any(k in low for k in _EVASION):
        return "scenarios/pgd_evasion.yaml"
    return None  # membership inference, extraction, prompt injection, ... = manual


def _is_sensitive(name: str, target: Dict[str, Any]) -> bool:
    low = name.lower()
    if any(k in low for k in ("prompt injection", "rag", "tool", "mcp", "extraction", "poison")):
        return True
    access = str(target.get("access", "")).lower()
    model = str(target.get("model_name", target.get("model", ""))).lower()
    return access not in ("white_box", "") or "://" in model


@dataclass
class ExperimentProposal:
    experiment_id: str
    name: str
    applicability: str                 # HIGH | CONDITIONAL | ...
    status: str = READY                # READY | NEEDS_INPUT | EXCLUDED | RUNNING | COMPLETED | FAILED
    scenario: Optional[str] = None
    reason: str = ""
    branch: str = "traditional_ml"
    sensitive: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    run_trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentProposal":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})


@dataclass
class ExperimentPlan:
    plan_id: str = field(default_factory=new_plan_id)
    source_type: str = "know_yourself"     # know_yourself | know_your_target
    source_analysis_id: Optional[str] = None
    threat_model_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    target: Dict[str, Any] = field(default_factory=dict)
    system_profile: Dict[str, Any] = field(default_factory=dict)
    threat_model: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    proposals: List[ExperimentProposal] = field(default_factory=list)
    status: str = PROPOSED
    approved_by_human: bool = False
    approved_at: Optional[str] = None
    approval_scope: Optional[str] = None

    # ---- buckets ----
    def approved_experiments(self) -> List[ExperimentProposal]:
        return [p for p in self.proposals if p.status in (READY, RUNNING, COMPLETED, FAILED)]

    def conditional_experiments(self) -> List[ExperimentProposal]:
        return [p for p in self.proposals if p.status == NEEDS_INPUT]

    def excluded_experiments(self) -> List[ExperimentProposal]:
        return [p for p in self.proposals if p.status == EXCLUDED]

    def get(self, experiment_id: str) -> Optional[ExperimentProposal]:
        return next((p for p in self.proposals if p.experiment_id == experiment_id), None)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "proposals"}
        d["proposals"] = [p.to_dict() for p in self.proposals]
        d["counts"] = {
            "approved": len(self.approved_experiments()),
            "conditional": len(self.conditional_experiments()),
            "excluded": len(self.excluded_experiments()),
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentPlan":
        plan = cls(**{k: d.get(k) for k in cls.__annotations__ if k in d and k != "proposals"})
        plan.proposals = [ExperimentProposal.from_dict(p) for p in d.get("proposals", [])]
        return plan

    def handoff(self) -> Dict[str, Any]:
        """The structured handoff object consumed by the Attack workspace."""
        return {
            "plan_id": self.plan_id,
            "source_analysis_id": self.source_analysis_id,
            "source_type": self.source_type,
            "target": self.target,
            "system_profile": self.system_profile,
            "threat_model_id": self.threat_model_id,
            "threat_model": self.threat_model,
            "evidence_ids": self.evidence_ids,
            "approved_experiments": [p.to_dict() for p in self.approved_experiments()],
            "conditional_experiments": [p.to_dict() for p in self.conditional_experiments()],
            "excluded_experiments": [p.to_dict() for p in self.excluded_experiments()],
            "approved_by_human": self.approved_by_human,
            "approved_at": self.approved_at,
            "status": "READY_FOR_ATTACK_WORKSPACE" if self.approved_by_human else "DRAFT",
        }


# --------------------------------------------------------------------------- #
# Persistence (survives navigation; artifacts/<plan_id>/plan.json)
# --------------------------------------------------------------------------- #
def save_plan(plan: ExperimentPlan, base_dir: str = DEFAULT_DIR) -> Path:
    pdir = Path(base_dir) / plan.plan_id
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2, default=str), encoding="utf-8")
    return pdir


def load_plan(plan_id: str, base_dir: str = DEFAULT_DIR) -> Optional[ExperimentPlan]:
    path = Path(base_dir) / plan_id / "plan.json"
    if not path.exists():
        return None
    return ExperimentPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Build a plan from an analysis result
# --------------------------------------------------------------------------- #
def _status_for(applicability: str) -> str:
    a = (applicability or "").upper()
    if a in ("HIGH", "APPLICABLE"):
        return READY
    if a in ("CONDITIONAL", "INSUFFICIENT_EVIDENCE", "MEDIUM"):
        return NEEDS_INPUT
    return EXCLUDED


def build_plan_from_know_yourself(result: Dict[str, Any]) -> ExperimentPlan:
    tml = result.get("traditional_ml") or {}
    gen = result.get("generative_ai") or {}
    fp = tml.get("fingerprint") or gen.get("fingerprint") or {}
    target = {
        "system_type": result.get("system_type"),
        "model_type": fp.get("model_type") or gen.get("fingerprint", {}).get("subtype"),
        "model": fp.get("model_name") or fp.get("model") or "unknown",
        "framework": fp.get("framework"),
        "access": fp.get("access", "unknown"),
        "input_type": fp.get("input_type"),
        "task": fp.get("task"),
    }
    plan = ExperimentPlan(
        source_type="know_yourself",
        source_analysis_id=result.get("trace_id"),
        target=target,
        system_profile=fp,
        threat_model={"system_type": result.get("system_type"),
                      "access": fp.get("access", "unknown")},
        evidence_ids=[result.get("trace_id")] if result.get("trace_id") else [],
    )
    # Build proposals from the full security posture (so excluded are recorded too).
    posture = {p["attack"]: p for p in result.get("security_posture", [])}
    recommended = {r["attack"] for r in result.get("recommended_experiments", [])}
    for idx, (attack, p) in enumerate(posture.items(), 1):
        status = {"APPLICABLE": READY, "CONDITIONAL": NEEDS_INPUT}.get(p["status"], EXCLUDED)
        scenario = _attack_to_scenario(attack)
        params: Dict[str, Any] = {}
        if scenario and status == READY:
            params = {"epsilon": 0.03, "iterations": 40}
            # White-box local artifact -> runnable directly against the model.
            if fp.get("artifact") and fp.get("access") == "white_box":
                params["weights_path"] = fp["artifact"]
                params["num_outputs"] = fp.get("num_classes") or 2
        plan.proposals.append(ExperimentProposal(
            experiment_id=f"EXP-{idx:02d}", name=attack, applicability=p["status"],
            status=status, scenario=scenario, reason=p.get("rationale", ""),
            branch=p.get("branch", "traditional_ml"),
            sensitive=_is_sensitive(attack, target),
            parameters=params, evidence_ids=p.get("evidence", []),
            missing=p.get("prerequisites_missing", []),
        ))
    return plan


def build_plan_from_target_analysis(assessment: Dict[str, Any]) -> ExperimentPlan:
    target = {
        "target": assessment.get("target"),
        "system_type": (assessment.get("target_types") or [{}])[0].get("type"),
        "access": (assessment.get("threat_model") or {}).get("adversary", {}).get("access", "unknown"),
        "ai_surface": (assessment.get("ai_surface") or {}).get("status"),
    }
    plan = ExperimentPlan(
        source_type="know_your_target",
        source_analysis_id=assessment.get("recon_run_id") or assessment.get("recon_run"),
        target=target,
        system_profile={"target_types": assessment.get("target_types"),
                        "ai_surface": assessment.get("ai_surface")},
        threat_model=assessment.get("threat_model", {}),
        evidence_ids=[e.get("id") for e in assessment.get("evidence", []) if e.get("id")],
    )
    for idx, e in enumerate(assessment.get("suggested_experiments", []), 1):
        status = {"APPLICABLE": READY, "INSUFFICIENT_EVIDENCE": NEEDS_INPUT}.get(
            e.get("applicability"), EXCLUDED)
        scenario = e.get("scenario") or _attack_to_scenario(e.get("name", ""))
        plan.proposals.append(ExperimentProposal(
            experiment_id=f"EXP-{idx:02d}", name=e.get("name", "experiment"),
            applicability=e.get("applicability", ""), status=status, scenario=scenario,
            reason=e.get("rationale", ""), branch="generative_ai" if e.get("ai_specific") else "traditional_ml",
            sensitive=_is_sensitive(e.get("name", ""), target),
            parameters={"epsilon": 0.03, "iterations": 40} if (scenario and status == READY) else {},
            evidence_ids=e.get("evidence", []), missing=e.get("missing", []),
        ))
    return plan


def approve_plan(plan: ExperimentPlan, scope: str = "all_approved") -> ExperimentPlan:
    """Record human approval. Prepares execution; never runs anything."""
    plan.approved_by_human = True
    plan.approved_at = _now()
    plan.approval_scope = scope
    plan.status = APPROVED
    return plan
