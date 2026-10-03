"""Experiment lifecycle: one stateful workflow over PLAN → ATTACK → MEASURE →
DEFEND → RETEST that orchestrates existing objects (plan, runs, replay, compare).

    Plan approval != attack execution.
    Defense applied != defense effective.
    Retest determines whether posture improved.

Transitions are deterministic and fail-closed; no attack logic is duplicated here.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stage status vocabularies (§32).
PROPOSED, APPROVED = "PROPOSED", "APPROVED"
PENDING, READY, RUNNING, COMPLETE, FAILED, APPLIED = (
    "PENDING", "READY", "RUNNING", "COMPLETE", "FAILED", "APPLIED")

DEFAULT_DIR = "artifacts"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_workspace_id() -> str:
    return "ORN-EXP-" + uuid.uuid4().hex[:8].upper()


class StageError(Exception):
    """Raised on a disallowed (fail-closed) stage transition."""


@dataclass
class ExperimentLifecycle:
    experiment_workspace_id: str = field(default_factory=new_workspace_id)
    plan_id: Optional[str] = None
    analysis_context_id: Optional[str] = None
    self_profile_id: Optional[str] = None
    target_profile_id: Optional[str] = None
    environment_profile_id: Optional[str] = None
    threat_model_id: Optional[str] = None
    active_experiment_id: Optional[str] = None
    current_stage: str = "plan"
    stages: Dict[str, str] = field(default_factory=lambda: {
        "plan": PROPOSED, "attack": PENDING, "measure": PENDING,
        "defend": PENDING, "retest": PENDING})
    attack_run_id: Optional[str] = None
    measurement_id: Optional[str] = None
    defense_id: Optional[str] = None
    retest_run_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["next_action"] = next_action(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentLifecycle":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Path:
    ws.updated_at = _now()
    wdir = Path(base_dir) / ws.experiment_workspace_id
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "workspace.json").write_text(json.dumps(ws.to_dict(), indent=2, default=str), encoding="utf-8")
    return wdir


def load(ws_id: str, base_dir: str = DEFAULT_DIR) -> Optional[ExperimentLifecycle]:
    path = Path(base_dir) / ws_id / "workspace.json"
    if not path.exists():
        return None
    return ExperimentLifecycle.from_dict(json.loads(path.read_text(encoding="utf-8")))


def list_workspaces(base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    base = Path(base_dir)
    out: List[Dict[str, Any]] = []
    if not base.exists():
        return out
    for d in base.iterdir():
        f = d / "workspace.json"
        if f.exists():
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            out.append({"experiment_workspace_id": j.get("experiment_workspace_id"),
                        "plan_id": j.get("plan_id"), "active_experiment_id": j.get("active_experiment_id"),
                        "current_stage": j.get("current_stage"), "stages": j.get("stages", {}),
                        "updated_at": j.get("updated_at", "")})
    out.sort(key=lambda w: w.get("updated_at", ""), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Create from an approved plan
# --------------------------------------------------------------------------- #
def create_from_plan(plan, base_dir: str = DEFAULT_DIR) -> ExperimentLifecycle:
    ws = ExperimentLifecycle(
        plan_id=plan.plan_id,
        analysis_context_id=plan.analysis_context_id,
        self_profile_id=plan.self_profile_id,
        target_profile_id=plan.target_profile_id,
        environment_profile_id=plan.environment_profile_id,
        threat_model_id=plan.threat_model_id,
    )
    ws.stages["plan"] = APPROVED if plan.approved_by_human else PROPOSED
    ready = [p for p in plan.proposals if p.status == "READY" and p.scenario]
    if ready:
        ws.active_experiment_id = ready[0].experiment_id
        ws.stages["attack"] = READY if plan.approved_by_human else PENDING
    ws.current_stage = "attack" if ready and plan.approved_by_human else "plan"
    save(ws, base_dir)
    return ws


def select_experiment(ws: ExperimentLifecycle, experiment_id: str, base_dir: str = DEFAULT_DIR) -> ExperimentLifecycle:
    ws.active_experiment_id = experiment_id
    # Reset the downstream stages for the newly-selected experiment.
    ws.stages.update({"attack": READY, "measure": PENDING, "defend": PENDING, "retest": PENDING})
    ws.attack_run_id = ws.measurement_id = ws.defense_id = ws.retest_run_id = None
    ws.current_stage = "attack"
    save(ws, base_dir)
    return ws


# --------------------------------------------------------------------------- #
# Stage transitions (deterministic, fail-closed)
# --------------------------------------------------------------------------- #
def run_attack(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    from orion import plans as PL
    plan = PL.load_plan(ws.plan_id, base_dir)
    if plan is None:
        raise StageError("plan not found")
    if not plan.approved_by_human:
        raise StageError("attack requires an approved plan")      # fail-closed
    if not ws.active_experiment_id:
        raise StageError("no active experiment selected")
    ws.stages["attack"] = RUNNING
    save(ws, base_dir)
    try:
        result = PL.run_experiment(plan, ws.active_experiment_id, base_dir=base_dir)
    except Exception:
        ws.stages["attack"] = FAILED
        save(ws, base_dir)
        raise
    ws.attack_run_id = result["trace_id"]
    ws.measurement_id = result["trace_id"]        # metrics live on the run record
    ws.stages["attack"] = COMPLETE
    ws.stages["measure"] = READY
    ws.current_stage = "measure"
    save(ws, base_dir)
    return result


def run_blackbox_attack(ws: ExperimentLifecycle, params: Dict[str, Any],
                        base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    """Run a real decision-based black-box evasion against a live endpoint."""
    from orion import plans as PL
    plan = PL.load_plan(ws.plan_id, base_dir)
    if plan is None or not plan.approved_by_human:
        raise StageError("black-box attack requires an approved plan")
    url = (params or {}).get("url")
    if not url:
        raise StageError("a live target URL is required for a black-box attack")
    provenance = {
        "source_type": "blackbox_attack", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id,
    }
    ws.stages["attack"] = RUNNING
    save(ws, base_dir)
    try:
        from orion.adversarial import run_blackbox_evasion
        rec = run_blackbox_evasion(
            url=url, path=params.get("path", "/"), field=params.get("field", "image"),
            image_path=params.get("image", "static/fake/0001_00_00_01_0.jpg"),
            epsilon=float(params.get("epsilon", 0.05)),
            max_queries=int(params.get("max_queries", 20)),
            base_dir=base_dir, provenance=provenance)
    except Exception:
        ws.stages["attack"] = FAILED
        save(ws, base_dir)
        raise
    ws.attack_run_id = rec.trace_id
    ws.measurement_id = rec.trace_id
    ws.stages["attack"] = COMPLETE
    ws.stages["measure"] = READY
    ws.current_stage = "measure"
    save(ws, base_dir)
    return {"trace_id": rec.trace_id, "status": rec.status, "mode": "blackbox"}


def attack_options(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    """Derive the default access level and the attack catalog for the console.

    White-box vs black-box is *derived* from the plan's origin (Know Yourself →
    weights; external live service → query-only) and offered for confirm/override.
    """
    from orion import plans as PL
    from orion.adversarial.catalog import catalog, derive_access_level
    plan = PL.load_plan(ws.plan_id, base_dir)
    target = (getattr(plan, "target", {}) or {}) if plan else {}
    url = target.get("target") if isinstance(target, dict) else None
    live = bool(url and str(url).startswith("http"))
    derived = derive_access_level(self_profile_id=ws.self_profile_id,
                                  target_url=url if live else None)
    return {"derived": derived, "catalog": catalog("image"),
            "live_url": url if live else None}


def run_whitebox_attack(ws: ExperimentLifecycle, params: Dict[str, Any],
                        base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    """Run a real white-box (torch + ART) adversarial-image attack on the weights."""
    from orion import plans as PL
    plan = PL.load_plan(ws.plan_id, base_dir)
    if plan is None or not plan.approved_by_human:
        raise StageError("white-box attack requires an approved plan")
    params = params or {}
    weights = params.get("weights_path")
    image = params.get("image")
    if not weights:
        raise StageError("white-box attack requires a model weights path")
    if not image:
        raise StageError("white-box attack requires an input image")
    attack = params.get("attack", "CarliniL2")
    provenance = {
        "source_type": "whitebox_attack", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id,
    }
    ws.stages["attack"] = RUNNING
    save(ws, base_dir)
    try:
        from orion.adversarial import infer_num_outputs, run_adversarial_experiment
        num_outputs = params.get("num_outputs") or infer_num_outputs(weights)
        rec = run_adversarial_experiment(
            weights_path=weights, num_outputs=int(num_outputs), image_path=image,
            attack=attack, params=params.get("params") or {},
            base_dir=base_dir, provenance=provenance)
    except Exception:
        ws.stages["attack"] = FAILED
        save(ws, base_dir)
        raise
    ws.attack_run_id = rec.trace_id
    ws.measurement_id = rec.trace_id
    ws.stages["attack"] = COMPLETE
    ws.stages["measure"] = READY
    ws.current_stage = "measure"
    save(ws, base_dir)
    return {"trace_id": rec.trace_id, "status": rec.status, "mode": "whitebox",
            "attack": attack, "num_outputs": int(num_outputs)}


def measure(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("attack") != COMPLETE or not ws.attack_run_id:
        raise StageError("measure requires a completed attack")
    from orion.evidence import EvidenceStore
    rec = EvidenceStore(base_dir).load(ws.attack_run_id)
    ws.stages["measure"] = COMPLETE
    ws.stages["defend"] = READY
    ws.current_stage = "defend"
    save(ws, base_dir)
    return {"measurement_id": ws.measurement_id, "status": rec.status,
            "metrics": rec.metrics, "baseline_result": rec.baseline_result,
            "adversarial_result": rec.adversarial_result}


def apply_defense(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("measure") != COMPLETE:
        raise StageError("defend requires a completed measurement")
    ws.defense_id = "ORN-DEF-" + uuid.uuid4().hex[:8].upper()
    ws.stages["defend"] = APPLIED
    ws.stages["retest"] = READY
    ws.current_stage = "retest"
    save(ws, base_dir)
    return {"defense_id": ws.defense_id, "status": "APPLIED",
            "note": "Defense applied. A defense is not validated until the attack is replayed (Retest)."}


def retest(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("defend") != APPLIED:
        raise StageError("retest requires an applied defense (posture change)")  # fail-closed
    if not ws.attack_run_id:
        raise StageError("retest requires a previous attack configuration")
    from orion.experiments import replay, compare
    ws.stages["retest"] = RUNNING
    save(ws, base_dir)
    try:
        hardened = replay(ws.attack_run_id, mode="hardened", base_dir=base_dir)
    except Exception:
        ws.stages["retest"] = FAILED
        save(ws, base_dir)
        raise
    ws.retest_run_id = hardened.trace_id
    # Stamp provenance so the retest run links back through the whole chain.
    try:
        from orion.evidence import EvidenceStore
        store = EvidenceStore(base_dir)
        rec = store.load(hardened.trace_id)
        rec.provenance = {
            "source_type": "experiment_retest", "plan_id": ws.plan_id,
            "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
            "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
            "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
            "defense_id": ws.defense_id, "retest_of": ws.attack_run_id,
            "experiment_workspace_id": ws.experiment_workspace_id,
        }
        store.save(rec)
    except Exception:  # noqa: BLE001
        pass
    ws.stages["retest"] = COMPLETE
    ws.current_stage = "retest"
    save(ws, base_dir)
    # Comparison is a *result* of retest (original measurement vs retest measurement).
    cmp = compare(ws.attack_run_id, ws.retest_run_id, base_dir=base_dir)
    return {"retest_run_id": ws.retest_run_id, "status": hardened.status, "comparison": cmp}


# --------------------------------------------------------------------------- #
# Deterministic next-action engine (§35) — never an LLM
# --------------------------------------------------------------------------- #
def next_action(ws: ExperimentLifecycle) -> Dict[str, str]:
    s = ws.stages
    if s.get("plan") not in (APPROVED,):
        return {"label": "REVIEW PLAN", "stage": "plan"}
    # No runnable experiment (e.g. AI surface only POSSIBLE → adversarial excluded).
    if not ws.active_experiment_id and s.get("attack") != COMPLETE:
        return {"label": "NO RUNNABLE ATTACK — RE-ANALYZE WITH ACTIVE PROBE", "stage": "plan"}
    if s.get("attack") in (PENDING, READY):
        return {"label": "RUN ATTACK", "stage": "attack"}
    if s.get("attack") == COMPLETE and s.get("measure") == READY:
        return {"label": "VIEW MEASUREMENTS", "stage": "measure"}
    if s.get("measure") == COMPLETE and s.get("defend") in (PENDING, READY):
        return {"label": "OPEN DEFEND", "stage": "defend"}
    if s.get("defend") == APPLIED and s.get("retest") in (PENDING, READY):
        return {"label": "RETEST SAME EXPERIMENT", "stage": "retest"}
    if s.get("retest") == COMPLETE:
        return {"label": "VIEW COMPARISON", "stage": "retest"}
    return {"label": "OPEN EVIDENCE", "stage": "evidence"}
