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
    assessment_id: Optional[str] = None
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
    attack_mode: Optional[str] = None          # adversarial | agentic
    attack_catalog_id: Optional[str] = None    # unified catalog attack id (agentic)
    measurement_id: Optional[str] = None
    defense_id: Optional[str] = None
    defense_control: Optional[str] = None       # catalog control id applied in DEFEND
    defense_implementation: Optional[str] = None  # concrete control implementation id
    finding_id: Optional[str] = None            # the Finding this experiment produced
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
    if ws.assessment_id:
        from orion.assessments.service import record_workspace
        record_workspace(ws, base_dir)
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
def _proposal_runnable(p) -> bool:
    """A proposal is runnable if it has a legacy scenario OR maps to a catalog
    attack with a runner (so GenAI/agentic experiments are runnable too)."""
    if p.status != "READY":
        return False
    if p.scenario:
        return True
    from orion.catalog import attacks as CAT
    a = CAT.get(p.name)
    return bool(a and a.runnable)


def create_from_plan(plan, base_dir: str = DEFAULT_DIR, assessment_id: Optional[str] = None) -> ExperimentLifecycle:
    ws = ExperimentLifecycle(
        assessment_id=assessment_id,
        plan_id=plan.plan_id,
        analysis_context_id=plan.analysis_context_id,
        self_profile_id=plan.self_profile_id,
        target_profile_id=plan.target_profile_id,
        environment_profile_id=plan.environment_profile_id,
        threat_model_id=plan.threat_model_id,
    )
    ws.stages["plan"] = APPROVED if plan.approved_by_human else PROPOSED
    ready = [p for p in plan.proposals if _proposal_runnable(p)]
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
    # If the active experiment maps to a GenAI/agentic catalog attack, dispatch to
    # the controlled lab (not the adversarial scenario runner).
    from orion.catalog import attacks as CAT
    active = next((p for p in plan.proposals if p.experiment_id == ws.active_experiment_id), None)
    a = CAT.get(active.name) if active else None
    if a and a.family in (CAT.GENERATIVE_AI, CAT.AGENTIC_AI):
        return run_agentic_attack(ws, {"attack_id": a.id}, base_dir=base_dir)
    ws.attack_mode = "adversarial"
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
    ws.attack_mode = "adversarial"
    provenance = {
        "source_type": "blackbox_attack", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id,
        **({"assessment_id": ws.assessment_id} if ws.assessment_id else {}),
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
    from orion.catalog import attacks as CAT
    # Suggest the family that matches the active experiment (e.g. an LLM target's
    # prompt-injection proposal → generative_ai), so the chooser opens on it.
    suggested = "traditional_ml"
    active = next((p for p in (getattr(plan, "proposals", []) or [])
                   if p.experiment_id == ws.active_experiment_id), None)
    if active:
        a = CAT.get(active.name)
        if a:
            suggested = a.family
    ai_endpoints = (target.get("ai_endpoints") or []) if isinstance(target, dict) else []
    return {"derived": derived, "catalog": catalog("image"),
            "live_url": url if live else None,
            "ai_endpoints": ai_endpoints,
            "families": CAT.grouped_by_family(), "suggested_family": suggested}


def run_agentic_attack(ws: ExperimentLifecycle, params: Dict[str, Any],
                       base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    """Run a controlled GenAI / Agentic experiment (prompt injection, tool abuse)."""
    from orion import plans as PL
    from orion.catalog import attacks as CAT
    plan = PL.load_plan(ws.plan_id, base_dir)
    if plan is None or not plan.approved_by_human:
        raise StageError("agentic attack requires an approved plan")
    attack_id = (params or {}).get("attack_id")
    attack = CAT.get(attack_id)
    if attack is None or attack.family == CAT.TRADITIONAL_ML:
        raise StageError("a GenAI/agentic attack id is required")
    trials = int((params or {}).get("trials", 3))
    provenance = {
        "source_type": "agentic_attack", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id, "attack_id": attack.id,
        **({"assessment_id": ws.assessment_id} if ws.assessment_id else {}),
    }
    ws.attack_mode = "agentic"
    ws.attack_catalog_id = attack.id
    ws.stages["attack"] = RUNNING
    save(ws, base_dir)
    try:
        from orion.agentic import run_agentic_experiment
        rec = run_agentic_experiment(attack.id, trials=trials, base_dir=base_dir,
                                     provenance=provenance)
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
    return {"trace_id": rec.trace_id, "status": rec.status, "mode": "agentic",
            "attack": attack.name, "family": attack.family}


def run_live_agentic_attack(ws: ExperimentLifecycle, params: Dict[str, Any],
                            base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    """Run a real prompt injection against a live LLM endpoint (the extension)."""
    from orion import plans as PL
    from orion.catalog import attacks as CAT
    plan = PL.load_plan(ws.plan_id, base_dir)
    if plan is None or not plan.approved_by_human:
        raise StageError("live attack requires an approved plan")
    params = params or {}
    url = params.get("url") or (getattr(plan, "target", {}) or {}).get("target")
    endpoint = params.get("endpoint")
    if not url:
        raise StageError("a live target URL is required")
    if not endpoint:
        raise StageError("a live AI endpoint (e.g. /api/chat) is required")
    attack_id = params.get("attack_id", "ORN-ATTACK-PI-001")
    attack = CAT.get(attack_id)
    if attack is None or attack.family == CAT.TRADITIONAL_ML:
        raise StageError("a GenAI/agentic attack id is required")
    provenance = {
        "source_type": "live_prompt_injection", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id, "attack_id": attack.id,
        **({"assessment_id": ws.assessment_id} if ws.assessment_id else {}),
    }
    ws.attack_mode = "agentic_live"
    ws.attack_catalog_id = attack.id
    ws.stages["attack"] = RUNNING
    save(ws, base_dir)
    try:
        from orion.agentic import run_live_prompt_injection
        rec = run_live_prompt_injection(
            url=url, endpoint=endpoint, field=params.get("field"),
            owasp=params.get("owasp"),
            trials=int(params["trials"]) if params.get("trials") else None,
            attack_id=attack.id, base_dir=base_dir, provenance=provenance)
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
    return {"trace_id": rec.trace_id, "status": rec.status, "mode": "agentic_live",
            "attack": attack.name, "family": attack.family, "endpoint": endpoint}


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
    ws.attack_mode = "adversarial"
    attack = params.get("attack", "CarliniL2")
    provenance = {
        "source_type": "whitebox_attack", "plan_id": ws.plan_id,
        "experiment_id": ws.active_experiment_id, "analysis_context_id": ws.analysis_context_id,
        "self_profile_id": ws.self_profile_id, "target_profile_id": ws.target_profile_id,
        "environment_profile_id": ws.environment_profile_id, "threat_model_id": ws.threat_model_id,
        "experiment_workspace_id": ws.experiment_workspace_id,
        **({"assessment_id": ws.assessment_id} if ws.assessment_id else {}),
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


def _chain_provenance(ws: ExperimentLifecycle) -> Dict[str, Any]:
    return {"plan_id": ws.plan_id, "experiment_id": ws.active_experiment_id,
            "analysis_context_id": ws.analysis_context_id, "self_profile_id": ws.self_profile_id,
            "target_profile_id": ws.target_profile_id, "environment_profile_id": ws.environment_profile_id,
            "threat_model_id": ws.threat_model_id, "experiment_workspace_id": ws.experiment_workspace_id,
            **({"assessment_id": ws.assessment_id} if ws.assessment_id else {})}


def _asr(rec) -> float:
    v = (rec.metrics.get("attack_success_rate", {}) or {}).get("value")
    if v is not None:
        return float(v)
    return 1.0 if rec.status == "ATTACK_SUCCESS" else 0.0


def _ensure_finding(ws: ExperimentLifecycle, base_dir: str):
    """Create or corroborate the finding for the attack run.

    Findings are keyed by (attack, target) so re-running the SAME experiment
    accumulates independent runs. A single successful run is OBSERVED; CONFIRMED
    only when the corroboration policy is met across independent runs. Returns the
    Finding or None.
    """
    from orion.evidence import EvidenceStore
    from orion.findings import (FindingStore, build_finding_from_record, corroborate,
                                evaluate_corroboration)
    store = EvidenceStore(base_dir)
    rec = store.load(ws.attack_run_id)
    if rec.status != "ATTACK_SUCCESS":
        return None
    fs = FindingStore(base_dir)
    attack_id = (rec.parameters or {}).get("attack_id") or rec.attack_technique
    target = (rec.target or {}).get("model_name") or (rec.target or {}).get("task", "")
    existing = (fs.load(ws.finding_id) if ws.finding_id else None) or \
        fs.find_by_attack_target(attack_id, target)

    if existing:
        if rec.trace_id not in existing.evidence_refs:
            existing.evidence_refs.append(rec.trace_id)
        records = [store.load(t) for t in existing.evidence_refs if _safe_exists(store, t)]
        corroborate(existing, rec, evidence_records=records)
        fs.save(existing)
        ws.finding_id = existing.id
        return existing

    f = build_finding_from_record(rec, provenance=_chain_provenance(ws))
    f.corroboration = evaluate_corroboration(f, [rec])
    fs.save(f)
    ws.finding_id = f.id
    return f


def _safe_exists(store, trace_id: str) -> bool:
    try:
        store.load(trace_id)
        return True
    except Exception:  # noqa: BLE001
        return False


def measure(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("attack") != COMPLETE or not ws.attack_run_id:
        raise StageError("measure requires a completed attack")
    from orion.evidence import EvidenceStore
    rec = EvidenceStore(base_dir).load(ws.attack_run_id)
    ws.stages["measure"] = COMPLETE
    ws.stages["defend"] = READY
    ws.current_stage = "defend"
    finding = None
    try:
        finding = _ensure_finding(ws, base_dir)
    except Exception:  # noqa: BLE001 - finding creation must not break measurement
        finding = None
    save(ws, base_dir)
    p = rec.parameters or {}
    return {"measurement_id": ws.measurement_id, "status": rec.status,
            "metrics": rec.metrics, "family": rec.family,
            "baseline_result": rec.baseline_result, "adversarial_result": rec.adversarial_result,
            # Explanation layer between evidence and the Finding (P1.5).
            "success_evaluation": p.get("success_evaluation"),
            "trust_boundary": p.get("trust_boundary"),
            "boundary_crossing": p.get("boundary_crossing"),
            "finding_id": ws.finding_id, "finding": finding.to_dict() if finding else None}


def apply_defense(ws: ExperimentLifecycle, params: Optional[Dict[str, Any]] = None,
                  base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("measure") != COMPLETE:
        raise StageError("defend requires a completed measurement")
    ws.defense_control = (params or {}).get("control_id") or ws.defense_control
    # Resolve the concrete implementation (definition != implementation).
    from orion.catalog import controls as CC
    impl = CC.get_implementation((params or {}).get("implementation_id") or "") or \
        CC.default_implementation(ws.defense_control or "")
    ws.defense_implementation = impl.id if impl else None
    ws.defense_id = "ORN-DEF-" + uuid.uuid4().hex[:8].upper()
    ws.stages["defend"] = APPLIED
    ws.stages["retest"] = READY
    ws.current_stage = "retest"
    save(ws, base_dir)
    return {"defense_id": ws.defense_id, "control_id": ws.defense_control,
            "control_implementation": ws.defense_implementation,
            "coverage": impl.coverage if impl else None, "status": "APPLIED",
            "note": "Control selected and implementation applied. Not validated until Retest."}


def _metric_comparison(before_rec, after_rec) -> Dict[str, Any]:
    """Before/after over the numeric metrics both runs share (for agentic runs)."""
    out: Dict[str, Any] = {}
    for k, bm in (before_rec.metrics or {}).items():
        am = (after_rec.metrics or {}).get(k)
        if not (isinstance(bm, dict) and isinstance(am, dict)):
            continue
        b, a = bm.get("value"), am.get("value")
        if isinstance(b, bool) or isinstance(a, bool) or not isinstance(b, (int, float)) or not isinstance(a, (int, float)):
            continue
        out[k] = {"before": b, "after": a, "delta": round(a - b, 4)}
    return {"metrics": out}


def _update_finding_on_retest(ws, before_rec, after_rec, base_dir):
    from orion.findings import FindingStore, apply_retest
    if not ws.finding_id:
        return None
    fs = FindingStore(base_dir)
    f = fs.load(ws.finding_id)
    if not f:
        return None
    apply_retest(f, _asr(before_rec), _asr(after_rec), ws.defense_control or "control", after_rec.trace_id)
    f.control_implementation = ws.defense_implementation
    fs.save(f)
    return f


def retest(ws: ExperimentLifecycle, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    if ws.stages.get("defend") != APPLIED:
        raise StageError("retest requires an applied defense (posture change)")  # fail-closed
    if not ws.attack_run_id:
        raise StageError("retest requires a previous attack configuration")
    from orion.evidence import EvidenceStore
    store = EvidenceStore(base_dir)
    before_rec = store.load(ws.attack_run_id)
    ws.stages["retest"] = RUNNING
    save(ws, base_dir)

    retest_provenance = {
        "source_type": "experiment_retest", "defense_id": ws.defense_id,
        "defense_control": ws.defense_control, "retest_of": ws.attack_run_id,
        **_chain_provenance(ws),
    }

    try:
        if ws.attack_mode == "agentic_live":
            # Replay the SAME live attack, applying a client-side input filter when
            # the chosen control is one a gateway can enforce.
            from orion.agentic import run_live_prompt_injection
            p = before_rec.parameters or {}
            mitigate = ws.defense_control in {"instruction_provenance", "context_isolation", "human_approval"}
            after_rec = run_live_prompt_injection(
                url=p.get("url"), endpoint=p.get("endpoint", "/api/chat"), field=p.get("field"),
                owasp=p.get("owasp"), trials=int(p["trials"]) if p.get("trials") else None,
                attack_id=ws.attack_catalog_id,
                mitigate=mitigate, mode="hardened", base_dir=base_dir, provenance=retest_provenance)
        elif ws.attack_mode == "agentic":
            # Replay the SAME agentic attack with the applied control.
            from orion.agentic import run_agentic_experiment
            controls = [ws.defense_control] if ws.defense_control else []
            bp = before_rec.parameters or {}
            # Retest reuses the SAME attack, target and payload — only the control changes.
            after_rec = run_agentic_experiment(
                ws.attack_catalog_id, controls=controls,
                user_request=bp.get("user_request"), external_content=bp.get("external_content"),
                mode="hardened", base_dir=base_dir, provenance=retest_provenance)
        else:
            from orion.experiments import replay
            after_rec = replay(ws.attack_run_id, mode="hardened", base_dir=base_dir)
            after_rec.provenance = retest_provenance
            store.save(after_rec)
    except Exception:
        ws.stages["retest"] = FAILED
        save(ws, base_dir)
        raise

    ws.retest_run_id = after_rec.trace_id
    ws.stages["retest"] = COMPLETE
    ws.current_stage = "retest"

    finding = None
    try:
        finding = _update_finding_on_retest(ws, before_rec, after_rec, base_dir)
    except Exception:  # noqa: BLE001
        finding = None
    save(ws, base_dir)

    # Comparison is a *result* of retest (original vs retest measurement).
    if ws.attack_mode in ("agentic", "agentic_live"):
        cmp = _metric_comparison(before_rec, after_rec)
    else:
        from orion.experiments import compare
        cmp = compare(ws.attack_run_id, ws.retest_run_id, base_dir=base_dir)
    return {"retest_run_id": ws.retest_run_id, "status": after_rec.status, "comparison": cmp,
            "finding_id": ws.finding_id, "finding": finding.to_dict() if finding else None}


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
