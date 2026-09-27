"""Execute scenarios, produce evidence, replay and compare."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion import metrics as M
from orion.defenses import get_defense
from orion.evidence import EvidenceStore, ExperimentRecord, ExperimentStatus
from orion.experiments.base import ExperimentMode, ExperimentOutcome
from orion.experiments import synthetic
from orion.scenarios.loader import load_scenario
from orion.scenarios.model import Scenario


# Defenses that harden the model itself (translated to a robustness margin).
_MARGIN_DEFENSES = {"adversarial_training", "input_preprocessing", "input_validation"}
# Per-example margin contribution of a hardening defense.
_MARGIN_STEP = 0.015


def _resolve_mode(mode: "str | ExperimentMode") -> ExperimentMode:
    return mode if isinstance(mode, ExperimentMode) else ExperimentMode(str(mode).lower())


def _run_synthetic(scenario: Scenario, mode: ExperimentMode) -> ExperimentOutcome:
    params = scenario.attack.params
    epsilon = float(params.get("epsilon", 0.03))
    n = int(scenario.dataset.get("size", 20))
    seed = int(scenario.dataset.get("seed", 1337))

    # Hardening -> robustness margin + blocking controls.
    robust_margin = 0.0
    controls: List[Dict[str, Any]] = []
    if mode is ExperimentMode.HARDENED:
        for d in scenario.hardening.defenses:
            name = d["name"]
            if name in _MARGIN_DEFENSES:
                robust_margin += _MARGIN_STEP
            controls.append({"name": name, "params": d.get("params", {}), "blocked": False,
                             "notes": "hardening control applied for retest"})

    # baseline mode: no attack -> epsilon 0.
    active_epsilon = 0.0 if mode is ExperimentMode.BASELINE else epsilon
    batch = synthetic.run_batch(active_epsilon, robust_margin, n=n, seed=seed)

    y_true = batch["y_true"]
    y_clean = batch["y_pred_clean"]
    y_adv = batch["y_pred_adv"]

    # Blocking controls (e.g. confidence_threshold): a blocked adversarial
    # prediction is rejected in production and thus not a successful evasion.
    blocked_count = 0
    if mode is ExperimentMode.HARDENED:
        for d in scenario.hardening.defenses:
            if d["name"] == "confidence_threshold":
                defense = get_defense("confidence_threshold", **d.get("params", {}))
                for i in range(len(y_adv)):
                    res = defense.apply(prediction={"confidence": batch["confidence_adv"][i]})
                    if res.blocked and y_adv[i] != y_true[i]:
                        y_adv[i] = y_true[i]  # rejected -> treated as not-evaded
                        blocked_count += 1
                for c in controls:
                    if c["name"] == "confidence_threshold":
                        c["blocked"] = blocked_count > 0
                        c["notes"] = f"blocked {blocked_count} low-confidence adversarial prediction(s)"

    clean_acc = M.calculate_clean_accuracy(y_true, y_clean)
    robust_acc = M.calculate_robust_accuracy(y_true, y_adv)
    asr = M.calculate_attack_success_rate(y_true, y_adv)
    avg_pert = sum(batch["perturbations"]) / len(batch["perturbations"]) if batch["perturbations"] else 0.0
    avg_conf_shift = sum(
        M.confidence_shift(c, a) for c, a in zip(batch["confidence_clean"], batch["confidence_adv"])
    ) / len(y_true) if y_true else 0.0

    metrics = {
        "clean_accuracy": M.MetricResult("clean_accuracy", round(clean_acc, 4)).to_dict(),
        "robust_accuracy": M.MetricResult("robust_accuracy", round(robust_acc, 4)).to_dict(),
        "attack_success_rate": M.MetricResult("attack_success_rate", round(asr, 4)).to_dict(),
        "perturbation_linf": M.MetricResult("perturbation_linf", round(avg_pert, 4)).to_dict(),
        "confidence_shift": M.MetricResult("confidence_shift", round(avg_conf_shift, 4)).to_dict(),
        "class_level_degradation": {"value": M.class_level_degradation(y_true, y_clean, y_adv)},
        "attack_cost": {"value": M.attack_cost(iterations=int(params.get("iterations", 0)) or None)},
    }

    baseline_result = {
        "clean_accuracy": round(clean_acc, 4),
        "n_examples": len(y_true),
        "backend": "synthetic",
    }
    adversarial_result = {
        "robust_accuracy": round(robust_acc, 4),
        "attack_success_rate": round(asr, 4),
        "blocked_predictions": blocked_count,
    }

    return ExperimentOutcome(
        mode=mode,
        baseline_result=baseline_result,
        adversarial_result=adversarial_result,
        metrics=metrics,
        controls_tested=controls,
        attack_success=asr > 0,
        model_version=f"synthetic-margin-{robust_margin:.3f}",
        limitations=[
            "Synthetic teaching backend: results illustrate the methodology, not a real model.",
            "Metrics depend on the selected dataset size/seed and attack parameters.",
        ],
        notes="Deterministic synthetic backend (no GPU/torch required).",
    )


def _decide_status(scenario: Scenario, outcome: ExperimentOutcome) -> ExperimentStatus:
    mode = outcome.mode
    if mode is ExperimentMode.BASELINE:
        return ExperimentStatus.NO_ATTACK
    asr = outcome.adversarial_result.get("attack_success_rate", 0.0)
    if asr <= 0:
        return ExperimentStatus.ATTACK_BLOCKED
    if mode is ExperimentMode.HARDENED:
        # Compare against the un-hardened attack to detect partial mitigation.
        ref = _run_synthetic(scenario, ExperimentMode.ATTACK)
        ref_asr = ref.adversarial_result.get("attack_success_rate", 0.0)
        if asr < ref_asr:
            return ExperimentStatus.PARTIALLY_MITIGATED
    return ExperimentStatus.ATTACK_SUCCESS


def run_scenario(
    scenario: "Scenario | str",
    mode: "str | ExperimentMode" = ExperimentMode.ATTACK,
    base_dir: str = "artifacts",
    persist: bool = True,
) -> ExperimentRecord:
    """Run a scenario in a given mode and (optionally) persist evidence."""
    sc = scenario if isinstance(scenario, Scenario) else load_scenario(scenario)
    m = _resolve_mode(mode)

    outcome = _run_synthetic(sc, m)
    status = _decide_status(sc, outcome)

    record = ExperimentRecord(
        scenario_name=sc.name,
        phase=sc.phase.value,
        mode=m.value,
        target=sc.threat_model.target.to_dict(),
        model_version=outcome.model_version,
        threat_model=sc.threat_model.to_dict(),
        attack_technique=sc.attack.technique,
        parameters=dict(sc.attack.params),
        baseline_result=outcome.baseline_result,
        adversarial_result=outcome.adversarial_result,
        metrics=outcome.metrics,
        mitre_atlas=[m_.to_dict() for m_ in sc.mitre_atlas],
        hardening=list(sc.hardening.defenses),
        controls_tested=outcome.controls_tested,
        status=status.value,
        limitations=outcome.limitations,
        notes=outcome.notes,
    )

    if persist:
        EvidenceStore(base_dir).save(record, images=outcome.images or None)
    return record


def replay(trace_id: str, mode: Optional[str] = None, base_dir: str = "artifacts") -> ExperimentRecord:
    """Replay a prior experiment using the *same* parameters and target.

    Reconstructs the scenario from the stored evidence so the same attack
    technique, parameters, target and model version are used. If ``mode`` is
    given, re-runs in that mode (e.g. replay an ``attack`` trace as ``hardened``).
    """
    store = EvidenceStore(base_dir)
    original = store.load(trace_id)

    # Rebuild a scenario dict from the stored record (same params -> same attack).
    scenario_dict = {
        "name": original.scenario_name or f"replay-{trace_id}",
        "description": f"Replay of {trace_id}",
        "phase": "retest" if (mode or original.mode) == "hardened" else original.phase,
        "target": original.target,
        "adversary": original.threat_model.get("adversary", {}),
        "assets": original.threat_model.get("assets", []),
        "surfaces": original.threat_model.get("surfaces", []),
        "attack": {"technique": original.attack_technique, **original.parameters},
        "metrics": list(original.metrics.keys()),
        "mitre_atlas": {"mappings": original.mitre_atlas},
        "hardening": {"defenses": original.hardening or [c.get("name") for c in original.controls_tested]},
        "dataset": {},
    }
    from orion.scenarios.loader import load_scenario_dict

    sc = load_scenario_dict(scenario_dict)
    replay_mode = mode or original.mode
    record = run_scenario(sc, mode=replay_mode, base_dir=base_dir)
    record.notes = f"Replay of {trace_id} (same parameters). {record.notes}"
    # Persist the updated note.
    store.save(record)
    return record


def compare(baseline_trace: str, hardened_trace: str, base_dir: str = "artifacts") -> Dict[str, Any]:
    """Compare two experiment records (e.g. attack vs hardened)."""
    store = EvidenceStore(base_dir)
    a = store.load(baseline_trace)
    b = store.load(hardened_trace)

    def _val(rec: ExperimentRecord, key: str) -> Optional[float]:
        entry = rec.metrics.get(key)
        if isinstance(entry, dict) and "value" in entry:
            try:
                return float(entry["value"])
            except (TypeError, ValueError):
                return None
        return None

    keys = ["clean_accuracy", "robust_accuracy", "attack_success_rate",
            "perturbation_linf", "confidence_shift"]
    deltas = {}
    for k in keys:
        va, vb = _val(a, k), _val(b, k)
        if va is not None and vb is not None:
            deltas[k] = {"before": va, "after": vb, "delta": round(vb - va, 4)}

    return {
        "before": {"trace_id": a.trace_id, "mode": a.mode, "status": a.status},
        "after": {"trace_id": b.trace_id, "mode": b.mode, "status": b.status},
        "metrics": deltas,
        "note": "One successful mitigation does not prove universal robustness. See docs/limitations.md.",
    }
