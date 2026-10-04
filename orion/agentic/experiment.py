"""Run a GenAI / Agentic experiment in the controlled lab and persist evidence.

Produces an :class:`ExperimentRecord` whose metrics are the ones the attack
declares in the unified catalog (not image-adversarial metrics), whose status is
derived from the observed trace, and whose ``execution_trace`` preserves every
trial so a Finding can point back to exactly what happened.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion.agentic import lab
from orion.catalog import attacks as CAT
from orion.evidence import EvidenceStore, ExperimentRecord, ExperimentStatus

# metric_id -> function(observations list) -> rate
_METRIC_FNS = {
    "attack_success_rate": lambda o: _rate(o, "attack_success"),
    "refusal_rate": lambda o: 1.0 - _rate(o, "injected_instruction_followed"),
    "policy_violation_rate": lambda o: _rate(o, "policy_violation"),
    "secret_leakage_rate": lambda o: _rate(o, "secret_leaked"),
    "instruction_following_rate": lambda o: _rate(o, "injected_instruction_followed"),
    "unauthorized_tool_call_rate": lambda o: _rate(o, "unauthorized_tool_action"),
    "privilege_boundary_violation_rate": lambda o: _rate(o, "privilege_boundary_crossed"),
    "unsafe_action_rate": lambda o: _rate(o, "unsafe_action"),
    "goal_hijack_rate": lambda o: _rate(o, "attack_success"),
    "approval_bypass_rate": lambda o: _rate(o, "approval_bypassed"),
}


def _rate(observations: List[Dict[str, Any]], key: str) -> float:
    if not observations:
        return 0.0
    return round(sum(1 for o in observations if o.get(key)) / len(observations), 4)


def run_agentic_experiment(
    attack_id: str,
    controls: Optional[List[Any]] = None,
    trials: int = 3,
    mode: str = "attack",
    scenario_name: Optional[str] = None,
    base_dir: str = "artifacts",
    provenance: Optional[Dict[str, Any]] = None,
) -> ExperimentRecord:
    """Execute a GenAI/agentic attack (optionally with controls) and save evidence."""
    attack = CAT.get(attack_id)
    if attack is None or attack.family == CAT.TRADITIONAL_ML:
        raise ValueError(f"not a GenAI/agentic attack: {attack_id!r}")

    trace = lab.run_trials(attack.id, controls=controls, trials=trials)
    obs = [t["observations"] for t in trace]

    metrics: Dict[str, Any] = {}
    for mid in attack.supported_metrics:
        if mid in _METRIC_FNS:
            metrics[mid] = {"name": mid, "value": _METRIC_FNS[mid](obs), "unit": "ratio"}
    asr = metrics.get("attack_success_rate", {}).get("value", 0.0)
    successes = sum(1 for o in obs if o.get("attack_success"))
    metrics["attack_success"] = {"value": asr > 0}
    metrics["trials"] = {"name": "trials", "value": len(trace), "unit": "runs"}
    metrics["successes"] = {"name": "successes", "value": successes, "unit": "runs"}

    control_ids = lab._norm_controls(controls)
    controls_tested = [{"name": cid, "blocked": asr == 0.0,
                        "notes": f"applied; attack_success_rate={asr}"} for cid in control_ids]

    atlas = [{"technique_id": tid, "technique": tid, "tactic": "", "confidence": "mapped"}
             for tid in attack.framework_mappings.get("mitre_atlas", [])]

    status = (ExperimentStatus.ATTACK_SUCCESS if asr > 0 else ExperimentStatus.ATTACK_BLOCKED).value
    rec = ExperimentRecord(
        scenario_name=scenario_name or attack.name.lower().replace(" ", "-").replace("/", ""),
        phase="attack", mode=mode, family=attack.family,
        target={"task": attack.family, "access": "black_box",
                "model_name": "controlled agentic lab"},
        threat_model={
            "adversary": {"goal": "evasion/abuse", "knowledge": "black_box",
                          "access": "black_box", "budget": "low"},
            "assets": ["tool_integrity", "data_confidentiality", "policy_enforcement"],
            "surfaces": ["untrusted_context", "agent_tools"]},
        attack_technique=attack.name,
        parameters={"attack_id": attack.id, "trials": len(trace),
                    "controls": list(control_ids.keys())},
        baseline_result={"benign_task": "export_report -> internal-reports",
                         "note": "agent's intended, authorised action"},
        adversarial_result={"injected_payload": trace[0]["injected_payload"],
                            "successes": f"{successes}/{len(trace)}"},
        metrics=metrics,
        mitre_atlas=atlas,
        controls_tested=controls_tested,
        status=status,
        execution_trace=trace,
        limitations=[
            "Controlled lab: deterministic agent and tools, not a live LLM/MCP deployment.",
            "Results characterise this scenario and control set, not the whole system.",
        ],
        notes=f"Controlled agentic experiment for {attack.id} "
              f"({'with controls: ' + ', '.join(control_ids) if control_ids else 'no controls'}).",
        provenance=provenance or {},
    )
    EvidenceStore(base_dir).save(rec)
    return rec
