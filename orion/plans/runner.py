"""Explicit execution of an approved plan's experiment (never automatic)."""
from __future__ import annotations

from typing import Any, Dict, Optional

from orion.plans import plan as P


class ExperimentNotRunnable(Exception):
    """Raised when an experiment cannot be executed automatically (manual only)."""


def run_experiment(plan: "P.ExperimentPlan", experiment_id: str,
                   base_dir: str = P.DEFAULT_DIR, image_path: Optional[str] = None) -> Dict[str, Any]:
    """Run one approved experiment explicitly and link provenance back to the plan.

    Requires the plan to be human-approved and the experiment to be READY.
    Dispatches to the real adversarial runner (white-box local artifact) or the
    deterministic scenario runner; manual-only experiments raise.
    """
    if not plan.approved_by_human:
        raise ExperimentNotRunnable("Plan is not human-approved; approve the plan first.")
    prop = plan.get(experiment_id)
    if prop is None:
        raise ExperimentNotRunnable(f"Unknown experiment {experiment_id}")
    if prop.status not in (P.READY, P.FAILED):
        raise ExperimentNotRunnable(f"Experiment {experiment_id} is {prop.status}, not runnable.")
    if not prop.scenario:
        raise ExperimentNotRunnable(f"{prop.name} is a manual experiment (no automated runner).")

    provenance = {
        "source_type": plan.source_type,
        "source_analysis_id": plan.source_analysis_id,
        "analysis_context_id": plan.analysis_context_id,
        "self_profile_id": plan.self_profile_id,
        "target_profile_id": plan.target_profile_id,
        "environment_profile_id": plan.environment_profile_id,
        "threat_model_id": plan.threat_model_id,
        "plan_id": plan.plan_id,
        "experiment_id": experiment_id,
    }

    prop.status = P.RUNNING
    P.save_plan(plan, base_dir)
    try:
        wp = prop.parameters.get("weights_path")
        if wp:
            # White-box local artifact -> real torch + ART attack.
            from orion.adversarial import run_adversarial_experiment
            record = run_adversarial_experiment(
                weights_path=wp,
                num_outputs=int(prop.parameters.get("num_outputs") or 2),
                image_path=image_path or "static/fake/0001_00_00_01_0.jpg",
                base_dir=base_dir, provenance=provenance,
            )
        else:
            # Deterministic scenario attack (no GPU required).
            from orion.experiments import run_scenario
            from orion.scenarios.loader import load_scenario
            record = run_scenario(load_scenario(prop.scenario), mode="attack",
                                  base_dir=base_dir, provenance=provenance)
    except Exception:
        prop.status = P.FAILED
        P.save_plan(plan, base_dir)
        raise

    prop.status = P.COMPLETED
    prop.run_trace_id = record.trace_id
    P.save_plan(plan, base_dir)
    return {"trace_id": record.trace_id, "status": record.status,
            "experiment_id": experiment_id, "plan_id": plan.plan_id}
