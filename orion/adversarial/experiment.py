"""Run a full adversarial-image experiment and persist evidence.

Ties the structured adversarial result (`generate_adversarial_evidence`) into
the methodology's evidence layer so the web UI, CLI and tests all share one
source of truth. Logic lives here, not in Flask routes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from orion.adversarial.image import generate_adversarial_evidence
from orion.evidence import EvidenceStore, ExperimentRecord, ExperimentStatus
from orion.metrics import MetricResult, confidence_shift


def run_adversarial_experiment(
    weights_path: str,
    num_outputs: int,
    image_path: str,
    labels: Optional[Dict[int, str]] = None,
    attack: str = "CarliniL2",
    params: Optional[Dict[str, Any]] = None,
    threat_model: Optional[Dict[str, Any]] = None,
    scenario_name: str = "adversarial-image",
    base_dir: str = "artifacts",
    output_dir: str = "static/adversarial",
    provenance: Optional[Dict[str, Any]] = None,
) -> ExperimentRecord:
    """Execute the real (torch + ART) adversarial-image attack and save evidence.

    Returns the persisted :class:`ExperimentRecord`. Raises ``RuntimeError`` if
    torch/ART are unavailable so the caller can surface a clean error.
    """
    params = params or {}
    res = generate_adversarial_evidence(
        weights_path=weights_path,
        num_outputs=int(num_outputs),
        image_path=image_path,
        labels=labels or {0: "fake", 1: "real"},
        attack=attack,
        params=params,
        output_dir=output_dir,
    )

    tm = threat_model or {
        "target": {"task": "image_classification", "access": "white_box"},
        "adversary": {"goal": "evasion", "knowledge": "full", "access": "white_box", "budget": "low"},
        "assets": ["model_integrity", "prediction_reliability"],
        "surfaces": ["inference_api", "model_artifact"],
    }

    weights_name = Path(weights_path).name
    from orion.catalog import attacks as _CAT
    _ad = _CAT.get(attack)
    record = ExperimentRecord(
        scenario_name=scenario_name,
        phase="attack",
        mode="attack",
        family="traditional_ml",
        target={
            "task": "image_classification",
            "access": "white_box",
            "model_name": "vit_base_patch16_224 (fine-tuned)",
            "model_version": weights_name,
        },
        model_version=weights_name,
        threat_model=tm,
        attack_technique=f"{attack} (white-box)",
        parameters={"nb_classes": int(num_outputs), "input_shape": "3x224x224",
                    "attack_id": _ad.id if _ad else attack,
                    "tuning": {k: v for k, v in params.items()}},
        baseline_result={
            "prediction": res.baseline_prediction,
            "confidence": res.baseline_confidence,
        },
        adversarial_result={
            "prediction": res.adversarial_prediction,
            "confidence": res.adversarial_confidence,
        },
        metrics={
            "perturbation_linf": MetricResult("perturbation_linf", res.perturbation_linf).to_dict(),
            "perturbation_l2": MetricResult("perturbation_l2", res.perturbation_l2).to_dict(),
            "confidence_shift": MetricResult(
                "confidence_shift",
                round(confidence_shift(res.baseline_confidence, res.adversarial_confidence), 4),
            ).to_dict(),
            "attack_success": {"value": res.attack_success},
        },
        status=(ExperimentStatus.ATTACK_SUCCESS if res.attack_success
                else ExperimentStatus.ATTACK_BLOCKED).value,
        limitations=[
            "Single-image, white-box attack; one input does not represent all adversaries or the full dataset.",
            "Results are specific to this model artifact and image.",
        ],
        notes=f"Real torch+ART {attack} attack against {weights_name}.",
        provenance=provenance or {},
    )

    EvidenceStore(base_dir).save(record, images={
        "original": res.artifacts.get("original"),
        "adversarial": res.artifacts.get("output_art"),
        "difference": res.artifacts.get("difference"),
    })
    return record
