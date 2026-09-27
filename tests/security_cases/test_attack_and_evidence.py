"""Security regression tests: attack, metrics and evidence properties.

These verify *properties* of the methodology, not merely that code runs.
"""
import json
from pathlib import Path

import pytest

from orion.evidence import EvidenceStore, ExperimentStatus
from orion.experiments import run_scenario
from orion.experiments.base import ExperimentMode
from orion.scenarios.loader import load_scenario

SCENARIO = "scenarios/pgd_evasion.yaml"


@pytest.fixture
def scenario():
    return load_scenario(SCENARIO)


def test_attack_changes_prediction(scenario, tmp_path):
    """An effective attack must degrade robust accuracy below clean accuracy."""
    rec = run_scenario(scenario, mode=ExperimentMode.ATTACK, base_dir=str(tmp_path))
    clean = rec.metrics["clean_accuracy"]["value"]
    robust = rec.metrics["robust_accuracy"]["value"]
    asr = rec.metrics["attack_success_rate"]["value"]
    assert clean > robust, "attack should reduce accuracy on adversarial inputs"
    assert asr > 0, "attack success rate should be positive for a successful attack"
    assert rec.status == ExperimentStatus.ATTACK_SUCCESS.value


def test_attack_metrics_are_generated(scenario, tmp_path):
    """All declared metrics must be present and numeric."""
    rec = run_scenario(scenario, mode=ExperimentMode.ATTACK, base_dir=str(tmp_path))
    for name in ("clean_accuracy", "robust_accuracy", "attack_success_rate",
                 "perturbation_linf", "confidence_shift"):
        assert name in rec.metrics, f"missing metric {name}"
        assert isinstance(rec.metrics[name]["value"], (int, float))
    # Perturbation must respect the L-infinity budget from the scenario.
    assert rec.metrics["perturbation_linf"]["value"] <= scenario.attack.params["epsilon"] + 1e-9


def test_attack_evidence_is_saved(scenario, tmp_path):
    """Evidence files (experiment.json + report.md) must be written."""
    rec = run_scenario(scenario, mode=ExperimentMode.ATTACK, base_dir=str(tmp_path))
    tdir = Path(tmp_path) / rec.trace_id
    assert (tdir / "experiment.json").exists()
    assert (tdir / "report.md").exists()
    data = json.loads((tdir / "experiment.json").read_text())
    # Evidence must carry the threat model and the ATLAS mappings.
    assert data["threat_model"]["adversary"]["goal"]
    assert data["attack_technique"] == scenario.attack.technique
    assert data["metrics"]


def test_normal_behavior_is_preserved(scenario, tmp_path):
    """Baseline (no attack) must preserve legitimate behaviour."""
    rec = run_scenario(scenario, mode=ExperimentMode.BASELINE, base_dir=str(tmp_path))
    assert rec.status == ExperimentStatus.NO_ATTACK.value
    assert rec.metrics["clean_accuracy"]["value"] == pytest.approx(1.0)
    assert rec.metrics["attack_success_rate"]["value"] == 0.0
