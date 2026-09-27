"""Scenario loading/validation properties."""
import pytest

from orion.scenarios import ScenarioValidationError, load_scenario, load_scenario_dict


def test_valid_scenario_loads():
    sc = load_scenario("scenarios/pgd_evasion.yaml")
    assert sc.name == "pgd-evasion"
    assert sc.attack.technique == "PGD"
    assert sc.threat_model.adversary.goal.value == "targeted_misclassification"
    assert "model_integrity" in sc.threat_model.assets


def test_scenario_without_adversary_is_rejected():
    """An attack with no threat model must be rejected."""
    with pytest.raises(ScenarioValidationError):
        load_scenario_dict({
            "name": "no-threat-model",
            "attack": {"technique": "PGD", "epsilon": 0.03},
        })


def test_scenario_without_attack_is_rejected():
    with pytest.raises(ScenarioValidationError):
        load_scenario_dict({
            "name": "no-attack",
            "adversary": {"goal": "evasion"},
        })
