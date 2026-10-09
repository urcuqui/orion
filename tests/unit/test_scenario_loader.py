"""Unit tests for orion.scenarios.loader — validation and error paths."""
import pytest

from orion.scenarios import loader as L
from orion.scenarios.model import ScenarioValidationError


def test_load_scenario_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        L.load_scenario("does/not/exist.yaml")


def test_validate_scenario_missing_returns_problem_list():
    problems = L.validate_scenario("does/not/exist.yaml")
    assert problems and isinstance(problems, list)


def test_load_scenario_dict_rejects_empty_mapping():
    with pytest.raises(ScenarioValidationError):
        L.load_scenario_dict({})


def test_list_scenarios_missing_directory_is_empty():
    assert L.list_scenarios("no/such/dir") == []


def test_valid_scenario_dict_roundtrips():
    sc = L.load_scenario_dict({
        "name": "unit-pgd", "description": "unit", "phase": "attack",
        "attack": {"technique": "PGD", "parameters": {"epsilon": 0.03}},
        "target": {"task": "image_classification", "access": "white_box"},
        "adversary": {"goal": "evasion", "knowledge": "full", "access": "white_box"},
    })
    assert sc.name == "unit-pgd" and sc.attack.technique == "PGD"


def test_list_scenarios_reads_real_directory():
    # The repo ships scenarios/pgd_evasion.yaml; the listing must find + parse it.
    entries = L.list_scenarios("scenarios")
    assert any(e["valid"] == "yes" and e.get("technique") for e in entries)
