"""Security regression tests: replay and hardening properties."""
import pytest

from orion.experiments import replay, run_scenario
from orion.experiments.base import ExperimentMode
from orion.scenarios.loader import load_scenario

SCENARIO = "scenarios/pgd_evasion.yaml"


@pytest.fixture
def scenario():
    return load_scenario(SCENARIO)


def test_replay_uses_same_parameters(scenario, tmp_path):
    """Replay must reuse the same target, technique and attack parameters."""
    original = run_scenario(scenario, mode=ExperimentMode.ATTACK, base_dir=str(tmp_path))
    replayed = replay(original.trace_id, mode="attack", base_dir=str(tmp_path))

    assert replayed.attack_technique == original.attack_technique
    assert replayed.parameters == original.parameters
    assert replayed.target == original.target
    # Same parameters + deterministic backend => identical measured degradation.
    assert replayed.metrics["attack_success_rate"]["value"] == \
        original.metrics["attack_success_rate"]["value"]


def test_hardened_model_improves_robust_metric(scenario, tmp_path):
    """Retesting a hardened configuration must not worsen robust accuracy, and
    should improve it (or reduce attack success) relative to the raw attack."""
    attack = run_scenario(scenario, mode=ExperimentMode.ATTACK, base_dir=str(tmp_path))
    hardened = run_scenario(scenario, mode=ExperimentMode.HARDENED, base_dir=str(tmp_path))

    robust_before = attack.metrics["robust_accuracy"]["value"]
    robust_after = hardened.metrics["robust_accuracy"]["value"]
    asr_before = attack.metrics["attack_success_rate"]["value"]
    asr_after = hardened.metrics["attack_success_rate"]["value"]

    assert robust_after >= robust_before
    assert asr_after <= asr_before
    assert robust_after > robust_before or asr_after < asr_before, \
        "hardening must measurably improve at least one robustness metric"
    # Clean behaviour must be preserved after mitigation.
    assert hardened.metrics["clean_accuracy"]["value"] == \
        pytest.approx(attack.metrics["clean_accuracy"]["value"])
