"""Unit tests for orion.metrics.core — pure functions, edge/error paths.

These complement the behavioral suite by pinning exact numeric behaviour and the
empty / mismatched / zero cases that integration tests never exercise.
"""
import pytest

from orion.metrics import core as M


# ------------------------------ accuracy ------------------------------------ #
def test_accuracy_perfect_and_partial():
    assert M.calculate_clean_accuracy([0, 1, 1], [0, 1, 1]) == 1.0
    assert M.calculate_robust_accuracy([0, 1, 1, 0], [0, 0, 1, 0]) == 0.75


def test_accuracy_empty_is_zero_not_error():
    assert M.calculate_clean_accuracy([], []) == 0.0
    assert M.calculate_robust_accuracy([], [1, 2]) == 0.0


def test_accuracy_mismatched_lengths_truncate_to_truth():
    # zip stops at the shorter sequence; denominator is len(y_true).
    assert M.calculate_clean_accuracy([1, 1], [1]) == 0.5


# -------------------------- attack success rate ----------------------------- #
def test_attack_success_rate_untargeted():
    assert M.calculate_attack_success_rate([0, 1, 2], [1, 1, 2]) == pytest.approx(1 / 3)


def test_attack_success_rate_targeted():
    asr = M.calculate_attack_success_rate([0, 1, 2], [1, 0, 1], y_target=[1, 1, 1])
    assert asr == pytest.approx(2 / 3)


def test_attack_success_rate_empty_is_zero():
    assert M.calculate_attack_success_rate([], []) == 0.0


# ---------------------------- perturbations --------------------------------- #
def test_perturbation_identical_inputs_are_zero():
    assert M.perturbation_linf([0.1, 0.2], [0.1, 0.2]) == 0.0
    assert M.perturbation_l2([0.1, 0.2], [0.1, 0.2]) == 0.0


def test_perturbation_linf_and_l2_math():
    assert M.perturbation_linf([0, 0, 0], [0.1, 0.3, -0.2]) == pytest.approx(0.3)
    assert M.perturbation_l2([0, 0], [3, 4]) == pytest.approx(5.0)


def test_calculate_perturbation_selects_norm():
    assert M.calculate_perturbation([0, 0], [3, 4], norm="l2") == pytest.approx(5.0)
    assert M.calculate_perturbation([0, 0], [3, 4], norm="linf") == pytest.approx(4.0)


def test_perturbation_empty_is_zero():
    assert M.perturbation_linf([], []) == 0.0
    assert M.perturbation_l2([], []) == 0.0


# --------------------------- confidence / cost ------------------------------ #
def test_confidence_shift_signed():
    assert M.confidence_shift(0.9, 0.5) == pytest.approx(0.4)
    assert M.confidence_shift(0.4, 0.9) == pytest.approx(-0.5)
    assert M.calculate_confidence_shift(1.0, 1.0) == 0.0


def test_attack_cost_passthrough_and_defaults():
    assert M.attack_cost(5, 10) == {"query_count": 5, "iterations": 10}
    assert M.attack_cost() == {"query_count": None, "iterations": None}


# --------------------- class-level degradation + summarize ------------------ #
def test_class_level_degradation_per_class():
    out = M.class_level_degradation([0, 0, 1], [0, 1, 1], [0, 0, 0])
    assert out[0]["clean_accuracy"] == 0.5 and out[0]["robust_accuracy"] == 1.0
    assert out[0]["degradation"] == -0.5 and out[0]["support"] == 2
    assert out[1]["robust_accuracy"] == 0.0


def test_summarize_keys_by_metric_name():
    s = M.summarize([M.MetricResult("linf", 0.03, "ratio"), M.MetricResult("l2", 1.2)])
    assert set(s) == {"linf", "l2"}
    assert s["linf"]["value"] == 0.03 and s["linf"]["unit"] == "ratio"
