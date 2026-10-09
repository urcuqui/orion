"""Unit tests for the built-in candidate defenses (pure transforms/decisions).

A defense is a *candidate* control: its effect is shown by retest, not asserted.
These tests pin the deterministic transform/decision each one makes.
"""
import pytest

from orion.defenses import get_defense
from orion.defenses.builtin import _flatten, _quantize


def test_input_preprocessing_quantizes_and_never_blocks():
    res = get_defense("input_preprocessing", levels=4).apply(input_data=[0.1, 0.9, 0.5])
    assert res.blocked is False
    assert res.transformed_input == pytest.approx([0.0, 1.0, 2 / 3])
    assert res.detail["levels"] == 4


def test_input_preprocessing_without_input_is_noop():
    res = get_defense("input_preprocessing").apply(input_data=None)
    assert res.transformed_input is None and res.blocked is False


def test_confidence_threshold_blocks_low_confidence():
    d = get_defense("confidence_threshold", threshold=0.6)
    assert d.apply(prediction={"confidence": 0.4}).blocked is True
    assert d.apply(prediction={"confidence": 0.8}).blocked is False
    assert d.apply(prediction=None).blocked is False          # default confidence 1.0


def test_rate_limit_blocks_after_max_queries():
    d = get_defense("rate_limit", max_queries=1, window_seconds=600)
    assert d.apply().blocked is False                          # 1st within limit
    assert d.apply().blocked is True                           # 2nd exceeds


def test_input_validation_blocks_out_of_range():
    d = get_defense("input_validation", min=-1.0, max=1.0)
    assert d.apply(input_data=[0.0, 0.5, -0.5]).blocked is False
    assert d.apply(input_data=[0.0, 9.0]).blocked is True


def test_monitoring_and_adversarial_training_never_block():
    assert get_defense("monitoring").apply(prediction={"x": 1}).blocked is False
    at = get_defense("adversarial_training", hardened_weights="w/hardened.pth").apply()
    assert at.blocked is False and at.detail["hardened_weights"] == "w/hardened.pth"


def test_quantize_clamps_levels_and_handles_nested():
    assert _quantize([[0.5]], 1) == [[0.0]]                    # levels<2 clamped to 2 → round(0.5)=0
    assert _quantize(0.5, 4) == pytest.approx(2 / 3)


def test_flatten_scalar_and_nested():
    assert _flatten(3) == [3.0]
    assert _flatten([[1, 2], [3]]) == [1.0, 2.0, 3.0]
