"""Unit tests for the metrics and ATLAS mapping layers."""
import pytest

from orion import metrics as M
from orion.mappings import validate_mappings, lookup_technique


def test_accuracy_metrics():
    y_true = [0, 1, 1, 0]
    y_clean = [0, 1, 1, 0]
    y_adv = [0, 0, 1, 1]
    assert M.calculate_clean_accuracy(y_true, y_clean) == 1.0
    assert M.calculate_robust_accuracy(y_true, y_adv) == 0.5
    assert M.calculate_attack_success_rate(y_true, y_adv) == 0.5


def test_targeted_attack_success_rate():
    y_true = [0, 0, 0]
    y_adv = [1, 1, 0]
    y_target = [1, 1, 1]
    # Two of three reached the target class.
    assert M.calculate_attack_success_rate(y_true, y_adv, y_target) == pytest.approx(2 / 3)


def test_perturbation_and_confidence_shift():
    orig = [[0.0, 0.0], [0.0, 0.0]]
    adv = [[0.03, 0.0], [0.0, -0.02]]
    assert M.perturbation_linf(orig, adv) == pytest.approx(0.03)
    assert M.perturbation_l2(orig, adv) == pytest.approx((0.03**2 + 0.02**2) ** 0.5)
    assert M.confidence_shift(0.99, 0.80) == pytest.approx(0.19)


def test_class_level_degradation():
    y_true = [0, 0, 1, 1]
    y_clean = [0, 0, 1, 1]
    y_adv = [1, 0, 1, 1]
    deg = M.class_level_degradation(y_true, y_clean, y_adv)
    assert deg[0]["degradation"] == pytest.approx(0.5)
    assert deg[1]["degradation"] == 0.0


def test_atlas_mappings_are_not_invented():
    """Known IDs resolve; unknown IDs are flagged, never silently accepted."""
    resolved = validate_mappings([
        {"technique_id": "AML.T0043.000", "confidence": "high"},
        {"technique_id": "AML.T9999", "confidence": "high"},
    ])
    assert resolved[0].known is True
    assert resolved[0].technique  # populated from the curated index
    assert resolved[1].known is False
    assert resolved[1].confidence == "low"  # uncertainty is explicit
    assert lookup_technique("AML.T0015") is not None
