"""Core metric implementations.

All functions accept plain Python sequences/numbers. Where an image tensor is
involved it may be a nested list or a numpy array; we only rely on ``abs`` and
element iteration, so numpy is optional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

Number = float


@dataclass
class MetricResult:
    """A named metric value with optional detail, ready for evidence JSON."""

    name: str
    value: float
    unit: str = ""
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "detail": self.detail}


def _flatten(values) -> List[float]:
    """Flatten an arbitrarily nested numeric structure (or numpy array)."""
    # numpy array fast path without importing numpy.
    flatten_attr = getattr(values, "flatten", None)
    if callable(flatten_attr):
        try:
            return [float(v) for v in flatten_attr().tolist()]
        except Exception:  # pragma: no cover - defensive
            pass
    out: List[float] = []
    if isinstance(values, (int, float)):
        return [float(values)]
    for v in values:
        if isinstance(v, (list, tuple)) or hasattr(v, "__iter__"):
            out.extend(_flatten(v))
        else:
            out.append(float(v))
    return out


# --------------------------------------------------------------------------- #
# Aggregate (dataset/batch) metrics
# --------------------------------------------------------------------------- #
def calculate_clean_accuracy(y_true: Sequence, y_pred_clean: Sequence) -> float:
    """Accuracy of the model on unperturbed inputs (0..1)."""
    return _accuracy(y_true, y_pred_clean)


def calculate_robust_accuracy(y_true: Sequence, y_pred_adv: Sequence) -> float:
    """Accuracy of the model on adversarial inputs (0..1).

    Robust accuracy is the fraction of examples still classified correctly
    *after* the attack. Higher is better for the defender.
    """
    return _accuracy(y_true, y_pred_adv)


def calculate_attack_success_rate(
    y_true: Sequence,
    y_pred_adv: Sequence,
    y_target: Optional[Sequence] = None,
) -> float:
    """Fraction of examples for which the attack met its goal (0..1).

    Untargeted (``y_target`` is None): success = adversarial prediction differs
    from the true label. Targeted: success = adversarial prediction equals the
    intended target label.
    """
    true = list(y_true)
    adv = list(y_pred_adv)
    if not true:
        return 0.0
    if y_target is None:
        successes = sum(1 for t, a in zip(true, adv) if a != t)
    else:
        tgt = list(y_target)
        successes = sum(1 for a, g in zip(adv, tgt) if a == g)
    return successes / len(true)


def class_level_degradation(
    y_true: Sequence,
    y_pred_clean: Sequence,
    y_pred_adv: Sequence,
) -> Dict[object, Dict[str, float]]:
    """Per-class clean vs robust accuracy and their delta."""
    classes = sorted(set(y_true), key=lambda x: str(x))
    out: Dict[object, Dict[str, float]] = {}
    for cls in classes:
        idx = [i for i, t in enumerate(y_true) if t == cls]
        if not idx:
            continue
        clean = sum(1 for i in idx if y_pred_clean[i] == y_true[i]) / len(idx)
        robust = sum(1 for i in idx if y_pred_adv[i] == y_true[i]) / len(idx)
        out[cls] = {
            "clean_accuracy": clean,
            "robust_accuracy": robust,
            "degradation": clean - robust,
            "support": len(idx),
        }
    return out


def _accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    true = list(y_true)
    pred = list(y_pred)
    if not true:
        return 0.0
    correct = sum(1 for t, p in zip(true, pred) if t == p)
    return correct / len(true)


# --------------------------------------------------------------------------- #
# Per-example metrics
# --------------------------------------------------------------------------- #
def perturbation_linf(original, adversarial) -> float:
    """L-infinity norm of the perturbation (max absolute pixel change)."""
    o = _flatten(original)
    a = _flatten(adversarial)
    if not o or len(o) != len(a):
        return 0.0
    return max(abs(x - y) for x, y in zip(o, a))


def perturbation_l2(original, adversarial) -> float:
    """L2 norm of the perturbation."""
    o = _flatten(original)
    a = _flatten(adversarial)
    if not o or len(o) != len(a):
        return 0.0
    return sum((x - y) ** 2 for x, y in zip(o, a)) ** 0.5


def calculate_perturbation(original, adversarial, norm: str = "linf") -> float:
    """Dispatch to a perturbation norm (``"linf"`` or ``"l2"``)."""
    if norm == "l2":
        return perturbation_l2(original, adversarial)
    return perturbation_linf(original, adversarial)


def confidence_shift(baseline_confidence: Number, adversarial_confidence: Number) -> float:
    """Signed change in confidence of the *baseline* class (baseline - adv).

    A positive value means the attack reduced confidence in the original
    prediction.
    """
    return float(baseline_confidence) - float(adversarial_confidence)


def calculate_confidence_shift(baseline_confidence: Number, adversarial_confidence: Number) -> float:
    """Alias of :func:`confidence_shift` (spec-facing name)."""
    return confidence_shift(baseline_confidence, adversarial_confidence)


def attack_cost(query_count: Optional[int] = None, iterations: Optional[int] = None) -> Dict[str, Optional[int]]:
    """Report the attack cost where applicable (queries / iterations)."""
    return {"query_count": query_count, "iterations": iterations}


def summarize(results: Iterable[MetricResult]) -> Dict[str, object]:
    """Turn a list of :class:`MetricResult` into a JSON-ready mapping."""
    return {r.name: r.to_dict() for r in results}
