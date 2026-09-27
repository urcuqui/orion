"""Built-in candidate defenses.

These are intentionally simple and dependency-light so they can be reasoned
about and tested. Each is a *candidate* control whose effect must be shown by
retesting, not asserted.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from orion.defenses.base import Defense, DefenseResult, register_defense


class InputPreprocessingDefense(Defense):
    """Reduce high-frequency adversarial perturbation via quantization.

    Rounds each feature to ``levels`` discrete steps in [0, 1]-ish space. This
    is a classic (and defeatable) preprocessing defense; it may blunt small
    L-infinity perturbations while degrading clean accuracy.
    """

    name = "input_preprocessing"
    description = "Feature-space quantization to blunt small perturbations."

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        levels = int(self.params.get("levels", 16))
        transformed = _quantize(input_data, levels) if input_data is not None else None
        return DefenseResult(
            name=self.name,
            blocked=False,
            transformed_input=transformed,
            notes=f"Quantized input to {levels} levels.",
            detail={"levels": levels},
        )


class ConfidenceThresholdDefense(Defense):
    """Reject low-confidence predictions as potential anomalies."""

    name = "confidence_threshold"
    description = "Block/flag predictions whose confidence is below a threshold."

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        threshold = float(self.params.get("threshold", 0.5))
        conf = float((prediction or {}).get("confidence", 1.0))
        blocked = conf < threshold
        return DefenseResult(
            name=self.name,
            blocked=blocked,
            transformed_input=input_data,
            notes=f"confidence={conf:.3f} threshold={threshold:.3f}",
            detail={"threshold": threshold, "confidence": conf},
        )


class RateLimitDefense(Defense):
    """Simple per-process query rate limiter to raise black-box attack cost."""

    name = "rate_limit"
    description = "Limit queries per window to increase attack cost."

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._window_start = time.monotonic()
        self._count = 0

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        max_queries = int(self.params.get("max_queries", 100))
        window_s = float(self.params.get("window_seconds", 60))
        now = time.monotonic()
        if now - self._window_start > window_s:
            self._window_start = now
            self._count = 0
        self._count += 1
        blocked = self._count > max_queries
        return DefenseResult(
            name=self.name,
            blocked=blocked,
            transformed_input=input_data,
            notes=f"{self._count}/{max_queries} queries this window",
            detail={"max_queries": max_queries, "window_seconds": window_s, "count": self._count},
        )


class InputValidationDefense(Defense):
    """Validate inputs are within an expected numeric range."""

    name = "input_validation"
    description = "Reject inputs whose values fall outside an expected range."

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        low = float(self.params.get("min", -5.0))
        high = float(self.params.get("max", 5.0))
        values = _flatten(input_data) if input_data is not None else []
        out_of_range = [v for v in values if v < low or v > high]
        blocked = len(out_of_range) > 0
        return DefenseResult(
            name=self.name,
            blocked=blocked,
            transformed_input=input_data,
            notes=f"{len(out_of_range)} value(s) outside [{low}, {high}]",
            detail={"min": low, "max": high, "out_of_range": len(out_of_range)},
        )


class MonitoringHookDefense(Defense):
    """Non-blocking observability hook: records the decision for later review."""

    name = "monitoring"
    description = "Emit a monitoring event; never blocks (detective control)."

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        return DefenseResult(
            name=self.name,
            blocked=False,
            transformed_input=input_data,
            notes="Recorded prediction for monitoring.",
            detail={"prediction": prediction or {}},
        )


class AdversarialTrainingDefense(Defense):
    """Declarative marker that the model artifact was adversarially trained.

    Orion does not train models for you; this defense records that a hardened
    model artifact (e.g. a different weights file) should be used for retesting.
    The actual robustness gain is proven by replaying the attack, not by this
    marker.
    """

    name = "adversarial_training"
    description = "Use an adversarially-trained model artifact for retest."

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:
        return DefenseResult(
            name=self.name,
            blocked=False,
            transformed_input=input_data,
            notes="Retest must load the hardened (adversarially-trained) artifact.",
            detail={"hardened_weights": self.params.get("hardened_weights")},
        )


def _flatten(values) -> list:
    flatten_attr = getattr(values, "flatten", None)
    if callable(flatten_attr):
        try:
            return [float(v) for v in flatten_attr().tolist()]
        except Exception:  # pragma: no cover
            pass
    out: list = []
    if isinstance(values, (int, float)):
        return [float(values)]
    try:
        for v in values:
            if isinstance(v, (list, tuple)) or hasattr(v, "__iter__"):
                out.extend(_flatten(v))
            else:
                out.append(float(v))
    except TypeError:  # pragma: no cover
        return []
    return out


def _quantize(values, levels: int):
    if levels < 2:
        levels = 2

    def q(x: float) -> float:
        return round(x * (levels - 1)) / (levels - 1)

    def walk(v):
        if isinstance(v, (list, tuple)):
            return [walk(x) for x in v]
        if hasattr(v, "tolist"):
            return walk(v.tolist())
        try:
            return q(float(v))
        except (TypeError, ValueError):
            return v

    return walk(values)


# Register built-ins.
for _factory in (
    InputPreprocessingDefense,
    ConfidenceThresholdDefense,
    RateLimitDefense,
    InputValidationDefense,
    MonitoringHookDefense,
    AdversarialTrainingDefense,
):
    register_defense(_factory.name, _factory)
