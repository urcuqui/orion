"""A deterministic, dependency-free adversarial backend.

This backend makes the full Understand->...->Retest loop reproducible without a
GPU, torch or ART. It is a *teaching* model, not a real classifier.

Model
-----
Each example has features whose mean sits a distance ``d`` from a decision
boundary at 0.5 (the class margin). The clean prediction is correct. An
L-infinity evasion attack of budget ``epsilon`` shifts every feature toward the
boundary; an example flips when its margin ``d`` is smaller than the attacker's
*effective* budget.

- Attack: reported perturbation is always the full ``epsilon`` (the L-infinity
  budget the attacker actually spent).
- Hardened model: a ``robust_margin`` widens the model's tolerant band, so the
  attacker's effective budget becomes ``epsilon - robust_margin``. The same
  attack therefore flips fewer examples — a robustness gain that is only
  provable by replaying the attack.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

FEATURE_DIM = 16
BOUNDARY = 0.5


def make_dataset(n: int = 20, seed: int = 1337) -> List[Tuple[List[float], int, float]]:
    """Deterministic dataset of (feature_vector, true_label, margin).

    Margins are drawn thin (0.01..0.09) so a small epsilon is meaningful, and
    feature noise is small enough that clean predictions stay correct.
    """
    rng = random.Random(seed)
    data: List[Tuple[List[float], int, float]] = []
    for _ in range(n):
        label = rng.randint(0, 1)
        margin = rng.uniform(0.01, 0.09)
        direction = 1.0 if label == 1 else -1.0
        center = BOUNDARY + direction * margin
        vec = [min(1.0, max(0.0, center + rng.uniform(-0.006, 0.006))) for _ in range(FEATURE_DIM)]
        data.append((vec, label, margin))
    return data


def _mean(vec: List[float]) -> float:
    return sum(vec) / len(vec) if vec else 0.0


def predict(vec: List[float]) -> Tuple[int, float]:
    """Clean prediction: (label, confidence)."""
    m = _mean(vec)
    label = 1 if m >= BOUNDARY else 0
    confidence = min(0.999, 0.5 + abs(m - BOUNDARY) * 5.0)
    return label, round(confidence, 4)


def pgd_like_evasion(vec: List[float], true_label: int, epsilon: float) -> List[float]:
    """Full L-infinity perturbation toward the boundary (reported perturbation)."""
    direction = -1.0 if true_label == 1 else 1.0
    return [min(1.0, max(0.0, x + direction * epsilon)) for x in vec]


def run_batch(epsilon: float, robust_margin: float, n: int, seed: int) -> Dict[str, object]:
    """Run clean + adversarial predictions over the synthetic dataset."""
    data = make_dataset(n=n, seed=seed)
    y_true: List[int] = []
    y_clean: List[int] = []
    y_adv: List[int] = []
    perts: List[float] = []
    conf_clean: List[float] = []
    conf_adv: List[float] = []

    effective_epsilon = max(0.0, epsilon - robust_margin)

    for vec, label, margin in data:
        y_true.append(label)
        c_lbl, c_conf = predict(vec)
        y_clean.append(c_lbl)
        conf_clean.append(c_conf)

        # Report the full epsilon actually spent, regardless of hardening.
        adv_vec = pgd_like_evasion(vec, label, epsilon)
        perts.append(max(abs(a - b) for a, b in zip(vec, adv_vec)) if vec else 0.0)

        # The (possibly hardened) model flips only if the margin < effective budget.
        flipped = margin < effective_epsilon
        if flipped:
            a_lbl = 1 - label
            a_conf = round(min(0.999, 0.5 + (effective_epsilon - margin) * 5.0), 4)
        else:
            a_lbl = label
            # Surviving-but-pushed examples lose confidence.
            residual = max(0.0, margin - effective_epsilon)
            a_conf = round(min(0.999, 0.5 + residual * 5.0), 4)
        y_adv.append(a_lbl)
        conf_adv.append(a_conf)

    return {
        "y_true": y_true,
        "y_pred_clean": y_clean,
        "y_pred_adv": y_adv,
        "perturbations": perts,
        "confidence_clean": conf_clean,
        "confidence_adv": conf_adv,
    }
