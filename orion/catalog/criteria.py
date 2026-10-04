"""Executable attack success criteria (P1.1).

An attack's success criteria are not just descriptive metadata: they are an
executable rule over the *observed* signals of a run. This answers, without
reading source code: **why did Orion classify this attack as successful?**

Only two operators are supported — ALL and ANY — by design.
"""
from __future__ import annotations

from typing import Any, Dict


def evaluate_attack_success(attack, observations: Dict[str, Any],
                            run_id: str = "") -> Dict[str, Any]:
    """Evaluate ``attack``'s success rule against observed signals.

    Returns the operator, a per-criterion breakdown and the overall result, so
    the Measure/Evidence UI can show exactly which criteria fired.
    """
    rule = attack.success_rule() if hasattr(attack, "success_rule") else {"all": []}
    op = "all" if "all" in rule else "any"
    criteria = rule.get(op, []) or []

    results = []
    for c in criteria:
        observed = bool(observations.get(c))
        results.append({"criterion": c, "expected": True, "observed": observed,
                        "result": observed, "evidence_reference": run_id or None})

    if op == "all":
        success = bool(results) and all(r["result"] for r in results)
    else:  # any
        success = any(r["result"] for r in results)

    return {
        "rule": op.upper(),
        "criteria": results,
        "satisfied": sum(1 for r in results if r["result"]),
        "total": len(results),
        "result": bool(success),
    }
