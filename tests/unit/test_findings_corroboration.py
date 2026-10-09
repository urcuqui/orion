"""Unit tests for corroboration + control classification — pure, in-memory.

Security-critical: one or two runs must never confirm a finding, mixed targets
must not corroborate, and control effectiveness is derived, never assigned.
"""
import pytest

from orion.evidence import ExperimentRecord
from orion.findings import model as FM


def _rec(trace_id, succeeded=True, attack="ORN-ATTACK-PI-002", target="agent"):
    return ExperimentRecord(
        trace_id=trace_id,
        status="ATTACK_SUCCESS" if succeeded else "ATTACK_BLOCKED",
        target={"model_name": target},
        parameters={"attack_id": attack,
                    "success_evaluation": {"result": succeeded,
                                           "criteria": [{"criterion": "unauthorized_tool_action",
                                                         "result": succeeded}]}},
        metrics={"attack_success_rate": {"value": 1.0 if succeeded else 0.0}},
    )


def _finding(refs):
    # Built from a successful run → OBSERVED (corroboration may promote to CONFIRMED).
    f = FM.Finding(attack_id="ORN-ATTACK-PI-002", affected_target="agent", status=FM.OBSERVED)
    f.evidence_refs = list(refs)
    return f


# ------------------------------ corroboration ------------------------------- #
def test_single_run_is_not_eligible():
    recs = [_rec("R1")]
    meta = FM.evaluate_corroboration(_finding(["R1"]), recs)
    assert meta["confirmation_eligible"] is False
    assert meta["trial_count"] == 1


def test_two_runs_are_not_eligible_by_default_policy():
    recs = [_rec("R1"), _rec("R2")]
    meta = FM.evaluate_corroboration(_finding(["R1", "R2"]), recs)
    assert meta["confirmation_eligible"] is False     # below minimum_trials=3


def test_three_independent_runs_are_eligible():
    recs = [_rec("R1"), _rec("R2"), _rec("R3")]
    meta = FM.evaluate_corroboration(_finding(["R1", "R2", "R3"]), recs)
    assert meta["confirmation_eligible"] is True
    assert meta["independent_run_count"] == 3
    assert "unauthorized_tool_action" in meta["criteria_reproduced"]


def test_mixed_target_does_not_corroborate():
    recs = [_rec("R1", target="agent"), _rec("R2", target="other"), _rec("R3", target="agent")]
    meta = FM.evaluate_corroboration(_finding(["R1", "R2", "R3"]), recs)
    assert meta["confirmation_eligible"] is False      # not the same target condition


def test_success_rate_below_threshold_blocks_confirmation():
    recs = [_rec("R1"), _rec("R2", succeeded=False), _rec("R3", succeeded=False)]
    meta = FM.evaluate_corroboration(_finding(["R1", "R2", "R3"]), recs)
    assert meta["success_rate"] < 0.66 and meta["confirmation_eligible"] is False


def test_policy_thresholds_are_honored():
    recs = [_rec("R1"), _rec("R2")]
    lenient = FM.CorroborationPolicy(minimum_trials=2, minimum_success_rate=0.5,
                                     require_independent_runs=False)
    meta = FM.evaluate_corroboration(_finding(["R1", "R2"]), recs, lenient)
    assert meta["confirmation_eligible"] is True


def test_corroborate_only_confirms_when_eligible():
    f = _finding([])
    for i in range(1, 3):                              # two runs: stays OBSERVED
        FM.corroborate(f, _rec(f"R{i}"), evidence_records=[_rec(f"R{j}") for j in range(1, i + 1)])
    assert f.status == FM.OBSERVED
    FM.corroborate(f, _rec("R3"), evidence_records=[_rec("R1"), _rec("R2"), _rec("R3")])
    assert f.status == FM.CONFIRMED and f.confidence == "HIGH"


# -------------------------- control classification -------------------------- #
def test_classify_control_spectrum():
    assert FM.classify_control(1.0, 0.0) == FM.EFFECTIVE
    assert FM.classify_control(1.0, 0.5) == FM.PARTIALLY_EFFECTIVE
    assert FM.classify_control(1.0, 1.0) == FM.INEFFECTIVE
    assert FM.classify_control(0.5, 0.8) == FM.INEFFECTIVE     # got worse → not effective


def test_classify_control_non_numeric_is_ineffective():
    assert FM.classify_control(None, None) == FM.INEFFECTIVE
    assert FM.classify_control("x", "y") == FM.INEFFECTIVE
