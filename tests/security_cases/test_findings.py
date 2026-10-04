"""Findings are first-class: evidence-backed, never auto-confirmed, retest-aware.

    Evidence = what happened.
    Finding  = the security meaning of what happened.
"""
import pytest

import app as orion_app
from orion import findings as F
from orion.agentic import run_agentic_experiment
from orion.adversarial.image import infer_num_outputs  # noqa: F401 (import guard)


@pytest.fixture
def client():
    return orion_app.app.test_client()


def _run(tmp_path, attack="ORN-ATTACK-AG-002", controls=None, trials=3):
    return run_agentic_experiment(attack, controls=controls, trials=trials, base_dir=str(tmp_path))


# --------------------------- derivation & status ---------------------------- #
def test_single_run_is_observed_not_confirmed(tmp_path):
    f = F.build_finding_from_record(_run(tmp_path))
    assert f.status == F.OBSERVED        # never auto-CONFIRMED from one run
    assert f.confidence == "MEDIUM"
    assert f.id.startswith("ORN-FND-")


def test_finding_points_back_to_evidence(tmp_path):
    rec = _run(tmp_path)
    f = F.build_finding_from_record(rec)
    assert f.evidence_refs == [rec.trace_id]
    # Boundary crossed comes from the attack trust boundary / observed crossing.
    assert f.boundary_crossed == "Tool Authorization → Privileged Capability"
    assert f.success_condition  # from the attack's success_criteria


def test_two_runs_do_not_confirm(tmp_path):
    # Two evidence refs are NOT enough (P0.1): confirmation needs the policy met.
    r1, r2 = _run(tmp_path), _run(tmp_path)
    f = F.build_finding_from_record(r1)
    F.corroborate(f, r2, evidence_records=[r1, r2])
    assert f.status == F.OBSERVED
    assert f.corroboration["confirmation_eligible"] is False


def test_corroboration_promotes_to_confirmed(tmp_path):
    runs = [_run(tmp_path) for _ in range(3)]
    f = F.build_finding_from_record(runs[0])
    for i, r in enumerate(runs[1:], start=2):
        F.corroborate(f, r, evidence_records=runs[:i])
    assert f.status == F.CONFIRMED and f.confidence == "HIGH"
    assert f.corroboration["trial_count"] == 3
    assert f.corroboration["success_rate"] >= 0.66
    assert "unauthorized_tool_action" in f.corroboration["criteria_reproduced"] \
        or f.corroboration["criteria_reproduced"]


def test_blocked_attack_does_not_assert_a_finding(tmp_path):
    rec = _run(tmp_path, controls=["tool_authorization"])   # blocked
    f = F.build_finding_from_record(rec)
    assert f.status == F.NOT_REPRODUCIBLE


# ---------------------------- control classification ------------------------ #
def test_classify_control_spectrum():
    assert F.classify_control(1.0, 0.0) == F.EFFECTIVE
    assert F.classify_control(1.0, 0.5) == F.PARTIALLY_EFFECTIVE
    assert F.classify_control(1.0, 1.0) == F.INEFFECTIVE


def test_apply_retest_updates_status_honestly(tmp_path):
    f = F.build_finding_from_record(_run(tmp_path))
    F.apply_retest(f, 1.0, 0.0, "tool_authorization", "ORN-RUN-AFTER")
    assert f.status == F.MITIGATED and f.retest_status == F.EFFECTIVE
    assert "ORN-RUN-AFTER" in f.retest_refs

    f2 = F.build_finding_from_record(_run(tmp_path))
    F.apply_retest(f2, 1.0, 1.0, "tool_authorization", "ORN-RUN-AFTER2")
    assert f2.retest_status == F.INEFFECTIVE and f2.status != F.MITIGATED


# -------------------------------- storage ----------------------------------- #
def test_store_roundtrip_and_find_by_evidence(tmp_path):
    rec = _run(tmp_path)
    f = F.build_finding_from_record(rec)
    F.save_finding(f, str(tmp_path))
    assert F.load_finding(f.id, str(tmp_path)).id == f.id
    assert F.FindingStore(str(tmp_path)).find_by_evidence(rec.trace_id).id == f.id
    assert [x.id for x in F.list_findings(str(tmp_path))] == [f.id]


# ------------------------------- pages -------------------------------------- #
def test_findings_pages_render(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    # The page routes read the real 'artifacts' dir; create + clean up there.
    import shutil, os
    rec = _run_artifacts()
    f = F.build_finding_from_record(rec); F.save_finding(f, "artifacts")
    try:
        client = orion_app.app.test_client()
        assert ">Findings<" in client.get("/findings").data.decode()
        body = client.get(f"/findings/{f.id}").data.decode()
        assert "Provenance" in body and f.id in body
    finally:
        from orion.evidence import EvidenceStore
        shutil.rmtree(EvidenceStore("artifacts").trace_dir(rec.trace_id), ignore_errors=True)
        os.remove(f"artifacts/findings/{f.id}.json")


def _run_artifacts():
    return run_agentic_experiment("ORN-ATTACK-AG-002", trials=2, base_dir="artifacts")
