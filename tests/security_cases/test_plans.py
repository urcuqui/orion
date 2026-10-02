"""Plan approval / handoff / explicit-execution tests (spec §30).

    Analysis proposes. Humans approve. Attack prepares. Humans execute.
    Approve Plan != Run Attack.
"""
import pytest

from orion import plans as PL
from orion.know_yourself import analyze as ky_analyze
from orion.target_analysis import build_assessment_from_summary


def _ky_plan():
    r = ky_analyze(descriptor={"model_name": "resnet50", "task": "image_classification",
                               "input_type": "image", "access": "white_box", "num_classes": 2,
                               "framework": "pytorch"}, persist=False)
    return PL.build_plan_from_know_yourself(r), r


def _kyt_plan():
    a = build_assessment_from_summary({
        "target": "model-api", "endpoints": ["/v1/predict", "/v1/infer"],
        "report_markdown": "pytorch model inference classifier"})
    return PL.build_plan_from_target_analysis(a), a


def test_approved_plan_creates_handoff():
    plan, _ = _ky_plan()
    PL.approve_plan(plan)
    h = plan.handoff()
    assert plan.approved_by_human is True
    assert h["status"] == "READY_FOR_ATTACK_WORKSPACE"
    assert h["approved_experiments"]


def test_plan_approval_does_not_execute_experiment():
    plan, _ = _ky_plan()
    PL.approve_plan(plan)
    # No experiment has a run trace after approval.
    assert all(p.run_trace_id is None for p in plan.proposals)
    assert all(p.status != PL.RUNNING for p in plan.proposals)


def test_handoff_preserves_target():
    plan, _ = _ky_plan()
    h = PL.approve_plan(plan).handoff()
    assert h["target"]["access"] == "white_box"
    assert h["target"]["system_type"] == "traditional_ml"


def test_handoff_preserves_threat_model():
    plan, _ = _kyt_plan()
    h = PL.approve_plan(plan).handoff()
    assert h["threat_model"]  # non-empty threat model carried from analysis


def test_handoff_preserves_evidence():
    plan, _ = _kyt_plan()
    h = PL.approve_plan(plan).handoff()
    assert isinstance(h["evidence_ids"], list) and len(h["evidence_ids"]) >= 1


def test_attack_workspace_loads_plan(tmp_path):
    plan, _ = _ky_plan()
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    loaded = PL.load_plan(plan.plan_id, base_dir=str(tmp_path))
    assert loaded is not None
    assert loaded.plan_id == plan.plan_id
    assert loaded.approved_by_human is True


def test_multiple_experiments_create_queue():
    plan, _ = _ky_plan()
    assert len(plan.approved_experiments()) >= 2  # FGSM/PGD/C&W/DeepFool


def test_conditional_experiment_not_executable(tmp_path):
    plan, _ = _ky_plan()
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    cond = plan.conditional_experiments()
    assert cond  # e.g. model extraction / membership inference
    with pytest.raises(PL.ExperimentNotRunnable):
        PL.run_experiment(plan, cond[0].experiment_id, base_dir=str(tmp_path))


def test_non_applicable_experiment_excluded():
    plan, _ = _ky_plan()
    excluded_names = [p.name.lower() for p in plan.excluded_experiments()]
    assert any("prompt injection" in n for n in excluded_names)
    # Excluded experiments are not in the approved (runnable) bucket.
    assert all(p.status == PL.EXCLUDED for p in plan.excluded_experiments())


def test_sensitive_experiment_requires_execution_confirmation():
    plan, _ = _kyt_plan()   # remote target -> sensitive
    # Generative/remote attacks are flagged sensitive for a 2nd confirmation.
    assert any(p.sensitive for p in plan.proposals)
    # A white-box local evasion experiment is not sensitive.
    ky, _ = _ky_plan()
    pgd = next(p for p in ky.proposals if "PGD" in p.name)
    assert pgd.sensitive is False


def test_local_experiment_can_be_run_explicitly(tmp_path):
    plan, _ = _kyt_plan()   # pgd_evasion scenario, synthetic (no torch needed)
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    ready = next(p for p in plan.approved_experiments() if p.scenario)
    result = PL.run_experiment(plan, ready.experiment_id, base_dir=str(tmp_path))
    assert result["trace_id"]
    reloaded = PL.load_plan(plan.plan_id, base_dir=str(tmp_path))
    prop = reloaded.get(ready.experiment_id)
    assert prop.status == PL.COMPLETED and prop.run_trace_id == result["trace_id"]


def test_measure_receives_attack_result(tmp_path):
    from orion.evidence import EvidenceStore
    plan, _ = _kyt_plan()
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    ready = next(p for p in plan.approved_experiments() if p.scenario)
    result = PL.run_experiment(plan, ready.experiment_id, base_dir=str(tmp_path))
    rec = EvidenceStore(str(tmp_path)).load(result["trace_id"])
    assert rec.metrics  # measurement available for the Measure phase
    assert "attack_success_rate" in rec.metrics


def test_defend_and_retest_carry_provenance(tmp_path):
    from orion.evidence import EvidenceStore
    plan, _ = _kyt_plan()
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    ready = next(p for p in plan.approved_experiments() if p.scenario)
    result = PL.run_experiment(plan, ready.experiment_id, base_dir=str(tmp_path))
    rec = EvidenceStore(str(tmp_path)).load(result["trace_id"])
    # Provenance links run -> plan -> analysis for Defend/Retest traceability.
    assert rec.provenance.get("plan_id") == plan.plan_id
    assert rec.provenance.get("experiment_id") == ready.experiment_id
