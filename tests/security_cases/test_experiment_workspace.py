"""Experiment workspace: lifecycle, stage transitions, backward compat, evidence.

    Plan before attack. Attack before measure. Measure before defense.
    Defense before retest. Retest verifies improvement. Evidence preserves it all.
"""
import pytest

import app as orion_app
from orion import plans as PL
from orion import experiments as EXP
from orion.target_analysis import build_assessment_from_summary


@pytest.fixture
def client():
    return orion_app.app.test_client()


def _approved_plan(tmp_path):
    a = build_assessment_from_summary({
        "target": "svc", "endpoints": ["/v1/predict"],
        "report_markdown": "pytorch model inference classifier"})
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


# ------------------------------ §45 navigation ------------------------------ #
def test_dashboard_shows_experiment_not_attack_measure_defend(client):
    body = client.get("/").data.decode()
    assert "EXPERIMENTATION" in body
    assert ">Experiment<" in body
    # Attack / Measure / Defend are no longer their own top-level menu items.
    assert ">Attack<" not in body and ">Measure<" not in body and ">Defend<" not in body


def test_evidence_remains_top_level(client):
    assert ">Evidence<" in client.get("/").data.decode()


def test_context_pillars_remain_top_level(client):
    body = client.get("/").data.decode()
    for name in ("Know Yourself", "Know Your Target", "Know The Environment"):
        assert name in body


# ------------------------- §46 workspace from plan -------------------------- #
def test_experiment_workspace_created_from_plan(tmp_path):
    plan = _approved_plan(tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    assert ws.experiment_workspace_id.startswith("ORN-EXP-")
    assert ws.plan_id == plan.plan_id
    assert ws.stages["plan"] == "APPROVED"
    assert ws.active_experiment_id is not None


def test_experiment_preserves_analysis_context(tmp_path):
    a = build_assessment_from_summary({"target": "t", "endpoints": ["/v1/predict"],
                                       "report_markdown": "pytorch inference classifier"})
    a["environment_profile_id"] = "ORN-ENV-XYZ"
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    assert ws.environment_profile_id == "ORN-ENV-XYZ"


def test_experiment_preserves_threat_model(tmp_path):
    plan = _approved_plan(tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    assert ws.threat_model_id == plan.threat_model_id


def test_experiment_stage_state_persisted(tmp_path):
    plan = _approved_plan(tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    back = EXP.load_workspace(ws.experiment_workspace_id, base_dir=str(tmp_path))
    assert back is not None and back.stages == ws.stages


# --------------------------- §47 stage transitions -------------------------- #
def test_attack_requires_approved_plan(tmp_path):
    a = build_assessment_from_summary({"target": "t", "endpoints": ["/v1/predict"],
                                       "report_markdown": "pytorch inference classifier"})
    plan = PL.build_plan_from_target_analysis(a)        # NOT approved
    PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_attack(ws, base_dir=str(tmp_path))


def test_measure_follows_attack(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.measure(ws, base_dir=str(tmp_path))          # before attack
    EXP.run_attack(ws, base_dir=str(tmp_path))
    m = EXP.measure(ws, base_dir=str(tmp_path))
    assert ws.stages["measure"] == "COMPLETE" and m["metrics"]


def test_defend_follows_measure(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_attack(ws, base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.apply_defense(ws, base_dir=str(tmp_path))     # before measure
    EXP.measure(ws, base_dir=str(tmp_path))
    d = EXP.apply_defense(ws, base_dir=str(tmp_path))
    assert ws.stages["defend"] == "APPLIED" and d["defense_id"]


def test_retest_follows_defense(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_attack(ws, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.retest(ws, base_dir=str(tmp_path))            # before defense
    EXP.apply_defense(ws, base_dir=str(tmp_path))
    r = EXP.retest(ws, base_dir=str(tmp_path))
    assert ws.stages["retest"] == "COMPLETE" and r["retest_run_id"]


def test_retest_reuses_original_attack_configuration(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_attack(ws, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    EXP.apply_defense(ws, base_dir=str(tmp_path))
    r = EXP.retest(ws, base_dir=str(tmp_path))
    from orion.evidence import EvidenceStore
    atk = EvidenceStore(str(tmp_path)).load(ws.attack_run_id)
    rt = EvidenceStore(str(tmp_path)).load(r["retest_run_id"])
    assert atk.attack_technique == rt.attack_technique       # same attack
    assert atk.parameters == rt.parameters                   # same parameters


# ---------------------------- §48 backward compat --------------------------- #
def test_attack_route_opens_experiment(client):
    body = client.get("/attack").data.decode()
    assert "ORION // EXPERIMENT" in body


def test_measure_route_opens_experiment_measure_stage(client):
    body = client.get("/measure/ORN-RUN-NONE").data.decode()
    assert 'data-compat-stage="measure"' in body


def test_defend_route_opens_experiment_defend_stage(client):
    body = client.get("/defend/ORN-RUN-NONE").data.decode()
    assert 'data-compat-stage="defend"' in body


# -------------------------------- §49 evidence ------------------------------ #
def test_comparison_uses_original_and_retest_measurements(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_attack(ws, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    EXP.apply_defense(ws, base_dir=str(tmp_path))
    r = EXP.retest(ws, base_dir=str(tmp_path))
    cmp = r["comparison"]["metrics"]
    assert "robust_accuracy" in cmp
    assert cmp["robust_accuracy"]["before"] == pytest.approx(
        __import__("orion.evidence", fromlist=["EvidenceStore"]).EvidenceStore(str(tmp_path))
        .load(ws.attack_run_id).metrics["robust_accuracy"]["value"])


def test_retest_evidence_tracks_full_chain(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    EXP.run_attack(ws, base_dir=str(tmp_path))
    EXP.measure(ws, base_dir=str(tmp_path))
    EXP.apply_defense(ws, base_dir=str(tmp_path))
    r = EXP.retest(ws, base_dir=str(tmp_path))
    from orion.evidence import EvidenceStore
    rec = EvidenceStore(str(tmp_path)).load(r["retest_run_id"])
    assert rec.provenance.get("plan_id") == ws.plan_id
    assert rec.provenance.get("defense_id") == ws.defense_id
    assert rec.provenance.get("retest_of") == ws.attack_run_id


def test_no_runnable_attack_guides_reanalysis(tmp_path):
    # A plan whose AI-specific experiments were all EXCLUDED (AI surface only
    # POSSIBLE) yields no active experiment, and the workspace guides re-analysis
    # instead of offering a dead RUN ATTACK.
    from orion.target_analysis import build_assessment_from_summary
    a = build_assessment_from_summary({"target": "http://svc", "endpoints": [],
                                       "report_markdown": "face detection"})  # POSSIBLE, not CONFIRMED
    assert a["ai_surface"]["status"] != "CONFIRMED"
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    # The only READY proposal (recon extension) has no scenario -> not runnable.
    assert ws.active_experiment_id is None
    na = EXP.next_action(ws)
    assert "RE-ANALYZE" in na["label"] and "RUN ATTACK" not in na["label"]


def test_confirmed_ai_service_yields_runnable_attack(tmp_path):
    # An active-probe summary that CONFIRMS an ML inference service produces a
    # runnable adversarial experiment and an active workspace.
    from orion.target_analysis import build_assessment_from_summary
    summary = {"target": "http://svc", "endpoints": ["POST /predict"],
               "report_markdown": "pytorch model inference classifier",
               "findings": [{"id": "f1", "title": "ml_inference_response", "severity": "info"}]}
    a = build_assessment_from_summary(summary)
    assert a["ai_surface"]["status"] == "CONFIRMED"
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan); PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    assert ws.active_experiment_id is not None
    assert ws.stages["attack"] == "READY"


def test_blackbox_signature_detects_decision_change():
    from orion.adversarial.blackbox import _signature
    a = _signature('{"label":"real"}', "application/json")
    b = _signature('{"label":"fake"}', "application/json")
    assert a != b                                   # decision change observable
    # Echoed base64 image must not create a spurious difference.
    h1 = _signature('<img src="data:image/png;base64,AAAA"> no faces', "text/html")
    h2 = _signature('<img src="data:image/png;base64,ZZZZ"> no faces', "text/html")
    assert h1 == h2


def test_blackbox_attack_advances_lifecycle(tmp_path, monkeypatch):
    # The live runner is monkeypatched (no network) to verify lifecycle wiring.
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    import orion.adversarial as adv

    def _stub(**kw):
        from orion.evidence import EvidenceStore, ExperimentRecord
        rec = ExperimentRecord(scenario_name="blackbox-evasion", attack_technique="Black-box query evasion",
                               status="ATTACK_SUCCESS",
                               metrics={"attack_success": {"value": True}, "query_count": {"value": 5}},
                               provenance=kw.get("provenance") or {})
        EvidenceStore(kw.get("base_dir", "artifacts")).save(rec)
        return rec

    monkeypatch.setattr(adv, "run_blackbox_evasion", _stub, raising=True)
    r = EXP.run_blackbox_attack(ws, {"url": "http://127.0.0.1:5999"}, base_dir=str(tmp_path))
    assert r["mode"] == "blackbox" and r["trace_id"]
    assert ws.stages["attack"] == "COMPLETE" and ws.stages["measure"] == "READY"
    assert ws.attack_run_id == r["trace_id"]
    from orion.evidence import EvidenceStore
    rec = EvidenceStore(str(tmp_path)).load(r["trace_id"])
    assert rec.provenance.get("plan_id") == ws.plan_id


def test_blackbox_attack_requires_url(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_blackbox_attack(ws, {}, base_dir=str(tmp_path))
