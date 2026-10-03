"""Attack selection by access level + base tuning.

Owning the weights (white-box) unlocks gradient attacks; a query-only live
service (black-box) unlocks decision-based ones. The access level is *derived*
from the plan's origin and offered for confirm/override — never silent — and
every attack ships base tuning so a run is one click away.
"""
import pytest

import app as orion_app
from orion import plans as PL
from orion import experiments as EXP
from orion.adversarial import catalog as CAT
from orion.target_analysis import build_assessment_from_summary


@pytest.fixture
def client():
    return orion_app.app.test_client()


def _plan(summary, tmp_path):
    a = build_assessment_from_summary(summary)
    plan = PL.build_plan_from_target_analysis(a)
    PL.approve_plan(plan)
    PL.save_plan(plan, base_dir=str(tmp_path))
    return plan


# -------------------------------- catalog ----------------------------------- #
def test_whitebox_offers_gradient_attacks():
    ids = {a["id"] for a in CAT.attacks_for(CAT.WHITE_BOX)}
    assert {"CarliniL2", "PGD", "FGSM"} <= ids


def test_blackbox_offers_decision_based_attack():
    ids = {a["id"] for a in CAT.attacks_for(CAT.BLACK_BOX)}
    assert "HopSkipJump" in ids


def test_every_attack_has_base_tuning():
    for level in (CAT.WHITE_BOX, CAT.BLACK_BOX):
        for a in CAT.attacks_for(level):
            assert a["base_params"], f"{a['id']} missing base tuning"


def test_non_image_modality_has_no_catalogued_attacks():
    assert CAT.attacks_for(CAT.WHITE_BOX, modality="tabular") == []


# ---------------------------- access derivation ----------------------------- #
def test_self_profile_derives_white_box():
    d = CAT.derive_access_level(self_profile_id="ORN-SELF-1")
    assert d["access_level"] == CAT.WHITE_BOX and d["source"] == "self_profile"


def test_live_target_derives_black_box():
    d = CAT.derive_access_level(target_url="http://svc")
    assert d["access_level"] == CAT.BLACK_BOX and d["source"] == "target_url"


def test_derivation_always_explains_itself():
    assert CAT.derive_access_level()["reason"]  # never a silent default


# ----------------------- workspace attack options --------------------------- #
def test_attack_options_derives_black_box_for_live_service(tmp_path):
    plan = _plan({"target": "http://svc", "endpoints": ["POST /predict"],
                  "report_markdown": "pytorch model inference classifier",
                  "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]},
                 tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    opt = EXP.attack_options(ws, base_dir=str(tmp_path))
    assert opt["derived"]["access_level"] == CAT.BLACK_BOX
    assert opt["live_url"] == "http://svc"
    assert "CarliniL2" in {a["id"] for a in opt["catalog"][CAT.WHITE_BOX]}


def test_attack_options_derives_white_box_when_self_owned(tmp_path):
    plan = _plan({"target": "local-model", "endpoints": ["/v1/predict"],
                  "report_markdown": "pytorch inference classifier"}, tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    ws.self_profile_id = "ORN-SELF-ABC"           # we own the model
    opt = EXP.attack_options(ws, base_dir=str(tmp_path))
    assert opt["derived"]["access_level"] == CAT.WHITE_BOX


# --------------------- white-box attack: fail-closed ------------------------ #
def _approved_plan(tmp_path):
    return _plan({"target": "t", "endpoints": ["/v1/predict"],
                  "report_markdown": "pytorch inference classifier"}, tmp_path)


def test_whitebox_requires_approved_plan(tmp_path):
    a = build_assessment_from_summary({"target": "t", "endpoints": ["/v1/predict"],
                                       "report_markdown": "pytorch inference classifier"})
    plan = PL.build_plan_from_target_analysis(a)          # NOT approved
    PL.save_plan(plan, base_dir=str(tmp_path))
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_whitebox_attack(ws, {"weights_path": "w", "image": "i"}, base_dir=str(tmp_path))


def test_whitebox_requires_weights_and_image(tmp_path):
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    with pytest.raises(EXP.StageError):
        EXP.run_whitebox_attack(ws, {"image": "i"}, base_dir=str(tmp_path))     # no weights
    with pytest.raises(EXP.StageError):
        EXP.run_whitebox_attack(ws, {"weights_path": "w"}, base_dir=str(tmp_path))  # no image


def test_whitebox_attack_advances_lifecycle(tmp_path, monkeypatch):
    # The heavy torch+ART runner is monkeypatched to verify lifecycle wiring and
    # that num_outputs is inferred from the weights, not demanded from the user.
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    import orion.adversarial as adv
    seen = {}

    def _stub(**kw):
        from orion.evidence import EvidenceStore, ExperimentRecord
        seen.update(kw)
        rec = ExperimentRecord(scenario_name="adversarial-image",
                               attack_technique=f"{kw.get('attack')} (white-box)",
                               status="ATTACK_SUCCESS",
                               artifacts={"original": "original.png", "adversarial": "adversarial.png",
                                          "difference": "difference.png"},
                               metrics={"attack_success": {"value": True},
                                        "perturbation_linf": {"value": 0.03}},
                               provenance=kw.get("provenance") or {})
        EvidenceStore(kw.get("base_dir", "artifacts")).save(rec)
        return rec

    monkeypatch.setattr(adv, "run_adversarial_experiment", _stub, raising=True)
    monkeypatch.setattr(adv, "infer_num_outputs", lambda *a, **k: 7, raising=True)

    r = EXP.run_whitebox_attack(ws, {"weights_path": "weights/x.pth", "image": "img.png",
                                     "attack": "PGD", "params": {"eps": 0.03}}, base_dir=str(tmp_path))
    assert r["mode"] == "whitebox" and r["attack"] == "PGD"
    assert r["num_outputs"] == 7                        # inferred, not supplied
    assert seen["num_outputs"] == 7
    assert ws.stages["attack"] == "COMPLETE" and ws.stages["measure"] == "READY"
    from orion.evidence import EvidenceStore
    rec = EvidenceStore(str(tmp_path)).load(r["trace_id"])
    assert rec.provenance.get("plan_id") == ws.plan_id
    assert rec.provenance.get("source_type") == "whitebox_attack"


# ------------------------------- route wiring ------------------------------- #
def test_attack_options_route(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    plan = _plan({"target": "http://svc", "endpoints": ["POST /predict"],
                  "report_markdown": "pytorch model inference classifier",
                  "findings": [{"id": "f", "title": "ml_inference_response", "severity": "info"}]},
                 tmp_path)
    ws = EXP.create_from_plan(plan, base_dir=str(tmp_path))
    body = orion_app.app.test_client().get(f"/api/experiment/{ws.experiment_workspace_id}/attack-options").get_json()
    assert body["derived"]["access_level"] in (CAT.WHITE_BOX, CAT.BLACK_BOX)
    assert "white_box" in body["catalog"] and "black_box" in body["catalog"]


def test_attack_whitebox_route_wired(tmp_path, monkeypatch):
    import orion.integrations.flask_blueprint as bp
    monkeypatch.setattr(bp, "ARTIFACT_DIR", str(tmp_path), raising=True)
    import orion.experiments as E
    called = {}

    def _stub(ws, params, base_dir):
        called["params"] = params
        return {"trace_id": "T", "status": "ATTACK_SUCCESS", "mode": "whitebox", "attack": "FGSM"}

    monkeypatch.setattr(E, "run_whitebox_attack", _stub, raising=True)
    ws = EXP.create_from_plan(_approved_plan(tmp_path), base_dir=str(tmp_path))
    r = orion_app.app.test_client().post(
        f"/api/experiment/{ws.experiment_workspace_id}/attack-whitebox",
        json={"weights_path": "w", "image": "i", "attack": "FGSM", "params": {"eps": 0.03}})
    assert r.status_code == 200
    assert called["params"]["attack"] == "FGSM"


# --------------------------- real ART construction -------------------------- #
def test_build_attack_maps_ids_to_real_art_attacks():
    torch = pytest.importorskip("torch")
    pytest.importorskip("art")
    from orion.adversarial.image import _build_attack
    from art.attacks.evasion import CarliniL2Method, FastGradientMethod, ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 2))
    clf = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(),
                            nb_classes=2, input_shape=(3, 224, 224))
    assert isinstance(_build_attack("FGSM", clf, {"eps": 0.03}), FastGradientMethod)
    assert isinstance(_build_attack("PGD", clf, {"eps": 0.03}), ProjectedGradientDescent)
    assert isinstance(_build_attack("CarliniL2", clf, {}), CarliniL2Method)
    # The base-tuning parameter actually reaches the attack.
    assert _build_attack("FGSM", clf, {"eps": 0.07}).eps == pytest.approx(0.07)
