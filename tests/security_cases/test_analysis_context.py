"""Analysis Context: profile persistence, provenance, and context-driven applicability."""
from orion.context import (
    save_profile, load_profile, build_self_profile, build_target_profile,
    build_environment_profile, build_analysis_context, build_context_assessment,
    build_context_threat_model, SelfProfile,
)
from orion.know_yourself import analyze as ky_analyze


def _env():
    return build_environment_profile({
        "target": "app.local", "endpoints": ["/mcp_tools", "/predict"],
        "report_markdown": "MCP server, ChromaDB vector store, retrieval, DeepSeek",
        "findings": [], "screenshots": []}, target_reference="app.local")


def _applic(a, key):
    e = next((x for x in a["suggested_experiments"] if x["key"] == key), None)
    return (e["applicability"] if e else None), (e["source_profiles"] if e else [])


# ------------------------------ §43 ownership ------------------------------- #
def test_recon_belongs_to_environment():
    env = _env()
    assert env.environment_profile_id.startswith("ORN-ENV-")


def test_target_analysis_does_not_own_recon():
    import orion.target_analysis as ta
    assert not hasattr(ta, "start_run") and not hasattr(ta, "RUNS")


def test_target_analysis_can_consume_environment_profile():
    from orion.context import environment_to_summary
    from orion.target_analysis import build_assessment_from_summary
    a = build_assessment_from_summary(environment_to_summary(_env()))
    assert a["ai_surface"]["status"] == "CONFIRMED"


# ------------------------------ §44 persistence ----------------------------- #
def test_environment_profile_persisted(tmp_path):
    env = _env()
    save_profile(env.environment_profile_id, env.to_dict(), str(tmp_path))
    back = load_profile(env.environment_profile_id, str(tmp_path))
    assert back and back["environment_profile_id"] == env.environment_profile_id


def test_target_profile_persisted(tmp_path):
    tp = build_target_profile({"name": "svc", "target_type": "generative_ai"})
    save_profile(tp.target_profile_id, tp.to_dict(), str(tmp_path))
    assert load_profile(tp.target_profile_id, str(tmp_path))["name"] == "svc"


def test_analysis_context_persisted(tmp_path):
    ctx = build_analysis_context(self_id="S", target_id="T", environment_id="E")
    save_profile(ctx.analysis_context_id, ctx.to_dict(), str(tmp_path))
    back = load_profile(ctx.analysis_context_id, str(tmp_path))
    assert back["self_profile_id"] == "S" and back["coverage"]["self"] == "COMPLETE"


def test_self_profile_persisted(tmp_path):
    r = ky_analyze(descriptor={"task": "image_classification", "input_type": "image",
                               "access": "white_box", "framework": "pytorch"},
                   base_dir=str(tmp_path))
    assert r["self_profile_id"].startswith("ORN-SELF-")
    sp = load_profile(r["self_profile_id"], str(tmp_path))
    assert sp and sp["system_type"] == "traditional_ml"


def test_self_profile_builder_normalizes_result():
    r = ky_analyze(descriptor={"rag": True, "mcp": True, "tools": [{"name": "t"}]},
                   context="llm agent", persist=False)
    sp = build_self_profile(r)
    assert isinstance(sp, SelfProfile)
    assert sp.system_type == "generative_ai"
    assert sp.capabilities.get("retrieval") is True


# ------------------------------ §46 provenance ------------------------------ #
def test_threat_model_tracks_environment_source():
    tm = build_context_threat_model(None, None, _env().to_dict(), ai_confirmed=True)
    assert tm["assets"] and all(a["source"] == "ENVIRONMENT" for a in tm["assets"])


def test_threat_model_tracks_self_source():
    self_p = {"system_type": "generative_ai", "capabilities": {"retrieval": True}}
    tm = build_context_threat_model(self_p, None, None, ai_confirmed=True)
    assert any(b["source"] == "SELF" for b in tm["trust_boundaries"])


def test_threat_model_tracks_target_source():
    tp = {"target_type": "generative_ai", "security_objective": "validate tool auth"}
    tm = build_context_threat_model(None, tp, None, ai_confirmed=True)
    assert any(o["source"] == "TARGET" for o in tm["objectives"])


def test_unknown_adversary_context_remains_unknown():
    tm = build_context_threat_model(None, None, _env().to_dict(), ai_confirmed=True)
    adv = tm["adversary"]
    assert adv["goal"] == "UNDEFINED" and adv["knowledge"] == "UNKNOWN" and adv["budget"] == "UNKNOWN"
    assert adv["credential_access"] == "NOT_OBSERVED"


# ------------------------------ §48 applicability --------------------------- #
def test_self_profile_affects_attack_applicability():
    a = build_context_assessment(
        {"system_type": "generative_ai", "capabilities": {"llm_interface": True, "retrieval": True}}, None, None)
    status, src = _applic(a, "rag_poisoning")
    assert status == "APPLICABLE" and "SELF" in src


def test_environment_profile_affects_attack_applicability():
    a = build_context_assessment(None, None, _env().to_dict())
    status, src = _applic(a, "tool_poisoning")
    assert status == "APPLICABLE" and "ENVIRONMENT" in src


def test_target_profile_affects_attack_applicability():
    a = build_context_assessment(None, {"target_type": "generative_ai"}, None)
    status, src = _applic(a, "prompt_injection")
    assert status == "APPLICABLE" and "TARGET" in src


def test_mcp_environment_enables_mcp_test():
    a = build_context_assessment(None, None, _env().to_dict())
    assert _applic(a, "tool_poisoning")[0] == "APPLICABLE"


def test_external_rag_source_enables_indirect_prompt_injection_test():
    # Spec example: SELF (LLM + RAG) + ENVIRONMENT (untrusted retrieval source).
    self_p = {"system_type": "generative_ai", "capabilities": {"llm_interface": True, "retrieval": True}}
    a = build_context_assessment(self_p, None, _env().to_dict())
    assert _applic(a, "rag_poisoning")[0] == "APPLICABLE"
    status, src = _applic(a, "prompt_injection")
    assert status == "APPLICABLE"
    # Retrieval source is corroborated by both SELF and ENVIRONMENT.
    assert "SELF" in src


def test_no_ai_context_does_not_generate_ai_specific_experiment():
    a = build_context_assessment({"system_type": "unknown", "capabilities": {}}, None, None)
    assert _applic(a, "prompt_injection")[0] == "NOT_APPLICABLE"
    assert _applic(a, "rag_poisoning")[0] == "NOT_APPLICABLE"
    assert _applic(a, "tool_poisoning")[0] == "NOT_APPLICABLE"


def test_plan_from_context_carries_ids():
    from orion import plans as PL
    a = build_context_assessment(
        {"system_type": "generative_ai", "capabilities": {"llm_interface": True, "retrieval": True}}, None, None)
    a["analysis_context_id"] = "ORN-CTX-TEST1"
    a["self_profile_id"] = "ORN-SELF-TEST1"
    plan = PL.build_plan_from_context(a)
    assert plan.source_type == "analysis_context"
    assert plan.analysis_context_id == "ORN-CTX-TEST1"
    assert plan.self_profile_id == "ORN-SELF-TEST1"
    assert plan.approved_experiments()  # applicable experiments became READY
    PL.approve_plan(plan)
    h = plan.handoff()
    assert h["analysis_context_id"] == "ORN-CTX-TEST1"
    assert h["self_profile_id"] == "ORN-SELF-TEST1"
