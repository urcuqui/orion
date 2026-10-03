"""Three context pillars: Recon ownership, Environment Profile, Analysis Context.

    Know Yourself + Know Your Target + Know the Environment -> Analysis Context.
    Recon is a data source that populates the Environment Profile; it is not the
    environment model, and Know Your Target does not duplicate it.
"""
from orion.context import (
    build_environment_profile, environment_to_summary, build_analysis_context,
    build_target_profile, EnvironmentProfile, COMPLETE, NOT_AVAILABLE,
)
from orion.target_analysis import build_assessment_from_summary


RECON_SUMMARY = {
    "target": "app.local",
    "endpoints": ["/mcp_tools", "/predict", "/v1/models", "/login"],
    "report_markdown": "LangGraph agent, MCP server, ChromaDB vector store, "
                       "retrieval-augmented RAG, DeepSeek model, TensorFlow Serving",
    "findings": [{"id": "f1", "title": "open upload endpoint", "severity": "medium"}],
    "screenshots": ["shot1.png"],
}
WEB_SUMMARY = {
    "target": "library.local",
    "endpoints": ["/wp-login.php", "/index.php"],
    "report_markdown": "WordPress site, nginx, php",
    "findings": [], "screenshots": [],
}


def _env(summary=RECON_SUMMARY):
    return build_environment_profile(summary, target_reference=summary["target"], recon_run_id="RUN-1")


# --------------------------- §46 Recon ownership ---------------------------- #
def test_recon_belongs_to_environment():
    # Recon output is normalized into an Environment Profile (the environment model).
    env = _env()
    assert env.environment_profile_id.startswith("ORN-ENV-")
    assert env.recon_run_id == "RUN-1"
    assert env.endpoints  # recon observations populate the profile


def test_target_analysis_does_not_duplicate_recon():
    # The target-analysis module must not own a reconnaissance engine.
    import orion.target_analysis as ta
    assert not hasattr(ta, "start_run")
    assert not hasattr(ta, "RUNS")


def test_environment_profile_created_from_recon():
    env = _env()
    assert env.counts()["ai_dependencies"] >= 1
    assert env.counts()["endpoints"] == 4


def test_target_analysis_consumes_environment_profile():
    # An env profile feeds the analyzer and can change applicability.
    env = _env()
    a = build_assessment_from_summary(environment_to_summary(env))
    assert a["ai_surface"]["status"] == "CONFIRMED"


# --------------------------- §47 Environment Profile ------------------------ #
def test_environment_assets_preserved():
    env = _env()
    assert any(x["kind"] == "endpoint" for x in env.assets)
    assert any(x["kind"] == "finding" for x in env.assets)


def test_environment_services_preserved():
    env = _env()
    assert "/login" in env.services
    assert any("/v1/models" == a or a == "/mcp_tools" for a in env.apis)


def test_environment_ai_dependencies_preserved():
    env = _env()
    types = {d["type"] for d in env.ai_dependencies}
    assert "MODEL_PROVIDER" in types or "MODEL_SERVER" in types
    assert "VECTOR_STORE" in types
    assert "AGENT_FRAMEWORK" in types


def test_environment_mcp_servers_preserved():
    env = _env()
    assert env.mcp_servers and env.mcp_servers[0]["type"] == "MCP_SERVER"


def test_environment_trust_relationships_preserved():
    env = _env()
    rels = {(r["source"], r["destination"]) for r in env.trust_relationships}
    assert any(dst == "MCP server" for _, dst in rels)


def test_environment_evidence_preserved():
    env = _env()
    assert env.evidence_ids
    assert env.screenshots == ["shot1.png"]


# --------------------------- §48 Analysis Context -------------------------- #
def test_context_self_only():
    c = build_analysis_context(self_id="ORN-SELF-1")
    assert c.coverage() == {"self": COMPLETE, "target": NOT_AVAILABLE, "environment": NOT_AVAILABLE}


def test_context_target_only():
    c = build_analysis_context(target_id="ORN-TARGET-1")
    assert c.coverage()["target"] == COMPLETE and c.coverage()["self"] == NOT_AVAILABLE


def test_context_environment_only():
    c = build_analysis_context(environment_id="ORN-ENV-1")
    assert c.coverage()["environment"] == COMPLETE


def test_context_self_target():
    c = build_analysis_context(self_id="S", target_id="T")
    assert c.coverage()["self"] == COMPLETE and c.coverage()["target"] == COMPLETE
    assert c.coverage()["environment"] == NOT_AVAILABLE


def test_context_target_environment():
    c = build_analysis_context(target_id="T", environment_id="E")
    assert c.status() == "PARTIAL CONTEXT"


def test_context_self_environment():
    c = build_analysis_context(self_id="S", environment_id="E")
    assert c.coverage()["self"] == COMPLETE and c.coverage()["environment"] == COMPLETE


def test_context_full():
    c = build_analysis_context(self_id="S", target_id="T", environment_id="E")
    assert c.status() == "READY FOR ANALYSIS"


def test_missing_context_not_invented():
    c = build_analysis_context()
    assert all(v == NOT_AVAILABLE for v in c.coverage().values())
    assert c.status() == "NO CONTEXT"


# --------------------------- §50 Attack applicability ---------------------- #
def test_environment_can_change_attack_applicability():
    ai_env = build_assessment_from_summary(environment_to_summary(_env(RECON_SUMMARY)))
    web_env = build_assessment_from_summary(environment_to_summary(_env(WEB_SUMMARY)))
    assert ai_env["ai_surface"]["status"] == "CONFIRMED"
    assert web_env["ai_surface"]["status"] == "NOT_OBSERVED"


def _applicable(a, key):
    e = next((x for x in a["suggested_experiments"] if x["key"] == key), None)
    return e and e["applicability"] == "APPLICABLE"


def test_mcp_dependency_enables_mcp_scenario():
    a = build_assessment_from_summary(environment_to_summary(_env(RECON_SUMMARY)))
    assert _applicable(a, "tool_poisoning")


def test_external_rag_source_enables_indirect_prompt_injection_scenario():
    a = build_assessment_from_summary(environment_to_summary(_env(RECON_SUMMARY)))
    assert _applicable(a, "rag_poisoning")
    assert _applicable(a, "prompt_injection")


def test_no_ai_dependency_does_not_generate_ai_environment_attack():
    a = build_assessment_from_summary(environment_to_summary(_env(WEB_SUMMARY)))
    assert not _applicable(a, "tool_poisoning")
    assert not _applicable(a, "rag_poisoning")
    assert not _applicable(a, "prompt_injection")


def test_target_profile_references_environment():
    env = _env()
    tp = build_target_profile({"name": "doc workflow", "target_type": "generative_ai",
                               "environment_profile_id": env.environment_profile_id})
    assert tp.environment_profile_id == env.environment_profile_id
