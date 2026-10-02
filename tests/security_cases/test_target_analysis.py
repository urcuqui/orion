"""Evidence-grounded Agent Analysis regression tests (spec §18).

Core principle: evidence -> applicability -> hypothesis -> test -> finding.
Never: generic knowledge -> finding. Orion knows when it does not know.
"""
from orion.target_analysis import (
    build_assessment_from_summary,
    detect_ai_surface,
    build_evidence,
    classify_target_types,
    HYPOTHESIS,
    NOT_APPLICABLE,
)


# --- fixtures: synthetic recon summaries ---
WEB = {
    "target": "university.edu",
    "endpoints": ["/wp-login.php", "/wp-admin/", "/index.php"],
    "report_markdown": "WordPress 6.4 identified. Public forms and login. nginx, php.",
}
WORDPRESS = {
    "target": "blog.example.com",
    "endpoints": ["/wp-login.php"],
    "report_markdown": "WordPress CMS, JavaScript, APIs, login form present.",
}
ML = {
    "target": "model-api.example.com",
    "endpoints": ["/v1/predict", "/v1/infer"],
    "report_markdown": "PyTorch model inference service. Returns class probabilities.",
}
LLM = {
    "target": "chat.example.com",
    "endpoints": ["/v1/chat/completions"],
    "report_markdown": "OpenAI-compatible LLM chat endpoint. Prompt in, completion out.",
}


def _exp(assessment, key):
    return next(e for e in assessment["suggested_experiments"] if e["key"] == key)


def test_web_target_does_not_imply_ai_surface():
    a = build_assessment_from_summary(WEB)
    assert a["ai_surface"]["status"] == "NOT_OBSERVED"
    # No AI hypotheses surfaced.
    assert all(not h.get("ai_specific") for h in a["threat_hypotheses"])
    assert a["abstention"]  # Orion abstains rather than inventing an AI threat


def test_wordpress_does_not_generate_prompt_injection_finding():
    a = build_assessment_from_summary(WORDPRESS)
    names = " ".join(h["statement"].lower() for h in a["threat_hypotheses"])
    assert "prompt injection" not in names
    assert "adversarial" not in names
    pi = _exp(a, "prompt_injection")
    assert pi["applicability"] == NOT_APPLICABLE


def test_no_ai_surface_disables_atlas_mapping():
    a = build_assessment_from_summary(WEB)
    assert a["mitre_atlas"]["applicable"] is False
    assert a["mitre_atlas"]["mappings"] == []
    assert "not applicable" in a["mitre_atlas"]["message"].lower()


def test_ai_hypothesis_without_evidence_is_low_confidence():
    # Only a weak generic signal ("chat" support) — must not confirm AI.
    a = build_assessment_from_summary({"target": "t", "report_markdown": "contact support via live chat"})
    assert a["ai_surface"]["status"] in ("POSSIBLE", "NOT_OBSERVED")
    for h in a["threat_hypotheses"]:
        if h.get("ai_specific"):
            assert h["confidence"] in ("LOW", "NONE")


def test_hypothesis_is_not_finding():
    a = build_assessment_from_summary(ML)
    # Hypotheses are never emitted as validated findings.
    assert a["validated_findings"] == []
    for h in a["threat_hypotheses"]:
        assert h["classification"] == HYPOTHESIS
        assert "candidate" in h["statement"].lower() or "validation" in h["statement"].lower()


def test_llm_claim_without_valid_evidence_is_downgraded():
    # Inject an AI hypothesis with a bogus evidence id on a non-AI target.
    a = build_assessment_from_summary(WEB)
    a["threat_hypotheses"].append({
        "statement": "Prompt injection (candidate)", "classification": HYPOTHESIS,
        "evidence": ["recon.ghost.999"], "confidence": "HIGH", "ai_specific": True,
        "rationale": "ungrounded",
    })
    from orion.target_analysis import validate_analysis
    validate_analysis(a)
    # The bogus AI hypothesis must be dropped (NOT_APPLICABLE on a non-AI target).
    assert all(not (h.get("ai_specific")) for h in a["threat_hypotheses"])


def test_confirmed_llm_endpoint_enables_prompt_injection_hypothesis():
    a = build_assessment_from_summary(LLM)
    assert a["ai_surface"]["status"] == "CONFIRMED"
    pi = _exp(a, "prompt_injection")
    assert pi["applicability"] == "APPLICABLE"


def test_confirmed_ml_endpoint_enables_adversarial_scenario():
    a = build_assessment_from_summary(ML)
    assert a["ai_surface"]["status"] == "CONFIRMED"
    adv = _exp(a, "adversarial_evasion")
    assert adv["applicability"] == "APPLICABLE"
    assert adv["atlas"] and all(m["known"] for m in adv["atlas"])


def test_agent_can_abstain():
    a = build_assessment_from_summary({"target": "static-site", "endpoints": ["/index.html"],
                                       "report_markdown": "static HTML site"})
    assert a["ai_surface"]["status"] == "NOT_OBSERVED"
    assert "insufficient evidence" in a["abstention"].lower()


def test_keywords_do_not_substring_match():
    # 'rag' must not match inside 'coverage'; 'ml' must not match inside 'html'.
    a = build_assessment_from_summary({
        "target": "report.example.com",
        "report_markdown": "Test coverage grade A. HTML pages. Average latency fine.",
    })
    tnames = {t["type"] for t in a["target_types"]}
    assert "rag_application" not in tnames
    assert "ml_inference_service" not in tnames
    assert a["ai_surface"]["status"] == "NOT_OBSERVED"


def test_target_type_classification_is_evidence_based():
    types = {t["type"]: t for t in classify_target_types(ML, build_evidence(ML))}
    assert "ml_inference_service" in types
    web_types = {t["type"] for t in classify_target_types(WEB, build_evidence(WEB))}
    assert "ml_inference_service" not in web_types
    assert "traditional_web_app" in web_types


# ---- AI-surface signal-strength regression tests (spec §4–6, §20) ----
from orion.target_analysis import detect_ai_surface, build_evidence  # noqa: E402


def _surface(summary):
    s = {"endpoints": [], "auth_indicators": [], "findings": [], "report_markdown": "",
         "target": "t", "objective": "", **summary}
    return detect_ai_surface(s, build_evidence(s))


def test_generic_chat_route_does_not_confirm_llm():
    s = _surface({"endpoints": ["/chat"]})
    assert s["status"] != "CONFIRMED"


def test_model_route_does_not_confirm_ml():
    s = _surface({"endpoints": ["/model"]})
    assert s["status"] != "CONFIRMED"


def test_tool_route_does_not_confirm_agent():
    s = _surface({"endpoints": ["/tool"]})
    assert s["status"] != "CONFIRMED"


def test_assistant_route_is_weak_signal_only():
    s = _surface({"endpoints": ["/assistant"]})
    strengths = {sig["strength"] for sig in s["signals"]}
    assert strengths == {"WEAK"}
    assert s["status"] != "CONFIRMED"


def test_single_weak_signal_returns_possible_or_not_observed():
    s = _surface({"endpoints": ["/chat"]})
    assert s["status"] in ("POSSIBLE", "NOT_OBSERVED")


def test_strong_signal_confirms_ai_surface():
    s = _surface({"endpoints": ["/v1/chat/completions"]})
    assert s["status"] == "CONFIRMED"
    assert s["signal_strength"]["strong"] >= 1


def test_multiple_medium_signals_can_confirm_ai_surface():
    # Two independent MEDIUM signals corroborate -> CONFIRMED.
    s = _surface({"endpoints": ["/predict"], "report_markdown": "uses a vector database for retrieval"})
    assert s["signal_strength"]["medium"] >= 2
    assert s["status"] == "CONFIRMED"


def test_wordpress_target_has_no_ai_specific_findings():
    a = build_assessment_from_summary(WORDPRESS)
    assert all(not h.get("ai_specific") for h in a["threat_hypotheses"])
    assert all(not f.get("ai_specific") for f in a["validated_findings"])


def test_traditional_site_has_no_atlas_mapping():
    a = build_assessment_from_summary(WEB)
    assert a["mitre_atlas"]["applicable"] is False


def test_non_ai_target_hides_ai_experiments_by_default():
    a = build_assessment_from_summary(WEB)
    ai_exps = [e for e in a["suggested_experiments"] if e["ai_specific"]]
    assert ai_exps  # they exist in the catalog
    assert all(e["applicability"] == "NOT_APPLICABLE" for e in ai_exps)


def test_adversary_properties_are_not_invented():
    a = build_assessment_from_summary(WEB)
    adv = a["threat_model"]["adversary"]
    assert adv["goal"] == "UNDEFINED"
    assert adv["knowledge"] == "UNKNOWN"
    assert adv["budget"] == "UNKNOWN"
    # Network reachability IS observable from recon endpoints.
    assert adv["access"] == "REMOTE_PUBLIC"
    assert a["threat_model"]["provenance"]["access"] == "OBSERVED"
    assert a["threat_model"]["access_detail"]["credential_access"] == "NOT_OBSERVED"


# ---- AI application detection (spec: an AI service must be detected) ----
ORION_LIKE_APP = {
    "target": "127.0.0.1:5001",
    "endpoints": ["/chat", "/chat_stream", "/mcp_tools", "/adverimage", "/api/agent/analyze-recon"],
    "report_markdown": ("ORION adversarial intelligence system. LLM chat via DeepSeek and "
                        "WhiteRabbitNeo. MCP tools registry. Adversarial machine learning. "
                        "MITRE ATLAS mappings. LangGraph supervisor agent."),
}


def test_ai_application_is_detected_as_confirmed():
    a = build_assessment_from_summary(ORION_LIKE_APP)
    assert a["ai_surface"]["status"] == "CONFIRMED", a["ai_surface"]
    assert a["ai_surface"]["signal_strength"]["strong"] >= 1
    # An LLM/agent app should enable at least one AI experiment hypothesis.
    assert any(e["ai_specific"] and e["applicability"] == "APPLICABLE"
               for e in a["suggested_experiments"])


def test_mcp_endpoint_alone_is_strong():
    s = _surface({"endpoints": ["/mcp_tools"]})
    assert s["status"] == "CONFIRMED"


def test_model_name_in_page_is_strong():
    s = _surface({"report_markdown": "powered by DeepSeek-R1 and Ollama"})
    assert s["status"] == "CONFIRMED"


# ---- direct URL probe ----
from orion.target_analysis import build_probe_summary, probe_url  # noqa: E402


def test_probe_summary_detects_ai_endpoints():
    obs = [
        {"path": "/", "status": 200, "content_type": "text/html", "snippet": "ORION dashboard"},
        {"path": "/mcp_tools", "status": 200, "content_type": "application/json"},
        {"path": "/chat", "status": 405},               # exists (POST-only)
        {"path": "/v1/models", "status": 200},
        {"path": "/nope", "status": 404},               # does not exist
    ]
    summary = build_probe_summary("http://127.0.0.1:5001", summary_obs := obs)
    assert "/mcp_tools" in summary["endpoints"]
    assert "/v1/models" in summary["endpoints"]
    assert "/nope" not in summary["endpoints"]
    a = build_assessment_from_summary(summary)
    assert a["ai_surface"]["status"] == "CONFIRMED"


def test_probe_url_rejects_bad_url():
    import pytest
    with pytest.raises(ValueError):
        probe_url("ftp://example.com")   # non-http scheme
    with pytest.raises(ValueError):
        probe_url("")                     # no host
