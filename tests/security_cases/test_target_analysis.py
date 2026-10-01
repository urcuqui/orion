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
    # Context mentions AI only as a weak textual signal (POSSIBLE, not CONFIRMED).
    a = build_assessment_from_summary({"target": "t", "report_markdown": "we might add an llm chatbot later"})
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
