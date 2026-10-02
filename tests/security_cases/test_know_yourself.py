"""Know Yourself profiling tests (spec §33–35).

Understand before attacking; profile before testing; unknown is valid;
applicability before execution; recommended attacks are proposals, not findings.
"""
import json

from orion.know_yourself import analyze, TRADITIONAL_ML, GENERATIVE_AI, HYBRID


def _posture(r, attack):
    return next((p for p in r["security_posture"] if p["attack"] == attack), None)


def _status(r, attack):
    p = _posture(r, attack)
    return p["status"] if p else None


# --------------------------- Traditional ML (§33) --------------------------- #
WB_CLASSIFIER = {"model_name": "resnet50", "task": "image_classification",
                 "input_type": "image", "framework": "pytorch", "access": "white_box",
                 "num_classes": 10, "input_shape": [3, 224, 224]}


def test_pytorch_classifier_detected_as_traditional_ml():
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    assert r["system_type"] == TRADITIONAL_ML
    assert r["traditional_ml"] is not None
    assert r["generative_ai"] is None


def test_model_fingerprint_contains_framework():
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    fp = r["traditional_ml"]["fingerprint"]
    assert fp["framework"] == "pytorch"
    assert fp["system_type"] == "traditional_ml"


def test_white_box_enables_gradient_attacks():
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    for atk in ("FGSM evasion", "PGD evasion", "C&W evasion", "DeepFool evasion"):
        assert _status(r, atk) == "APPLICABLE"


def test_pgd_not_applicable_without_gradient_or_query_path():
    r = analyze(descriptor={"task": "image_classification", "input_type": "image"},
                force_type=TRADITIONAL_ML, persist=False)
    assert r["capabilities"]["supports_gradients"] is False
    assert r["capabilities"]["query_access"] is False
    assert _status(r, "PGD evasion") in ("NOT_APPLICABLE", "INSUFFICIENT_EVIDENCE")


def test_prompt_injection_not_applicable_to_classifier():
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    assert _status(r, "Prompt injection") == "NOT_APPLICABLE"


def test_robustness_metrics_not_fabricated():
    # Know Yourself reports applicability, never fabricated robustness numbers.
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    blob = json.dumps(r)
    assert "clean_accuracy" not in blob
    assert "robust_accuracy" not in blob
    assert "attack_success_rate" not in blob


def test_unknown_training_provenance_stays_unknown():
    r = analyze(descriptor=WB_CLASSIFIER, persist=False)
    assumptions = r["traditional_ml"]["assumptions"]
    prov = [a for a in assumptions if "training-data provenance" in a["statement"].lower()]
    assert prov and prov[0]["classification"] == "UNKNOWN"


# --------------------------- Generative AI (§34) ---------------------------- #
def test_llm_app_detected_as_generative_ai():
    r = analyze(context="openai /v1/chat/completions LLM chat application", persist=False)
    assert r["system_type"] == GENERATIVE_AI
    assert r["generative_ai"] is not None


def test_rag_detection_adds_retrieval_boundary():
    r = analyze(descriptor={"rag": True}, context="llm chat", persist=False)
    boundaries = [b["boundary"] for b in r["generative_ai"]["trust_boundaries"]]
    assert any("Retrieved Context" in b for b in boundaries)


def test_agent_tools_add_tool_boundary():
    r = analyze(descriptor={"tools": [{"name": "shell"}]}, context="llm agent", persist=False)
    boundaries = [b["boundary"] for b in r["generative_ai"]["trust_boundaries"]]
    assert any("Agent -> Tool" in b for b in boundaries)


def test_mcp_adds_mcp_trust_boundary():
    r = analyze(descriptor={"mcp": True}, context="llm agent", persist=False)
    boundaries = [b["boundary"] for b in r["generative_ai"]["trust_boundaries"]]
    assert any("MCP Server" in b for b in boundaries)


def test_prompt_injection_requires_llm_surface():
    gen = analyze(context="openai llm chat", persist=False)
    trad = analyze(descriptor=WB_CLASSIFIER, persist=False)
    assert _status(gen, "Prompt injection") == "APPLICABLE"
    assert _status(trad, "Prompt injection") == "NOT_APPLICABLE"


def test_rag_poisoning_requires_retrieval():
    with_rag = analyze(descriptor={"rag": True}, context="llm", persist=False)
    no_rag = analyze(context="openai llm chat", persist=False)
    assert _status(with_rag, "RAG poisoning") == "APPLICABLE"
    assert _status(no_rag, "RAG poisoning") == "NOT_APPLICABLE"


def test_tool_poisoning_requires_tool_or_mcp_surface():
    with_tool = analyze(descriptor={"mcp": True}, context="llm agent", persist=False)
    no_tool = analyze(context="openai llm chat", persist=False)
    assert _status(with_tool, "Tool / MCP poisoning") == "APPLICABLE"
    assert _status(no_tool, "Tool / MCP poisoning") == "NOT_APPLICABLE"


def test_pgd_not_suggested_for_generic_llm_app():
    r = analyze(context="openai /v1/chat/completions llm chat", persist=False)
    recs = [x["attack"] for x in r["recommended_experiments"]]
    assert not any("evasion" in a.lower() or a.startswith("PGD") for a in recs)


def test_unknown_control_is_not_marked_missing():
    r = analyze(context="openai llm chat", persist=False)
    controls = {c["control"]: c["status"] for c in r["generative_ai"]["controls"]}
    # A control with no evidence must be UNKNOWN, never NOT_FOUND.
    assert controls.get("audit_logging") == "UNKNOWN"


# ------------------------------- Hybrid (§35) ------------------------------- #
def test_hybrid_system_runs_both_branches():
    r = analyze(descriptor={"task": "image_classification", "input_type": "image",
                            "access": "white_box", "framework": "pytorch", "mcp": True,
                            "tools": [{"name": "interpret"}]},
                context="image classifier plus an llm agent with mcp tools",
                force_type=HYBRID, persist=False)
    assert r["system_type"] == HYBRID
    assert r["traditional_ml"] is not None
    assert r["generative_ai"] is not None
    assert _status(r, "PGD evasion") == "APPLICABLE"            # from the ML branch
    assert _status(r, "Tool / MCP poisoning") == "APPLICABLE"   # from the GenAI branch


def test_local_artifact_fingerprint_enables_whitebox_handoff(tmp_path):
    # A local model artifact => white-box fingerprint with hash/size, PGD applicable.
    art = tmp_path / "model.pth"
    art.write_bytes(b"\x00\x01\x02fake-weights")
    r = analyze(descriptor={"artifact": str(art), "task": "image_classification",
                            "input_type": "image", "access": "white_box", "num_classes": 2},
                persist=False)
    fp = r["traditional_ml"]["fingerprint"]
    assert fp["artifact"] == str(art)
    assert fp["framework"] == "pytorch"          # from .pth extension
    assert fp["model_hash"] and fp["model_size_bytes"] > 0
    assert _status(r, "PGD evasion") == "APPLICABLE"
