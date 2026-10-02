"""Evidence-grounded target interpretation.

Core principle::

    Evidence -> applicability -> hypothesis -> test -> finding
    (never: generic knowledge -> finding)

Orion must know when it does not know. Agent Analysis never treats generic
AI-security knowledge as evidence about *this* target: an AI-specific threat is
only surfaced when reconnaissance actually observed an AI/ML surface. Every
conclusion is classified (OBSERVED / INFERRED / HYPOTHESIS / NOT_APPLICABLE),
carries evidence references + confidence + rationale, and passes a deterministic
validation layer that downgrades claims the evidence does not support.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# In-memory registry of built assessments + human approval state.
ANALYSES: Dict[str, Dict[str, Any]] = {}

# --- classifications & confidence ------------------------------------------- #
OBSERVED = "OBSERVED"
INFERRED = "INFERRED"
HYPOTHESIS = "HYPOTHESIS"
NOT_APPLICABLE = "NOT_APPLICABLE"

HIGH, MEDIUM, LOW, NONE = "HIGH", "MEDIUM", "LOW", "NONE"


def display_run_id(run_id: str, kind: str = "RECON") -> str:
    tail = (run_id or "0000").replace("-", "")[:4].upper()
    return f"ORN-{kind}-{tail}"


def _confidence_for(evidence_count: int) -> str:
    if evidence_count >= 2:
        return HIGH
    if evidence_count == 1:
        return MEDIUM
    return NONE


# --------------------------------------------------------------------------- #
# Recon summary (unchanged contract, with light technology extraction)
# --------------------------------------------------------------------------- #
def summarize_recon(run_id: str) -> Optional[Dict[str, Any]]:
    try:
        from libs.recon import web as recon_web
    except Exception:  # pragma: no cover
        return None
    session = recon_web.get_run(run_id)
    if session is None:
        return None
    with session.lock:
        events = list(session.events)
        status = session.status
        final_markdown = session.final_markdown

    objective = target = ""
    findings: Dict[str, Dict[str, Any]] = {}
    endpoints: List[str] = []
    auth_indicators: List[str] = []
    screenshots: List[str] = []
    report_markdown = final_markdown or ""

    for ev in events:
        etype = ev.get("type")
        if etype == "start":
            objective = ev.get("objective", "") or objective
            target = ev.get("target", "") or target
        elif etype == "evaluate":
            for f in ev.get("findings", []) or []:
                fid = f.get("id") or f.get("title")
                if fid:
                    findings[fid] = f
        elif etype in ("execute", "browser_tool") or ev.get("browser_tool"):
            for ep in ev.get("api_endpoints", []) or []:
                if ep not in endpoints:
                    endpoints.append(ep)
            for ai in ev.get("auth_indicators", []) or []:
                if ai not in auth_indicators:
                    auth_indicators.append(ai)
            shot = ev.get("screenshot_filename")
            if shot and shot not in screenshots:
                screenshots.append(shot)
        elif etype == "report":
            report_markdown = ev.get("report_markdown") or report_markdown

    findings_list = list(findings.values())
    return {
        "run_id": run_id,
        "display_id": display_run_id(run_id),
        "status": status,
        "objective": objective,
        "target": target or "unknown",
        "endpoints": endpoints,
        "auth_indicators": auth_indicators,
        "screenshots": screenshots,
        "findings": findings_list,
        "report_markdown": report_markdown,
        "counts": {
            "endpoints": len(endpoints),
            "findings": len(findings_list),
            "screenshots": len(screenshots),
            "auth_flows": len(auth_indicators),
        },
    }


# --------------------------------------------------------------------------- #
# Evidence index: every observation gets a stable, referenceable id
# --------------------------------------------------------------------------- #
def build_evidence(summary: Dict[str, Any]) -> List[Dict[str, str]]:
    evidence: List[Dict[str, str]] = []
    for i, ep in enumerate(summary.get("endpoints", []), 1):
        evidence.append({"id": f"recon.endpoint.{i:03d}", "kind": "endpoint", "value": str(ep)})
    for i, au in enumerate(summary.get("auth_indicators", []), 1):
        evidence.append({"id": f"recon.auth.{i:03d}", "kind": "auth", "value": str(au)})
    for i, f in enumerate(summary.get("findings", []), 1):
        evidence.append({"id": f"recon.finding.{i:03d}", "kind": "finding",
                         "value": str(f.get("title") or f.get("description") or "finding"),
                         "severity": str(f.get("severity") or "info")})
    report = summary.get("report_markdown") or ""
    for i, tech in enumerate(_extract_technologies(report), 1):
        evidence.append({"id": f"recon.tech.{i:03d}", "kind": "technology", "value": tech})
    return evidence


_TECH_PATTERNS = [
    "wordpress", "drupal", "joomla", "nginx", "apache", "php", "node.js", "express",
    "django", "flask", "react", "vue", "angular", "mysql", "postgres", "cloudflare",
    "openai", "anthropic", "langchain", "langgraph", "llamaindex", "huggingface",
    "pytorch", "tensorflow", "chromadb", "pinecone", "weaviate", "ollama", "vllm",
    "triton", "torchserve", "onnx", "deepseek", "llama", "mistral", "whiterabbitneo",
    "mcp",
]


def _extract_technologies(text: str) -> List[str]:
    low = (text or "").lower()
    return [t for t in _TECH_PATTERNS if t in low]


# --------------------------------------------------------------------------- #
# Signal keywords
# --------------------------------------------------------------------------- #
_LLM_KW = ["llm", "gpt", "openai", "anthropic", "claude", "chat model", "chatbot",
           "completion", "prompt", "langchain", "langgraph", "assistant"]
_RAG_KW = ["rag", "retrieval", "embedding", "vector db", "vector store", "knowledge base",
           "chromadb", "pinecone", "weaviate", "document ingest"]
_AGENT_KW = ["agent", "tool-calling", "tool calling", "function calling"]
_MCP_KW = ["mcp", "model context protocol"]
_ML_KW = ["predict", "inference", "classifier", "ml model", "machine learning", "neural",
          ".pth", ".onnx", ".h5", ".safetensors", "pytorch", "tensorflow", "huggingface",
          "model artifact", "model api",
          # Computer-vision tasks / libs (an image classifier/detector is ML)
          "face detection", "object detection", "image classification",
          "face recognition", "classification model", "computer vision",
          "yolo", "opencv", "mediapipe", "keras", "sklearn", "scikit-learn", "onnxruntime",
          # Behavioural confirmation (observed a prediction/inference response)
          "ml_inference_response", "ml_prediction_response", "model_version_status"]
_WEB_KW = ["html", "wordpress", "cms", "login", "form", "javascript", "cookie", "session",
           "drupal", "joomla", "nginx", "apache", "php"]
_API_KW = ["/api", "json", "rest", "/v1/", "endpoint", "swagger", "openapi", "graphql"]

# --- AI-surface signal tiers (strength-scored, inspectable) ---------------- #
# STRONG: unambiguous AI/ML inference surfaces, serving stacks or model artifacts.
_STRONG_AI = [
    # Inference / serving endpoints (direct evidence of an AI service)
    "/v1/chat/completions", "/chat/completions", "/v1/completions", "/v1/embeddings",
    "/embeddings", "/invocations", "/predictions", "/v2/models", "/v1/models",
    # Model Context Protocol (unambiguous AI/agent tooling)
    "model context protocol", "mcp server", "mcp", "/mcp", "/mcp_tools",
    # SDKs / serving stacks / frameworks
    "openai", "anthropic", "ollama", "huggingface", "hugging face", "pytorch",
    "tensorflow serving", "torchserve", "triton", "onnx runtime", "vllm",
    "langchain", "langgraph", "llamaindex", "tensorflow", "text-generation-inference",
    # Specific model names/versions (imply a real deployed model)
    "deepseek", "whiterabbitneo", "gpt-4", "gpt-3",
    # ML model-serving stacks / deep-learning frameworks
    "kserve", "seldon", "bentoml", "mlflow", "sagemaker", "keras",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "model-server",
    "signature_name", "model_version_status", "ml_prediction_response",
    # Computer-vision models / libraries (specific = strong)
    "yolo", "yolov", "ultralytics", "mediapipe", "mtcnn", "facenet", "dlib",
    "opencv", "detectron", "retinaface", "haar cascade",
    # Behavioural signals observed in a response
    "softmax", "logits", "ml_inference_response",
    # Model artifacts
    ".pt", ".pth", ".onnx", ".safetensors", ".gguf", ".ckpt", ".h5",
    ".joblib", ".pkl",
]
_MEDIUM_AI = [
    # AI-ish routes (need corroboration)
    "/predict", "/inference", "/generate", "/rag", "/embedding", "/vector",
    "/agent", "/api/agent", "/chat_stream", "/completions",
    "/classify", "/score", "/predictions", "/api/predict", "/infer",
    # Terminology without confirmed implementation
    "llm", "large language model", "language model", "foundation model",
    "generative ai", "adversarial machine learning", "machine learning",
    "deep learning", "neural network", "transformer", "classifier",
    "classification model", "model inference", "inference endpoint",
    "probabilities", "class probabilities", "feature vector",
    # Computer-vision tasks
    "face detection", "object detection", "image classification",
    "face recognition", "image recognition", "computer vision",
    "segmentation", "bounding box", "upload image",
    "chatbot", "assistant interface", "inference", "retrieval-augmented",
    "rag pipeline", "vector database", "vector store", "prompt template",
    "prompt injection", "completion api", "model api",
    # Generic model families
    "gpt", "claude", "gemini", "llama", "mistral", "qwen", "gemma",
]
_WEAK_AI = ["/chat", "/model", "/tool", "/assistant", "/ai", "assistant", "chat"]

_STRONG, _MEDIUM, _WEAK = "STRONG", "MEDIUM", "WEAK"


def _signal_strength(value: str) -> Optional[str]:
    """Classify a single value into STRONG/MEDIUM/WEAK, strongest first."""
    v = (value or "").lower()
    if any(_kw_in(v, k) for k in _STRONG_AI):
        return _STRONG
    if any(_kw_in(v, k) for k in _MEDIUM_AI):
        return _MEDIUM
    if any(_kw_in(v, k) for k in _WEAK_AI):
        return _WEAK
    return None


def _haystack(summary: Dict[str, Any]) -> str:
    parts = list(summary.get("endpoints", [])) + list(summary.get("auth_indicators", []))
    for f in summary.get("findings", []):
        parts.append(str(f.get("title", "")))
        parts.append(str(f.get("description", "")))
    parts.append(summary.get("report_markdown", "") or "")
    parts.append(summary.get("target", "") or "")
    parts.append(summary.get("objective", "") or "")
    return " \n ".join(parts).lower()


def _kw_in(text: str, kw: str) -> bool:
    """Match a keyword, using word boundaries for plain words to avoid
    substring false positives (e.g. 'rag' inside 'coverage'). Keywords that
    contain path/extension characters (/, ., -) are matched as substrings."""
    if re.search(r"[^\w ]", kw):
        return kw in text
    return re.search(r"\b" + re.escape(kw) + r"\b", text) is not None


def _match_evidence(evidence: List[Dict[str, str]], keywords: List[str]) -> List[str]:
    """Return evidence ids whose value matches any keyword."""
    ids = []
    for e in evidence:
        v = e["value"].lower()
        if any(_kw_in(v, k) for k in keywords):
            ids.append(e["id"])
    return ids


# --------------------------------------------------------------------------- #
# Target-type classification + AI-surface detection
# --------------------------------------------------------------------------- #
def classify_target_types(summary: Dict[str, Any], evidence: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    hay = _haystack(summary)
    types: List[Dict[str, Any]] = []

    def add(tname, kws):
        ids = _match_evidence(evidence, kws)
        # also count a text-only signal (weaker) if nothing in discrete evidence
        text_hit = any(_kw_in(hay, k) for k in kws)
        if ids:
            types.append({"type": tname, "confidence": _confidence_for(len(ids)), "evidence": ids})
        elif text_hit:
            types.append({"type": tname, "confidence": LOW, "evidence": []})

    add("traditional_web_app", _WEB_KW)
    add("api_service", _API_KW)
    add("ml_inference_service", _ML_KW)
    add("llm_application", _LLM_KW)
    add("rag_application", _RAG_KW)
    add("agentic_application", _AGENT_KW)
    add("mcp_enabled_system", _MCP_KW)

    if not types:
        types.append({"type": "unknown", "confidence": NONE, "evidence": []})
    return types


def detect_ai_surface(summary: Dict[str, Any], evidence: List[Dict[str, str]]) -> Dict[str, Any]:
    """Deterministic, strength-scored AI-surface detection.

    Each matched signal is STRONG / MEDIUM / WEAK and references the evidence it
    came from. Classification (inspectable, not a black box):

      CONFIRMED : >=1 STRONG signal, or >=2 MEDIUM signals
      POSSIBLE  : exactly 1 MEDIUM, or >=2 WEAK signals
      NOT_OBSERVED : otherwise (a lone WEAK signal never confirms AI)

    A WEAK generic route (/chat, /model, /tool, /assistant) on its own can never
    produce CONFIRMED — it requires corroborating strong/medium evidence.
    """
    signals: List[Dict[str, Any]] = []
    seen: set = set()

    # Evidence-backed signals (endpoints, technologies, findings).
    for e in evidence:
        strength = _signal_strength(e["value"])
        if strength:
            key = e["value"].lower()
            if key not in seen:
                seen.add(key)
                signals.append({"value": e["value"], "strength": strength, "evidence_id": e["id"]})

    # Strong/medium textual signals not already captured as discrete evidence.
    hay = _haystack(summary)
    for kw in _STRONG_AI + _MEDIUM_AI:
        if kw in seen:
            continue
        if _kw_in(hay, kw):
            strength = _STRONG if kw in _STRONG_AI else _MEDIUM
            signals.append({"value": kw, "strength": strength, "evidence_id": None})
            seen.add(kw)

    strong = [s for s in signals if s["strength"] == _STRONG]
    medium = [s for s in signals if s["strength"] == _MEDIUM]
    weak = [s for s in signals if s["strength"] == _WEAK]
    score = len(strong) * 4 + len(medium) * 2 + len(weak) * 1

    if strong or len(medium) >= 2:
        status, conf = "CONFIRMED", HIGH
    elif len(medium) == 1 or len(weak) >= 2:
        status, conf = "POSSIBLE", (MEDIUM if medium else LOW)
    else:
        status, conf = "NOT_OBSERVED", NONE

    ev_ids = sorted({s["evidence_id"] for s in (strong + medium) if s["evidence_id"]})
    rationale = {
        "CONFIRMED": ("Confirmed by " + (f"{len(strong)} strong" if strong else f"{len(medium)} medium")
                      + " AI signal(s)."),
        "POSSIBLE": "Indirect AI signal(s) only; corroboration required before treating as confirmed.",
        "NOT_OBSERVED": "No AI/ML attack surface was observed in the reconnaissance evidence.",
    }[status]
    return {"status": status, "confidence": conf, "evidence": ev_ids,
            "signals": signals, "score": score,
            "signal_strength": {"strong": len(strong), "medium": len(medium), "weak": len(weak)},
            "rationale": rationale}


def _derive_capabilities(summary: Dict[str, Any], evidence: List[Dict[str, str]],
                         ai_surface: Dict[str, Any], types: List[str]) -> Dict[str, List[str]]:
    """Map evidence to attack-prerequisite capability flags -> supporting ids."""
    caps: Dict[str, List[str]] = {}
    ai_confirmed = ai_surface["status"] == "CONFIRMED"
    endpoint_ids = [e["id"] for e in evidence if e["kind"] == "endpoint"]
    # A CONFIRMED surface is itself the evidence; fall back to a marker when the
    # confirming signals were behavioural/text (no discrete evidence id) so the
    # capability is not wrongly treated as missing.
    surf_ev = ai_surface.get("evidence") or (["ai_surface:confirmed"] if ai_confirmed else [])
    if "ml_inference_service" in types and ai_confirmed:
        caps["inference_input"] = surf_ev
        caps["model_behavior_observable"] = surf_ev
        caps["inference_api"] = surf_ev
        caps["query_access"] = endpoint_ids or surf_ev
    if ("llm_application" in types or "agentic_application" in types) and ai_confirmed:
        caps["llm_interface"] = surf_ev
    if "rag_application" in types and ai_confirmed:
        caps["retrieval"] = surf_ev
        caps["knowledge_source"] = surf_ev
    if ("agentic_application" in types or "mcp_enabled_system" in types) and ai_confirmed:
        caps["agent_tool"] = surf_ev
    if endpoint_ids:
        caps["http_endpoint"] = endpoint_ids
    return caps


# --------------------------------------------------------------------------- #
# Attack catalog with explicit prerequisites
# --------------------------------------------------------------------------- #
ATTACK_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "adversarial_evasion", "name": "Adversarial input robustness (PGD / C&W)",
        "ai_specific": True, "risk": "LOW", "scenario": "scenarios/pgd_evasion.yaml",
        "target_types": ["ml_inference_service"],
        "requires": ["inference_input", "model_behavior_observable"],
        "atlas": ["AML.T0043.000", "AML.T0015"],
    },
    {
        "id": "model_extraction", "name": "Model extraction / inference-API abuse",
        "ai_specific": True, "risk": "MEDIUM", "scenario": None,
        "target_types": ["ml_inference_service", "api_service"],
        "requires": ["inference_api", "query_access"], "atlas": ["AML.T0040"],
    },
    {
        "id": "prompt_injection", "name": "Prompt injection", "ai_specific": True,
        "risk": "MEDIUM", "scenario": None,
        "target_types": ["llm_application", "agentic_application"],
        "requires": ["llm_interface"], "atlas": [],
    },
    {
        "id": "rag_poisoning", "name": "RAG poisoning", "ai_specific": True,
        "risk": "MEDIUM", "scenario": None, "target_types": ["rag_application"],
        "requires": ["retrieval", "knowledge_source"], "atlas": ["AML.T0020"],
    },
    {
        "id": "tool_poisoning", "name": "Tool / MCP poisoning", "ai_specific": True,
        "risk": "MEDIUM", "scenario": None,
        "target_types": ["agentic_application", "mcp_enabled_system"],
        "requires": ["agent_tool"], "atlas": [],
    },
    {
        "id": "api_boundary", "name": "API boundary validation", "ai_specific": False,
        "risk": "MEDIUM", "scenario": None, "target_types": ["api_service"],
        "requires": ["http_endpoint"], "atlas": [],
    },
    {
        "id": "recon_extension", "name": "Reconnaissance extension", "ai_specific": False,
        "risk": "LOW", "scenario": None, "target_types": [], "requires": [], "atlas": [],
    },
]


def check_applicability(attack: Dict[str, Any], types: List[str],
                        ai_surface: Dict[str, Any], caps: Dict[str, List[str]]) -> Dict[str, Any]:
    """Decide whether an attack's prerequisites are met by the evidence."""
    missing: List[str] = []
    # AI-specific attacks require a confirmed AI surface.
    if attack["ai_specific"] and ai_surface["status"] != "CONFIRMED":
        return {"status": "NOT_APPLICABLE", "missing": ["ai_surface"],
                "evidence": [], "reason": "No confirmed AI/ML attack surface."}
    # Target-type prerequisite (empty = any).
    if attack["target_types"] and not any(t in types for t in attack["target_types"]):
        missing.append("target_type:" + "|".join(attack["target_types"]))
    # Capability prerequisites.
    ev: List[str] = []
    for req in attack["requires"]:
        if req in caps and caps[req]:
            ev.extend(caps[req])
        else:
            missing.append(req)
    if missing:
        status = "NOT_APPLICABLE" if attack["ai_specific"] else "INSUFFICIENT_EVIDENCE"
        return {"status": status, "missing": missing, "evidence": sorted(set(ev)),
                "reason": "Prerequisites not satisfied: " + ", ".join(missing)}
    return {"status": "APPLICABLE", "missing": [], "evidence": sorted(set(ev)),
            "reason": "Prerequisites satisfied by observed evidence."}


# --------------------------------------------------------------------------- #
# Observations / inferences / hypotheses / experiments / findings
# --------------------------------------------------------------------------- #
def _observations(summary: Dict[str, Any], evidence: List[Dict[str, str]],
                  types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    obs: List[Dict[str, Any]] = []
    eps = [e for e in evidence if e["kind"] == "endpoint"]
    if eps:
        obs.append({"statement": f"{len(eps)} endpoint(s) discovered", "classification": OBSERVED,
                    "evidence": [e["id"] for e in eps], "confidence": _confidence_for(len(eps)),
                    "rationale": "Endpoints were directly observed by recon tools."})
    techs = [e for e in evidence if e["kind"] == "technology"]
    if techs:
        obs.append({"statement": "Technologies identified: " + ", ".join(e["value"] for e in techs),
                    "classification": OBSERVED, "evidence": [e["id"] for e in techs],
                    "confidence": _confidence_for(len(techs)),
                    "rationale": "Technology fingerprints appeared in recon output."})
    for e in [e for e in evidence if e["kind"] == "finding"]:
        obs.append({"statement": e["value"], "classification": OBSERVED, "evidence": [e["id"]],
                    "confidence": MEDIUM, "rationale": f"Reported by a recon tool (severity {e.get('severity','info')})."})
    primary = types[0]["type"] if types else "unknown"
    obs.append({"statement": f"Target presents as: {primary}", "classification": OBSERVED
                if types and types[0]["confidence"] != NONE else INFERRED,
                "evidence": types[0]["evidence"] if types else [],
                "confidence": types[0]["confidence"] if types else NONE,
                "rationale": "Derived from observed technologies/endpoints."})
    return obs


def _inferences(summary: Dict[str, Any], evidence: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    inf: List[Dict[str, Any]] = []
    auth_ids = [e["id"] for e in evidence if e["kind"] == "auth"] + \
               [e["id"] for e in evidence if e["kind"] == "endpoint"
                and any(k in e["value"].lower() for k in ("login", "signin", "auth", "session"))]
    if auth_ids:
        inf.append({"statement": "Public authentication surface exists", "classification": INFERRED,
                    "evidence": sorted(set(auth_ids)), "confidence": _confidence_for(len(set(auth_ids))),
                    "rationale": "A login/authentication indicator was observed."})
    api_ids = [e["id"] for e in evidence if e["kind"] == "endpoint"
               and any(k in e["value"].lower() for k in _API_KW)]
    if api_ids:
        inf.append({"statement": "Programmatic API surface exists", "classification": INFERRED,
                    "evidence": api_ids, "confidence": _confidence_for(len(api_ids)),
                    "rationale": "API-style endpoints were observed."})
    return inf


def threat_hypotheses(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Public helper: classified threat hypotheses (evidence-gated)."""
    evidence = build_evidence(summary)
    types = classify_target_types(summary, evidence)
    tnames = [t["type"] for t in types]
    ai_surface = detect_ai_surface(summary, evidence)
    caps = _derive_capabilities(summary, evidence, ai_surface, tnames)
    return _threat_hypotheses(summary, evidence, tnames, ai_surface, caps)


def _threat_hypotheses(summary, evidence, tnames, ai_surface, caps) -> List[Dict[str, Any]]:
    hyps: List[Dict[str, Any]] = []
    # Web/API hypotheses (not AI) — only when the surface exists.
    api_ev = caps.get("http_endpoint", [])
    if api_ev:
        hyps.append({"statement": "Input validation / API abuse (candidate)", "classification": HYPOTHESIS,
                     "evidence": api_ev, "confidence": _confidence_for(len(api_ev)),
                     "severity": "MEDIUM", "ai_specific": False,
                     "rationale": "Endpoints exist; boundary handling requires validation."})
    # High/critical recon findings become hypotheses to validate.
    for e in [e for e in evidence if e["kind"] == "finding" and e.get("severity", "info").upper() in ("HIGH", "CRITICAL")]:
        hyps.append({"statement": e["value"] + " (requires validation)", "classification": HYPOTHESIS,
                     "evidence": [e["id"]], "confidence": MEDIUM, "severity": e["severity"].upper(),
                     "ai_specific": False, "rationale": "Recon flagged this; confirm before treating as a finding."})
    # AI hypotheses ONLY when an AI surface is confirmed/possible.
    for attack in ATTACK_CATALOG:
        if not attack["ai_specific"]:
            continue
        appl = check_applicability(attack, tnames, ai_surface, caps)
        if appl["status"] == "APPLICABLE":
            hyps.append({"statement": f"{attack['name']} (candidate — not established)",
                         "classification": HYPOTHESIS, "evidence": appl["evidence"],
                         "confidence": _confidence_for(len(appl["evidence"])) or LOW,
                         "severity": "MEDIUM", "ai_specific": True,
                         "rationale": "An AI surface and the attack's prerequisites were observed."})
    return hyps


def propose_experiments(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence = build_evidence(summary)
    types = classify_target_types(summary, evidence)
    tnames = [t["type"] for t in types]
    ai_surface = detect_ai_surface(summary, evidence)
    caps = _derive_capabilities(summary, evidence, ai_surface, tnames)
    return _suggested_experiments(summary, evidence, tnames, ai_surface, caps)


def _suggested_experiments(summary, evidence, tnames, ai_surface, caps) -> List[Dict[str, Any]]:
    from orion.mappings import validate_mappings

    exps: List[Dict[str, Any]] = []
    idx = 1
    for attack in ATTACK_CATALOG:
        appl = check_applicability(attack, tnames, ai_surface, caps)
        # ATLAS mappings only for AI-specific attacks whose prerequisites are
        # actually satisfied — never force a mapping onto a non-applicable attack.
        atlas: List[Dict[str, Any]] = []
        if (attack["ai_specific"] and appl["status"] == "APPLICABLE"
                and ai_surface["status"] == "CONFIRMED" and attack["atlas"]):
            atlas = [m.to_dict() for m in validate_mappings([{"technique_id": i} for i in attack["atlas"]])]
        exps.append({
            "id": f"{idx:02d}", "key": attack["id"], "name": attack["name"],
            "risk": attack["risk"], "ai_specific": attack["ai_specific"],
            "scenario": attack["scenario"], "atlas": atlas,
            "applicability": appl["status"], "missing": appl["missing"],
            "evidence": appl["evidence"], "rationale": appl["reason"],
        })
        idx += 1
    # Primary recommendations first (applicable), then gated ones.
    exps.sort(key=lambda e: {"APPLICABLE": 0, "INSUFFICIENT_EVIDENCE": 1, "NOT_APPLICABLE": 2}.get(e["applicability"], 3))
    return exps


def _validated_findings(evidence: List[Dict[str, str]], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, f in enumerate(summary.get("findings", []), 1):
        vs = (f.get("validation_status") or "").lower()
        if vs in ("validated", "confirmed", "verified"):
            out.append({"statement": f.get("title", "finding"), "classification": OBSERVED,
                        "evidence": [f"recon.finding.{i:03d}"], "confidence": HIGH,
                        "severity": (f.get("severity") or "info").upper(),
                        "rationale": "Validated by a recon tool."})
    return out


def propose_threat_model(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Evidence-based threat model (assets/adversary reflect what was observed)."""
    evidence = build_evidence(summary)
    types = classify_target_types(summary, evidence)
    tnames = [t["type"] for t in types]
    ai_surface = detect_ai_surface(summary, evidence)

    assets: List[str] = ["availability"]
    surfaces: List[str] = []
    if any(e["kind"] == "endpoint" for e in evidence):
        surfaces.append("inference_api" if ai_surface["status"] == "CONFIRMED" else "application")
        assets.append("inference_api" if ai_surface["status"] == "CONFIRMED" else "application")
    if any(e["kind"] == "auth" for e in evidence) or "login" in _haystack(summary):
        assets.append("confidentiality")
    if ai_surface["status"] == "CONFIRMED":
        assets += ["model_integrity", "prediction_reliability"]
        surfaces.append("model_artifact")

    # Honesty: Orion does not invent adversary properties. Only what recon can
    # establish is marked OBSERVED; everything else is UNKNOWN/UNDEFINED and
    # requires analyst input.
    has_endpoint = any(e["kind"] == "endpoint" for e in evidence)
    network_access = "REMOTE_PUBLIC" if has_endpoint else "UNKNOWN"
    adversary = {
        "goal": "UNDEFINED",
        "knowledge": "UNKNOWN",          # model/system knowledge is not observable from recon
        "access": network_access,        # network reachability only
        "budget": "UNKNOWN",
    }
    access_detail = {
        "network_access": network_access,
        "model_knowledge": "UNKNOWN",
        "credential_access": "NOT_OBSERVED",
    }
    # Per-field provenance so the UI can label OBSERVED vs UNKNOWN/UNDEFINED.
    provenance = {
        "goal": "UNDEFINED",
        "knowledge": "UNKNOWN",
        "access": "OBSERVED" if has_endpoint else "UNKNOWN",
        "budget": "UNKNOWN",
        "network_access": "OBSERVED" if has_endpoint else "UNKNOWN",
        "model_knowledge": "UNKNOWN",
        "credential_access": "OBSERVED",  # we observed its ABSENCE (not present)
    }
    return {
        "status": "PROPOSED",
        "target": {"task": tnames[0] if tnames else "unknown", "access": network_access,
                   "model_name": summary.get("target", "unknown")},
        "adversary": adversary,
        "access_detail": access_detail,
        "provenance": provenance,
        "assets": sorted(set(assets)),
        "surfaces": sorted(set(surfaces)),
        "note": "Adversary properties require analyst input.",
        "rationale": (f"Derived from recon of {summary.get('target','the target')}: "
                      f"target type(s) {', '.join(tnames)}; AI surface {ai_surface['status']}. "
                      f"Adversary knowledge/goal/budget are not observable from recon alone."),
    }


def _atlas_section(ai_surface: Dict[str, Any], exps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if ai_surface["status"] == "NOT_OBSERVED":
        return {"applicable": False, "mappings": [],
                "message": "Not applicable with current evidence."}
    mappings = []
    for e in exps:
        for m in e.get("atlas", []):
            if m not in mappings:
                mappings.append(m)
    return {"applicable": bool(mappings), "mappings": mappings,
            "message": "" if mappings else "No AI-specific mappings for the applicable experiments."}


# --------------------------------------------------------------------------- #
# Deterministic validation pass (do not let the LLM police itself)
# --------------------------------------------------------------------------- #
def validate_analysis(assessment: Dict[str, Any]) -> List[str]:
    """Downgrade/reject claims the evidence does not support. Returns notes."""
    notes: List[str] = []
    valid_ids = {e["id"] for e in assessment.get("evidence", [])}
    ai_confirmed = assessment.get("ai_surface", {}).get("status") == "CONFIRMED"

    def fix_item(item: Dict[str, Any], section: str):
        # Drop evidence refs that don't exist.
        refs = [r for r in item.get("evidence", []) if r in valid_ids]
        if len(refs) != len(item.get("evidence", [])):
            notes.append(f"[{section}] removed non-existent evidence ref in '{item.get('statement','')[:40]}'")
        item["evidence"] = refs
        n = len(refs)
        # Confidence must be consistent with evidence count.
        if item.get("confidence") == HIGH and n < 2:
            item["confidence"] = MEDIUM if n == 1 else LOW
            notes.append(f"[{section}] downgraded HIGH→{item['confidence']} (evidence={n})")
        if item.get("confidence") == MEDIUM and n == 0:
            item["confidence"] = LOW
            notes.append(f"[{section}] downgraded MEDIUM→LOW (evidence=0)")
        # AI-specific claim without a confirmed AI surface -> NOT_APPLICABLE / NONE.
        if item.get("ai_specific") and not ai_confirmed:
            item["classification"] = NOT_APPLICABLE
            item["confidence"] = NONE
            notes.append(f"[{section}] AI claim without AI surface → NOT_APPLICABLE: '{item.get('statement','')[:40]}'")
        # A zero-evidence hypothesis can be at most LOW.
        if item.get("classification") == HYPOTHESIS and n == 0 and item.get("confidence") not in (LOW, NONE):
            item["confidence"] = LOW
            notes.append(f"[{section}] zero-evidence hypothesis capped at LOW")

    for sect in ("observations", "inferences", "threat_hypotheses", "validated_findings"):
        for item in assessment.get(sect, []):
            fix_item(item, sect)

    # Drop AI hypotheses that became NOT_APPLICABLE from the primary list.
    assessment["threat_hypotheses"] = [
        h for h in assessment.get("threat_hypotheses", [])
        if not (h.get("ai_specific") and h.get("classification") == NOT_APPLICABLE)
    ]
    return notes


# --------------------------------------------------------------------------- #
# Assemble the assessment
# --------------------------------------------------------------------------- #
def _assemble(summary: Dict[str, Any], source: str, run_id: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
    evidence = build_evidence(summary)
    types = classify_target_types(summary, evidence)
    tnames = [t["type"] for t in types]
    ai_surface = detect_ai_surface(summary, evidence)
    caps = _derive_capabilities(summary, evidence, ai_surface, tnames)

    observations = _observations(summary, evidence, types)
    inferences = _inferences(summary, evidence)
    hyps = _threat_hypotheses(summary, evidence, tnames, ai_surface, caps)
    exps = _suggested_experiments(summary, evidence, tnames, ai_surface, caps)
    validated = _validated_findings(evidence, summary)
    tm = propose_threat_model(summary)
    atlas = _atlas_section(ai_surface, exps)

    abstention = None
    if ai_surface["status"] == "NOT_OBSERVED":
        abstention = ("Insufficient evidence to determine whether this target exposes an "
                      "AI/ML attack surface. If the engagement concerns AI systems, identify "
                      "the actual model API, chatbot, RAG pipeline or agent environment first.")

    assessment = {
        "source": source,
        "recon_run": summary.get("display_id"),
        "recon_run_id": run_id,
        "target": summary.get("target", "unknown"),
        "summary": summary,
        "evidence": evidence,
        "target_types": types,
        "ai_surface": ai_surface,
        "observations": observations,
        "inferences": inferences,
        "threat_hypotheses": hyps,
        "suggested_experiments": exps,
        "validated_findings": validated,
        "threat_model": tm,
        "mitre_atlas": atlas,
        "atlas_count": len(atlas["mappings"]),
        "abstention": abstention,
        "next_action": "Human review required",
        "approved": {"threat_model": False, "experiment_plan": False},
        "limitations": [
            "Proposed artifacts are derived from recon evidence by rules and are not confirmed.",
            "A hypothesis is not a finding; a suggested experiment is not evidence of vulnerability.",
        ],
    }
    assessment.update(extra)
    assessment["validation_notes"] = validate_analysis(assessment)
    # Recompute atlas gating post-validation (AI surface may bar it).
    assessment["mitre_atlas"] = _atlas_section(assessment["ai_surface"], assessment["suggested_experiments"])
    assessment["atlas_count"] = len(assessment["mitre_atlas"]["mappings"])
    return assessment


def _normalize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(summary)
    s.setdefault("endpoints", [])
    s.setdefault("auth_indicators", [])
    s.setdefault("screenshots", [])
    s.setdefault("findings", [])
    s.setdefault("report_markdown", "")
    s.setdefault("target", "unknown")
    s.setdefault("objective", "")
    s["counts"] = {
        "endpoints": len(s["endpoints"]), "findings": len(s["findings"]),
        "screenshots": len(s["screenshots"]), "auth_flows": len(s["auth_indicators"]),
    }
    return s


def build_assessment_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a full assessment from a raw recon-style summary (no live run)."""
    return _assemble(_normalize_summary(summary), "summary", None, {})


def build_assessment(run_id: str) -> Optional[Dict[str, Any]]:
    summary = summarize_recon(run_id)
    if summary is None:
        return None
    assessment = _assemble(summary, "recon", run_id, {})
    ANALYSES[run_id] = assessment
    return assessment


def build_assessment_from_context(context: str, target: str = "manual-context") -> Dict[str, Any]:
    """Analysis without recon: evidence is only the user-declared context text."""
    text = context or ""
    # Treat the declared context as a single, weak evidence source.
    summary = {
        "run_id": "", "display_id": display_run_id("manual", "CTX"), "status": "manual",
        "objective": "manual target context", "target": target,
        "endpoints": [], "auth_indicators": [], "screenshots": [], "findings": [],
        "report_markdown": text,
        "counts": {"endpoints": 0, "findings": 0, "screenshots": 0, "auth_flows": 0},
    }
    assessment = _assemble(summary, "context", None, {"context": context})
    assessment["limitations"].insert(
        0, "No recon evidence loaded; this assessment is from user-provided context only.")
    key = "ctx:" + target
    ANALYSES[key] = assessment
    return assessment


def approve_threat_model(run_id: str, approved: bool = True) -> Optional[Dict[str, Any]]:
    a = ANALYSES.get(run_id) or build_assessment(run_id)
    if a is None:
        return None
    a["approved"]["threat_model"] = bool(approved)
    a["threat_model"]["status"] = "APPROVED" if approved else "REJECTED"
    return a
