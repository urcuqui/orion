"""Shared Know Yourself types, system-type detection, and the posture engine.

Know Yourself characterizes an AI system *before* any attack:

    Profile -> Baseline -> Surface -> Controls -> Posture -> Experiments

Core principles: understand before attacking; profile before testing; unknown is
a valid answer; applicability before execution; recommended attacks are
proposals, not findings. Traditional ML is model-centric; Generative AI is
system-centric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---- system types ---------------------------------------------------------- #
TRADITIONAL_ML = "traditional_ml"
GENERATIVE_AI = "generative_ai"
HYBRID = "hybrid"
UNKNOWN_SYSTEM = "unknown"

# ---- status vocabularies --------------------------------------------------- #
# Provenance of an observation/assumption.
OBSERVED, INFERRED, UNKNOWN = "OBSERVED", "INFERRED", "UNKNOWN"
# Control inventory status.
ENABLED, PARTIAL, NOT_FOUND, NOT_APPLICABLE = "ENABLED", "PARTIAL", "NOT_FOUND", "NOT_APPLICABLE"
# Attack applicability.
APPLICABLE, CONDITIONAL, NA, INSUFFICIENT = "APPLICABLE", "CONDITIONAL", "NOT_APPLICABLE", "INSUFFICIENT_EVIDENCE"
# Confidence.
HIGH, MEDIUM, LOW, NONE = "HIGH", "MEDIUM", "LOW", "NONE"


@dataclass
class Item:
    """A classified statement with evidence (observation / assumption / etc.)."""
    statement: str
    classification: str = OBSERVED
    confidence: str = MEDIUM
    evidence: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "classification": self.classification,
                "confidence": self.confidence, "evidence": self.evidence, "rationale": self.rationale}


@dataclass
class PostureItem:
    """Applicability of one attack against the profiled system."""
    attack: str
    status: str = NA
    branch: str = TRADITIONAL_ML
    rationale: str = ""
    evidence: List[str] = field(default_factory=list)
    prerequisites_met: List[str] = field(default_factory=list)
    prerequisites_missing: List[str] = field(default_factory=list)
    risk: str = "LOCAL / SAFE"

    def to_dict(self) -> Dict[str, Any]:
        return {"attack": self.attack, "status": self.status, "branch": self.branch,
                "rationale": self.rationale, "evidence": self.evidence,
                "prerequisites_met": self.prerequisites_met,
                "prerequisites_missing": self.prerequisites_missing, "risk": self.risk}


# --------------------------------------------------------------------------- #
# System-type auto-detection (reuses the evidence-grounded target_analysis)
# --------------------------------------------------------------------------- #
_GEN_TYPES = {"llm_application", "rag_application", "agentic_application", "mcp_enabled_system"}
_TRAD_TYPES = {"ml_inference_service"}


def detect_system_type(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a profiled system as traditional_ml / generative_ai / hybrid / unknown."""
    from orion.target_analysis import classify_target_types, detect_ai_surface, build_evidence

    evidence = build_evidence(summary)
    types = classify_target_types(summary, evidence)
    ai = detect_ai_surface(summary, evidence)
    tnames = {t["type"] for t in types}

    is_gen = bool(tnames & _GEN_TYPES)
    is_trad = bool(tnames & _TRAD_TYPES)
    # Confirmed AI image/tabular classifier with no gen markers is traditional.
    if ai["status"] == "CONFIRMED" and not is_gen and not is_trad:
        is_trad = True

    if is_gen and is_trad:
        system_type, conf = HYBRID, MEDIUM
    elif is_gen:
        system_type, conf = GENERATIVE_AI, (HIGH if ai["status"] == "CONFIRMED" else MEDIUM)
    elif is_trad:
        system_type, conf = TRADITIONAL_ML, (HIGH if ai["status"] == "CONFIRMED" else MEDIUM)
    else:
        system_type, conf = UNKNOWN_SYSTEM, NONE

    rationale = {
        TRADITIONAL_ML: "Model-centric predictive ML evidence (classifier/inference service).",
        GENERATIVE_AI: "System-centric generative evidence (LLM / RAG / agent / MCP).",
        HYBRID: "Both predictive-ML and generative-AI evidence observed.",
        UNKNOWN_SYSTEM: "Insufficient evidence to classify the system type. Manual classification required.",
    }[system_type]
    return {"system_type": system_type, "confidence": conf,
            "evidence": ai.get("evidence", []),
            "target_types": types, "ai_surface": ai, "rationale": rationale}


# --------------------------------------------------------------------------- #
# Capabilities derived from evidence + an optional analyst descriptor
# --------------------------------------------------------------------------- #
def derive_capabilities(summary: Dict[str, Any], descriptor: Optional[Dict[str, Any]],
                        system_type: str) -> Dict[str, Any]:
    """Facts that gate attack applicability. UNKNOWN stays UNKNOWN."""
    d = descriptor or {}
    hay = " ".join(str(summary.get(k, "")) for k in ("report_markdown", "target", "objective")).lower()
    endpoints = summary.get("endpoints", []) or []

    # Access / gradients: a local artifact => white-box; a remote service => black-box.
    access = d.get("access")
    if not access:
        if d.get("artifact") or d.get("model_path"):
            access = "white_box"
        elif endpoints or "://" in str(summary.get("target", "")):
            access = "black_box"
        else:
            access = "unknown"
    gradients = d.get("supports_gradients")
    if gradients is None:
        gradients = (access == "white_box")

    retrieval = bool(d.get("rag") or any(k in hay for k in ("rag", "retrieval", "vector", "embedding")))
    tools = d.get("tools")
    has_tools = bool(tools) or any(k in hay for k in ("tool", "function calling"))
    mcp = bool(d.get("mcp") or "mcp" in hay)
    memory = bool(d.get("persistent_memory") or "memory" in hay)
    llm = system_type in (GENERATIVE_AI, HYBRID)

    return {
        "access": access,
        "supports_gradients": bool(gradients),
        "query_access": bool(endpoints) or access in ("black_box", "white_box"),
        "input_type": d.get("input_type") or _infer_input_type(hay),
        "retrieval": retrieval,
        "tools": has_tools,
        "mcp": mcp,
        "persistent_memory": memory,
        "llm_interface": llm,
        "external_actions": bool(d.get("external_actions") or has_tools or mcp),
        "training_data_known": d.get("training_data_known", None),  # None => UNKNOWN
    }


def _infer_input_type(hay: str) -> str:
    if any(k in hay for k in ("image", "vision", "face", "photo")):
        return "image"
    if any(k in hay for k in ("tabular", "csv", "feature")):
        return "tabular"
    if any(k in hay for k in ("text", "nlp", "sentence", "token")):
        return "text"
    return "unknown"


# --------------------------------------------------------------------------- #
# Deterministic attack-applicability (security posture)
# --------------------------------------------------------------------------- #
def build_posture(caps: Dict[str, Any], system_type: str) -> List[PostureItem]:
    """Evidence-based applicability of each attack against the profiled system."""
    items: List[PostureItem] = []
    trad = system_type in (TRADITIONAL_ML, HYBRID)
    gen = system_type in (GENERATIVE_AI, HYBRID)
    grad = caps.get("supports_gradients")
    query = caps.get("query_access")
    acc = caps.get("access")
    ev_grad = ["access:white_box"] if grad else []
    ev_query = ["query_access"] if query else []

    def ev(*flags):
        out = []
        for f in flags:
            if f:
                out.append(f)
        return out

    # ---- Traditional ML / evasion family ----
    for name in ("FGSM", "PGD", "C&W", "DeepFool"):
        it = PostureItem(attack=f"{name} evasion", branch=TRADITIONAL_ML)
        if not trad:
            it.status, it.rationale = NA, "No traditional-ML inference surface."
            it.prerequisites_missing = ["traditional_ml"]
        elif grad:
            it.status = APPLICABLE
            it.rationale = "White-box gradients available for optimization-based evasion."
            it.prerequisites_met = ["traditional_ml", "gradients"]
            it.evidence = ev_grad
        elif query:
            it.status = CONDITIONAL
            it.rationale = "No gradients; black-box (score/decision-based) evasion only, higher query cost."
            it.prerequisites_met = ["traditional_ml", "query_access"]
            it.prerequisites_missing = ["gradients"]
            it.evidence = ev_query
        else:
            it.status = INSUFFICIENT
            it.rationale = "No gradient or query path established."
            it.prerequisites_missing = ["gradients", "query_access"]
        items.append(it)

    # Model extraction
    me = PostureItem(attack="Model extraction", branch=TRADITIONAL_ML)
    if (trad or gen) and query:
        me.status, me.rationale = CONDITIONAL, "Query access enables extraction; cost depends on output granularity."
        me.prerequisites_met, me.evidence = ["query_access"], ev_query
    else:
        me.status, me.rationale = NA, "No query access to the model."
        me.prerequisites_missing = ["query_access"]
    items.append(me)

    # Membership inference
    mi = PostureItem(attack="Membership inference", branch=TRADITIONAL_ML)
    if trad and (query or caps.get("training_data_known") is None):
        mi.status = CONDITIONAL
        mi.rationale = "Query access and/or unknown training-data provenance make this worth testing."
        mi.prerequisites_met = ev("query_access" if query else "", "training_provenance_unknown")
        mi.evidence = ev_query
    else:
        mi.status = NA
        mi.rationale = "Not a predictive model with a query path."
        mi.prerequisites_missing = ["traditional_ml", "query_access"]
    items.append(mi)

    # ---- Generative AI family ----
    pi = PostureItem(attack="Prompt injection", branch=GENERATIVE_AI)
    if gen and caps.get("llm_interface"):
        pi.status, pi.rationale = APPLICABLE, "LLM/prompt surface present."
        pi.prerequisites_met, pi.evidence = ["llm_interface"], ["llm_interface"]
    else:
        pi.status, pi.rationale = NA, "No LLM/prompt surface."
        pi.prerequisites_missing = ["llm_interface"]
    items.append(pi)

    rp = PostureItem(attack="RAG poisoning", branch=GENERATIVE_AI)
    if gen and caps.get("retrieval"):
        rp.status, rp.rationale = APPLICABLE, "Retrieval/RAG pipeline observed; untrusted context can influence output."
        rp.prerequisites_met, rp.evidence = ["retrieval"], ["retrieval"]
    else:
        rp.status, rp.rationale = NA, "No retrieval/RAG pipeline."
        rp.prerequisites_missing = ["retrieval"]
    items.append(rp)

    tp = PostureItem(attack="Tool / MCP poisoning", branch=GENERATIVE_AI)
    if gen and (caps.get("tools") or caps.get("mcp")):
        tp.status, tp.rationale = APPLICABLE, "Tool/MCP surface with potential external side effects."
        tp.prerequisites_met = ev("tools" if caps.get("tools") else "", "mcp" if caps.get("mcp") else "")
        tp.evidence = tp.prerequisites_met
    else:
        tp.status, tp.rationale = NA, "No tool/MCP surface."
        tp.prerequisites_missing = ["tools_or_mcp"]
    items.append(tp)

    mp = PostureItem(attack="Memory poisoning", branch=GENERATIVE_AI)
    if gen and caps.get("persistent_memory"):
        mp.status, mp.rationale = CONDITIONAL, "Persistent memory can retain attacker influence across sessions."
        mp.prerequisites_met, mp.evidence = ["persistent_memory"], ["persistent_memory"]
    else:
        mp.status, mp.rationale = NA, "No persistent memory."
        mp.prerequisites_missing = ["persistent_memory"]
    items.append(mp)

    return items


def recommended_experiments(posture: List[PostureItem], unknowns: List[Item]) -> List[Dict[str, Any]]:
    """Propose next experiments from APPLICABLE/CONDITIONAL posture + unknowns.

    Proposals, not findings; never auto-executed.
    """
    recs: List[Dict[str, Any]] = []
    order = {APPLICABLE: 0, CONDITIONAL: 1}
    for it in sorted([p for p in posture if p.status in order], key=lambda p: order[p.status]):
        recs.append({
            "attack": it.attack,
            "applicability": "HIGH" if it.status == APPLICABLE else "CONDITIONAL",
            "reason": it.rationale,
            "evidence_count": len(it.evidence),
            "risk": it.risk,
            "branch": it.branch,
        })
    # Surface a privacy test when training provenance is unknown.
    if any("training" in u.statement.lower() and u.classification == UNKNOWN for u in unknowns):
        if not any(r["attack"] == "Membership inference" for r in recs):
            recs.append({"attack": "Membership inference", "applicability": "CONDITIONAL",
                         "reason": "Training-data provenance is unknown.", "evidence_count": 0,
                         "risk": "LOCAL / SAFE", "branch": TRADITIONAL_ML})
    return recs
