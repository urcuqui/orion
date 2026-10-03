"""Correlate Self + Target + Environment profiles into one Analysis Context.

Attack applicability and the threat model derive from whatever profiles are
present; each conclusion carries its **source profile** (SELF / TARGET /
ENVIRONMENT) and evidence. Unknown adversary context stays UNKNOWN — Orion does
not invent it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SELF, TARGET, ENVIRONMENT = "SELF", "TARGET", "ENVIRONMENT"


def _derive_caps_and_types(self_p, target_p, env_p):
    caps: Dict[str, List[str]] = {}          # capability -> source tags (evidence)
    types: set = set()
    ai_confirmed = False

    def setcap(cap, src):
        caps.setdefault(cap, [])
        if src not in caps[cap]:
            caps[cap].append(src)

    if self_p:
        st = self_p.get("system_type")
        cp = self_p.get("capabilities", {}) or {}
        if st in ("generative_ai", "hybrid"):
            ai_confirmed = True
            types.add("llm_application"); setcap("llm_interface", SELF)
            if cp.get("retrieval"):
                types.add("rag_application"); setcap("retrieval", SELF); setcap("knowledge_source", SELF)
            if cp.get("tools") or cp.get("mcp"):
                types.add("agentic_application"); setcap("agent_tool", SELF)
            if cp.get("mcp"):
                types.add("mcp_enabled_system")
            if cp.get("persistent_memory"):
                setcap("persistent_memory", SELF)
        if st in ("traditional_ml", "hybrid"):
            ai_confirmed = True
            types.add("ml_inference_service"); setcap("inference_api", SELF)
            if cp.get("supports_gradients"):
                setcap("inference_input", SELF); setcap("model_behavior_observable", SELF)
            if cp.get("query_access"):
                setcap("query_access", SELF)

    if env_p:
        for d in env_p.get("ai_dependencies", []) or []:
            t = d.get("type")
            ai_confirmed = True
            if t == "MCP_SERVER":
                types.add("mcp_enabled_system"); setcap("agent_tool", ENVIRONMENT)
            elif t in ("VECTOR_STORE", "RAG_SOURCE", "MEMORY_STORE", "EMBEDDING_PROVIDER"):
                types.add("rag_application"); setcap("retrieval", ENVIRONMENT); setcap("knowledge_source", ENVIRONMENT)
            elif t in ("MODEL_PROVIDER", "MODEL_SERVER", "AI_API"):
                types.add("ml_inference_service"); setcap("inference_api", ENVIRONMENT); setcap("query_access", ENVIRONMENT)
            elif t in ("AGENT_FRAMEWORK", "EXTERNAL_TOOL"):
                types.add("agentic_application"); setcap("agent_tool", ENVIRONMENT)
        if env_p.get("endpoints"):
            setcap("http_endpoint", ENVIRONMENT); setcap("query_access", ENVIRONMENT)

    if target_p:
        tt = (target_p.get("target_type") or "").lower()
        if "generative" in tt or "llm" in tt or "agent" in tt:
            ai_confirmed = True; types.add("llm_application"); setcap("llm_interface", TARGET)
        if any(k in tt for k in ("ml", "classif", "image", "regress", "vision")):
            ai_confirmed = True; types.add("ml_inference_service")

    return caps, types, ai_confirmed


def build_context_assessment(self_p: Optional[Dict[str, Any]], target_p: Optional[Dict[str, Any]],
                             env_p: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Applicability + source-aware threat model derived from available profiles."""
    from orion.target_analysis import ATTACK_CATALOG, check_applicability

    caps, types, ai_confirmed = _derive_caps_and_types(self_p, target_p, env_p)
    tnames = sorted(types)
    ai_surface = {"status": "CONFIRMED" if ai_confirmed else "NOT_OBSERVED",
                  "evidence": [], "confidence": "HIGH" if ai_confirmed else "NONE"}

    experiments: List[Dict[str, Any]] = []
    idx = 1
    for attack in ATTACK_CATALOG:
        appl = check_applicability(attack, tnames, ai_surface, caps)
        # Source profiles = those that supplied the satisfied prerequisites.
        srcs: set = set()
        for req in attack.get("requires", []):
            for s in caps.get(req, []):
                srcs.add(s)
        experiments.append({
            "id": f"EXP-{idx:02d}", "key": attack["id"], "name": attack["name"],
            "ai_specific": attack["ai_specific"], "applicability": appl["status"],
            "rationale": appl["reason"], "missing": appl["missing"],
            "source_profiles": sorted(srcs), "evidence": appl["evidence"],
        })
        idx += 1
    experiments.sort(key=lambda e: {"APPLICABLE": 0, "INSUFFICIENT_EVIDENCE": 1,
                                    "NOT_APPLICABLE": 2}.get(e["applicability"], 3))

    return {
        "system_types": tnames,
        "ai_surface": ai_surface,
        "capabilities": {k: sorted(set(v)) for k, v in caps.items()},
        "suggested_experiments": experiments,
        "threat_model": build_context_threat_model(self_p, target_p, env_p, ai_confirmed),
    }


def build_context_threat_model(self_p, target_p, env_p, ai_confirmed: bool) -> Dict[str, Any]:
    """Source-aware threat model. Every item records its origin profile."""
    assets: List[Dict[str, Any]] = []
    boundaries: List[Dict[str, Any]] = []
    objectives: List[Dict[str, Any]] = []

    if env_p:
        for d in env_p.get("ai_dependencies", []) or []:
            assets.append({"asset": d.get("name"), "kind": d.get("type"),
                           "source": ENVIRONMENT, "evidence": d.get("evidence", [])})
        for r in env_p.get("trust_relationships", []) or []:
            boundaries.append({"boundary": f"{r.get('source')} -> {r.get('destination')}",
                               "source": ENVIRONMENT, "evidence": r.get("evidence_ids", []),
                               "trust": r.get("trust")})
    if self_p:
        st = self_p.get("system_type")
        cp = self_p.get("capabilities", {}) or {}
        if st in ("generative_ai", "hybrid"):
            boundaries.append({"boundary": "User -> Model", "source": SELF, "evidence": []})
            if cp.get("retrieval"):
                boundaries.append({"boundary": "Retrieved Context -> Agent", "source": SELF, "evidence": []})
            if cp.get("tools") or cp.get("mcp"):
                boundaries.append({"boundary": "Agent -> Tool", "source": SELF, "evidence": []})
        if st in ("traditional_ml", "hybrid"):
            assets.append({"asset": "model integrity", "kind": "model", "source": SELF, "evidence": []})
    if target_p:
        if target_p.get("security_objective"):
            objectives.append({"objective": target_p["security_objective"], "source": TARGET})
        if target_p.get("objective"):
            objectives.append({"objective": target_p["objective"], "source": TARGET})

    has_endpoint = bool((env_p or {}).get("endpoints"))
    adversary = {
        "goal": "UNDEFINED", "knowledge": "UNKNOWN", "budget": "UNKNOWN",
        "credential_access": "NOT_OBSERVED",
        "network_access": "REMOTE_PUBLIC" if has_endpoint else "UNKNOWN",
    }
    return {"assets": assets, "trust_boundaries": boundaries, "objectives": objectives,
            "adversary": adversary,
            "note": "Adversary properties require analyst input; unknown stays unknown."}
