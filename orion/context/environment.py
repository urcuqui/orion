"""Environment Profile: the normalized model of the terrain around a target.

    Recon collects -> observations -> environment evidence -> Environment Profile.

Recon is a *data source*, not the environment model. This module reuses the
existing recon workflow output (via ``orion.target_analysis.summarize_recon``)
and the evidence-grounded signal detection, and normalizes it into a first-class
Environment Profile — assets, technologies, AI dependencies, MCP/tools,
identity context and trust relationships. No new reconnaissance engine.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_env_id() -> str:
    return "ORN-ENV-" + uuid.uuid4().hex[:8].upper()


# AI-dependency detection: tech/keyword -> (display name, dependency type).
_AI_DEP_MAP = [
    ("openai", "OpenAI", "MODEL_PROVIDER"),
    ("anthropic", "Anthropic", "MODEL_PROVIDER"),
    ("huggingface", "Hugging Face", "MODEL_PROVIDER"),
    ("ollama", "Ollama", "MODEL_SERVER"),
    ("vllm", "vLLM", "MODEL_SERVER"),
    ("triton", "Triton Inference Server", "MODEL_SERVER"),
    ("torchserve", "TorchServe", "MODEL_SERVER"),
    ("tensorflow serving", "TensorFlow Serving", "MODEL_SERVER"),
    ("deepseek", "DeepSeek", "MODEL_PROVIDER"),
    ("whiterabbitneo", "WhiteRabbitNeo", "MODEL_PROVIDER"),
    ("chromadb", "ChromaDB", "VECTOR_STORE"),
    ("pinecone", "Pinecone", "VECTOR_STORE"),
    ("weaviate", "Weaviate", "VECTOR_STORE"),
    ("vector database", "Vector store", "VECTOR_STORE"),
    ("vector store", "Vector store", "VECTOR_STORE"),
    ("langchain", "LangChain", "AGENT_FRAMEWORK"),
    ("langgraph", "LangGraph", "AGENT_FRAMEWORK"),
    ("llamaindex", "LlamaIndex", "AGENT_FRAMEWORK"),
    ("model context protocol", "MCP server", "MCP_SERVER"),
    ("mcp", "MCP server", "MCP_SERVER"),
    ("retrieval-augmented", "RAG source", "RAG_SOURCE"),
    ("rag pipeline", "RAG source", "RAG_SOURCE"),
    ("embedding", "Embedding provider", "EMBEDDING_PROVIDER"),
]


@dataclass
class EnvironmentProfile:
    environment_profile_id: str = field(default_factory=new_env_id)
    target_reference: str = ""
    recon_run_id: Optional[str] = None
    assets: List[Dict[str, Any]] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    apis: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    ai_dependencies: List[Dict[str, Any]] = field(default_factory=list)
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    identity_context: List[Dict[str, Any]] = field(default_factory=list)
    trust_relationships: List[Dict[str, Any]] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def counts(self) -> Dict[str, int]:
        return {
            "assets": len(self.assets), "endpoints": len(self.endpoints),
            "technologies": len(self.technologies), "external_services": len(self.services),
            "ai_dependencies": len(self.ai_dependencies), "mcp_servers": len(self.mcp_servers),
            "trust_relationships": len(self.trust_relationships), "findings": len(self.findings),
        }

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["counts"] = self.counts()
        d["topology"] = topology(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentProfile":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})


def build_environment_profile(summary: Dict[str, Any], target_reference: str = "",
                              recon_run_id: Optional[str] = None) -> EnvironmentProfile:
    """Normalize a recon/probe summary into an Environment Profile (no re-recon)."""
    from orion.target_analysis import build_evidence, detect_ai_surface

    evidence = build_evidence(summary)
    endpoints = list(summary.get("endpoints", []) or [])
    report = str(summary.get("report_markdown", "") or "")
    low = (report + " " + " ".join(endpoints)).lower()

    # Technologies from the recon report.
    from orion.target_analysis.analysis import _extract_technologies
    technologies = _extract_technologies(report)

    # APIs vs services from endpoints.
    apis = [e for e in endpoints if any(k in e.lower() for k in ("/api", "/v1", "/v2", "graphql", "swagger"))]
    services = [e for e in endpoints if e not in apis]

    # AI dependency inventory (dedup by (name,type)).
    ai_deps: List[Dict[str, Any]] = []
    seen = set()
    for kw, name, dtype in _AI_DEP_MAP:
        hit_ids = [e["id"] for e in evidence if kw in e["value"].lower()]
        if kw in low or hit_ids:
            key = (name, dtype)
            if key in seen:
                continue
            seen.add(key)
            ai_deps.append({
                "name": name, "type": dtype,
                "evidence": hit_ids, "trust": "UNKNOWN",
                "authentication": "UNKNOWN",
                "source": "ENVIRONMENT_PROFILE",
                "confidence": "HIGH" if hit_ids else "MEDIUM",
            })

    mcp_servers = [d for d in ai_deps if d["type"] == "MCP_SERVER"]
    tools = [{"name": e, "source": "endpoint"} for e in endpoints if "tool" in e.lower() or "mcp" in e.lower()]

    # Identity / auth context.
    auth = summary.get("auth_indicators", []) or []
    identity_context = [{"type": "AUTHENTICATED", "evidence": a} for a in auth]
    if not identity_context and endpoints:
        identity_context = [{"type": "PUBLIC", "evidence": "reachable without observed auth"}]
    if not endpoints and not auth:
        identity_context = [{"type": "UNKNOWN", "evidence": None}]

    # Assets = endpoints + findings (named observations).
    assets = [{"name": e, "kind": "endpoint"} for e in endpoints]
    findings = list(summary.get("findings", []) or [])
    for f in findings:
        assets.append({"name": f.get("title", "finding"), "kind": "finding",
                       "severity": f.get("severity", "info")})

    trust = _trust_relationships(ai_deps, endpoints)

    return EnvironmentProfile(
        target_reference=target_reference or summary.get("target", ""),
        recon_run_id=recon_run_id,
        assets=assets, endpoints=endpoints, technologies=technologies,
        services=services, apis=apis,
        external_dependencies=[d["name"] for d in ai_deps if d["type"] in ("MODEL_PROVIDER", "AI_API")],
        ai_dependencies=ai_deps, mcp_servers=mcp_servers, tools=tools,
        identity_context=identity_context, trust_relationships=trust,
        screenshots=list(summary.get("screenshots", []) or []),
        findings=findings,
        evidence_ids=[e["id"] for e in evidence],
    )


def environment_to_summary(profile: "EnvironmentProfile") -> Dict[str, Any]:
    """Convert an Environment Profile back into an analysis summary so the
    evidence-grounded analyzer can let the environment change applicability."""
    report_bits = [f"environment of {profile.target_reference}"]
    report_bits += profile.technologies
    for d in profile.ai_dependencies:
        report_bits.append(d["name"])
        report_bits.append(d["type"].lower().replace("_", " "))
    return {
        "target": profile.target_reference or "environment",
        "objective": "environment profile",
        "endpoints": list(profile.endpoints),
        "auth_indicators": [i.get("evidence", "") for i in profile.identity_context
                            if i.get("type") == "AUTHENTICATED"],
        "screenshots": list(profile.screenshots),
        "findings": list(profile.findings),
        "report_markdown": " \n ".join(str(b) for b in report_bits if b),
        "counts": {"endpoints": len(profile.endpoints), "findings": len(profile.findings),
                   "screenshots": len(profile.screenshots), "auth_flows": 0},
    }


def _trust_relationships(ai_deps: List[Dict[str, Any]], endpoints: List[str]) -> List[Dict[str, Any]]:
    """Derive evidence-based trust relationships from observed dependencies."""
    rels: List[Dict[str, Any]] = []

    def add(src, dst, rel, auth, trust, ev):
        rels.append({"source": src, "destination": dst, "relationship": rel,
                     "authentication": auth, "trust": trust, "evidence_ids": ev,
                     "inferred": not ev})

    by_type = {d["type"]: d for d in ai_deps}
    if "MODEL_PROVIDER" in by_type or "MODEL_SERVER" in by_type:
        d = by_type.get("MODEL_PROVIDER") or by_type["MODEL_SERVER"]
        add("application", d["name"], "inference", "UNKNOWN", "PARTIAL", d.get("evidence", []))
    if "VECTOR_STORE" in by_type:
        add("rag", by_type["VECTOR_STORE"]["name"], "retrieval", "UNKNOWN", "PARTIAL",
            by_type["VECTOR_STORE"].get("evidence", []))
    if "AGENT_FRAMEWORK" in by_type:
        add("application", by_type["AGENT_FRAMEWORK"]["name"], "orchestration", "UNKNOWN", "PARTIAL",
            by_type["AGENT_FRAMEWORK"].get("evidence", []))
    if "MCP_SERVER" in by_type:
        add("agent", by_type["MCP_SERVER"]["name"], "tool_access", "service_account", "PARTIAL",
            by_type["MCP_SERVER"].get("evidence", []))
    return rels


def topology(profile: "EnvironmentProfile") -> Dict[str, Any]:
    """Lightweight evidence-based topology (nodes + edges). No graph framework."""
    nodes = [{"id": "user", "label": "USER"}, {"id": "application", "label": "APPLICATION"}]
    edges = [{"source": "user", "destination": "application", "inferred": True}]
    seen = {"user", "application"}
    for d in profile.ai_dependencies:
        nid = d["name"]
        if nid not in seen:
            nodes.append({"id": nid, "label": d["name"], "type": d["type"]})
            seen.add(nid)
    for r in profile.trust_relationships:
        for n in (r["source"], r["destination"]):
            if n not in seen:
                nodes.append({"id": n, "label": n})
                seen.add(n)
        edges.append({"source": r["source"], "destination": r["destination"],
                      "relationship": r["relationship"], "inferred": r.get("inferred", True)})
    return {"nodes": nodes, "edges": edges}
