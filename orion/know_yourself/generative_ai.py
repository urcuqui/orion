"""Generative AI profiling: system-centric (prompts, context, memory, tools, identity)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orion.know_yourself.common import (
    ENABLED, PARTIAL, NOT_FOUND, NOT_APPLICABLE, UNKNOWN, OBSERVED, INFERRED,
)


@dataclass
class SystemFingerprint:
    system_type: str = "generative_ai"
    subtype: str = "llm_application"      # rag_application / agentic_application / agentic_rag_application
    model_provider: str = "unknown"
    model: str = "unknown"
    rag: bool = False
    persistent_memory: bool = False
    tools: int = 0
    mcp: bool = False
    external_actions: bool = False
    human_approval: str = "UNKNOWN"       # none / partial / full / UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def build_system_fingerprint(summary: Dict[str, Any], descriptor: Optional[Dict[str, Any]],
                             caps: Dict[str, Any]) -> SystemFingerprint:
    d = descriptor or {}
    fp = SystemFingerprint()
    fp.rag = bool(caps.get("retrieval"))
    fp.persistent_memory = bool(caps.get("persistent_memory"))
    fp.mcp = bool(caps.get("mcp"))
    fp.external_actions = bool(caps.get("external_actions"))
    tools = d.get("tools")
    fp.tools = len(tools) if isinstance(tools, list) else int(tools or (1 if caps.get("tools") else 0))
    agentic = caps.get("tools") or caps.get("mcp")
    if fp.rag and agentic:
        fp.subtype = "agentic_rag_application"
    elif agentic:
        fp.subtype = "agentic_application"
    elif fp.rag:
        fp.subtype = "rag_application"
    else:
        fp.subtype = "llm_application"
    fp.model_provider = d.get("model_provider", _provider_from_text(summary))
    fp.model = d.get("model", "unknown")
    fp.human_approval = str(d.get("human_approval", "UNKNOWN"))
    return fp


def _provider_from_text(summary: Dict[str, Any]) -> str:
    hay = str(summary.get("report_markdown", "")).lower()
    for kw, prov in (("openai", "openai"), ("anthropic", "anthropic"), ("ollama", "local"),
                     ("deepseek", "local"), ("whiterabbitneo", "local"), ("langchain", "local"),
                     ("langgraph", "local")):
        if kw in hay:
            return prov
    return "unknown"


# --------------------------------------------------------------------------- #
# Prompt / context surface — classify each source's trust
# --------------------------------------------------------------------------- #
def context_surface(fp: SystemFingerprint) -> List[Dict[str, str]]:
    sources = [
        {"source": "system_prompt", "trust": "TRUSTED"},
        {"source": "user_input", "trust": "UNTRUSTED"},
    ]
    if fp.rag:
        sources.append({"source": "retrieved_context", "trust": "UNTRUSTED"})
    if fp.persistent_memory:
        sources.append({"source": "memory", "trust": "MIXED"})
    if fp.tools:
        sources.append({"source": "tool_output", "trust": "UNTRUSTED"})
    if fp.mcp:
        sources.append({"source": "mcp_server", "trust": "UNKNOWN"})
    return sources


# --------------------------------------------------------------------------- #
# Trust boundaries — a first-class capability for GenAI
# --------------------------------------------------------------------------- #
def trust_boundaries(fp: SystemFingerprint) -> List[Dict[str, Any]]:
    b: List[Dict[str, Any]] = []

    def add(name, data, trust, impact, evidence):
        b.append({"boundary": name, "data_crossing": data, "trust": trust,
                  "controls": "UNKNOWN", "impact": impact, "evidence": evidence})

    add("User -> Model", "user prompt", "UNTRUSTED", "prompt injection", ["user_input"])
    if fp.rag:
        add("Retrieved Context -> Model", "documents/chunks", "UNTRUSTED",
            "indirect prompt injection / RAG poisoning", ["retrieval"])
    if fp.persistent_memory:
        add("Memory -> Future Decision", "stored state", "MIXED",
            "persistent attacker influence", ["persistent_memory"])
    if fp.tools:
        add("Agent -> Tool", "tool arguments", "UNTRUSTED", "tool misuse / overprivilege", ["tools"])
        add("Tool -> Agent", "tool output", "UNTRUSTED", "output-driven hijack", ["tools"])
    if fp.mcp:
        add("MCP Server -> Agent", "tool definitions/output", "UNKNOWN",
            "tool poisoning via MCP", ["mcp"])
    if fp.external_actions:
        add("Agent -> External System", "actions", "SENSITIVE", "real-world side effects", ["external_actions"])
    return b


def tool_inventory(descriptor: Optional[Dict[str, Any]], fp: SystemFingerprint) -> List[Dict[str, Any]]:
    tools = (descriptor or {}).get("tools")
    if isinstance(tools, list):
        out = []
        for t in tools:
            if isinstance(t, dict):
                out.append({"name": t.get("name", "tool"), "capability": t.get("capability", "UNKNOWN"),
                            "permissions": t.get("permissions", "UNKNOWN"),
                            "sensitive": bool(t.get("sensitive", False)),
                            "external_side_effects": bool(t.get("external_side_effects", False)),
                            "approval_required": t.get("approval_required", "UNKNOWN"),
                            "trust": t.get("trust", "UNKNOWN")})
            else:
                out.append({"name": str(t), "capability": "UNKNOWN", "permissions": "UNKNOWN",
                            "sensitive": False, "external_side_effects": False,
                            "approval_required": "UNKNOWN", "trust": "UNKNOWN"})
        return out
    if fp.tools:
        return [{"name": f"tool_{i+1}", "capability": "UNKNOWN", "permissions": "UNKNOWN",
                 "sensitive": False, "external_side_effects": fp.external_actions,
                 "approval_required": "UNKNOWN", "trust": "UNKNOWN"} for i in range(fp.tools)]
    return []


def identity_profile(descriptor: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    d = (descriptor or {}).get("identity", {}) or {}
    fields = ["agent_identity", "credentials", "scopes", "delegated_permissions",
              "tenant_boundaries", "standing_privilege", "just_in_time_privilege"]
    out = []
    for f in fields:
        # Never expose secrets: only the status, never the value.
        status = OBSERVED if f in d else UNKNOWN
        out.append({"field": f, "status": status})
    return out


_GEN_CONTROLS = [
    "prompt_context_separation", "retrieval_provenance", "input_validation",
    "output_validation", "tool_authorization", "argument_validation",
    "least_privilege", "human_approval", "identity_verification",
    "mcp_trust_verification", "memory_isolation", "memory_expiration",
    "audit_logging", "egress_controls", "sandboxing",
]


def control_inventory(descriptor: Optional[Dict[str, Any]], fp: SystemFingerprint) -> List[Dict[str, str]]:
    d = (descriptor or {}).get("controls", {}) or {}
    out: List[Dict[str, str]] = []
    for c in _GEN_CONTROLS:
        if c in d:
            status = str(d[c]).upper()
            if status not in (ENABLED, PARTIAL, NOT_FOUND, NOT_APPLICABLE, UNKNOWN):
                status = ENABLED if d[c] else NOT_FOUND
        elif c == "human_approval" and fp.human_approval not in ("UNKNOWN", ""):
            status = {"full": ENABLED, "partial": PARTIAL, "none": NOT_FOUND}.get(
                fp.human_approval.lower(), UNKNOWN)
        elif c in ("retrieval_provenance", "memory_isolation", "memory_expiration") and not (fp.rag or fp.persistent_memory):
            status = NOT_APPLICABLE
        elif c in ("tool_authorization", "argument_validation") and not fp.tools:
            status = NOT_APPLICABLE
        elif c in ("mcp_trust_verification",) and not fp.mcp:
            status = NOT_APPLICABLE
        else:
            status = UNKNOWN
        out.append({"control": c, "status": status})
    return out
