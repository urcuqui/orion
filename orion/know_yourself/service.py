"""Know Yourself orchestration: profile -> baseline -> surface -> controls -> posture."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from orion.know_yourself import common as C
from orion.know_yourself import traditional_ml as T
from orion.know_yourself import generative_ai as G


def _summary_from_input(url: Optional[str], context: Optional[str],
                        descriptor: Optional[Dict[str, Any]], active: bool) -> Dict[str, Any]:
    """Build a recon-style evidence summary from whatever the analyst provided."""
    if url:
        from orion.target_analysis import probe_url
        return probe_url(url, active=active)
    text_parts = []
    if context:
        text_parts.append(context)
    if descriptor:
        # Fold descriptor hints into the text so detection can use them.
        for k in ("task", "input_type", "framework", "model", "model_provider"):
            if descriptor.get(k):
                text_parts.append(str(descriptor[k]))
        if descriptor.get("rag"):
            text_parts.append("rag retrieval vector")
        if descriptor.get("mcp"):
            text_parts.append("mcp")
        if descriptor.get("tools"):
            text_parts.append("tool function calling")
    target = (descriptor or {}).get("model_name") or (context or "manual-profile")[:60]
    return {
        "target": target, "objective": "know-yourself profile",
        "endpoints": [], "auth_indicators": [], "screenshots": [], "findings": [],
        "report_markdown": " \n ".join(text_parts),
        "counts": {"endpoints": 0, "findings": 0, "screenshots": 0, "auth_flows": 0},
    }


def _type_from_descriptor(d: Dict[str, Any]) -> Optional[str]:
    gen = bool(d.get("rag") or d.get("mcp") or d.get("tools") or d.get("persistent_memory")
               or d.get("model_provider"))
    task = str(d.get("task", "")).lower()
    trad = bool(d.get("artifact") or d.get("model_path") or d.get("framework")
                or d.get("input_shape") or d.get("num_classes")
                or d.get("input_type") in ("image", "tabular")
                or any(k in task for k in ("class", "regress", "detection", "recognition")))
    if gen and trad:
        return C.HYBRID
    if gen:
        return C.GENERATIVE_AI
    if trad:
        return C.TRADITIONAL_ML
    return None


def analyze(url: Optional[str] = None, context: Optional[str] = None,
            descriptor: Optional[Dict[str, Any]] = None, active: bool = False,
            force_type: Optional[str] = None, persist: bool = True,
            base_dir: str = "artifacts") -> Dict[str, Any]:
    """Profile an AI system and produce an evidence-based security posture."""
    summary = _summary_from_input(url, context, descriptor, active)

    detection = C.detect_system_type(summary)
    system_type = force_type or detection["system_type"]
    # Descriptor-driven fallback: an explicit descriptor is direct evidence of the
    # system type even when free-text detection is inconclusive.
    if not force_type and system_type == C.UNKNOWN_SYSTEM and descriptor:
        system_type = _type_from_descriptor(descriptor) or system_type
        if system_type != C.UNKNOWN_SYSTEM:
            detection = {**detection, "system_type": system_type, "confidence": C.MEDIUM,
                         "rationale": "Classified from the analyst-provided descriptor."}
    caps = C.derive_capabilities(summary, descriptor, system_type)

    result: Dict[str, Any] = {
        "system_type": system_type,
        "detection": detection,
        "capabilities": caps,
        "traditional_ml": None,
        "generative_ai": None,
    }

    assumptions: List[C.Item] = []

    if system_type in (C.TRADITIONAL_ML, C.HYBRID):
        fp = T.build_model_fingerprint(summary, descriptor, caps)
        tml_assumptions = T.model_assumptions(fp, caps, descriptor)
        assumptions += tml_assumptions
        result["traditional_ml"] = {
            "fingerprint": fp.to_dict(),
            "input_surface": T.input_surface(fp, descriptor),
            "assumptions": [a.to_dict() for a in tml_assumptions],
            "controls": T.control_inventory(descriptor, summary),
        }

    if system_type in (C.GENERATIVE_AI, C.HYBRID):
        gfp = G.build_system_fingerprint(summary, descriptor, caps)
        result["generative_ai"] = {
            "fingerprint": gfp.to_dict(),
            "context_surface": G.context_surface(gfp),
            "trust_boundaries": G.trust_boundaries(gfp),
            "tools": G.tool_inventory(descriptor, gfp),
            "identity": G.identity_profile(descriptor),
            "controls": G.control_inventory(descriptor, gfp),
        }

    posture = C.build_posture(caps, system_type)
    result["security_posture"] = [p.to_dict() for p in posture]
    result["recommended_experiments"] = C.recommended_experiments(posture, assumptions)
    result["summary_card"] = _summary_card(result, posture)
    result["limitations"] = [
        "Profile reflects observed/declared evidence; UNKNOWN fields require analyst input.",
        "Recommended experiments are proposals, not findings, and are never auto-executed.",
    ]

    if persist:
        result["trace_id"] = _save(result, base_dir)
        # Persist a first-class Self Profile (ORN-SELF-…) for the Analysis Context.
        try:
            from orion.context import build_self_profile, save_profile
            sp = build_self_profile(result)
            save_profile(sp.self_profile_id, sp.to_dict(), base_dir)
            result["self_profile_id"] = sp.self_profile_id
        except Exception:  # noqa: BLE001 - profiling must not fail the analysis
            pass
    return result


def _summary_card(result: Dict[str, Any], posture: List[C.PostureItem]) -> Dict[str, Any]:
    st = result["system_type"]
    applicable = sum(1 for p in posture if p.status == C.APPLICABLE)
    conditional = sum(1 for p in posture if p.status == C.CONDITIONAL)
    card: Dict[str, Any] = {"system_type": st, "applicable_attacks": applicable,
                            "conditional_attacks": conditional}
    if result.get("traditional_ml"):
        fp = result["traditional_ml"]["fingerprint"]
        unknown_assumptions = sum(1 for a in result["traditional_ml"]["assumptions"]
                                  if a["classification"] == C.UNKNOWN)
        unknown_controls = sum(1 for c in result["traditional_ml"]["controls"]
                               if c["status"] == C.UNKNOWN)
        card.update({"model_type": fp["model_type"], "framework": fp["framework"],
                     "access": fp["access"], "unknown_assumptions": unknown_assumptions,
                     "unknown_controls": unknown_controls})
    if result.get("generative_ai"):
        gfp = result["generative_ai"]["fingerprint"]
        sensitive_tools = sum(1 for t in result["generative_ai"]["tools"] if t.get("sensitive"))
        unknown_controls = sum(1 for c in result["generative_ai"]["controls"]
                               if c["status"] == C.UNKNOWN)
        card.update({"subtype": gfp["subtype"], "context_sources": len(result["generative_ai"]["context_surface"]),
                     "trust_boundaries": len(result["generative_ai"]["trust_boundaries"]),
                     "tools": gfp["tools"], "sensitive_tools": sensitive_tools,
                     "mcp": gfp["mcp"], "human_approval": gfp["human_approval"],
                     "unknown_controls": unknown_controls})
    return card


def _save(result: Dict[str, Any], base_dir: str) -> str:
    from orion.evidence import new_trace_id
    trace_id = new_trace_id()
    tdir = Path(base_dir) / trace_id
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "know_yourself.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")
    return trace_id
