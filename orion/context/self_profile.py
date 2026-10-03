"""Self Profile: the first-class persisted output of Know Yourself.

Normalizes a Know Yourself analysis result into a stable ``ORN-SELF-…`` object so
it can be referenced by the Analysis Context, threat model and plan — without
duplicating the branch-specific fingerprint structures.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_self_id() -> str:
    return "ORN-SELF-" + uuid.uuid4().hex[:8].upper()


@dataclass
class SelfProfile:
    self_profile_id: str = field(default_factory=new_self_id)
    system_type: str = "unknown"            # traditional_ml | generative_ai | hybrid
    subtype: str = ""
    profile: Dict[str, Any] = field(default_factory=dict)       # fingerprint(s)
    capabilities: Dict[str, Any] = field(default_factory=dict)  # gradients/rag/tools/mcp…
    controls: List[Dict[str, Any]] = field(default_factory=list)
    security_posture: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelfProfile":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})


def build_self_profile(ky_result: Dict[str, Any]) -> SelfProfile:
    """Normalize a Know Yourself ``analyze()`` result into a Self Profile."""
    tml = ky_result.get("traditional_ml") or {}
    gen = ky_result.get("generative_ai") or {}
    fp = tml.get("fingerprint") or {}
    gfp = gen.get("fingerprint") or {}
    controls = (tml.get("controls") or []) + (gen.get("controls") or [])
    subtype = gfp.get("subtype") or fp.get("model_type") or ""
    return SelfProfile(
        system_type=ky_result.get("system_type", "unknown"),
        subtype=subtype,
        profile={"traditional_ml": fp or None, "generative_ai": gfp or None},
        capabilities=ky_result.get("capabilities", {}),
        controls=controls,
        security_posture=ky_result.get("security_posture", []),
        evidence_ids=[ky_result.get("trace_id")] if ky_result.get("trace_id") else [],
    )
