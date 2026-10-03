"""Analysis Context: where Self + Target + Environment profiles converge.

Any profile may be absent — the system works progressively. Coverage is
categorical (COMPLETE / PARTIAL / NOT_AVAILABLE), never a fabricated percentage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

COMPLETE, PARTIAL, NOT_AVAILABLE = "COMPLETE", "PARTIAL", "NOT_AVAILABLE"


def new_context_id() -> str:
    return "ORN-CTX-" + uuid.uuid4().hex[:8].upper()


@dataclass
class AnalysisContext:
    analysis_context_id: str = field(default_factory=new_context_id)
    self_profile_id: Optional[str] = None
    target_profile_id: Optional[str] = None
    environment_profile_id: Optional[str] = None
    scope: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["coverage"] = self.coverage()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisContext":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})

    def coverage(self) -> Dict[str, str]:
        return {
            "self": COMPLETE if self.self_profile_id else NOT_AVAILABLE,
            "target": COMPLETE if self.target_profile_id else NOT_AVAILABLE,
            "environment": COMPLETE if self.environment_profile_id else NOT_AVAILABLE,
        }

    def status(self) -> str:
        cov = self.coverage()
        present = [k for k, v in cov.items() if v == COMPLETE]
        if len(present) == 3:
            return "READY FOR ANALYSIS"
        if present:
            return "PARTIAL CONTEXT"
        return "NO CONTEXT"


def build_analysis_context(self_id: Optional[str] = None, target_id: Optional[str] = None,
                           environment_id: Optional[str] = None,
                           scope: Optional[Dict[str, Any]] = None) -> AnalysisContext:
    return AnalysisContext(self_profile_id=self_id, target_profile_id=target_id,
                           environment_profile_id=environment_id, scope=scope or {})
