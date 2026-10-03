"""Target Profile: what exactly is being assessed, and why.

Know Your Target *consumes* Self and Environment profiles; it does not generate
them. It defines the object and objective of the assessment.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_target_id() -> str:
    return "ORN-TARGET-" + uuid.uuid4().hex[:8].upper()


@dataclass
class TargetProfile:
    target_profile_id: str = field(default_factory=new_target_id)
    name: str = ""
    target_type: str = "unknown"
    objective: str = ""
    security_objective: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    authorization: Dict[str, Any] = field(default_factory=dict)
    known_interfaces: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    self_profile_id: Optional[str] = None
    environment_profile_id: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetProfile":
        return cls(**{k: d.get(k) for k in cls.__annotations__ if k in d})


def build_target_profile(data: Dict[str, Any]) -> TargetProfile:
    return TargetProfile(
        name=str(data.get("name", "") or ""),
        target_type=str(data.get("target_type", "unknown") or "unknown"),
        objective=str(data.get("objective", "") or ""),
        security_objective=str(data.get("security_objective", "") or ""),
        scope=dict(data.get("scope", {}) or {}),
        constraints=dict(data.get("constraints", {}) or {}),
        authorization=dict(data.get("authorization", {}) or {}),
        known_interfaces=list(data.get("known_interfaces", []) or []),
        self_profile_id=data.get("self_profile_id"),
        environment_profile_id=data.get("environment_profile_id"),
    )
