"""A bounded validation effort; completion is not a security verdict."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import uuid

STATUSES = ('DRAFT', 'ACTIVE', 'COMPLETED', 'ARCHIVED')


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Assessment:
    assessment_id: str = field(default_factory=lambda: 'ORN-ASMT-' + uuid.uuid4().hex[:8].upper())
    name: str = ''
    description: str = ''
    status: str = 'DRAFT'
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    system_profile_id: Optional[str] = None
    target_id: Optional[str] = None
    environment_id: Optional[str] = None
    scope: dict = field(default_factory=dict)
    analysis_context_ids: list = field(default_factory=list)
    threat_model_ids: list = field(default_factory=list)
    plan_ids: list = field(default_factory=list)
    experiment_ids: list = field(default_factory=list)
    finding_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Historical applied-control pointers; catalog definitions and evidence stay independent.
    control_refs: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})
