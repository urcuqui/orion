"""Boundaries: attack surfaces and the target under evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AttackSurface:
    """A place where an adversary can act against the system."""

    key: str
    name: str
    description: str


# Canonical attack surfaces of an ML pipeline.
ATTACK_SURFACES: Dict[str, AttackSurface] = {
    s.key: s
    for s in [
        AttackSurface("training_pipeline", "Training pipeline", "Code/orchestration that trains the model."),
        AttackSurface("dataset", "Dataset", "Data collection and labeling."),
        AttackSurface("feature_pipeline", "Feature pipeline", "Preprocessing and feature engineering."),
        AttackSurface("inference_api", "Inference API", "The deployed serving endpoint."),
        AttackSurface("model_artifact", "Model artifact", "The stored weights / serialized model."),
        AttackSurface("feedback_loop", "Feedback loop", "Human/automated feedback that re-enters training."),
    ]
}


@dataclass
class Target:
    """The system under evaluation."""

    task: str = "image_classification"
    access: str = "white_box"
    model_name: str = "unknown"
    model_version: str = "unknown"
    description: str = ""
    surfaces: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "access": self.access,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "description": self.description,
            "surfaces": list(self.surfaces),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Target":
        return cls(
            task=str(data.get("task", "image_classification")),
            access=str(data.get("access", "white_box")),
            model_name=str(data.get("model_name", data.get("model", "unknown"))),
            model_version=str(data.get("model_version", "unknown")),
            description=str(data.get("description", "")),
            surfaces=list(data.get("surfaces", []) or []),
        )
