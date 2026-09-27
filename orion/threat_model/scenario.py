"""The structured ThreatModel object that ties target, assets and adversary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from orion.threat_model.adversary import Adversary
from orion.threat_model.assets import Asset, resolve_assets
from orion.threat_model.boundaries import ATTACK_SURFACES, Target


@dataclass
class ThreatModel:
    """A complete, LLM-independent threat model.

    Example::

        ThreatModel(
            target=Target(task="image_classification", access="white_box"),
            adversary=Adversary(goal=Goal.TARGETED_MISCLASSIFICATION,
                                knowledge=KnowledgeLevel.FULL, budget=Budget.LOW),
            assets=["model_integrity", "prediction_reliability"],
            surfaces=["inference_api", "model_artifact"],
        )
    """

    target: Target
    adversary: Adversary
    assets: List[str] = field(default_factory=list)
    surfaces: List[str] = field(default_factory=list)
    rationale: str = ""

    def resolved_assets(self) -> List[Asset]:
        return resolve_assets(self.assets)

    def unknown_surfaces(self) -> List[str]:
        return [s for s in self.surfaces if s not in ATTACK_SURFACES]

    def to_dict(self) -> Dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "adversary": self.adversary.to_dict(),
            "assets": list(self.assets),
            "surfaces": list(self.surfaces),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ThreatModel":
        return cls(
            target=Target.from_dict(dict(data.get("target", {}) or {})),
            adversary=Adversary.from_dict(dict(data.get("adversary", {}) or {})),
            assets=list(data.get("assets", []) or []),
            surfaces=list(data.get("surfaces", []) or []),
            rationale=str(data.get("rationale", "")),
        )
