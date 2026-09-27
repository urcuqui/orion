"""Assets: what the defender is trying to protect."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Asset:
    """A protectable asset of an AI system."""

    key: str
    name: str
    description: str
    # Which classic security properties this asset primarily concerns.
    properties: tuple = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "properties": list(self.properties),
        }


# A catalog of common AI-system assets. Scenarios reference these by key.
ASSET_CATALOG: Dict[str, Asset] = {
    a.key: a
    for a in [
        Asset(
            "model_integrity",
            "Model integrity",
            "The model behaves as intended and has not been tampered with.",
            ("integrity",),
        ),
        Asset(
            "prediction_reliability",
            "Prediction reliability",
            "Predictions remain correct and trustworthy under adversarial input.",
            ("integrity", "availability"),
        ),
        Asset(
            "training_data",
            "Training data",
            "The data used to train or fine-tune the model.",
            ("integrity", "confidentiality"),
        ),
        Asset(
            "model_weights",
            "Model weights",
            "The learned parameters / model artifact.",
            ("confidentiality", "integrity"),
        ),
        Asset(
            "inference_api",
            "Inference API",
            "The serving interface exposed to consumers.",
            ("availability", "integrity"),
        ),
        Asset(
            "confidentiality",
            "Confidentiality",
            "Secrecy of data, weights, prompts and outputs.",
            ("confidentiality",),
        ),
        Asset(
            "availability",
            "Availability",
            "The system remains usable and responsive.",
            ("availability",),
        ),
    ]
}


def resolve_assets(keys: List[str]) -> List[Asset]:
    """Resolve asset keys to :class:`Asset` objects.

    Unknown keys are preserved as ad-hoc assets so a scenario is never rejected
    solely for naming an asset outside the catalog.
    """
    resolved: List[Asset] = []
    for key in keys:
        if key in ASSET_CATALOG:
            resolved.append(ASSET_CATALOG[key])
        else:
            resolved.append(Asset(key, key.replace("_", " ").title(), "Custom asset.", ()))
    return resolved
