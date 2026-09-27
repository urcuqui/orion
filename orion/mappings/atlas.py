"""A small, curated slice of the MITRE ATLAS matrix.

Only a subset of tactics/techniques relevant to Orion's current scenarios is
included. IDs follow the public ATLAS matrix (https://atlas.mitre.org/).

Do not invent mappings. When a scenario's mapping is approximate, record the
uncertainty in the mapping's ``rationale`` and set ``confidence`` accordingly.
The authoritative, complete data lives at atlas.mitre.org; this module is a
convenience index, not a mirror.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AtlasTactic:
    id: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class AtlasTechnique:
    id: str
    name: str
    tactic_id: str
    description: str = ""

    @property
    def url(self) -> str:
        # Sub-techniques use the form AML.T0043.000; the page lives at the parent.
        base = self.id.split(".")
        slug = ".".join(base[:2]) if len(base) > 2 else self.id
        return f"https://atlas.mitre.org/techniques/{slug}"


# --- Curated tactics (subset) ------------------------------------------------
ATLAS_TACTICS: Dict[str, AtlasTactic] = {
    t.id: t
    for t in [
        AtlasTactic("AML.TA0002", "Reconnaissance", "Gather information about the target ML system."),
        AtlasTactic("AML.TA0000", "ML Model Access", "Gain some level of access to the ML model."),
        AtlasTactic("AML.TA0004", "Initial Access", "Gain access to the system hosting the ML artifacts."),
        AtlasTactic("AML.TA0006", "ML Attack Staging", "Prepare an attack against the target ML model."),
        AtlasTactic("AML.TA0007", "Impact", "Manipulate, interrupt, or erode confidence in the ML system."),
    ]
}


# --- Curated techniques (subset) --------------------------------------------
ATLAS_TECHNIQUES: Dict[str, AtlasTechnique] = {
    t.id: t
    for t in [
        AtlasTechnique(
            "AML.T0043", "Craft Adversarial Data", "AML.TA0006",
            "Craft adversarial inputs that cause the model to produce incorrect outputs.",
        ),
        AtlasTechnique(
            "AML.T0043.000", "Craft Adversarial Data: White-Box Optimization", "AML.TA0006",
            "Use full knowledge of the model (gradients) to optimize a perturbation.",
        ),
        AtlasTechnique(
            "AML.T0043.001", "Craft Adversarial Data: Black-Box Optimization", "AML.TA0006",
            "Use only query access to search for an effective perturbation.",
        ),
        AtlasTechnique(
            "AML.T0015", "Evade ML Model", "AML.TA0007",
            "Use adversarial inputs at inference time to evade the model.",
        ),
        AtlasTechnique(
            "AML.T0040", "ML Model Inference API Access", "AML.TA0000",
            "Interact with the model via its inference API.",
        ),
        AtlasTechnique(
            "AML.T0020", "Poison Training Data", "AML.TA0006",
            "Introduce manipulated samples into the training data.",
        ),
        AtlasTechnique(
            "AML.T0000", "Search for Victim's Publicly Available Research Materials", "AML.TA0002",
            "Reconnaissance of publicly available information about the target.",
        ),
    ]
}


@dataclass
class Mapping:
    """A scenario's mapping to a single ATLAS technique."""

    technique_id: str
    tactic: str = ""
    technique: str = ""
    description: str = ""
    rationale: str = ""
    confidence: str = "medium"  # low | medium | high
    known: bool = True          # whether the technique_id exists in our index

    def to_dict(self) -> Dict[str, object]:
        return {
            "technique_id": self.technique_id,
            "tactic": self.tactic,
            "technique": self.technique,
            "description": self.description,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "known": self.known,
        }


def lookup_technique(technique_id: str) -> Optional[AtlasTechnique]:
    return ATLAS_TECHNIQUES.get(technique_id)


def validate_mappings(raw_mappings: List[dict]) -> List[Mapping]:
    """Resolve/validate scenario mappings against the curated ATLAS index.

    Unknown technique IDs are preserved but flagged with ``known=False`` and a
    low confidence, so uncertainty is explicit rather than silently accepted.
    """
    resolved: List[Mapping] = []
    for raw in raw_mappings or []:
        tid = str(raw.get("technique_id") or raw.get("technique") or "").strip()
        tech = lookup_technique(tid)
        if tech is not None:
            tactic = ATLAS_TACTICS.get(tech.tactic_id)
            resolved.append(
                Mapping(
                    technique_id=tech.id,
                    tactic=(tactic.name if tactic else tech.tactic_id),
                    technique=tech.name,
                    description=raw.get("description", tech.description),
                    rationale=raw.get("rationale", ""),
                    confidence=str(raw.get("confidence", "medium")),
                    known=True,
                )
            )
        else:
            resolved.append(
                Mapping(
                    technique_id=tid,
                    tactic=str(raw.get("tactic", "")),
                    technique=str(raw.get("technique", "")),
                    description=str(raw.get("description", "")),
                    rationale=str(raw.get("rationale", "unverified: not in Orion's curated ATLAS index")),
                    confidence="low",
                    known=False,
                )
            )
    return resolved
