"""Scenario data model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from orion.mappings.atlas import Mapping, validate_mappings
from orion.methodology import Phase, coerce_phase
from orion.threat_model import ThreatModel


class ScenarioValidationError(ValueError):
    """Raised when a scenario file is missing required fields or malformed."""


@dataclass
class AttackSpec:
    """The attack to run, technique-agnostic with free-form parameters."""

    technique: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"technique": self.technique, "params": dict(self.params)}


@dataclass
class HardeningSpec:
    """Defenses to apply in the harden/retest phases."""

    defenses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {"defenses": list(self.defenses)}


@dataclass
class Scenario:
    """A complete, validated Orion scenario."""

    name: str
    description: str
    threat_model: ThreatModel
    attack: AttackSpec
    metrics: List[str] = field(default_factory=list)
    mitre_atlas: List[Mapping] = field(default_factory=list)
    hardening: HardeningSpec = field(default_factory=HardeningSpec)
    phase: Phase = Phase.ATTACK
    dataset: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "phase": self.phase.value,
            "target": self.threat_model.target.to_dict(),
            "adversary": self.threat_model.adversary.to_dict(),
            "assets": list(self.threat_model.assets),
            "surfaces": list(self.threat_model.surfaces),
            "attack": self.attack.to_dict(),
            "metrics": list(self.metrics),
            "mitre_atlas": [m.to_dict() for m in self.mitre_atlas],
            "hardening": self.hardening.to_dict(),
            "dataset": dict(self.dataset),
        }


def build_scenario(data: Dict[str, Any]) -> Scenario:
    """Build and validate a :class:`Scenario` from a raw mapping."""
    if not isinstance(data, dict):
        raise ScenarioValidationError("Scenario must be a mapping.")

    name = data.get("name")
    if not name:
        raise ScenarioValidationError("Scenario is missing required field 'name'.")

    attack_raw = data.get("attack")
    if not isinstance(attack_raw, dict) or not attack_raw.get("technique"):
        raise ScenarioValidationError("Scenario is missing required 'attack.technique'.")

    # The threat model is assembled from target + adversary (+ assets/surfaces).
    tm_source = {
        "target": data.get("target", {}),
        "adversary": data.get("adversary", {}),
        "assets": data.get("assets", []),
        "surfaces": data.get("surfaces", data.get("attack_surfaces", [])),
        "rationale": data.get("rationale", ""),
    }
    if not tm_source["adversary"]:
        raise ScenarioValidationError(
            "Scenario is missing an 'adversary' block. "
            "An attack algorithm without a threat model is only an experiment."
        )
    threat_model = ThreatModel.from_dict(tm_source)

    technique = str(attack_raw["technique"])
    params = {k: v for k, v in attack_raw.items() if k != "technique"}

    hardening_raw = data.get("hardening", {}) or {}
    hardening = HardeningSpec(defenses=_normalize_defenses(hardening_raw.get("defenses", [])))

    atlas_raw = (data.get("mitre_atlas", {}) or {}).get("mappings", [])
    mappings = validate_mappings(atlas_raw)

    phase = coerce_phase(data.get("phase", Phase.ATTACK.value))

    return Scenario(
        name=str(name),
        description=str(data.get("description", "")),
        threat_model=threat_model,
        attack=AttackSpec(technique=technique, params=params),
        metrics=list(data.get("metrics", []) or []),
        mitre_atlas=mappings,
        hardening=hardening,
        phase=phase,
        dataset=dict(data.get("dataset", {}) or {}),
    )


def _normalize_defenses(defenses: Any) -> List[Dict[str, Any]]:
    """Accept either ['name', ...] or [{'name':..., 'params':...}, ...]."""
    out: List[Dict[str, Any]] = []
    for d in defenses or []:
        if isinstance(d, str):
            out.append({"name": d, "params": {}})
        elif isinstance(d, dict) and d.get("name"):
            params = {k: v for k, v in d.items() if k not in ("name", "params")}
            params.update(d.get("params", {}) or {})
            out.append({"name": d["name"], "params": params})
    return out
