"""Access-level view over the unified Attack Catalog (white-box vs black-box).

This is a thin projection of ``orion.catalog.attacks`` for the image-modality
adversarial-ML attacks, grouped by the attacker's access level. The attack
metadata (names, base tuning, descriptions) lives once in the unified catalog —
this module only reshapes it for the ATTACK-stage chooser and derives the
default access level from the plan's origin.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from orion.catalog import attacks as _unified
from orion.catalog.attacks import BLACK_BOX, GRAY_BOX, WHITE_BOX  # noqa: F401


def _as_option(a: _unified.AttackDefinition) -> Dict[str, Any]:
    """Project a unified attack into the chooser's shape (engine key as id)."""
    return {"id": a.engine or a.id, "name": a.name, "access": a.required_access[0] if a.required_access else WHITE_BOX,
            "backend": a.backend, "base_params": dict(a.parameters), "about": a.description,
            "catalog_id": a.id}


def attacks_for(access_level: str, modality: str = "image") -> List[Dict[str, Any]]:
    """Attacks available for an access level (image modality for now)."""
    out = []
    for a in _unified.by_family(_unified.TRADITIONAL_ML):
        if a.modality == modality and access_level in a.required_access:
            out.append(_as_option(a))
    return out


def catalog(modality: str = "image") -> Dict[str, List[Dict[str, Any]]]:
    return {WHITE_BOX: attacks_for(WHITE_BOX, modality),
            BLACK_BOX: attacks_for(BLACK_BOX, modality)}


def find_attack(attack_id: str, modality: str = "image") -> Optional[Dict[str, Any]]:
    a = _unified.get(attack_id)
    if a and a.modality == modality:
        return _as_option(a)
    return None


def derive_access_level(*, self_profile_id: Optional[str] = None,
                        target_url: Optional[str] = None) -> Dict[str, str]:
    """Derive the default access level from the plan's origin.

    Know Yourself (you own the model) → white-box weights; an external live
    service → query-only black-box. The console shows this as a default the
    analyst confirms or overrides — never a silent decision.
    """
    if self_profile_id:
        return {"access_level": WHITE_BOX, "source": "self_profile",
                "reason": "Plan originates from Know Yourself — you own the model, so weights "
                          "(white-box) access is assumed. Confirm or override below."}
    if target_url:
        return {"access_level": BLACK_BOX, "source": "target_url",
                "reason": "Target is an external live service — query-only (black-box) access "
                          "is assumed. Provide weights to switch to white-box."}
    return {"access_level": WHITE_BOX, "source": "default",
            "reason": "No access evidence on the plan; defaulting to white-box. "
                      "Confirm or override below."}
