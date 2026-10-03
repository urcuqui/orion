"""Attack catalog keyed by attacker *access level*, with base tuning.

The access level is the operational form of the threat model's
``adversary.knowledge``: owning the weights (white-box) unlocks gradient-based
attacks; a query-only live service (black-box) unlocks decision-based ones.
Each entry ships sensible **base tuning** so a run is one click away, while the
console still lets an analyst override every parameter.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

WHITE_BOX = "white_box"
BLACK_BOX = "black_box"

# modality -> access level -> list of attack specs
_IMAGE_ATTACKS: Dict[str, List[Dict[str, Any]]] = {
    WHITE_BOX: [
        {"id": "CarliniL2", "name": "Carlini & Wagner (L2)", "access": WHITE_BOX,
         "backend": "whitebox_image", "base_params": {"max_iter": 10, "confidence": 0.0},
         "about": "Strong minimal-L2 optimization attack; the standard white-box benchmark."},
        {"id": "PGD", "name": "Projected Gradient Descent (L∞)", "access": WHITE_BOX,
         "backend": "whitebox_image", "base_params": {"eps": 0.03, "eps_step": 0.005, "max_iter": 20},
         "about": "Iterative L∞ gradient attack; strong, widely-used baseline."},
        {"id": "FGSM", "name": "Fast Gradient Sign Method (L∞)", "access": WHITE_BOX,
         "backend": "whitebox_image", "base_params": {"eps": 0.03},
         "about": "Single-step L∞ attack; fast but weaker than PGD/C&W."},
    ],
    BLACK_BOX: [
        {"id": "HopSkipJump", "name": "Decision-based query evasion", "access": BLACK_BOX,
         "backend": "blackbox_query", "base_params": {"epsilon": 0.05, "max_queries": 20},
         "about": "Decision-based: needs only the endpoint's top label, no gradients."},
    ],
}


def attacks_for(access_level: str, modality: str = "image") -> List[Dict[str, Any]]:
    """Attacks available for an access level (image modality for now)."""
    if modality != "image":
        return []
    return [dict(a) for a in _IMAGE_ATTACKS.get(access_level, [])]


def catalog(modality: str = "image") -> Dict[str, List[Dict[str, Any]]]:
    return {WHITE_BOX: attacks_for(WHITE_BOX, modality),
            BLACK_BOX: attacks_for(BLACK_BOX, modality)}


def find_attack(attack_id: str, modality: str = "image") -> Optional[Dict[str, Any]]:
    for level in (WHITE_BOX, BLACK_BOX):
        for a in _IMAGE_ATTACKS.get(level, []):
            if a["id"].lower() == (attack_id or "").lower():
                return dict(a)
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
