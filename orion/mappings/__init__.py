"""MITRE ATLAS mapping helpers.

MITRE ATLAS (Adversarial Threat Landscape for AI Systems) describes
AI-specific adversary behaviour. It is distinct from MITRE ATT&CK, which covers
enterprise IT. Orion uses ATLAS terminology for AI threats and only references
ATT&CK where a genuine ATT&CK mapping exists.
"""
from __future__ import annotations

from orion.mappings.atlas import (
    AtlasTactic,
    AtlasTechnique,
    ATLAS_TACTICS,
    ATLAS_TECHNIQUES,
    lookup_technique,
    validate_mappings,
    Mapping,
)

__all__ = [
    "AtlasTactic",
    "AtlasTechnique",
    "ATLAS_TACTICS",
    "ATLAS_TECHNIQUES",
    "lookup_technique",
    "validate_mappings",
    "Mapping",
]
