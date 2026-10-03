"""The three context pillars converging into a shared Analysis Context.

    Know Yourself      -> Self Profile        (what AI system do I have?)
    Know Your Target   -> Target Profile      (what exactly am I evaluating?)
    Know the Environment -> Environment Profile (what surrounds the target?)

    Self + Target + Environment -> Analysis Context -> Threat Model -> Plan

Recon is a data source that populates the Environment Profile; it is not the
environment model itself.
"""
from __future__ import annotations

from orion.context.environment import (
    EnvironmentProfile, build_environment_profile, environment_to_summary, new_env_id, topology,
)
from orion.context.target import TargetProfile, build_target_profile, new_target_id
from orion.context.analysis_context import (
    AnalysisContext, build_analysis_context, new_context_id,
    COMPLETE, PARTIAL, NOT_AVAILABLE,
)
from orion.context.store import save_profile, load_profile

__all__ = [
    "EnvironmentProfile", "build_environment_profile", "environment_to_summary",
    "new_env_id", "topology",
    "TargetProfile", "build_target_profile", "new_target_id",
    "AnalysisContext", "build_analysis_context", "new_context_id",
    "COMPLETE", "PARTIAL", "NOT_AVAILABLE",
    "save_profile", "load_profile",
]
