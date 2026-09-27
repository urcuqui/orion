"""Human oversight policies.

Formalizes ``human_approval`` and ``require_sensitive_approval``. Sensitive
operations are explicitly classified, and the system fails closed when approval
is required but unavailable.
"""
from __future__ import annotations

from orion.policies.approval import (
    SensitiveAction,
    SENSITIVE_ACTIONS,
    ApprovalPolicy,
    ApprovalRequired,
    ApprovalDenied,
    classify,
    is_sensitive,
)

__all__ = [
    "SensitiveAction",
    "SENSITIVE_ACTIONS",
    "ApprovalPolicy",
    "ApprovalRequired",
    "ApprovalDenied",
    "classify",
    "is_sensitive",
]
