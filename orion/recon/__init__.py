"""Reconnaissance: the 'Know the Environment' (Understand) phase.

This package re-exports the existing reconnaissance workflow (``libs.recon``)
so it can be referenced as ``orion.recon`` under the methodology, while the
implementation continues to live in ``libs/recon`` for backward compatibility.

Recon requires human approval for sensitive actions (see ``orion.policies``) and
generates structured evidence (see ``orion.evidence``).
"""
from __future__ import annotations

try:
    from libs.recon import web as web  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    web = None  # type: ignore

__all__ = ["web"]
