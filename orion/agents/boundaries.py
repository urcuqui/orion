"""Explicit boundaries for agent behaviour.

Agents assist; they do not decide security. A security *verdict* (secure /
robust / safe) must come from metrics and experiments, never from an agent's
narrative. :func:`assert_not_security_verdict` gives callers a programmatic
guard so an agent's text is never mistaken for a quantitative result.
"""
from __future__ import annotations

import re
from typing import List

ALLOWED_AGENT_ROLES: List[str] = [
    "suggest experiments",
    "explain results",
    "correlate findings with MITRE ATLAS",
    "summarize evidence",
    "recommend candidate controls",
    "assist with report generation",
]

FORBIDDEN_AGENT_CLAIMS: List[str] = [
    "deciding whether a model is secure",
    "replacing quantitative evaluation",
    "proving robustness",
]


class AgentBoundaryError(RuntimeError):
    """Raised when agent output is used as if it were a security verdict."""


# Phrases that would constitute an (unbacked) security verdict.
_VERDICT_PATTERNS = [
    r"\bthe model is (now )?(secure|safe|robust)\b",
    r"\bproven?ly (secure|robust)\b",
    r"\bguarantee[sd]? (security|robustness|safety)\b",
    r"\b100% (secure|robust|safe)\b",
    r"\bfully (secure|robust|protected)\b",
]


def looks_like_security_verdict(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(p, lowered) for p in _VERDICT_PATTERNS)


def assert_not_security_verdict(text: str) -> str:
    """Return ``text`` unchanged, or raise if it asserts a security verdict.

    Use this at the boundary where agent output would otherwise be treated as a
    conclusion about the target's security. The determination of security must
    come from the metrics/evidence layer, not the agent.
    """
    if looks_like_security_verdict(text):
        raise AgentBoundaryError(
            "Agent output asserts a security verdict; verdicts must come from "
            "metrics/experiments, not the agent. See docs/agent-role.md."
        )
    return text
