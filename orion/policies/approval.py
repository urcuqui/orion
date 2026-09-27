"""Sensitive-action classification and fail-closed approval gate."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional


class SensitiveAction(str, Enum):
    """Categories of operations that require human oversight."""

    EXTERNAL_EXECUTION = "external_execution"      # run code / tools against external systems
    DESTRUCTIVE = "destructive"                    # delete/overwrite/modify state
    CREDENTIAL_USE = "credential_use"              # use secrets/credentials
    EXPLOIT_ATTEMPT = "exploit_attempt"            # active exploitation
    EXTERNAL_CHANGE = "external_change"            # changes to external systems


# Human-readable descriptions of each sensitive category.
SENSITIVE_ACTIONS: Dict[SensitiveAction, str] = {
    SensitiveAction.EXTERNAL_EXECUTION: "Execute code or tools against an external target.",
    SensitiveAction.DESTRUCTIVE: "Delete, overwrite, or otherwise destroy data or state.",
    SensitiveAction.CREDENTIAL_USE: "Use credentials or secrets.",
    SensitiveAction.EXPLOIT_ATTEMPT: "Attempt to exploit a vulnerability.",
    SensitiveAction.EXTERNAL_CHANGE: "Make a change to an external system.",
}


# Keyword heuristics mapping tool/action names to sensitivity.
_KEYWORDS = {
    SensitiveAction.EXPLOIT_ATTEMPT: ("nuclei", "exploit", "attack", "payload", "cve"),
    SensitiveAction.EXTERNAL_EXECUTION: ("execute_code", "curl", "http", "request", "browser", "playwright", "scan", "scrape"),
    SensitiveAction.DESTRUCTIVE: ("delete", "remove", "drop", "overwrite", "write_file", "edit_file", "rm "),
    SensitiveAction.CREDENTIAL_USE: ("login", "password", "credential", "token", "auth", "secret"),
    SensitiveAction.EXTERNAL_CHANGE: ("post", "put", "patch", "deploy", "modify"),
}


def classify(action_name: str) -> Optional[SensitiveAction]:
    """Classify an action/tool name into a sensitive category, or None."""
    name = (action_name or "").lower()
    for category, keywords in _KEYWORDS.items():
        if any(k in name for k in keywords):
            return category
    return None


def is_sensitive(action_name: str) -> bool:
    return classify(action_name) is not None


class ApprovalRequired(Exception):
    """Raised (fail-closed) when approval is required but unavailable."""


class ApprovalDenied(Exception):
    """Raised when a human explicitly denied an action."""


@dataclass
class ApprovalPolicy:
    """Gate sensitive operations behind human approval.

    ``require_human_approval`` gates *all* actions; ``require_sensitive_approval``
    gates only actions classified as sensitive. ``approver`` is an optional
    callback ``(action_name, category) -> bool``. If approval is required and no
    approver is configured, the gate fails closed (raises
    :class:`ApprovalRequired`).
    """

    require_human_approval: bool = False
    require_sensitive_approval: bool = True
    approver: Optional[Callable[[str, Optional[SensitiveAction]], bool]] = None

    def needs_approval(self, action_name: str) -> bool:
        if self.require_human_approval:
            return True
        if self.require_sensitive_approval and is_sensitive(action_name):
            return True
        return False

    def authorize(self, action_name: str) -> bool:
        """Return True if the action may proceed; fail closed otherwise.

        Raises :class:`ApprovalRequired` if approval is needed but no approver is
        available, and :class:`ApprovalDenied` if the approver rejects it.
        """
        if not self.needs_approval(action_name):
            return True
        category = classify(action_name)
        if self.approver is None:
            raise ApprovalRequired(
                f"Action {action_name!r} requires human approval "
                f"({category.value if category else 'policy'}), but no approver is configured."
            )
        approved = bool(self.approver(action_name, category))
        if not approved:
            raise ApprovalDenied(f"Action {action_name!r} was denied by the approver.")
        return True
