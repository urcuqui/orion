"""Security regression tests: human oversight and agent boundaries."""
import pytest

from orion.agents.boundaries import AgentBoundaryError, assert_not_security_verdict
from orion.policies import (
    ApprovalDenied,
    ApprovalPolicy,
    ApprovalRequired,
    SensitiveAction,
    classify,
    is_sensitive,
)


def test_sensitive_action_requires_approval():
    """Sensitive actions must fail closed when no approver is available."""
    policy = ApprovalPolicy(require_sensitive_approval=True, approver=None)
    assert is_sensitive("nuclei_scan")
    with pytest.raises(ApprovalRequired):
        policy.authorize("nuclei_scan")

    # A denying approver blocks the action.
    deny = ApprovalPolicy(require_sensitive_approval=True, approver=lambda name, cat: False)
    with pytest.raises(ApprovalDenied):
        deny.authorize("exploit_payload")

    # An approving approver lets it through.
    allow = ApprovalPolicy(require_sensitive_approval=True, approver=lambda name, cat: True)
    assert allow.authorize("nuclei_scan") is True

    # Non-sensitive actions pass without approval.
    assert policy.authorize("read_docs") is True


def test_sensitive_classification_covers_expected_categories():
    assert classify("execute_code") == SensitiveAction.EXTERNAL_EXECUTION
    assert classify("delete_records") == SensitiveAction.DESTRUCTIVE
    assert classify("use_credential") == SensitiveAction.CREDENTIAL_USE
    assert classify("nuclei") == SensitiveAction.EXPLOIT_ATTEMPT


def test_agent_does_not_override_metric_result():
    """Agent narrative must not be accepted as a security verdict."""
    # Benign explanatory text is fine.
    text = "The attack reduced robust accuracy; consider input preprocessing."
    assert assert_not_security_verdict(text) == text

    # A verdict-style claim is rejected — verdicts come from metrics, not agents.
    for claim in [
        "The model is now secure.",
        "This proves the model is robust.",
        "The system is 100% secure.",
    ]:
        with pytest.raises(AgentBoundaryError):
            assert_not_security_verdict(claim)
