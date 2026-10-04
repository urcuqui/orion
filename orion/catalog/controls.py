"""The unified Control Catalog — defenses that can be applied and *retested*.

A control is a hypothesis about how to reduce risk, not a guarantee. Applying a
control never means the system is secure; only a retest decides that. These
definitions give the Defend stage concrete, replayable controls — including the
agentic ones (tool authorization, least privilege, allowlists) the new agentic
experiments need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orion.catalog.attacks import AGENTIC_AI, GENERATIVE_AI, TRADITIONAL_ML


@dataclass
class ControlDefinition:
    """The *concept* of a control — what it defends, not how a system enforces it."""
    id: str
    name: str
    description: str
    families: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description,
                "families": list(self.families), "parameters": dict(self.parameters)}


# Coverage a concrete implementation actually provides — never claim FULL loosely.
PARTIAL, SUBSTANTIAL, FULL = "PARTIAL", "SUBSTANTIAL", "FULL"


@dataclass
class ControlImplementation:
    """How a *specific* environment enforces a ControlDefinition.

    A string injection filter is an implementation of instruction_provenance with
    PARTIAL coverage — it is not the control itself. Keeping these separate stops
    Orion from implying guarantees the enforcement does not provide.
    """
    id: str
    control_id: str
    name: str
    enforcement_point: str                 # input_gateway | tool_authorizer | model | ...
    coverage: str = PARTIAL                 # PARTIAL | SUBSTANTIAL | FULL
    description: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    limitations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "control_id": self.control_id, "name": self.name,
                "enforcement_point": self.enforcement_point, "coverage": self.coverage,
                "description": self.description, "configuration": dict(self.configuration),
                "limitations": self.limitations}


_CONTROLS: List[ControlDefinition] = [
    # --------------------------- Traditional ML ----------------------------- #
    ControlDefinition("input_preprocessing", "Input preprocessing",
                      "Squeeze / quantise inputs to blunt small adversarial perturbations.",
                      [TRADITIONAL_ML]),
    ControlDefinition("adversarial_training", "Adversarial training",
                      "Train on adversarial examples to widen the robust margin.",
                      [TRADITIONAL_ML]),
    ControlDefinition("confidence_threshold", "Confidence threshold",
                      "Reject low-confidence predictions that often accompany evasion.",
                      [TRADITIONAL_ML]),
    ControlDefinition("rate_limit", "Rate limiting",
                      "Throttle queries to raise the cost of black-box search.",
                      [TRADITIONAL_ML]),
    # --------------------------- Generative AI ------------------------------ #
    ControlDefinition("instruction_provenance", "Instruction provenance",
                      "Treat instructions from retrieved/untrusted content as data, never commands.",
                      [GENERATIVE_AI, AGENTIC_AI]),
    ControlDefinition("context_isolation", "Context isolation",
                      "Isolate untrusted context so it cannot alter system/developer instructions.",
                      [GENERATIVE_AI, AGENTIC_AI]),
    # ----------------------------- Agentic AI ------------------------------- #
    ControlDefinition("tool_authorization", "Tool-level authorization",
                      "Authorise every tool call against the agent's identity and privilege.",
                      [AGENTIC_AI], parameters={"allowed_tools": []}),
    ControlDefinition("least_privilege", "Least privilege",
                      "Grant the agent only the minimum tools/privileges its task requires.",
                      [AGENTIC_AI]),
    ControlDefinition("destination_allowlist", "Destination allowlist",
                      "Permit tool actions only to vetted destinations (recipients, hosts, paths).",
                      [AGENTIC_AI], parameters={"allowed_destinations": []}),
    ControlDefinition("human_approval", "Human approval",
                      "Require explicit human approval before privileged or irreversible actions.",
                      [GENERATIVE_AI, AGENTIC_AI]),
]

CONTROLS: Dict[str, ControlDefinition] = {c.id: c for c in _CONTROLS}


# Concrete implementations Orion can actually apply and retest. Coverage is set to
# what the enforcement truly provides — the filter is PARTIAL, not FULL.
_IMPLEMENTATIONS: List[ControlImplementation] = [
    ControlImplementation(
        "gateway_injection_filter", "instruction_provenance", "Gateway injection filter",
        enforcement_point="input_gateway", coverage=PARTIAL,
        description="Detects known prompt-injection patterns in untrusted content before it "
                    "reaches the agent.",
        limitations="Pattern-based; does not provide cryptographic or end-to-end instruction "
                    "provenance, and can be bypassed by novel phrasings."),
    ControlImplementation(
        "context_isolation_wrapper", "context_isolation", "Untrusted-context isolation",
        enforcement_point="input_gateway", coverage=PARTIAL,
        description="Wraps retrieved content as inert data so injected instructions are not "
                    "treated as commands.",
        limitations="Heuristic separation; a determined injection may still influence reasoning."),
    ControlImplementation(
        "tool_authorization_policy", "tool_authorization", "Tool-level authorization",
        enforcement_point="tool_authorizer", coverage=SUBSTANTIAL,
        description="Authorizes every tool call against the agent's identity and privilege; "
                    "denies tools outside the agent's allowed set.",
        configuration={"allowed_tools": ["search_logs", "export_report"]},
        limitations="Enforces the configured allow-list only; does not prevent the agent from "
                    "being influenced, only from acting outside its authority."),
    ControlImplementation(
        "least_privilege_policy", "least_privilege", "Least-privilege enforcement",
        enforcement_point="tool_authorizer", coverage=SUBSTANTIAL,
        description="Denies any tool whose required privilege exceeds the agent's effective "
                    "privilege.",
        limitations="Depends on correct privilege labelling of tools and agents."),
    ControlImplementation(
        "destination_allowlist_policy", "destination_allowlist", "Destination allow-list",
        enforcement_point="tool_authorizer", coverage=SUBSTANTIAL,
        description="Permits tool actions only to vetted destinations.",
        configuration={"allowed_destinations": ["internal-reports"]},
        limitations="Only as good as the allow-list; new legitimate destinations need review."),
    ControlImplementation(
        "human_approval_gate", "human_approval", "Human approval gate",
        enforcement_point="tool_authorizer", coverage=SUBSTANTIAL,
        description="Requires explicit human approval before privileged or irreversible actions.",
        limitations="Subject to approval fatigue; out of scope to model human error here."),
]

IMPLEMENTATIONS: Dict[str, ControlImplementation] = {i.id: i for i in _IMPLEMENTATIONS}


def get(control_id: str) -> Optional[ControlDefinition]:
    return CONTROLS.get(control_id)


def get_implementation(impl_id: str) -> Optional[ControlImplementation]:
    return IMPLEMENTATIONS.get(impl_id)


def implementations_for(control_id: str) -> List[ControlImplementation]:
    return [i for i in _IMPLEMENTATIONS if i.control_id == control_id]


def default_implementation(control_id: str) -> Optional[ControlImplementation]:
    impls = implementations_for(control_id)
    return impls[0] if impls else None


def for_family(family: str) -> List[ControlDefinition]:
    return [c for c in _CONTROLS if family in c.families]


def for_attack(attack_id: str) -> List[ControlDefinition]:
    from orion.catalog.attacks import get as get_attack
    a = get_attack(attack_id)
    if not a:
        return []
    out = [CONTROLS[cid] for cid in a.recommended_controls if cid in CONTROLS]
    return out or for_family(a.family)
