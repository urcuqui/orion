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
    id: str
    name: str
    description: str
    families: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "description": self.description,
                "families": list(self.families), "parameters": dict(self.parameters)}


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


def get(control_id: str) -> Optional[ControlDefinition]:
    return CONTROLS.get(control_id)


def for_family(family: str) -> List[ControlDefinition]:
    return [c for c in _CONTROLS if family in c.families]


def for_attack(attack_id: str) -> List[ControlDefinition]:
    from orion.catalog.attacks import get as get_attack
    a = get_attack(attack_id)
    if not a:
        return []
    out = [CONTROLS[cid] for cid in a.recommended_controls if cid in CONTROLS]
    return out or for_family(a.family)
