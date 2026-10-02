"""Experiment registry: shared metadata for plans and the Attack workspace.

One source of truth for each experiment's domain, category, risk, prerequisites,
default parameters, scenario and whether an automated runner exists. The plan
builder and the Attack workspace both read this so they never disagree.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Risk levels.
LOCAL = "LOCAL"          # safe, local, no external side effects
CONTROLLED = "CONTROLLED"  # controlled but touches model behaviour / content
REMOTE = "REMOTE"        # remote target / external side effects

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # ---- Traditional ML / evasion family ----
    "pgd_evasion": {
        "id": "pgd_evasion", "name": "PGD evasion", "domain": "traditional_ml",
        "category": "evasion", "scenario": "scenarios/pgd_evasion.yaml",
        "runner": "adversarial", "risk": LOCAL,
        "prerequisites": ["traditional_ml", "gradients_or_query"],
        "parameters": {"epsilon": 0.03, "iterations": 40, "step_size": "auto"},
    },
    "fgsm_evasion": {
        "id": "fgsm_evasion", "name": "FGSM evasion", "domain": "traditional_ml",
        "category": "evasion", "scenario": "scenarios/pgd_evasion.yaml",
        "runner": "adversarial", "risk": LOCAL,
        "prerequisites": ["traditional_ml", "gradients"],
        "parameters": {"epsilon": 0.03},
    },
    "cw_evasion": {
        "id": "cw_evasion", "name": "C&W evasion", "domain": "traditional_ml",
        "category": "evasion", "scenario": "scenarios/pgd_evasion.yaml",
        "runner": "adversarial", "risk": LOCAL,
        "prerequisites": ["traditional_ml", "gradients"],
        "parameters": {"confidence": 0.0, "iterations": 40},
    },
    "deepfool_evasion": {
        "id": "deepfool_evasion", "name": "DeepFool evasion", "domain": "traditional_ml",
        "category": "evasion", "scenario": "scenarios/pgd_evasion.yaml",
        "runner": "adversarial", "risk": LOCAL,
        "prerequisites": ["traditional_ml", "gradients"], "parameters": {},
    },
    "model_extraction": {
        "id": "model_extraction", "name": "Model extraction", "domain": "traditional_ml",
        "category": "extraction", "scenario": None, "runner": None, "risk": REMOTE,
        "prerequisites": ["query_access"], "parameters": {},
    },
    "membership_inference": {
        "id": "membership_inference", "name": "Membership inference", "domain": "traditional_ml",
        "category": "privacy", "scenario": None, "runner": None, "risk": CONTROLLED,
        "prerequisites": ["query_access", "training_distribution_context"], "parameters": {},
    },
    # ---- Generative AI family ----
    "indirect_prompt_injection": {
        "id": "indirect_prompt_injection", "name": "Prompt injection", "domain": "generative_ai",
        "category": "context_manipulation", "scenario": None, "runner": None, "risk": CONTROLLED,
        "prerequisites": ["llm_interface"], "parameters": {},
    },
    "rag_poisoning": {
        "id": "rag_poisoning", "name": "RAG poisoning", "domain": "generative_ai",
        "category": "context_manipulation", "scenario": None, "runner": None, "risk": CONTROLLED,
        "prerequisites": ["retrieval"], "parameters": {},
    },
    "tool_poisoning": {
        "id": "tool_poisoning", "name": "Tool / MCP poisoning", "domain": "generative_ai",
        "category": "tool_abuse", "scenario": None, "runner": None, "risk": REMOTE,
        "prerequisites": ["tools_or_mcp"], "parameters": {},
    },
    "memory_poisoning": {
        "id": "memory_poisoning", "name": "Memory poisoning", "domain": "generative_ai",
        "category": "persistence", "scenario": None, "runner": None, "risk": CONTROLLED,
        "prerequisites": ["persistent_memory"], "parameters": {},
    },
}

# Map a free-text attack/experiment name to a registry entry.
_ALIASES = {
    "pgd evasion": "pgd_evasion", "pgd": "pgd_evasion",
    "fgsm evasion": "fgsm_evasion", "fgsm": "fgsm_evasion",
    "c&w evasion": "cw_evasion", "c&w": "cw_evasion", "carlini": "cw_evasion",
    "deepfool evasion": "deepfool_evasion", "deepfool": "deepfool_evasion",
    "adversarial input robustness (pgd / c&w)": "pgd_evasion",
    "adversarial input robustness": "pgd_evasion",
    "model extraction": "model_extraction",
    "model extraction / inference-api abuse": "model_extraction",
    "membership inference": "membership_inference",
    "prompt injection": "indirect_prompt_injection",
    "rag poisoning": "rag_poisoning",
    "tool / mcp poisoning": "tool_poisoning", "tool poisoning": "tool_poisoning",
    "memory poisoning": "memory_poisoning",
}


def lookup(name_or_id: str) -> Optional[Dict[str, Any]]:
    key = (name_or_id or "").strip().lower()
    if key in EXPERIMENTS:
        return EXPERIMENTS[key]
    alias = _ALIASES.get(key)
    return EXPERIMENTS.get(alias) if alias else None


def list_experiments(domain: Optional[str] = None) -> List[Dict[str, Any]]:
    items = list(EXPERIMENTS.values())
    return [e for e in items if e["domain"] == domain] if domain else items
