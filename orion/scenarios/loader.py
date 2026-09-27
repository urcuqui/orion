"""Load and validate scenario files from disk."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from orion.scenarios.model import Scenario, ScenarioValidationError, build_scenario

# Default directory scanned by ``orion list``.
DEFAULT_SCENARIO_DIR = Path("scenarios")


def _load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dep
        raise ScenarioValidationError(
            "PyYAML is required to load scenario files (pip install pyyaml)."
        ) from exc
    data = yaml.safe_load(text)
    if data is None:
        raise ScenarioValidationError("Scenario file is empty.")
    if not isinstance(data, dict):
        raise ScenarioValidationError("Scenario file must contain a YAML mapping at the top level.")
    return data


def load_scenario_dict(data: Dict[str, Any]) -> Scenario:
    """Validate an already-parsed scenario mapping."""
    return build_scenario(data)


def load_scenario(path: "str | Path") -> Scenario:
    """Load and validate a scenario from a YAML (or JSON) file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario file not found: {p}")
    text = p.read_text(encoding="utf-8")
    data = _load_yaml(text)
    scenario = build_scenario(data)
    return scenario


def validate_scenario(path: "str | Path") -> List[str]:
    """Return a list of validation problems (empty if the scenario is valid)."""
    try:
        load_scenario(path)
        return []
    except (ScenarioValidationError, FileNotFoundError) as exc:
        return [str(exc)]


def list_scenarios(directory: "str | Path" = DEFAULT_SCENARIO_DIR) -> List[Dict[str, str]]:
    """List scenario files in a directory with basic metadata."""
    d = Path(directory)
    if not d.exists():
        return []
    results: List[Dict[str, str]] = []
    for path in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
        entry = {"path": str(path), "name": path.stem, "description": "", "valid": "yes"}
        try:
            sc = load_scenario(path)
            entry["name"] = sc.name
            entry["description"] = sc.description
            entry["phase"] = sc.phase.value
            entry["technique"] = sc.attack.technique
        except (ScenarioValidationError, FileNotFoundError) as exc:
            entry["valid"] = "no"
            entry["description"] = f"INVALID: {exc}"
        results.append(entry)
    return results
