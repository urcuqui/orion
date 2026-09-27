"""Defense base class and registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class DefenseResult:
    """The outcome of applying a defense to a single input/decision.

    ``blocked`` indicates the defense refused/flagged the input (e.g. anomaly or
    confidence check). ``transformed_input`` is the (possibly) modified input to
    feed to the model. ``notes`` carries human-readable rationale.
    """

    name: str
    blocked: bool = False
    transformed_input: Any = None
    notes: str = ""
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "blocked": self.blocked,
            "notes": self.notes,
            "detail": self.detail,
        }


class Defense:
    """Base class for a pluggable defense.

    Subclasses override :meth:`apply`. A defense may transform the input,
    inspect a prediction, and/or decide to block. It must never assert that the
    model is now secure — only that it applied a control.
    """

    name: str = "defense"
    description: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = params

    def apply(self, *, input_data: Any = None, prediction: Optional[dict] = None) -> DefenseResult:  # noqa: D401
        raise NotImplementedError

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "description": self.description, "params": self.params}


DEFENSE_REGISTRY: Dict[str, Callable[..., Defense]] = {}


def register_defense(name: str, factory: Callable[..., Defense]) -> None:
    DEFENSE_REGISTRY[name] = factory


def get_defense(name: str, **params: Any) -> Defense:
    if name not in DEFENSE_REGISTRY:
        available = ", ".join(sorted(DEFENSE_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown defense {name!r}. Registered: {available}")
    return DEFENSE_REGISTRY[name](**params)
