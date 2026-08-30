"""Abstract base class for every recon tool.

Subclasses implement ``_run``; the base class handles timing, error wrapping
and producing a uniform ``ToolResult`` so the executor stays agnostic to the
concrete tool.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from libs.recon.models import ToolResult

import logging

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Common behaviour for every tool.

    Attributes:
        name: Unique registry key used by planners/executors.
        description: One-line human description shown in the tool catalogue.
        argument_help: Mapping of argument name -> short description.
        sensitive: True for tools that perform a real, state-changing or
            network-touching action (e.g. a POST request). Specialist agents
            gate these behind human approval when required.
    """

    name: str = "base"
    description: str = "Abstract tool"
    argument_help: dict[str, str] = {}
    sensitive: bool = False

    @abstractmethod
    def _run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool's logic and return a structured result."""
        raise NotImplementedError

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool, measuring duration and capturing failures."""
        start = time.perf_counter()
        logger.info("Running tool '%s' with args=%s", self.name, kwargs)
        try:
            result = self._run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately defensive at the seam
            logger.exception("Tool '%s' raised an exception", self.name)
            return ToolResult.fail(self.name, str(exc))
        result.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return result

    def is_call_sensitive(self, **kwargs: Any) -> bool:
        """Whether *this specific call* should be gated behind approval.

        Defaults to the static ``sensitive`` flag. Tools whose risk depends on
        the arguments (e.g. Nuclei escalating from a safe default profile to
        custom templates/high severity) override this so the same approval gate
        in :mod:`libs.recon.specialists` can be call-aware without a second
        approval mechanism.
        """
        return self.sensitive

    def catalogue_entry(self) -> str:
        """Return a one-line catalogue description including its arguments."""
        args = ", ".join(f"{k} ({v})" for k, v in self.argument_help.items()) or "none"
        return f"- {self.name}: {self.description} | args: {args}"
