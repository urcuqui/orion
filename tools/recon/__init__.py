"""Mock reconnaissance tools, ported from HexAgent (github.com/urcuqui/HexAgent).

Every tool here is deterministic and offline: outputs are derived from a
per-target "site profile" (see fixtures.py) rather than real network activity.
"""

from tools.recon.registry import ToolRegistry, default_registry

__all__ = ["ToolRegistry", "default_registry"]
