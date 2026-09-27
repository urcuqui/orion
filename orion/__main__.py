"""Entry point for ``python -m orion``."""
from __future__ import annotations

import sys

from orion.cli import main

if __name__ == "__main__":
    sys.exit(main())
