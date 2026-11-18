"""Executable module entry point for the poker-equity CLI.

Allows running the command-line interface directly with:
    python -m poker_equity <command> [options]
"""

from __future__ import annotations

import sys

from poker_equity.cli import main

if __name__ == "__main__":
    sys.exit(main())
