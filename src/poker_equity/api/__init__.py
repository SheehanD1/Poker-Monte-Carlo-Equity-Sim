"""FastAPI REST API package for poker equity calculations and range analysis."""

from __future__ import annotations

from poker_equity.api.app import app, create_app

__all__ = ["app", "create_app"]
