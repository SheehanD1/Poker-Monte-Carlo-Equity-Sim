"""Shared test fixtures and configuration for the poker-equity test suite."""

from __future__ import annotations

import pytest


@pytest.fixture()
def seed() -> int:
    """Fixed random seed for deterministic tests."""
    return 42
