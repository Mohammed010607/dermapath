"""Shared pytest fixtures."""

from pathlib import Path

import pytest

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def fixture_root() -> Path:
    """Root of the tiny synthetic image/mask dataset committed under tests/fixtures."""
    return FIXTURE_ROOT
