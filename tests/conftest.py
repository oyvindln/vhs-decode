from pathlib import Path

import pytest


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir():
    """Path to the tests/data directory containing test fixtures."""
    return DATA_DIR
