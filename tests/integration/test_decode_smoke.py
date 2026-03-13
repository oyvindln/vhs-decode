"""Smoke tests for end-to-end decoding.

These require real RF data and are intended for CI or manual runs.
Mark with @pytest.mark.integration so they can be selected/excluded.
"""

import pytest


@pytest.mark.integration
class TestDecodeSmoke:
    pass
