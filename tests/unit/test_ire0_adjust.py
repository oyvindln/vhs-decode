"""Tests for the --ire0_adjust hsync submode's per-field hz_ire.

On a degenerate field (dropout / severe sync collapse) the backporch and hsync
measurement windows read the same level, so

    hz_ire = (ire0 - hsync_level) / -vsync_ire

comes out exactly 0. hz_to_output_array then divides out_scale by it; numba's
default error model raises rather than returning inf, so an unguarded field
aborts the entire decode with ZeroDivisionError.
"""

import logging
import types

import numpy as np
import pytest

import vhsdecode.field as vf
from vhsdecode.field import FieldShared

OUTLINECOUNT = 263
OUTLINELEN = 910
GLOBAL_HZ_IRE = 8571.43


@pytest.fixture(autouse=True)
def _logger():
    if getattr(vf.ldd, "logger", None) is None:
        vf.ldd.logger = logging.getLogger("test")


def _field(ire0_adjust="backporch,hsync"):
    """A field stub carrying only what hz_to_output touches."""
    rf = types.SimpleNamespace(
        DecoderParams={
            "ire0": 4_542_857.0,
            "hz_ire": GLOBAL_HZ_IRE,
            "vsync_ire": -40.0,
            "track_ire0_offset": [0.0, 0.0],
        },
        SysParams={"outputZero": 1024.0},
        options=types.SimpleNamespace(ire0_adjust=ire0_adjust, export_raw_tbc=False),
    )
    return types.SimpleNamespace(
        rf=rf,
        ire0_backporch=[74, 124],
        outlinecount=OUTLINECOUNT,
        outlinelen=OUTLINELEN,
        out_scale=358.4,
        field_number=0,
        track_phase=None,
    )


def _flat_input(level=5_000_000.0):
    """A flat field: every measurement window reads the identical level."""
    return np.full(OUTLINECOUNT * OUTLINELEN, level, dtype=np.float32)


def test_degenerate_field_does_not_abort_the_decode():
    """ire0 == hsync_level -> hz_ire == 0. Previously ZeroDivisionError."""
    out = FieldShared.hz_to_output(_field(), _flat_input())

    assert out.dtype == np.uint16
    assert len(out) == OUTLINECOUNT * OUTLINELEN


def test_degenerate_field_falls_back_to_the_global_hz_ire():
    """The guarded field decodes as if the hsync adjustment were unavailable.

    Same flat input through the backporch-only submode never computes a per-field
    hz_ire at all, so it already uses the global one. If the fallback works, the
    two must agree.
    """
    guarded = FieldShared.hz_to_output(_field("backporch,hsync"), _flat_input())
    backporch_only = FieldShared.hz_to_output(_field("backporch"), _flat_input())

    np.testing.assert_array_equal(guarded, backporch_only)


def test_healthy_field_still_uses_its_own_hz_ire():
    """The guard must not disturb a field whose windows read different levels.

    hz_to_output measures hsync over [4, ire0_backporch[0] - 4) and black over
    [ire0_backporch[0] + 4, ire0_backporch[1] - 4), so the sync level has to
    cover the whole first window to move the median.
    """
    inp = _flat_input()
    for line in range(OUTLINECOUNT):
        base = line * OUTLINELEN
        inp[base : base + 74] = 4_000_000.0

    # hz_ire = (5e6 - 4e6) / 40 = 25000, which is not the global 8571.43 --
    # so a field that never hits the guard must not decode like one that does.
    out = FieldShared.hz_to_output(_field(), inp)
    backporch_only = FieldShared.hz_to_output(_field("backporch"), inp)

    assert out.dtype == np.uint16
    assert not np.array_equal(out, backporch_only)
