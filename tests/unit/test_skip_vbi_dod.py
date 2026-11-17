"""Tests for the --skip_vbi_dod flag (DodOptions.skip_vbi).

detect_dropouts_rf normally scans the whole field, including the vertical
blanking interval (VBI). With skip_vbi set it starts at the first active
field line (23 for 625-line, 21 for 525-line) instead of line 1, so VBI
sync/equalization pulses and test signals are not flagged as dropouts.

end_line must stay put when the start line moves, otherwise the detection
window runs past the bottom of the field and flags phantom dropouts there.
"""

import types

import numpy as np

from vhsdecode.doc import DodOptions, detect_dropouts_rf


# One field line is this many RF samples wide in the synthetic data below.
LINE_LEN = 100
# 625-line field: the active region starts at line 23.
FRAME_LINES = 625
FIRST_ACTIVE_LINE = 23
# Active line count, as process.py reports it (frame_lines // 2 - a few).
LINECOUNT = 300
# More linelocs than the detection window needs, so end_line is driven by the
# linecount+start_line term and is NOT clamped to the array length. This mirrors
# the real decode and is what lets a shifted start_line move end_line.
NUM_LINES = 350


class _RfLevelStub:
    """Stand-in for field.rf.field_averages.rf_level; only push() is used."""

    def push(self, _value):
        pass


def _field(dropout_lines):
    """A field stub carrying only what detect_dropouts_rf touches.

    dropout_lines: field line indices to drop the envelope on. Each dropout
    sits in the middle of its line and is wider than DOD_MIN_LENGTH so it
    survives the short-dropout filter.
    """
    # linelocs maps line index -> RF sample position.
    linelocs = np.arange(NUM_LINES + 1, dtype=np.float64) * LINE_LEN
    total_rf = int(linelocs[-1])

    # Flat envelope well above threshold, with notches on the dropout lines.
    env = np.full(total_rf, 100.0, dtype=np.float64)
    for line in dropout_lines:
        start = line * LINE_LEN + 20
        env[start : start + 40] = 1.0

    rf = types.SimpleNamespace(
        SysParams={"frame_lines": FRAME_LINES},
        field_averages=types.SimpleNamespace(rf_level=_RfLevelStub()),
    )

    return types.SimpleNamespace(
        data={"video": {"envelope": env}},
        rf=rf,
        linelocs=linelocs,
        lineoffset=0,
        linecount=LINECOUNT,
        outlinelen=LINE_LEN,
    )


def _options(skip_vbi):
    # Absolute threshold of 50 flags the notches (level 1.0) and nothing else.
    return DodOptions(
        dod_threshold_p=0.5,
        dod_threshold_a=50.0,
        dod_hysteresis=1.0,
        skip_vbi=skip_vbi,
    )


def _reported_lines(result):
    # detect_dropouts_rf returns (lines, starts, ends).
    return list(result[0])


def test_vbi_dropout_detected_by_default():
    """A dropout on a VBI line is reported when skip_vbi is off."""
    field = _field(dropout_lines=[5])
    lines = _reported_lines(detect_dropouts_rf(field, _options(skip_vbi=False)))
    assert 5 in lines


def test_skip_vbi_drops_vbi_dropout():
    """The same VBI dropout is gone when skip_vbi is on."""
    field = _field(dropout_lines=[5])
    lines = _reported_lines(detect_dropouts_rf(field, _options(skip_vbi=True)))
    assert 5 not in lines


def test_skip_vbi_keeps_active_dropout():
    """A dropout in the active picture is still reported with skip_vbi on."""
    active_line = FIRST_ACTIVE_LINE + 50
    field = _field(dropout_lines=[5, active_line])
    lines = _reported_lines(detect_dropouts_rf(field, _options(skip_vbi=True)))
    assert 5 not in lines
    assert active_line in lines


def test_skip_vbi_does_not_extend_detection_window():
    """Regression: skipping VBI must not move the detection window end.

    end_line is derived from linecount + start_line, so an unfixed version that
    shifts start_line before computing end_line pushes the window end past the
    real field end. Dropouts beyond the field (here lines above linecount) must
    not be reported whether skip_vbi is on or off; the window end is identical
    both ways.
    """
    # end_line (fixed, either mode) = min(NUM_LINES-1, linecount + lineoffset + 2).
    # A dropout just past that is outside the window with skip_vbi off. The bug
    # grew end_line when skipping, so it would wrongly flag this line as a
    # phantom dropout past the field end; the fix keeps the window end put.
    beyond_line = LINECOUNT + 5
    off = _reported_lines(detect_dropouts_rf(_field([beyond_line]), _options(skip_vbi=False)))
    on = _reported_lines(detect_dropouts_rf(_field([beyond_line]), _options(skip_vbi=True)))
    assert beyond_line not in off
    assert beyond_line not in on

    # And an in-window active dropout is reported identically either way.
    active_line = FIRST_ACTIVE_LINE + 100
    off2 = _reported_lines(detect_dropouts_rf(_field([active_line]), _options(skip_vbi=False)))
    on2 = _reported_lines(detect_dropouts_rf(_field([active_line]), _options(skip_vbi=True)))
    assert active_line in off2
    assert active_line in on2


def test_skip_vbi_525_line_start():
    """On a 525-line field the active region starts at line 21, not 23."""
    field = _field(dropout_lines=[21])
    field.rf.SysParams["frame_lines"] = 525
    lines = _reported_lines(detect_dropouts_rf(field, _options(skip_vbi=True)))
    assert 21 in lines
    # line 20 is still in the VBI for 525-line and should be skipped
    field = _field(dropout_lines=[19])
    field.rf.SysParams["frame_lines"] = 525
    lines = _reported_lines(detect_dropouts_rf(field, _options(skip_vbi=True)))
    assert 19 not in lines
