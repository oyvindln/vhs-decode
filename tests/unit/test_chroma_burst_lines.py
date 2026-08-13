"""Burst-indexed chroma passes must stay inside the field.

The burst lock runs while the field is decoded, before PAL fields set their
lineoffset (2 or 3), so field.phase_sequence starts at line 1 while the chroma
passes run with lineoffset + 1 = 3 or 4. The first burst entries then map to
negative rows, which numba (like numpy) wraps around to the bottom of the field.
"""

import numpy as np
import pytest
from numba.typed import List

from vhsdecode.chroma.heterodyne import BurstInfo, chroma_automatic_gain, upconvert_chroma

OUTWIDTH = 1135
LINESOUT = 313
LOCK_LINEOFFSET = 1  # field.lineoffset (0) + 1 when the burst lock runs


def _phase_sequence(amplitude=1.0, vbi_lines=0, vbi_amplitude=0.01):
    seq = List()
    for line in range(LOCK_LINEOFFSET, LOCK_LINEOFFSET + LINESOUT):
        amplitude_here = vbi_amplitude if line <= vbi_lines else amplitude
        start = (line - LOCK_LINEOFFSET) * OUTWIDTH
        burst = BurstInfo(
            line, start, start + 60, start + 30, 0.0, amplitude_here, amplitude_here, 0.0, 4.43361875e6, 0.0, 0.0
        )
        burst.phase_rotation = line % 4
        seq.append(burst)
    return seq


@pytest.mark.parametrize("lineoffset", [3, 4])
def test_upconvert_chroma_does_not_wrap_early_bursts(lineoffset):
    rng = np.random.default_rng(0)
    chroma = rng.standard_normal(LINESOUT * OUTWIDTH).astype(np.float32)
    heterodyne = rng.standard_normal((4, LINESOUT * OUTWIDTH)).astype(np.float32)
    uphet = np.zeros(LINESOUT * OUTWIDTH, dtype=np.float32)

    upconvert_chroma(chroma, uphet, lineoffset, OUTWIDTH, _phase_sequence(), heterodyne)

    rows = uphet.reshape(LINESOUT, OUTWIDTH)
    last_line = LOCK_LINEOFFSET + LINESOUT - 1
    last_row = last_line - lineoffset
    for row in range(last_row + 1):
        rotation = (row + lineoffset) % 4
        expected = chroma[row * OUTWIDTH : (row + 1) * OUTWIDTH] * heterodyne[rotation][
            row * OUTWIDTH : (row + 1) * OUTWIDTH
        ]
        np.testing.assert_array_equal(rows[row], expected)
    # Rows past the last locked line have no burst entry of their own (on a
    # real field they are inside the burst blanking around vsync) and must not
    # pick up the up-conversion of the entries above the first output line.
    assert not rows[last_row + 1 :].any()


BURST_REF = 8.0


def _gained_rows(lineoffset, burst_detected_line):
    chroma = np.ones(LINESOUT * OUTWIDTH, dtype=np.float32)
    # The first entries fall inside the burst blanking around vsync on a real
    # field, so their measured amplitude is tiny and their gain hits the clamp.
    seq = _phase_sequence(amplitude=1.0, vbi_lines=3)
    chroma_automatic_gain(chroma, BURST_REF, seq, burst_detected_line, 50, lineoffset, OUTWIDTH)
    return chroma.reshape(LINESOUT, OUTWIDTH)


@pytest.mark.parametrize("lineoffset", [3, 4])
def test_chroma_agc_does_not_regain_bottom_rows(lineoffset):
    rows = _gained_rows(lineoffset, burst_detected_line=0)
    # Every row gets the single clamped gain of BURST_REF / 1.0; wrapped
    # entries used to apply a second gain on top to the bottom rows.
    np.testing.assert_allclose(rows, BURST_REF, rtol=1e-5)


@pytest.mark.parametrize("lineoffset", [3, 4])
def test_chroma_agc_colour_killer_does_not_blank_bottom_rows(lineoffset):
    burst_detected_line = 6
    rows = _gained_rows(lineoffset, burst_detected_line)
    first_coloured_row = burst_detected_line - lineoffset
    assert not rows[:first_coloured_row].any()
    np.testing.assert_allclose(rows[first_coloured_row:], BURST_REF, rtol=1e-5)
