"""Burst-indexed chroma passes must stay inside the field.

The burst lock runs while the field is decoded, before PAL fields set their
lineoffset (2 or 3), so field.phase_sequence starts at line 1 while the chroma
passes run with lineoffset + 1 = 3 or 4. The first burst entries then map to
negative rows, which numba (like numpy) wraps around to the bottom of the field.
"""

import numpy as np
import pytest
from numba.typed import List

from vhsdecode.chroma import BurstInfo, upconvert_chroma

OUTWIDTH = 1135
LINESOUT = 313
LOCK_LINEOFFSET = 1  # field.lineoffset (0) + 1 when the burst lock runs


def _phase_sequence(amplitude=1.0):
    seq = List()
    for line in range(LOCK_LINEOFFSET, LOCK_LINEOFFSET + LINESOUT):
        start = (line - LOCK_LINEOFFSET) * OUTWIDTH
        burst = BurstInfo(
            line, start, start + 60, start + 30, 0.0, amplitude, amplitude, 0.0, 4.43361875e6, 0.0, 0.0
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
