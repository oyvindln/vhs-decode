"""shift_chroma_and_remove_dc: shift with wraparound, then remove the DC
measured over an optional range."""

import numpy as np

from vhsdecode.chroma import shift_chroma_and_remove_dc


def test_shift_and_remove_whole_mean():
    x = np.arange(10, dtype=np.float32)
    expected = np.roll(x, 3) - x.mean()
    shift_chroma_and_remove_dc(x, 3)
    np.testing.assert_allclose(x, expected)


def test_dc_from_range_ignores_edges():
    # a tone with big transients at both ends, like sosfiltfilt at block edges
    n = 4096
    x = (100 * np.sin(np.arange(n) * 2 * np.pi / 64)).astype(np.float32)
    x[:200] += 5000
    x[-200:] -= 2000
    shifted = np.roll(x, 5).astype(np.float64)

    y = x.copy()
    shift_chroma_and_remove_dc(y, 5, 512, n - 512)

    np.testing.assert_allclose(y, shifted - shifted[512 : n - 512].mean(), atol=1e-3)
    assert abs(y[512 : n - 512].mean()) < 1e-3
