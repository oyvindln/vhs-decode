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


def test_demod_chroma_is_clean_at_block_edges():
    """Colour-under chroma from demodblock should match filtering the whole
    signal in one go, right up to the edges of each block's kept region.
    (sosfiltfilt per block left transients as large as the chroma there.)"""
    import scipy.signal as sps

    import vhsdecode.process as process

    decoder = process.VHSRFDecode(inputfreq=40, system="PAL")
    n = decoder.blocklen
    step = n - decoder.blockcut - decoder.blockcut_end
    rng = np.random.default_rng(0)
    total = step * 6 + n
    t = np.arange(total) / decoder.freq_hz
    phase = 2 * np.pi * np.cumsum(4.3e6 + 0.45e6 * np.sin(2 * np.pi * 15625 * t)) / decoder.freq_hz
    sig = (8000 * np.cos(phase) + 600 * np.cos(2 * np.pi * 626.9e3 * t)
           + rng.normal(0, 50, total)).astype(np.float32)

    ref = np.roll(sps.sosfiltfilt(decoder.Filters["FVideoBurst"], sig.astype(np.float64)),
                  int(decoder.options.chroma_offset))

    for b in range(1, 5):
        start = b * step
        chroma = decoder.demodblock(data=sig[start : start + n].copy(), cut=True)["video"]["demod_burst"]
        expected = ref[start + decoder.blockcut : start + n - decoder.blockcut_end]
        err = np.abs(chroma - expected)
        assert err.max() < 0.01 * np.abs(expected).max(), f"block {b}: max error {err.max()}"
