import numpy as np
import scipy.fft as npfft

from lddecode.core import RFDecode
from lddecode.utils import unwrap_hilbert


def _legacy_outputs(rf, signal):
    indata_fft = npfft.fft(signal)
    hilbert = npfft.ifft(indata_fft * rf.Filters["RFVideo"])
    demod = unwrap_hilbert(hilbert, rf.freq_hz)
    demod_fft = npfft.fft(np.clip(demod, 1500000, rf.freq_hz * 0.75))

    video = [
        npfft.ifft(demod_fft * rf.Filters["FVideo"]).real,
        np.roll(
            npfft.ifft(demod_fft * rf.Filters["FVideo05"]).real,
            -rf.Filters["F05_offset"],
        ),
        np.roll(
            npfft.ifft(demod_fft * rf.Filters["FVideoBurst"]).real,
            -rf.Filters["FVideoBurst_offset"],
        ),
    ]
    if rf.system == "PAL":
        video.append(npfft.ifft(demod_fft * rf.Filters["FVideoPilot"]).real)

    rfhpf = npfft.ifft(indata_fft * rf.Filters["Frfhpf"]).real
    return demod, video, rfhpf


def _check_outputs(system):
    rf = RFDecode(system=system)
    rng = np.random.default_rng(12345)
    signal = rng.integers(0, 16384, rf.blocklen).astype(np.float64)

    demod, expected_video, expected_rfhpf = _legacy_outputs(rf, signal)
    actual = rf.demodblock(data=signal)
    sync = rf.demodblock_sync(data=signal)

    names = ["demod", "demod_05", "demod_burst"]
    if system == "PAL":
        names.append("demod_pilot")

    np.testing.assert_allclose(actual["video"]["demod_raw"], demod, rtol=1e-6)
    for name, expected in zip(names, expected_video):
        np.testing.assert_allclose(actual["video"][name], expected, rtol=1e-6)
    np.testing.assert_allclose(sync, expected_video[1], rtol=1e-6)

    rotdelay = rf.delays.get("video_rot", 0)
    expected_rfhpf = expected_rfhpf[
        rf.blockcut - rotdelay : -rf.blockcut_end - rotdelay
    ]
    np.testing.assert_allclose(actual["rfhpf"], expected_rfhpf, rtol=1e-6)


def test_ntsc_outputs_match_full_fft():
    _check_outputs("NTSC")


def test_pal_outputs_match_full_fft():
    _check_outputs("PAL")
