"""GPU paths (vhsdecode.gpu) against the CPU code they replace.

Skipped when CuPy or a CUDA device isn't available.
"""

import io
import types

import numpy as np
import pytest

from vhsdecode import gpu

ok, reason = gpu.gpu_available()
pytestmark = pytest.mark.skipif(not ok, reason=f"GPU not available: {reason}")

if ok:
    import cupy as cp

import lddecode.utils as lddu
import vhsdecode.process as process


def _fm_blocks(decoder, nblocks, seed=0):
    """Blocks of a PAL VHS-like FM signal: a carrier swinging between sync and
    white, plus a colour-under tone and a little noise."""
    rng = np.random.default_rng(seed)
    n = decoder.blocklen * nblocks
    t = np.arange(n) / decoder.freq_hz
    inst = 4.3e6 + 0.45e6 * np.sin(2 * np.pi * 15625 * t)
    phase = 2 * np.pi * np.cumsum(inst) / decoder.freq_hz
    sig = 8000 * np.cos(phase) + 600 * np.cos(2 * np.pi * 626.9e3 * t) + rng.normal(0, 50, n)
    return sig.astype(np.float32).reshape(nblocks, decoder.blocklen)


def test_demod_matches_cpu():
    decoder = process.VHSRFDecode(inputfreq=40, system="PAL")
    assert gpu.GPUDemod.unsupported_reason(decoder) is None
    rows = _fm_blocks(decoder, 4)

    cpu = [decoder.demodblock(data=r.copy(), cut=True)["video"] for r in rows]
    out, packed = gpu.GPUDemod(decoder).demod(cp.asarray(rows))

    for name, atol in (("demod", 2.0), ("demod_05", 2.0), ("envelope", 0.05)):
        ref = np.stack([c[name] for c in cpu])
        np.testing.assert_allclose(out[name], ref, rtol=0, atol=atol, err_msg=name)

    # Chroma: the |H|^2 mask equals sosfiltfilt away from the block edges.
    ref = np.stack([c["demod_burst"] for c in cpu]).astype(np.float64)
    got = out["demod_burst"].astype(np.float64)
    mid = slice(4096, -4096)
    assert np.abs(got - ref)[:, mid].max() < 2e-3 * np.abs(ref[:, mid]).max()

    # the GPU copy holds the same data
    for name, (dtype, col) in gpu.GPU_CHANNELS.items():
        np.testing.assert_array_equal(packed.view(dtype)[:, :, col].get(), out[name])


def test_resampler_matches_cpu_loader():
    raw = np.random.default_rng(1).integers(-20000, 20000, 400_000).astype("<i2")
    infile = io.BytesIO(raw.tobytes())
    loader = lddu.LoadResampled("s16", 20)

    starts = [5000 + i * 31712 for i in range(6)]
    rows, cpu_rows = gpu.GPUResampler(loader).load_rows(infile, starts, 32768)

    for i, s in enumerate(starts):
        ref = loader(infile, s, 32768)
        np.testing.assert_allclose(cpu_rows[i], ref, rtol=0, atol=0.02)
    np.testing.assert_array_equal(rows.get(), cpu_rows)


def test_resampler_end_of_file():
    infile = io.BytesIO(np.zeros(40_000, "<i2").tobytes())
    rows, _ = gpu.GPUResampler(lddu.LoadResampled("s16", 20)).load_rows(infile, [0, 31712, 63424], 32768)
    assert rows is None


def test_scale_field_matches_cpu():
    rng = np.random.default_rng(2)
    lut = np.load(lddu.__file__.replace("utils.py", "sinc_lut.npz"))["downscale_sinc_lut"]
    outwidth, lines, lineoffset = 1135, 313, 0
    inlinelen = 2560.0
    linelocs = np.arange(lines + 4) * inlinelen + 3000 + rng.normal(0, 2, lines + 4)
    count = (lines + lineoffset + 1) * outwidth
    locs, wow = lddu.linear_wow(linelocs, inlinelen, inlinelen / outwidth, count)
    buf = rng.normal(4.3e6, 2e5, int(linelocs[-1]) + 100).astype(np.float32)

    ref = np.zeros(lines * outwidth, dtype=np.float32)
    lddu.scale_field(buf, ref, locs, wow, lut, lineoffset, outwidth, wow_level_adjust_smoothing=312.5)

    field = types.SimpleNamespace(wow_level_adjust_smoothing=312.5, data={})
    got = np.zeros_like(ref)
    gpu.GPUScaler(lut).scale_field(field, "demod", buf, got, locs, wow, lineoffset, outwidth)

    np.testing.assert_allclose(got, ref, rtol=1e-6)
