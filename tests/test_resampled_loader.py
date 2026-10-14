"""Tests for LoadResampled, the in-process replacement for resampling raw
files through ffmpeg.

DemodCache reads overlapping blocks, so the loader must return the same
samples for the same output range no matter where a read starts.
"""

import io

import numpy as np
import pytest
import scipy.signal as sps

from lddecode.utils import LoadFFmpeg, LoadResampled, make_loader


def _file(samples, dtype="<i2"):
    return io.BytesIO(np.asarray(samples).astype(dtype).tobytes())


@pytest.mark.parametrize(
    "name,fmt",
    [
        ("a.s16", "s16"),
        ("a.raw", "s16"),
        ("a.u16", "u16"),
        ("a.r16", "u16"),
        ("a.tbc", "u16"),
        ("a.rf", "f32"),
        ("a.s8", "s8"),
        ("a.u8", "u8"),
        ("a.r8", "u8"),
    ],
)
def test_raw_formats_resample_in_process(name, fmt):
    loader = make_loader(name, inputfreq=20)

    assert isinstance(loader, LoadResampled)
    assert loader.dtype == np.dtype(LoadResampled.FORMATS[fmt][0])


def test_other_formats_still_use_ffmpeg():
    assert isinstance(make_loader("a.flac", inputfreq=20), LoadFFmpeg)


def test_40mhz_input_is_not_resampled():
    assert not isinstance(make_loader("a.s16", inputfreq=40), LoadResampled)


@pytest.mark.parametrize("inputfreq", [20, 28.636363, 35.8, 50])
def test_overlapping_reads_are_identical(inputfreq):
    rng = np.random.default_rng(1)
    infile = _file(rng.integers(-20000, 20000, 400_000))
    loader = LoadResampled("s16", inputfreq)

    whole = loader(infile, 1000, 200_000)
    for start in (1000, 1001, 37_123, 100_000):
        part = loader(infile, start, 32768)
        np.testing.assert_array_equal(part, whole[start - 1000 : start - 1000 + 32768])


def test_matches_resampling_the_whole_file():
    rng = np.random.default_rng(2)
    raw = rng.integers(-20000, 20000, 100_000)
    loader = LoadResampled("s16", 20)

    ref = sps.resample_poly(raw.astype(np.float32), 2, 1, window=loader.filter)

    np.testing.assert_array_equal(loader(_file(raw), 0, 32768), ref[:32768])
    np.testing.assert_array_equal(loader(_file(raw), 50_000, 32768), ref[50_000:82_768])


def test_passes_a_tone_at_unity_gain():
    """A 3 MHz tone sampled at 20 MHz comes out at 40 MHz with the same amplitude."""
    t = np.arange(200_000) / 20e6
    raw = np.round(10000 * np.sin(2 * np.pi * 3e6 * t))
    out = LoadResampled("s16", 20)(_file(raw), 1000, 100_000)

    t_out = (np.arange(100_000) + 1000) / 40e6
    expected = 10000 * np.sin(2 * np.pi * 3e6 * t_out)
    assert np.abs(out - expected).max() < 10


@pytest.mark.parametrize(
    "fmt,dtype,raw,expected",
    [
        ("u16", "<u2", [32768, 32769], [0, 1]),
        ("f32", "<f4", [0.5, -0.25], [16384, -8192]),
        ("s8", "i1", [1, -2], [256, -512]),
        ("u8", "u1", [129, 126], [256, -512]),
    ],
)
def test_samples_are_converted_like_ffmpeg(fmt, dtype, raw, expected):
    """Values are mapped to the s16 range the same way ffmpeg's pcm_s16le is."""
    loader = LoadResampled(fmt, 20)
    np.testing.assert_array_equal(loader._read(_file(raw, dtype), 0, len(raw)), expected)


def test_short_read_at_end_of_file_returns_none():
    infile = _file(np.zeros(50_000))
    loader = LoadResampled("s16", 20)

    assert loader(infile, 0, 99_000) is not None
    assert loader(infile, 90_000, 32768) is None
