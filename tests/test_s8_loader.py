"""Tests for reading .s8 (signed 8-bit) source files.

make_loader() recognised .s8 only on its resampling path, where a --frequency
was given and ffmpeg does the reading. Without --frequency the extension fell
through every branch to the bare LoadFFmpeg() fallback, which reads from stdin
with no format arguments: it produced no samples, so the decode ended with
"Completed without handling any frames" without ever having read the file.

See test_json_dumper_empty.py for the crash that empty decode then triggered.
"""

import io

import numpy as np

from lddecode.utils import (
    LoadFFmpeg,
    load_unpacked_data_s8,
    make_loader,
)


def test_s8_maps_to_its_own_loader_without_a_frequency():
    """.s8 with no --frequency must not land on the stdin ffmpeg fallback."""
    loader = make_loader("capture.s8")

    assert loader is load_unpacked_data_s8
    assert not isinstance(loader, LoadFFmpeg)


def test_s8_still_uses_ffmpeg_when_resampling():
    """With a --frequency, ffmpeg does the reading and must be told the format."""
    loader = make_loader("capture.s8", inputfreq=62.5)

    assert isinstance(loader, LoadFFmpeg)
    assert loader.input_args == ["-f", "s8"]


def test_s8_samples_are_scaled_to_the_16_bit_range():
    """Match the ffmpeg path, which converts s8 to pcm_s16le by scaling by 256.

    The same file has to decode identically whichever loader reads it, so the
    signed 8-bit values are widened rather than passed through at their native
    amplitude.
    """
    raw = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    infile = io.BytesIO(raw.tobytes())

    out = load_unpacked_data_s8(infile, 0, len(raw))

    np.testing.assert_array_equal(out, raw.astype(np.int16) * 256)


def test_s8_reads_from_the_right_offset():
    """Sample offsets are one byte apart, not two -- s8 is a one-byte format."""
    raw = np.arange(-8, 8, dtype=np.int8)
    infile = io.BytesIO(raw.tobytes())

    out = load_unpacked_data_s8(infile, 3, 4)

    np.testing.assert_array_equal(out, raw[3:7].astype(np.int16) * 256)


def test_s8_short_read_returns_none():
    """A truncated read is EOF, reported the same way as the other loaders."""
    raw = np.zeros(4, dtype=np.int8)
    infile = io.BytesIO(raw.tobytes())

    assert load_unpacked_data_s8(infile, 0, 64) is None
