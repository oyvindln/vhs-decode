#!/usr/bin/python3
#
# Tests for AC3-RF demodulation.
#
# The loopback tests modulate a random symbol stream as differential QPSK
# on a 2.88 MHz carrier and check that lddecode/ac3rf.py recovers it.
#
# The end-to-end test additionally synthesizes a complete NTSC LD signal
# carrying that subcarrier and runs ld-decode --AC3 over it, covering the
# demodulator's integration into the decoder (per-field demodulation,
# .ac3sym output and the per-field symbol counts recorded in the metadata).

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

from lddecode.ac3rf import Ac3RfDemodulator, SYMBOL_RATE, CARRIER_FREQ


def dqpsk_modulate(
    symbols, sample_rate, amplitude=1.0, carrier_freq=CARRIER_FREQ, symbol_rate=SYMBOL_RATE
):
    """Generate a real-valued DQPSK signal at carrier_freq.

    The demodulator mixes with exp(+j w t), so a transmitted phase phi is
    recovered as -phi at baseband; the differential decoder maps symbol 1
    to a -90 degree step, 2 to +90, 3 to 180.

    The carrier and symbol rate default to the demodulator's own, which
    is what a loopback test wants; pass them explicitly to modulate
    against fixed values independent of what ac3rf.py believes."""
    phase_step = {0: 0.0, 1: -np.pi / 2, 2: np.pi / 2, 3: np.pi}
    phases = np.cumsum([phase_step[s] for s in symbols])

    n_samples = int(len(symbols) * sample_rate / symbol_rate)
    t = np.arange(n_samples)
    symbol_index = (t * symbol_rate / sample_rate).astype(int)
    return (
        amplitude * np.cos(2 * np.pi * carrier_freq / sample_rate * t + phases[symbol_index])
    ).astype(np.float32)


def find_and_compare(tx, rx, probe_start=500, probe_len=64):
    """Align rx against tx using a probe subsequence, then return the
    match rate over the overlapping tail."""
    probe = tx[probe_start : probe_start + probe_len]
    rx_bytes = rx.tobytes()
    pos = rx_bytes.find(probe.tobytes())
    assert pos >= 0, "transmitted probe sequence not found in demodulated output"

    tx_tail = tx[probe_start:]
    rx_tail = rx[pos : pos + len(tx_tail)]
    n = min(len(tx_tail), len(rx_tail))
    assert n > 1000
    return np.mean(tx_tail[:n] == rx_tail[:n])


def test_loopback_clean():
    rng = np.random.default_rng(12345)
    tx = rng.integers(0, 4, 20000, dtype=np.uint8)
    rf = dqpsk_modulate(tx, 40e6)

    demod = Ac3RfDemodulator(40e6)
    assert demod.input_sample_alignment() == 1

    # Feed the signal in uneven block sizes to exercise the streaming state
    rx = b""
    pos = 0
    for blocklen in [100001, 65536, 999999] * 100:
        if pos >= len(rf):
            break
        rx += demod.demodulate_to_symbols(rf[pos : pos + blocklen])
        pos += blocklen

    rx = np.frombuffer(rx, np.uint8)
    match_rate = find_and_compare(tx, rx)
    assert match_rate > 0.999, f"symbol match rate {match_rate}"


def test_loopback_noisy():
    rng = np.random.default_rng(999)
    tx = rng.integers(0, 4, 20000, dtype=np.uint8)
    rf = dqpsk_modulate(tx, 40e6)
    rf += rng.normal(0, 0.25, len(rf)).astype(np.float32)

    demod = Ac3RfDemodulator(40e6)
    rx = np.frombuffer(demod.demodulate_to_symbols(rf), np.uint8)

    match_rate = find_and_compare(tx, rx)
    assert match_rate > 0.99, f"symbol match rate {match_rate}"


# NTSC LD signal parameters, matching SysParams_NTSC in lddecode/core.py.
SAMPLE_RATE = 40e6
HALFLINE_US = 63.55555555555555 / 2
HALFLINES_PER_FIELD = 525  # 262.5 lines
IRE0_HZ = 8100000.0
HZ_IRE = 1700000 / 140.0
SYNC_IRE = -40.0
BLACK_IRE = 7.5
HSYNC_US = 4.7
EQ_US = 2.3
VSYNC_US = 27.1
NUM_PULSES = 6

# Subcarrier level relative to the video FM carrier.  On a real disc the
# audio carriers sit well below the video carrier.
QPSK_AMPLITUDE = 0.15

# The AC3 subcarrier as specified for LD, stated here rather than taken
# from ac3rf.py so that the end-to-end test is modulating against the
# spec and would notice if the demodulator's own values drifted from it.
AC3_CARRIER_HZ = 2.88e6
AC3_SYMBOL_RATE = 288e3

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def build_field_ire():
    """Return one NTSC field as an IRE waveform, over a black raster.

    Each field is 262.5 lines: 6 equalizing pulses, 6 serrated vertical
    sync pulses and 6 more equalizing pulses (all at twice line rate),
    then normal lines each with a horizontal sync pulse.  Fields are
    identical, and interlace falls out of a field being an odd number of
    half-lines, so successive fields start on alternating whole-line and
    half-line boundaries."""
    total = int(round(HALFLINES_PER_FIELD * HALFLINE_US * SAMPLE_RATE / 1e6))
    ire = np.full(total, BLACK_IRE, dtype=np.float64)

    def punch(start_us, dur_us):
        start = int(round(start_us * SAMPLE_RATE / 1e6))
        end = int(round((start_us + dur_us) * SAMPLE_RATE / 1e6))
        ire[start : min(end, total)] = SYNC_IRE

    for halfline in range(HALFLINES_PER_FIELD):
        t = halfline * HALFLINE_US
        if halfline < NUM_PULSES:
            punch(t, EQ_US)
        elif halfline < 2 * NUM_PULSES:
            punch(t, VSYNC_US)
        elif halfline < 3 * NUM_PULSES:
            punch(t, EQ_US)
        elif (halfline - 3 * NUM_PULSES) % 2 == 0:
            punch(t, HSYNC_US)

    return ire


def synthesize_disc_rf(n_fields, rng):
    """Return (samples, symbols) for a synthetic AC3 disc RF signal.

    The video is FM modulated onto the LD video carrier, and the AC3 QPSK
    subcarrier is mixed in underneath it."""
    ire = np.tile(build_field_ire(), n_fields)
    video = np.cos(np.cumsum(2 * np.pi * (IRE0_HZ + ire * HZ_IRE) / SAMPLE_RATE))

    n_symbols = int(np.ceil(len(video) * AC3_SYMBOL_RATE / SAMPLE_RATE))
    symbols = rng.integers(0, 4, n_symbols, dtype=np.uint8)
    qpsk = dqpsk_modulate(
        symbols,
        SAMPLE_RATE,
        amplitude=QPSK_AMPLITUDE,
        carrier_freq=AC3_CARRIER_HZ,
        symbol_rate=AC3_SYMBOL_RATE,
    )

    n = min(len(video), len(qpsk))
    combined = video[:n] + qpsk[:n]
    combined *= 0.8 / np.max(np.abs(combined))
    return (combined * 32767).astype(np.int16), symbols


def run_ld_decode(rf_path, out_base):
    """Run ld-decode --AC3 over rf_path, writing output to out_base."""
    env = dict(os.environ)
    # Run against this checkout, whether or not lddecode is installed.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "ld-decode"), "--AC3", str(rf_path), str(out_base)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"ld-decode failed:\n{result.stderr}"


def test_decode_ac3_end_to_end(tmp_path):
    rng = np.random.default_rng(1234)
    samples, tx = synthesize_disc_rf(10, rng)

    # .s16 is raw 16-bit samples at 40 MHz, which ld-decode reads directly.
    rf_path = tmp_path / "ac3-synthetic.s16"
    samples.tofile(rf_path)
    out_base = tmp_path / "ac3-synthetic"
    run_ld_decode(rf_path, out_base)

    rx = np.fromfile(str(out_base) + ".ac3sym", np.uint8)
    assert len(rx) > 0, "no symbols were written"
    assert set(np.unique(rx)) <= {0, 1, 2, 3}, f"symbols out of range: {np.unique(rx)}"

    # ld-decode starts part way into the file, so the demodulated stream is
    # a subsequence of the transmitted one: probe from it to find where.
    match_rate = find_and_compare(rx, tx)
    assert match_rate > 0.99, f"symbol match rate {match_rate}"

    # Each field records how many symbols were demodulated during it, so
    # that consumers can reconstruct each field's range by summation; the
    # counts must therefore account for the .ac3sym file exactly.
    with open(str(out_base) + ".tbc.json") as f:
        fields = json.load(f)["fields"]
    counts = [field["ac3Symbols"] for field in fields]
    assert len(counts) > 1, "expected more than one field to be decoded"
    assert all(count > 0 for count in counts), f"field with no symbols: {counts}"
    assert sum(counts) == len(rx), (
        f"per-field counts sum to {sum(counts)}, but .ac3sym holds {len(rx)} symbols"
    )


if __name__ == "__main__":
    test_loopback_clean()
    test_loopback_noisy()
    print("OK")
