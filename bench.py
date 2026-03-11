"""Benchmark suite for vhs-decode signal processing pipeline.

Run via: make bench
Or directly: python bench.py
"""

import time
import sys
import os
import statistics
import numpy as np

ITERATIONS_DEMOD = 50
ITERATIONS_DSP = 500


def _fmt(values_us):
    """Format a list of microsecond timings as a stats line."""
    mean = statistics.mean(values_us)
    stdev = statistics.stdev(values_us) if len(values_us) > 1 else 0
    lo = min(values_us)
    hi = max(values_us)
    return f"{mean:10.1f} us  (std {stdev:8.1f}, min {lo:10.1f}, max {hi:10.1f})"


def _section(title):
    print()
    print(f"{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def bench_demodblock():
    """Benchmark the full demodblock pipeline (PAL + NTSC)."""
    import vhsdecode.process as process
    import vhsdecode.utils as utils

    _section("demodblock — full signal chain")

    for system, max_hz, min_hz in [("PAL", 4800000, 3800000), ("NTSC", 4400000, 3400000)]:
        samplerate_mhz = 40
        decoder = process.VHSRFDecode(inputfreq=samplerate_mhz, system=system)
        wavemax = utils.gen_wave_at_frequency(
            max_hz / 1e6, samplerate_mhz, decoder.blocklen // 2
        )
        wavemin = utils.gen_wave_at_frequency(
            min_hz / 1e6, samplerate_mhz, decoder.blocklen // 2
        )
        wave = np.concatenate((wavemax, wavemin))

        # Warmup
        for _ in range(3):
            decoder.demodblock(data=wave)

        timings = []
        for _ in range(ITERATIONS_DEMOD):
            t0 = time.perf_counter()
            decoder.demodblock(data=wave)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1e6)

        print(f"  {system:5s} ({decoder.blocklen} samples): {_fmt(timings)}")

        budget_ms = 40.0  # 25fps frame = 40ms
        mean_ms = statistics.mean(timings) / 1000
        blocks_per_frame = 2 * 25  # ~50 blocks/frame for PAL
        frame_cost_ms = mean_ms * blocks_per_frame
        print(f"         est. {frame_cost_ms:.1f} ms/frame of {budget_ms:.0f} ms budget "
              f"({frame_cost_ms / budget_ms * 100:.0f}%) @ {blocks_per_frame} blocks/frame")


def bench_dsp_primitives():
    """Benchmark individual DSP functions using hilbert_data.npz."""
    _section("DSP primitives — 32K complex samples")

    fixture = "hilbert_data.npz"
    if not os.path.exists(fixture):
        print(f"  SKIPPED — {fixture} not found")
        return

    loaded = np.load(fixture)
    complex_data = loaded["data"]
    n = len(complex_data)
    real_data = np.angle(complex_data).astype(np.float64)
    rfft_data = np.fft.rfft(real_data)

    benchmarks = {}

    # unwrap_hilbert (Rust — the hot path for VHS)
    try:
        import vhsd_rust

        def _unwrap_hilbert():
            vhsd_rust.unwrap_hilbert(complex_data, 40e6)

        benchmarks["unwrap_hilbert (rust)"] = _unwrap_hilbert
    except ImportError:
        pass

    # complex_angle (Rust)
    try:
        from vhsd_rust import complex_angle_py

        def _complex_angle():
            complex_angle_py(complex_data)

        benchmarks["complex_angle (rust)"] = _complex_angle
    except ImportError:
        pass

    # complex_angle (numpy baseline)
    def _np_angle():
        np.angle(complex_data)

    benchmarks["np.angle (numpy)"] = _np_angle

    # diff_forward_in_place (Rust)
    try:
        from vhsd_rust import diff_forward_in_place

        def _diff_fwd():
            buf = real_data.copy()
            diff_forward_in_place(buf)

        benchmarks["diff_forward_in_place (rust)"] = _diff_fwd
    except ImportError:
        pass

    # np.ediff1d (numpy baseline)
    def _ediff():
        np.ediff1d(real_data, to_begin=0)

    benchmarks["np.ediff1d (numpy)"] = _ediff

    # np.unwrap
    def _unwrap():
        np.unwrap(real_data)

    benchmarks["np.unwrap"] = _unwrap

    # sosfiltfilt (Rust)
    try:
        from vhsd_rust import sosfiltfilt
        import scipy.signal as sps

        sos_sections = sps.butter(4, 0.1, output="sos")
        sos = sos_sections.flatten()
        order = sos_sections.shape[0]

        def _sosfilt_rust():
            sosfiltfilt(order, sos, real_data)

        benchmarks["sosfiltfilt (rust)"] = _sosfilt_rust

        def _sosfilt_scipy():
            sps.sosfiltfilt(sps.butter(4, 0.1, output="sos"), real_data)

        benchmarks["sosfiltfilt (scipy)"] = _sosfilt_scipy
    except ImportError:
        pass

    # Envelope: old (sps.hilbert) vs new (_envelope_from_rfft)
    import scipy.signal as sps
    from vhsdecode.nonlinear_filter import _envelope_from_rfft

    def _envelope_old():
        hf_part = np.fft.irfft(rfft_data)
        np.abs(sps.hilbert(hf_part))

    def _envelope_new():
        hf_part = np.fft.irfft(rfft_data)
        _envelope_from_rfft(rfft_data, n)

    benchmarks["envelope: sps.hilbert (old)"] = _envelope_old
    benchmarks["envelope: rfft direct  (new)"] = _envelope_new

    # Run all benchmarks
    for name, fn in benchmarks.items():
        # Warmup
        for _ in range(5):
            fn()
        timings = []
        for _ in range(ITERATIONS_DSP):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1e6)
        print(f"  {name:38s} {_fmt(timings)}")


def bench_pulse_detection():
    """Benchmark pulse detection using PAL fixture data."""
    _section("Pulse detection — PAL field data")

    fixture = "PAL_GOOD.txt.gz"
    if not os.path.exists(fixture):
        print(f"  SKIPPED — {fixture} not found")
        return

    demod_data = np.loadtxt(fixture)
    print(f"  Fixture: {fixture} — {len(demod_data)} samples")

    try:
        from vhsdecode.addons.resync import _findpulses_numba_raw

        # Warmup (includes numba JIT compilation)
        _findpulses_numba_raw(demod_data, 3954307.8, 11.625, 1588.125)
        _findpulses_numba_raw(demod_data, 3954307.8, 11.625, 1588.125)

        timings = []
        for _ in range(ITERATIONS_DSP):
            t0 = time.perf_counter()
            _findpulses_numba_raw(demod_data, 3954307.8, 11.625, 1588.125)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1e6)

        print(f"  {'_findpulses_numba_raw':38s} {_fmt(timings)}")
    except Exception as e:
        print(f"  SKIPPED — {e}")


def main():
    print(f"vhs-decode benchmark")
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    try:
        import scipy
        print(f"SciPy {scipy.__version__}", end="")
    except ImportError:
        pass
    try:
        import numba
        print(f", Numba {numba.__version__}", end="")
    except ImportError:
        pass
    try:
        import vhsd_rust
        print(f", vhsd_rust loaded", end="")
    except ImportError:
        print(f", vhsd_rust NOT FOUND", end="")
    print()
    print(f"Iterations: demodblock={ITERATIONS_DEMOD}, primitives={ITERATIONS_DSP}")

    bench_demodblock()
    bench_dsp_primitives()
    bench_pulse_detection()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
