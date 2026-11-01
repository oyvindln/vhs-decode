"""The colour burst Gauss-Newton fit should recover a clean burst's parameters
and stop once it has converged rather than running to max_iter."""

import numpy as np

from vhsdecode.chroma import _tune_burst_measurements

FSC = 4433618.75  # PAL


def _burst(A, phi, dc, f, start, n):
    t = (np.arange(n) + start) / (4.0 * FSC)
    return A * np.cos(2 * np.pi * f * t - phi) + dc


def test_recovers_burst_parameters():
    burst = _burst(150.0, 0.7, 12.0, FSC, 400, 240)
    A, phi, dc, f = _tune_burst_measurements(burst, 400, 140.0, 0.6, 10.0, FSC)

    assert abs(A - 150.0) < 1e-6
    assert abs(phi - 0.7) < 1e-9
    assert abs(dc - 12.0) < 1e-6
    assert abs(f - FSC) < 1e-3


def test_converges_well_before_max_iter():
    # With noise the updates settle at float rounding level instead of exactly
    # zero, which an absolute tolerance on f (in Hz) never got below.
    rng = np.random.default_rng(3)
    burst = _burst(150.0, -2.0, 0.0, FSC, 1000, 240) + rng.normal(0, 5, 240)
    full = _tune_burst_measurements(burst, 1000, 140.0, -1.9, 1.0, FSC, max_iter=32)
    few = _tune_burst_measurements(burst, 1000, 140.0, -1.9, 1.0, FSC, max_iter=10)

    # if it stopped within 10 iterations, both runs return the same result
    assert full == few
