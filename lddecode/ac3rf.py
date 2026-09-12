#!/usr/bin/python3
#
# ac3rf.py - AC3-RF QPSK demodulation for LaserDisc
# Copyright (C) 2025-2026 Staffan Ulfberg
#
# This file is part of ld-decode.
#
# ac3rf.py is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# On AC3 discs the analog right audio channel is replaced by a QPSK
# signal at 2.88 MHz carrying Dolby Digital data at 288 kbaud.  This
# module demodulates the RF signal to a stream of raw differential QPSK
# symbols (one symbol per output byte, values 0-3).  Framing, Reed-Solomon
# error correction, and AC3 frame assembly happen downstream (decode-orc's
# ac3rf_sink stage reads the .ac3sym file written by ld-decode).
#
# This is a Python port of the Ac3RfDemodulator/Ac3DPLL classes from
# https://github.com/staffanu/museld (player/src/ac3/).

import numpy as np
import numba

try:
    from numba.experimental import jitclass
except ImportError:
    # Prior to numba 0.49
    from numba import jitclass


SYMBOL_RATE = 288e3
CARRIER_FREQ = 2.88e6

# Kaiser beta for ~80 dB stopband attenuation: beta = 0.1102 * (A_dB - 8.7)
KAISER_BETA_80DB = 7.857


def halfband_kaiser(ntaps, beta=KAISER_BETA_80DB):
    """Kaiser-windowed half-band lowpass FIR with cutoff Fs/4.

    ntaps must be 4k+3 (3, 7, 11, ...) so the every-other-zero half-band
    structure holds exactly."""
    assert ntaps % 4 == 3, "half-band length must be 4k+3"
    M = (ntaps - 1) // 2
    n = np.arange(-M, M + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sin(n * np.pi / 2) / (n * np.pi)
    h[M] = 0.5
    h[n % 2 == 0] = 0
    h[M] = 0.5
    return h * np.kaiser(ntaps, beta)


def rrc_filter(ntaps, samples_per_symbol, beta):
    """Root-raised cosine FIR, normalized to unit energy (matched-filter
    convention).  samples_per_symbol need not be an integer."""
    if ntaps % 2 == 0:
        ntaps += 1
    T = samples_per_symbol
    t = np.arange(ntaps) - ntaps // 2

    h = np.empty(ntaps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 / T * (1 + beta * (4 / np.pi - 1))
        elif abs(abs(4 * beta * ti / T) - 1) < 1e-10:
            h[i] = (
                beta
                / (T * np.sqrt(2))
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )
        else:
            h[i] = (
                1
                / T
                * (
                    np.sin(np.pi * ti / T * (1 - beta))
                    + 4 * beta * ti / T * np.cos(np.pi * ti / T * (1 + beta))
                )
                / (np.pi * ti / T * (1 - (4 * beta * ti / T) ** 2))
            )

    return h / np.sqrt(np.sum(h * h))


@numba.njit(cache=True, nogil=True, fastmath=True)
def _fir_decimate(data, tap_values, tap_offsets, ntaps, history, phase, decimation):
    """FIR filter + decimate in one pass, computing only the retained
    output samples and only the non-zero taps (half-band filters are
    nearly half zeros).

    tap_values/tap_offsets: the non-zero filter coefficients and their
    indices.  history: the last ntaps-1 samples of the previous call.
    Returns (output, new_history, new_phase)."""
    n = len(data)
    x = np.empty(n + ntaps - 1, data.dtype)
    x[: ntaps - 1] = history
    x[ntaps - 1 :] = data

    n_out = (n - phase + decimation - 1) // decimation
    output = np.empty(n_out, data.dtype)
    for j in range(n_out):
        base = phase + j * decimation + ntaps - 1
        acc = tap_values[0] * x[base - tap_offsets[0]]
        for k in range(1, len(tap_values)):
            acc += tap_values[k] * x[base - tap_offsets[k]]
        output[j] = acc

    return output, x[n:], (phase - n) % decimation


class _FirDecimStage:
    """Streaming FIR filter + decimator.  Carries the filter state and the
    decimation phase across calls, so input blocks may have any length."""

    def __init__(self, taps, decimation, dtype):
        taps = np.asarray(taps, np.float32)
        nonzero = np.nonzero(taps)[0]
        self.tap_values = np.ascontiguousarray(taps[nonzero])
        self.tap_offsets = np.ascontiguousarray(nonzero)
        self.ntaps = len(taps)
        self.decimation = decimation
        self.history = np.zeros(self.ntaps - 1, dtype)
        # Start decimation at the filter's group delay so the sampling
        # phase matches an implementation that end-aligns the taps.
        self.phase = (self.ntaps - 1) % decimation

    def process(self, data):
        output, self.history, self.phase = _fir_decimate(
            data,
            self.tap_values,
            self.tap_offsets,
            self.ntaps,
            self.history,
            self.phase,
            self.decimation,
        )
        return output


class _ComplexFirDecimStage:
    """Filters I and Q with the same real taps, as two independent real
    filter stages (real float32 loops vectorize much better than complex
    arithmetic)."""

    def __init__(self, taps, decimation):
        self.re_filter = _FirDecimStage(taps, decimation, np.float32)
        self.im_filter = _FirDecimStage(taps, decimation, np.float32)

    def process(self, data_re, data_im):
        return self.re_filter.process(data_re), self.im_filter.process(data_im)


# Attribute types of AC3_DPLL for numba.
AC3_DPLL_spec = [
    ("symbol_distance", numba.int32),
    ("nominal_add", numba.int32),
    ("prev_i", numba.float32[:]),
    ("prev_q", numba.float32[:]),
    ("prev_symbol", numba.uint8),
    ("error_sum", numba.int32),
    ("filter_out", numba.int32),
    ("clk_counter", numba.int32),
    ("toggle_position", numba.int32),
    ("number_of_toggles", numba.int32),
]

# DPLL counter is c_counter_bits wide; it wraps once per symbol.
_DPLL_COUNTER_BITS = 10
_DPLL_ERROR_SUM_BITS = 15

# The symbol reclocking loop filter is designed as a second-order loop
# with natural frequency _DPLL_OMEGA and damping factor _DPLL_ZETA,
# updated at the symbol rate.
_DPLL_OMEGA = 2 * np.pi * 1800  # undamped natural frequency [rad/s]
_DPLL_ZETA = 0.6  # damping factor
_DPLL_TS = 1.0 / SYMBOL_RATE
# Combined phase detector and VCO gain; the detector's average gain is
# well below unity since only cycles with exactly one symbol transition
# update the loop.
_DPLL_GPD_GVCO = 0.3
# Proportional and integral gains
_DPLL_G1 = (1 - np.exp(-2 * _DPLL_ZETA * _DPLL_OMEGA * _DPLL_TS)) / _DPLL_GPD_GVCO
_DPLL_G2 = (
    1
    + np.exp(-2 * _DPLL_OMEGA * _DPLL_ZETA * _DPLL_TS)
    - 2
    * np.exp(-_DPLL_OMEGA * _DPLL_ZETA * _DPLL_TS)
    * np.cos(_DPLL_OMEGA * _DPLL_TS * np.sqrt(1 - _DPLL_ZETA**2))
) / _DPLL_GPD_GVCO


@jitclass(AC3_DPLL_spec)
class AC3_DPLL:
    """Symbol timing recovery for the differentially encoded QPSK signal.

    The phase detector observes symbol transitions (changes in the decoded
    differential symbol) and steers a wrapping counter so that it overflows
    once per symbol, in the middle between transitions."""

    def __init__(self, input_sample_frequency):
        self.symbol_distance = np.int32(round(input_sample_frequency / SYMBOL_RATE))
        self.nominal_add = np.int32(
            (1 << _DPLL_COUNTER_BITS) * SYMBOL_RATE / input_sample_frequency
        )

        # Tail of the previous input so the differential symbol for the
        # first samples of a block can be computed.
        self.prev_i = np.zeros(self.symbol_distance, np.float32)
        self.prev_q = np.zeros(self.symbol_distance, np.float32)

        self.prev_symbol = 0
        self.error_sum = 0
        self.filter_out = 0
        self.clk_counter = 0
        self.toggle_position = 0
        self.number_of_toggles = 0

    def reclock_symbols(self, input_i, input_q):
        """input_i/input_q: baseband I/Q at the final (decimated) sample
        rate.  Returns an array of QPSK symbols (uint8, values 0-3)."""
        n = len(input_i)
        output = np.empty(n, np.uint8)
        count = 0

        counter_mask = (1 << _DPLL_COUNTER_BITS) - 1
        sd = self.symbol_distance

        for i in range(n):
            # The current symbol is the phase difference between the
            # current IQ value and the one symbol_distance samples back.
            if i >= sd:
                d_i = input_i[i - sd]
                d_q = input_q[i - sd]
            else:
                d_i = self.prev_i[i]
                d_q = self.prev_q[i]

            # (i + jq) * conj(d_i + j d_q)
            re = input_i[i] * d_i + input_q[i] * d_q
            im = input_q[i] * d_i - input_i[i] * d_q

            # Decode the quadrant to a symbol
            if abs(re) >= abs(im):
                current_symbol = 0 if re >= 0 else 3
            else:
                current_symbol = 1 if im >= 0 else 2

            if current_symbol != self.prev_symbol:
                self.toggle_position = self.clk_counter
                self.number_of_toggles += 1
            self.prev_symbol = current_symbol

            new_counter = (
                self.clk_counter + self.nominal_add + self.filter_out
            ) & counter_mask
            self.filter_out = 0
            if new_counter < self.clk_counter:
                output[count] = self.prev_symbol
                count += 1

                if self.number_of_toggles == 1:
                    error = -(
                        self.toggle_position - (1 << (_DPLL_COUNTER_BITS - 1))
                    )
                else:
                    error = 0

                # Truncate the error sum to a signed _DPLL_ERROR_SUM_BITS
                # bit value (two's complement)
                error_sum = (self.error_sum + error) & (
                    (1 << _DPLL_ERROR_SUM_BITS) - 1
                )
                if error_sum >= 1 << (_DPLL_ERROR_SUM_BITS - 1):
                    error_sum -= 1 << _DPLL_ERROR_SUM_BITS
                self.error_sum = error_sum

                self.filter_out = int(error * _DPLL_G1 + error_sum * _DPLL_G2)
                self.number_of_toggles = 0

            self.clk_counter = new_counter

        self.prev_i = input_i[n - sd :].copy()
        self.prev_q = input_q[n - sd :].copy()

        return output[:count]


class Ac3RfDemodulator:
    """Demodulates the 2.88 MHz AC3-RF QPSK signal to raw symbols.

    The input signal (float32, at the raw capture sample rate) is decimated
    by half-band stages to at least 7 MHz, mixed down with a 2.88 MHz
    carrier, decimated further to at least 5 samples per symbol, matched
    filtered (root-raised cosine), and finally reclocked by a DPLL.

    All filter and timing state is carried across calls, so the signal can
    be fed in blocks of any length (>= one symbol period at the final rate).
    """

    # Phase accumulator width for the mixer NCO
    _PHASE_ACCUM_BITS = 14

    def __init__(self, input_sample_frequency, logger=None):
        self.input_sample_frequency = input_sample_frequency

        # Decimate so that the sample rate is at least 7 MHz before mixing
        # (the signal occupies 2.88 MHz +- 150 kHz)
        pre_mix_stages = int(np.log2(input_sample_frequency / 7e6))
        assert pre_mix_stages >= 0, "input sample frequency must be >= 7 MHz"
        mix_frequency = input_sample_frequency / (1 << pre_mix_stages)

        # After mixing, decimate to at least 5 samples per QPSK symbol
        post_mix_stages = int(np.log2(mix_frequency / 5 / SYMBOL_RATE))
        final_frequency = mix_frequency / (1 << post_mix_stages)

        self.pre_mix_filters = [
            _FirDecimStage(halfband_kaiser(19), 2, np.float32)
            for _ in range(pre_mix_stages)
        ]

        # The last decimating stage runs closest to the signal band and
        # gets a slightly longer filter
        self.post_mix_filters = [
            _ComplexFirDecimStage(
                halfband_kaiser(23 if i == post_mix_stages - 1 else 19), 2
            )
            for i in range(post_mix_stages)
        ]

        # Matched filter for the QPSK pulse shape
        sps = final_frequency / SYMBOL_RATE
        rrc_ntaps = int(np.ceil(3 * sps)) * 2 + 1
        self.post_mix_filters.append(
            _ComplexFirDecimStage(rrc_filter(rrc_ntaps, sps, 0.7), 1)
        )

        # Mixer NCO: an integer phase accumulator indexing cos/sin tables
        lut_size = 1 << self._PHASE_ACCUM_BITS
        lut_phases = 2 * np.pi * np.arange(lut_size) / lut_size
        self._cos_lut = np.cos(lut_phases).astype(np.float32)
        self._sin_lut = np.sin(lut_phases).astype(np.float32)
        self._phase_step = int(lut_size * CARRIER_FREQ / mix_frequency)
        self._phase_accumulator = 0

        self.dpll = AC3_DPLL(final_frequency)

        if logger is not None:
            logger.debug(
                "AC3 demodulator: mix Fs=%.0f, final Fs=%.0f, %.1f samples/symbol"
                % (mix_frequency, final_frequency, final_frequency / SYMBOL_RATE)
            )

    def input_sample_alignment(self):
        """Input blocks may have any length (kept for API compatibility
        with the previous C++ extension module)."""
        return 1

    def demodulate_to_symbols(self, input_buffer):
        """input_buffer: 1-D float32 numpy array of raw RF samples.
        Returns the demodulated QPSK symbols as bytes (values 0-3)."""
        data = np.asarray(input_buffer, np.float32)

        for stage in self.pre_mix_filters:
            data = stage.process(data)

        # Mix with exp(i * 2 * pi * CARRIER_FREQ * t) to shift the signal
        # band down to 0
        lut_mask = (1 << self._PHASE_ACCUM_BITS) - 1
        phases = (
            self._phase_accumulator + self._phase_step * np.arange(len(data))
        ) & lut_mask
        self._phase_accumulator = (
            self._phase_accumulator + self._phase_step * len(data)
        ) & lut_mask
        data_re = data * self._cos_lut[phases]
        data_im = data * self._sin_lut[phases]

        for stage in self.post_mix_filters:
            data_re, data_im = stage.process(data_re, data_im)

        symbols = self.dpll.reclock_symbols(data_re, data_im)
        return symbols.tobytes()
