#!/usr/bin/python3
#
# efm.py - LDS sample to EFM data processing
# Copyright (C) 2019 Simon Inns
# Copyright (C) 2019 Adam Sampson
#
# This file is part of ld-decode.
#
# efm.py is free software: you can redistribute it and/or
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

# Note: The PLL implementation is based on original code provided to
# the ld-decode project by Olivier "Sarayan" Galibert.  Many thanks
# for the assistance!

import numba
import numpy as np
import sys

import scipy.interpolate as spi

try:
    from numba.experimental import jitclass
except ImportError:
    # Prior to numba 0.49
    from numba import jitclass


def computeefmfilter(freq_hz=40000000, blocklen=65536):
    """Frequency-domain equalisation filter for the LaserDisc EFM signal.
    This was inspired by the input signal equaliser in WSJT-X, described in
    Steven J. Franke and Joseph H. Taylor, "The MSK144 Protocol for
    Meteor-Scatter Communication", QEX July/August 2017.
    <http://physics.princeton.edu/pulsar/k1jt/MSK144_Protocol_QEX.pdf>

    This improved EFM filter was devised by Adam Sampson (@atsampson)
    """
    # Frequency bands
    freqs = np.linspace(0.0e6, 2.0e6, num=11)
    freq_per_bin = freq_hz / blocklen
    # Amplitude and phase adjustments for each band.
    # These values were adjusted empirically based on a selection of NTSC and PAL samples.
    amp = np.array([0.0, 0.215, 0.41, 0.73, 0.98, 1.03, 0.99, 0.81, 0.59, 0.42, 0.0])
    phase = np.array(
        [0.0, -0.92, -1.03, -1.11, -1.2, -1.2, -1.2, -1.2, -1.05, -0.95, -0.8]
    )
    phase = [p * 1.25 for p in phase]

    """Compute filter coefficients for the given FFTFilter."""
    # Anything above the highest frequency is left as zero.
    coeffs = np.zeros(blocklen, dtype=np.complex)

    # Generate the frequency-domain coefficients by cubic interpolation between the equaliser values.
    a_interp = spi.interp1d(freqs, amp, kind="cubic")
    p_interp = spi.interp1d(freqs, phase, kind="cubic")

    nonzero_bins = int(freqs[-1] / freq_per_bin) + 1

    bin_freqs = np.arange(nonzero_bins) * freq_per_bin
    bin_amp = a_interp(bin_freqs)
    bin_phase = p_interp(bin_freqs)

    # Scale by the amplitude, rotate by the phase
    coeffs[:nonzero_bins] = bin_amp * (
        np.cos(bin_phase) + (complex(0, -1) * np.sin(bin_phase))
    )

    return coeffs * 8


# Attribute types of EFM_PLL for numba.
EFM_PLL_spec = [
    ("zcPreviousInput", numba.int16),
    ("delta", numba.float64),
    ("pllResult", numba.int8[:]),
    ("pllResultCount", numba.uintp),
    ("basePeriod", numba.float64),
    ("minimumPeriod", numba.float64),
    ("maximumPeriod", numba.float64),
    ("periodAdjustBase", numba.float64),
    ("currentPeriod", numba.float64),
    ("phaseAdjust", numba.float64),
    ("refClockTime", numba.float64),
    ("frequencyHysteresis", numba.int32),
    ("tCounter", numba.int8),
]


@jitclass(EFM_PLL_spec)
class EFM_PLL:
    def __init__(self):
        # ZC detector state
        self.zcPreviousInput = 0
        self.delta = 0.0

        # PLL output buffer
        self.pllResult = np.empty(1 << 16, np.int8)
        self.pllResultCount = 0

        # PLL state
        self.basePeriod = 40000000.0 / 4321800.0  # T1 clock period 40MSPS / bit-rate

        self.minimumPeriod = self.basePeriod * 0.90  # -10% minimum
        self.maximumPeriod = self.basePeriod * 1.10  # +10% maximum
        self.periodAdjustBase = self.basePeriod * 0.0001  # Clock adjustment step

        # PLL Working parameters
        self.currentPeriod = self.basePeriod
        self.phaseAdjust = 0.0
        self.refClockTime = 0.0
        self.frequencyHysteresis = 0
        self.tCounter = 1

    def process(self, inputBuffer):
        """This method performs interpolated zero-crossing detection and stores
        the result as sample deltas (the number of samples between each
        zero-crossing).  Interpolation of the zero-crossing point provides a
        result with sub-sample resolution.

        Since the EFM data is NRZ-I (non-return to zero inverted) the polarity
        of the input signal is not important (only the frequency); therefore we
        can simply store the delta information.  The resulting delta
        information is fed to the phase-locked loop which is responsible for
        correcting jitter errors from the ZC detection process.

        inputBuffer is a numpy.ndarray of np.int16 samples.
        Returns a view into a numpy.ndarray of np.int8 times."""

        # print(len(inputBuffer), min(inputBuffer), max(inputBuffer))

        # Ensure the PLL result buffer is big enough, and clear it
        if len(self.pllResult) < len(inputBuffer):
            self.pllResult = np.empty(len(inputBuffer), np.int8)
        self.pllResultCount = 0

        for curr in inputBuffer:
            prev = self.zcPreviousInput

            # Have we seen a zero-crossing?
            if (prev < 0 and curr >= 0) or (prev >= 0 and curr < 0):
                # Interpolate to get the ZC sub-sample position fraction
                fraction = (-prev) / (curr - prev)

                # Feed the sub-sample accurate result to the PLL
                self.pushEdge(self.delta + fraction)

                # Offset the next delta by the fractional part of the result
                # in order to maintain accuracy
                self.delta = 1.0 - fraction
            else:
                # No ZC, increase delta by 1 sample
                self.delta += 1.0

            # Keep the previous input (so we can work across buffer boundaries)
            self.zcPreviousInput = curr

        return self.pllResult[: self.pllResultCount]

    def pushEdge(self, sampleDelta):
        """Called when a ZC happens on a sample number."""

        while sampleDelta >= self.refClockTime:
            nextTime = self.refClockTime + self.currentPeriod + self.phaseAdjust
            self.refClockTime = nextTime

            # Note: the tCounter < 3 check causes an 'edge push' if T is 1 or 2 (which
            # are invalid timing lengths for the NRZI data).  We also 'edge pull' values
            # greater than T11
            if (sampleDelta > nextTime or self.tCounter < 3) and self.tCounter < 11:
                self.phaseAdjust = 0.0
                self.tCounter += 1
            else:
                edgeDelta = sampleDelta - (nextTime - self.currentPeriod / 2.0)
                self.phaseAdjust = edgeDelta * 0.005

                # Adjust frequency based on error
                if edgeDelta < 0:
                    if self.frequencyHysteresis < 0:
                        self.frequencyHysteresis -= 1
                    else:
                        self.frequencyHysteresis = -1
                elif edgeDelta > 0:
                    if self.frequencyHysteresis > 0:
                        self.frequencyHysteresis += 1
                    else:
                        self.frequencyHysteresis = 1
                else:
                    self.frequencyHysteresis = 0

                # Update the reference clock?
                if self.frequencyHysteresis < -1.0 or self.frequencyHysteresis > 1.0:
                    aper = self.periodAdjustBase * edgeDelta / self.currentPeriod

                    # If there's been a substantial gap since the last edge (e.g.
                    # a dropout), edgeDelta can be very large here, so we need to
                    # limit how much of an adjustment we're willing to make
                    if aper < -self.periodAdjustBase:
                        aper = -self.periodAdjustBase
                    elif aper > self.periodAdjustBase:
                        aper = self.periodAdjustBase

                    self.currentPeriod += aper

                    if self.currentPeriod < self.minimumPeriod:
                        self.currentPeriod = self.minimumPeriod
                    elif self.currentPeriod > self.maximumPeriod:
                        self.currentPeriod = self.maximumPeriod

                self.pllResult[self.pllResultCount] = self.tCounter
                self.pllResultCount += 1

                self.tCounter = 1

        # Reset refClockTime ready for the next delta but
        # keep any error to maintain accuracy
        self.refClockTime -= sampleDelta

        # Use this debug if you want to monitor the PLL output frequency
        # print("Base =", self.basePeriod, "current = ", self.currentPeriod, file=sys.stderr)


if __name__ == "__main__":
    # If invoked as a script, test the PLL by reading from stdin and writing to stdout.

    pll = EFM_PLL()

    while True:
        inputBuffer = sys.stdin.buffer.read(1 << 10)
        if len(inputBuffer) == 0:
            break

        inputData = np.frombuffer(inputBuffer, np.int16)
        outputData = pll.process(inputData)
        sys.stdout.buffer.write(outputData.tobytes())
