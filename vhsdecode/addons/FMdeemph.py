"""
    FM deemphasis filter borrowed from GNURadio
    Copyright 2005,2007,2021 Free Software Foundation, Inc.
    SPDX-License-Identifier: GPL-3.0-or-later
"""
import math


def gen_high_shelf(f0, dbgain, qfactor, fs):
    """Generate high shelving filter coeficcients (digital).
    f0: The center frequency where the gain in decibel is at half the maximum value.
       Normalized to sampling frequency, i.e output will be filter from 0 to 2pi.
    dbgain: gain at the top of the shelf in decibels
    qfactor: determines shape of filter TODO: Document better
    fs: sampling frequency

    TODO: Generate based on -3db
    Based on: https://www.w3.org/2011/audio/audio-eq-cookbook.html
    """
    a = 10 ** (dbgain / 40.0)
    w0 = 2 * math.pi * (f0 / fs)
    alpha = math.sin(w0) / (2 * qfactor)

    cosw0 = math.cos(w0)
    asquared = math.sqrt(a)

    b0 = a * ((a + 1) + (a - 1) * cosw0 + 2 * asquared * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cosw0)
    b2 = a * ((a + 1) + (a - 1) * cosw0 - 2 * asquared * alpha)
    a0 = (a + 1) - (a - 1) * cosw0 + 2 * asquared * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cosw0)
    a2 = (a + 1) - (a - 1) * cosw0 - 2 * asquared * alpha
    return [b0, b1, b2], [a0, a1, a2]


class FMDeEmphasisB:
    r"""
    Deemphasis using a low-shelf filter.
    """

    def __init__(self, fs, dBgain, mid_point, Q=1 / 2):
        #        print("corner freq", corner_freq)
        #        self.ataps, self.btaps = bq.shelf((corner_freq) / (fs / 2), 14 ,
        #                                          btype="high", ftype="inner", analog=False, output="ba")
        #        self.ataps, self.btaps = bq.shelf(260000 / (fs / 2), dBgain,1/2,
        #                                          btype="high", ftype="mid", analog=False, output="ba")

        # We generate a low-shelf filter by inverting a high-shelf one that corresponds to the
        # specified preemphasis filter.
        # TODO:
        # We want to generate this based on time constant and 'X' value
        # currently we use the mid frequency and a gain to get the correct shape
        # with eyeballed mid value. Q=1/2 seems to give the corret slope.
        self.ataps, self.btaps = gen_high_shelf(mid_point, dBgain, Q, fs)

    def get(self):
        return self.btaps, self.ataps


class FMDeEmphasis:
    r"""
      FM Deemphasis IIR filter

      Args:
          fs: sampling frequency in Hz (float)
          tau: Time constant in seconds (75us in US, 50us in EUR) (float)

    An analog deemphasis filter:

                 R
    o------/\/\/\/---+----o
                     |
                    = C
                     |
                    ---

    Has this transfer function:

                 1             1
                ----          ---
                 RC          tau
    H(s) = ---------- = ----------
                   1             1
              s + ----      s + ---
                   RC           tau

    And has its -3 dB response, due to the pole, at

    |H(j w_c)|^2 = 1/2  =>  s = j w_c = j (1/(RC))

    Historically, this corner frequency of analog audio deemphasis filters
    been specified by the RC time constant used, called tau.
    So w_c = 1/tau.

    FWIW, for standard tau values, some standard analog components would be:
    tau = 75 us = (50K)(1.5 nF) = (50 ohms)(1.5 uF)
    tau = 50 us = (50K)(1.0 nF) = (50 ohms)(1.0 uF)

    In specifying tau for this digital deemphasis filter, tau specifies
    the *digital* corner frequency, w_c, desired.

    The digital deemphasis filter design below, uses the
    "bilinear transformation" method of designing digital filters:

    1. Convert digital specifications into the analog domain, by prewarping
       digital frequency specifications into analog frequencies.

       w_a = (2/T)tan(wT/2)

    2. Use an analog filter design technique to design the filter.

    3. Use the bilinear transformation to convert the analog filter design to a
       digital filter design.

       H(z) = H(s)|
                       s = (2/T)(1-z^-1)/(1+z^-1)


           w_ca         1          1 - (-1) z^-1
    H(z) = ---- * ----------- * -----------------------
           2 fs        -w_ca             -w_ca
                   1 - -----         1 + -----
                        2 fs              2 fs
                                 1 - ----------- z^-1
                                         -w_ca
                                     1 - -----
                                          2 fs

    We use this design technique, because it is an easy way to obtain a filter
    design with the -6 dB/octave roll-off required of the deemphasis filter.

    Jackson, Leland B., _Digital_Filters_and_Signal_Processing_Second_Edition_,
      Kluwer Academic Publishers, 1989, pp 201-212

    Orfanidis, Sophocles J., _Introduction_to_Signal_Processing_, Prentice Hall,
      1996, pp 573-583
    """

    def __init__(self, fs, tau=1.25e-6):

        # Digital corner frequency
        w_c = 1.0 / tau

        # Prewarped analog corner frequency
        w_ca = 2.0 * fs * math.tan(w_c / (2.0 * fs))

        # Resulting digital pole, zero, and gain term from the bilinear
        # transformation of H(s) = w_ca / (s + w_ca) to
        # H(z) = b0 (1 - z1 z^-1)/(1 - p1 z^-1)
        k = -w_ca / (2.0 * fs)
        z1 = -1.0
        p1 = (1.0 + k) / (1.0 - k)
        b0 = -k / (1.0 - k)

        self.btaps = [b0 * 1.0, b0 * -z1]
        self.ataps = [1.0, -p1]

        # Since H(s = 0) = 1.0, then H(z = 1) = 1.0 and has 0 dB gain at DC

    def get(self):
        return self.btaps, self.ataps
