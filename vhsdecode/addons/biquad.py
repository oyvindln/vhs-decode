"""
Audio EQ biquad filter design from cookbook formulae.

Created on Thu Mar 28 18:51:00 2013

UNFINISHED AND BUGGY

Python/SciPy implementation of the filters described in
"Cookbook formulae for audio EQ biquad filter coefficients"
by Robert Bristow-Johnson
https://www.musicdsp.org/en/latest/Filters/197-rbj-audio-eq-cookbook.html

These functions will output analog or digital transfer functions, deriving
the latter using the bilinear transform, as is done in the reference.

Overall gain parameters are not included.

"BLT frequency warping has been taken into account for
both significant frequency relocation (this is the normal "prewarping" that
is necessary when using the BLT) and for bandwidth readjustment (since the
bandwidth is compressed when mapped from analog to digital using the BLT)."

TODO: combine lowpass and highpass? and bandpass?

TODO: generate analog poles/zeros prototypes and convert them or output them
directly?  Would be useful to have an analog shelving filter designer, for
instance.

TODO: Use ordinary frequency instead of rad/s for analog filters?  angular
matches scipy, but these are usually used in audio. Compare with CSound
functions, etc.

http://www.dsprelated.com/showcode/170.php defines it as
function [b, a]  = shelving(G, fc, fs, Q, type)

TODO: sane defaults for Q for all filters

TODO: Try to think of better names than "outer", "constantq", "skirt", etc

TODO: Bandwidth is wrong for high-frequency peaking digital filters,
despite using the equations in the cookbook

TODO: functions should accept Q, BW, or S directly, since these are not
trivially derived otherwise?

Q (the EE kind of definition, except for peakingEQ in which A*Q is
    the classic EE Q.  That adjustment in definition was made so that
    a boost of N dB followed by a cut of N dB for identical Q and
    f0/Fs results in a precisely flat unity gain filter or "wire".)

 _or_ BW, the bandwidth in octaves (between -3 dB frequencies for BPF
    and notch or between midpoint (dBgain/2) gain frequencies for
    peaking EQ)

 _or_ S, a "shelf slope" parameter (for shelving EQ only).  When S = 1,
    the shelf slope is as steep as it can be and remain monotonically
    increasing or decreasing gain with frequency.  The shelf slope, in
    dB/octave, remains proportional to S for all other values for a
    fixed f0/Fs and dBgain.

Then compute a few intermediate variables:

alpha = sin(w0)/(2*Q)                                       (case: Q)
      = sin(w0)*sinh( ln(2)/2 * BW * w0/sin(w0) )           (case: BW)
      = sin(w0)/2 * sqrt( (A + 1/A)*(1/S - 1) + 2 )         (case: S)

    FYI: The relationship between bandwidth and Q is
         1/Q = 2*sinh(ln(2)/2*BW*w0/sin(w0))     (digital filter w BLT)
    or   1/Q = 2*sinh(ln(2)/2*BW)             (analog filter prototype)
"""

from math import pi, tan, sinh
from math import log as ln
from cmath import sqrt
import numpy as np
from scipy.signal import tf2zpk, tf2ss, lp2lp, bilinear


def _transform(b, a, Wn, analog, output):
    """
    Convert analog prototype filter to desired output format.

    Shift prototype filter to desired frequency, convert to digital with
    pre-warping, and return in various formats.
    """
    Wn = np.asarray(Wn)
    if not analog:
        if np.any(Wn < 0) or np.any(Wn > 1):
            raise ValueError(
                "Digital filter critical frequencies " "must be 0 <= Wn <= 1"
            )
        fs = 2.0
        warped = 2 * fs * tan(pi * Wn / fs)
    else:
        warped = Wn

    # Shift frequency
    b, a = lp2lp(b, a, wo=warped)

    # Find discrete equivalent if necessary
    if not analog:
        b, a = bilinear(b, a, fs=fs)

    # Transform to proper out type (pole-zero, numer-denom, state-space)
    if output in ("zpk", "zp"):
        return tf2zpk(b, a)
    elif output in ("ba", "tf"):
        return b, a
    elif output in ("ss", "abcd"):
        return tf2ss(b, a)
    else:
        raise ValueError("Unknown output type {0}".format(output))


def lowpass(Wn, Q=1 / sqrt(2), analog=False, output="ba"):
    """
    Design an analog or digital biquad lowpass filter with variable Q.

    Analog prototype: H(s) = 1 / (s**2 + s/Q + 1)

    Parameters
    ----------
    Wn : float
        Corner frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    Q : float
        Quality factor of the filter.  Examples:

        * 1/sqrt(2) (default) is a Butterworth filter, with maximally-flat
          passband
        * 1/sqrt(3) is a Bessel filter, with maximally-flat group delay.
        * 1/2 is a Linkwitz-Riley filter, used to make lowpass and highpass
          sections that sum flat to unity gain.

    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.

    """
    # H(s) = 1 / (s**2 + s/Q + 1)
    b = np.array([1])
    a = np.array([1, 1 / Q, 1])

    return _transform(b, a, Wn, analog, output)


def highpass(Wn, Q=1 / sqrt(2), analog=False, output="ba"):
    """
    Design an analog or digital biquad highpass filter with variable Q.

    Analog prototype: H(s) = s**2 / (s**2 + s/Q + 1)

    Parameters
    ----------
    Wn : float
        Corner frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    Q : float
        Quality factor of the filter.  Examples:

        * 1/sqrt(2) (default) is a Butterworth filter, with maximally-flat
          passband
        * 1/sqrt(3) is a Bessel filter, with maximally-flat group delay.
        * 1/2 is a Linkwitz-Riley filter, used to make lowpass and highpass
          sections that sum flat to unity gain.

    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.

    """
    # H(s) = s**2 / (s**2 + s/Q + 1)
    b = np.array([1, 0, 0])
    a = np.array([1, 1 / Q, 1])

    return _transform(b, a, Wn, analog, output)


def bandpass(Wn, Q=1, type="skirt", analog=False, output="ba"):
    """
    Design an analog or digital biquad bandpass filter with variable Q.

    Parameters
    ----------
    Wn : float
        Center frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    Q : float
        Quality factor of the filter.  Examples:

        * sqrt(2) is 1 octave wide

    type : {'skirt', 'peak'}, optional
        The type of filter.

        ``skirt``
            Type 1 (default), has a constant skirt gain, with peak gain = Q
            Transfer function: H(s) = s / (s**2 + s/Q + 1)
        ``peak``
            Type 2, has a constant peak gain of 0 dB, and the skirt changes
            with the Q.
            Transfer function: H(s) = (s/Q) / (s**2 + s/Q + 1)

    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.

    """
    if type in (1, "skirt"):
        # H(s) = s     / (s**2 + s/Q + 1)
        b = np.array([0, 1, 0])
    elif type in (2, "peak"):
        # H(s) = (s/Q) / (s**2 + s/Q + 1)
        b = np.array([0, 1 / Q, 0])
    else:
        raise ValueError('"%s" is not a known bandpass type' % type)

    a = np.array([1, 1 / Q, 1])

    return _transform(b, a, Wn, analog, output)


def notch(Wn, Q=10, analog=False, output="ba"):
    """
    Design an analog or digital biquad notch filter with variable Q.

    The notch differs from a peaking cut filter in that the gain at the
    notch center frequency is 0, or -Inf dB.

    Transfer function: H(s) = (s**2 + 1) / (s**2 + s/Q + 1)

    Parameters
    ----------
    Wn : float
        Center frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    Q : float
        Quality factor of the filter.  Examples:

        * sqrt(2) is 1 octave wide
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.

    """
    # H(s) = (s**2 + 1) / (s**2 + s/Q + 1)
    b = np.array([1, 0, 1])
    a = np.array([1, 1 / Q, 1])

    return _transform(b, a, Wn, analog, output)


def allpass(Wn, Q=1, analog=False, output="ba"):
    """
    Design an analog or digital biquad allpass filter with variable Q.

    Transfer function:  H(s) = (s**2 - s/Q + 1) / (s**2 + s/Q + 1)

    Parameters
    ----------
    Wn : float
        Center frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    Q : float
        Quality factor of the filter.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.

    """
    # H(s) = (s**2 - s/Q + 1) / (s**2 + s/Q + 1)
    b = np.array([1, -1 / Q, 1])
    a = np.array([1, 1 / Q, 1])

    return _transform(b, a, Wn, analog, output)


def peaking(Wn, dBgain, Q=None, BW=None, type="half", analog=False, output="ba"):
    """
    Design an analog or digital biquad peaking filter with variable Q.

    Transfer function: H(s) = (s**2 + s*(Az/Q) + 1) / (s**2 + s/(Ap*Q) + 1)

    Used in graphic or parametric EQs.

    Parameters
    ----------
    Wn : float
        Center frequency of the filter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    dBgain : float
        The gain at the center frequency, in dB.  Positive for boost,
        negative for cut.
    Q : float
        Quality factor of the filter.  Examples:

        * Q = sqrt(2) (default) produces a bandwidth of 1 octave
    ftype : {'half', 'constant'}, optional
        Where on the curve to measure the bandwidth of the filter.

        ``half``
            Bandwidth is defined using the points on the curve at which the
            gain in dB is half of the peak gain.  This is the method used in
            "Cookbook formulae for audio EQ biquad filter coefficients"
        ``constant``
            Bandwidth is defined using the points -3 dB down from the peak
            gain (or +3 dB up from the cut gain), maintaining constant Q
            regardless of center frequency or boost gain.  This is
            symmetrical in dB, so that a boost and cut with identical
            parameters sum to unity gain.
            This is the method used in "Constant-Q" hardware equalizers.
            [ref: http://www.rane.com/note101.html]
            Klark Teknik calls this "symmetrical Q"
            http://www.klarkteknik.com/faq-06.php
        constant Q asymmetrical
            constant Q for both boost and cut, which makes them asymmetrical
            (not implemented)
        Half-gain  Hybrid
            Defined symmetrical at half gain point except for 3 dB or less
            (not implemented)
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    output : {'ba', 'zpk', 'ss'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        state-space ('ss').
        Default is 'ba'.

    Notes
    -----
    Due to bilinear transform, this is always 0 dB at fs/2, but it would be
    better if the curve fell off symmetrically.

    Orfanidis describes a digital filter that more accurately matches the
    analog filter, but it is far more complicated.
    Orfanidis, Sophocles J., "Digital Parametric Equalizer Design with
    Prescribed Nyquist-Frequency Gain"

    """
    
    if Q is None and BW is None:
        BW = 1  # octave

    if Q is None:
        # w0 = Wn
        # Q = 1/(2*sinh(ln(2)/2*BW*w0/sin(w0))) # digital filter w BLT
        Q = 1 / (2 * sinh(ln(2) / 2 * BW))  # analog filter prototype
        # TODO: In testing, neither of these is even close to correct near
        # fs/2, and the difference between them is very small

    if type in ("half"):
        A = 10.0 ** (dBgain / 40.0)  # for peaking and shelving EQ filters only
        Az = A
        Ap = A
    elif type in ("constantq"):
        A = 10.0 ** (dBgain / 20.0)
        if dBgain > 0:  # boost
            Az = A
            Ap = 1
        else:  # cut
            Az = 1
            Ap = A
    else:
        raise ValueError('"%s" is not a known peaking type' % type)

    # H(s) = (s**2 + s*(Az/Q) + 1) / (s**2 + s/(Ap*Q) + 1)
    b = np.array([1, Az / Q, 1])
    a = np.array([1, 1 / (Ap * Q), 1])

    return _transform(b, a, Wn, analog, output)


def shelf(
    Wn, dBgain, S=1, btype="low", ftype="half", analog=False, output="ba", Q=None
):
    """
    Design an analog or digital biquad shelving filter with variable slope.

    Parameters
    ----------
    Wn : float
        Turnover frequency of the filter, defined by the `ftype` parameter.
        For digital filters, `Wn` is normalized from 0 to 1, where 1 is the
        Nyquist frequency, pi radians/sample.  (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g. rad/s).
    dBgain : float
        The gain at the center frequency, in dB.  Positive for boost,
        negative for cut.
    Q : float
        Quality factor of the filter.  Examples:

        * Q fdsafda
    ftype : {'half', 'outer', 'inner'}, optional
    fpoint?
    fdef?
        Definition of the filter's turnover frequency

        ``half``
            Wn is defined as the point on the curve at which the
            gain in dB is half of the shelf gain, or midway between the
            filter's pole and zero.  This method is used in
            "Cookbook formulae for audio EQ biquad filter coefficients"
        ``outer``
            Wn is defined as the point 3 dB up or down from the shelf's
            plateau.
            This is symmetrical in dB, so that a boost and cut with identical
            parameters sum to unity gain.
            This is defined using the location of the outer pole or zero of
            the filter (the lower of the two for a low shelf, higher of the
            two for a high shelf), so will not be exactly 3 dB at lower shelf
            gains.  This method is used in ____ hardware audio equalizers.
        ``inner``
            Wn is defined as the point 3 dB up or down from unity gain.
            This is symmetrical in dB, so that a boost and cut with identical
            parameters sum to unity gain.
    btype : {'low', 'high'}, optional
        Band type of the filter, low shelf or high shelf.


    ftype is the meaning of f, either midpoint of slope, fstop or fturnover
    turnover frequency at large boost/cuts, this is 3 dB away from unity gain
    stop frequency at large boost/cuts, this is 3 dB away from plateau

    tonmeister defines outer as fstop and inner as fturnover
        as does http://www.soundonsound.com/sos/dec05/articles/qa1205_3.htm

    Understanding Audio defines turnover as outer
        as does ems.music.utexas.edu/dwnld/mus329j10/Filter%20Basics.ppt
            also calls it knee

    R is transition ratio fstop/fturnover.  at R=1, fstop = fturnover
    If the transition ratio is less than 1, then the filter is a low shelving
    filter. If the transition ratio is greater than 1, then the filter is a
    high shelving filter.

    highShelf:
        H(s) = A * (A*s**2 + (sqrt(A)/Q)*s + 1)/(  s**2 + (sqrt(A)/Q)*s + A)
    lowShelf:
        H(s) = A * (  s**2 + (sqrt(A)/Q)*s + A)/(A*s**2 + (sqrt(A)/Q)*s + 1)

    2*sqrt(A)*alpha  =  sin(w0) * sqrt( (A**2 + 1)*(1/S - 1) + 2*A )
        is a handy intermediate variable for shelving EQ filters.

        The relationship between shelf slope and Q is
             1/Q = sqrt((A + 1/A)*(1/S - 1) + 2)

    f0 shelf midpoint frequency

    _or_ S, a "shelf slope" parameter (for shelving EQ only).  When S = 1,
        the shelf slope is as steep as it can be and remain monotonically
        increasing or decreasing gain with frequency.  The shelf slope, in
        dB/octave, remains proportional to S for all other values for a
        fixed f0/Fs and dBgain.

    """
    Q = None  # TODO: Maybe this should be a function parameter?

    if ftype in ("mid", "half"):
        A = 10.0 ** (dBgain / 40.0)  # for peaking and shelving EQ filters only

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        Az = A
        Ap = A

    elif ftype in ("outer"):
        A = 10.0 ** (dBgain / 20.0)

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        if dBgain > 0:  # boost
            Az = A
            Ap = 1
        else:  # cut
            Az = 1
            Ap = A

    elif ftype in ("inner"):
        A = 10.0 ** (dBgain / 20.0)

        if Q is None:
            Q = 1 / sqrt((A + 1 / A) * (1 / S - 1) + 2)

        if dBgain > 0:  # boost
            Az = 1
            Ap = A
        else:  # cut
            Az = A
            Ap = 1
    else:
        raise ValueError('"%s" is not a known shelf type' % ftype)

    if btype == "low":
        # H(s) = A * (  s**2 + (sqrt(A)/Q)*s + A)/(A*s**2 + (sqrt(A)/Q)*s + 1)
        b = Ap * np.array([1, sqrt(Az) / Q, Az])
        a = np.array([Ap, sqrt(Ap) / Q, 1])
    elif btype == "high":
        # H(s) = A * (A*s**2 + (sqrt(A)/Q)*s + 1)/(  s**2 + (sqrt(A)/Q)*s + A)
        b = Ap * np.array([Az, sqrt(Az) / Q, 1])
        a = np.array([1, sqrt(Ap) / Q, Ap])
    else:
        raise ValueError('"%s" is not a known shelf type' % btype)

    return _transform(b, a, Wn, analog, output)
