import math
from numba import njit


@njit(cache=True)
def gen_shelf(f0, dbgain, type, fs, qfactor):
    """Generate shelving filter coefficients (digital).
    * f0:
        The center frequency where the gain in decibel is at half the maximum value.
        Normalized to sampling frequency, i.e output will be filter from 0 to 2pi.
    * dbgain:
        gain at the top of the shelf in decibels
    * fs:
        sampling frequency
    * type:
        "high" for high shelf, "low" for low shelf
    * qfactor:
        determines shape of filter TODO: Document better

    Based on: https://www.w3.org/2011/audio/audio-eq-cookbook.html
    """
    a = 10 ** (dbgain / 40.0)
    w0 = 2 * math.pi * (f0 / fs)
    alpha = math.sin(w0) / (2 * qfactor)

    cosw0 = math.cos(w0)
    asquared = math.sqrt(a)

    if type == "low":
        b0 = a * ((a + 1) - (a - 1) * cosw0 + 2 * asquared * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cosw0)
        b2 = a * ((a + 1) - (a - 1) * cosw0 - 2 * asquared * alpha)
        a0 = (a + 1) + (a - 1) * cosw0 + 2 * asquared * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cosw0)
        a2 = (a + 1) + (a - 1) * cosw0 - 2 * asquared * alpha
    elif type == "high":
        b0 = a * ((a + 1) + (a - 1) * cosw0 + 2 * asquared * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cosw0)
        b2 = a * ((a + 1) + (a - 1) * cosw0 - 2 * asquared * alpha)
        a0 = (a + 1) - (a - 1) * cosw0 + 2 * asquared * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cosw0)
        a2 = (a + 1) - (a - 1) * cosw0 - 2 * asquared * alpha
    else:
        raise Exception(
            "Must specify 'high' or 'low' for shelf type, instead got: ", type
        )

    return [b0, b1, b2], [a0, a1, a2]
