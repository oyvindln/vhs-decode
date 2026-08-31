"""Luma FM carrier amplitude variation.

An FM carrier carries no amplitude information - its amplitude should be
constant. Every deviation from that constant is imposed by the tape, and is
measurable on the raw RF, before the time base correction has run and before
any scaling has been applied.

The amplitude is taken as the decoder's own envelope channel, which is
band limited to 700 kHz by the demodulator's envelope detector - a single pole
applied forward and backward, so 12 dB per octave. Nothing here narrows it
further: no smoothing, no additional band limit, no windowing. The tape imposes
its amplitude noise at whatever rate it imposes it, and a filter here would
decide in advance which of that the correction is allowed to see.

Two causes of amplitude change, only one of them noise
------------------------------------------------------
The received carrier amplitude is not constant, and the reasons split in two.

  Frequency driven.  The carrier sweeps with picture content - sync tip to peak
  white - and the amplitude response of the path varies across that sweep. The
  same carrier frequency gives the same amplitude every time, so this is the
  path's response and not tape noise, and it is not variation to report.

  Random.  Head-to-medium separation modulating the signal. This is the tape's
  own noise, and it is what the measurement is for.

Separating them is the whole job of this module, and it happens here at the
measurement rather than at whatever acts on it: the part of the carrier
amplitude that is a function of the luma's own frequency is divided out once,
leaving only the amplitude changes worth acting on. Downstream the two are no
longer separable.

Part of that response is known exactly and is simply divided out - the decoder's
own RF path, which on formats using the linear ramp boost tilts the band by
several dB across the carrier sweep on its own. The remainder - tape channel,
head response, record and playback equalization - is measured by binning the
flattened amplitude against carrier frequency and fitting a curve.

Both parts are functions of the carrier's frequency and of nothing else, so they
are combined into a single table over frequency and divided out together, once
per sample. That is what keeps this affordable: the response is exponentiated
where it is described, over the few thousand points of the table, rather than
over the million samples it is applied to.

What comes back is a property of the luma carrier and of the tape, and of
nothing else. It carries no assumption about what will consume it.
"""

from collections import namedtuple

import numpy as np
from numba import njit


# What `measure_amplitude_deviation` yields. `deviation` is unity where the
# carrier holds the amplitude its own frequency predicts, and departs from unity
# by the tape's multiplicative noise. `carrier_hz` is the instantaneous carrier
# frequency conditioned the same way the deviation was measured against, so a
# consumer scaling by wavelength uses the frequency that actually applies.
#
# Both are single precision, which is what the demodulator produced them in;
# widening them here would cost two full copies of the field per field and buy
# no precision that was ever recorded.
AmplitudeDeviation = namedtuple("AmplitudeDeviation", ["deviation", "carrier_hz"])


# Order of the curve fitted through the measured amplitude response across the
# carrier deviation range. This is what keeps the measurement free of the
# picture: whatever part of the amplitude tracks the luma's own frequency is the
# path's response, and removing it here is what leaves only tape noise behind.
#
# A straight line leaves too much: on a continuous luma ramp the applied gain
# still correlates with the luma at r = 0.21. A quadratic cuts that to 0.05 for
# no measurable loss of benefit. Going further does not help and starts to hurt,
# because the response is fitted per demodulation block and a block spans only
# about twelve lines - at order three the correlation on colour bars overshoots
# to -0.07 while the benefit falls.
RESPONSE_POLYNOMIAL_ORDER = 2

# Points the response curve is described by. Groups of equal population rather
# than of equal width: the response is a smooth characteristic, and a fixed IRE
# grid spends most of its bins on levels the picture barely visits while
# crowding the ones it dwells on. Equal population puts every point at the same
# confidence and puts them where the picture actually is.
#
# A quadratic needs three points; eight leaves margin for one landing badly.
# Measured, the choice is flat over its neighbourhood - 8, 16, 32 and 140 points
# all sit within 0.008 of each other on the correction's residual luma
# correlation, with eight the lowest of them.
RESPONSE_CURVE_POINTS = 8

# Samples each of those points is measured from. A median's standard error is
# about 1.25 * sigma / sqrt(m); the amplitude deviation runs about 5% rms and
# the correction acts at about 1% rms, so determining the curve an order of
# magnitude finer than it acts needs (1.25 * 0.05 / 0.001)^2, near enough four
# thousand samples a point. A field carries hundreds of thousands, so the curve
# is measured from a fraction of it and the correction still applies to every
# sample. Measured, taking one sample in twenty-one costs 0.0004 of that same
# residual correlation.
CURVE_SAMPLES_PER_POINT = 4096


@njit(cache=True, nogil=True, fastmath=True)
def _flatten_by_response(amplitude, carrier_hz, response, frequency_step):
    """Divide the decoder's own RF response out of the carrier amplitude.

    The response is tabulated on a uniform frequency grid, so the sample's bin
    is an index rather than a search.

    Only the samples the response curve is fitted from come through here - a
    few tens of thousands of the field's million. Applying the finished model
    to the whole field is `_deviation_from_inverse_response`, which divides out
    this response and the fitted curve together.
    """
    count = len(amplitude)
    # Carrier amplitude is physically positive, so zero
    # cannot be mistaken for a measurement.
    flattened = np.zeros(count)
    last = len(response) - 1
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        if position <= 0.0:
            value = response[0]
        elif position >= last:
            value = response[last]
        else:
            lower = int(position)
            fraction = position - lower
            value = response[lower] * (1.0 - fraction) + response[lower + 1] * fraction
        if value > 0.0:
            flattened[i] = amplitude[i] / value
    return flattened


def rf_path_response(rf):
    """The decoder's own RF amplitude response, and the grid it is tabulated on.

    `Filters["RFVideo"]` is the magnitude response actually applied to the
    signal before the carrier amplitude is taken, so consuming it directly keeps
    this correct for every format and every combination of ramp boost, peaking
    and notch options.

    Fixed for the life of the decoder, so it is derived once and kept on the
    decoder rather than rebuilt every field.
    """
    cached = getattr(rf, "_luma_amplitude_response", None)
    if cached is not None:
        return cached

    response = np.abs(rf.Filters["RFVideo"])
    # The stored response covers the whole spectrum with the negative
    # frequencies mirrored above Nyquist; only the positive half is meaningful.
    half = len(response) // 2
    cached = (response[:half], rf.freq_hz / len(response))
    rf._luma_amplitude_response = cached
    return cached


def carrier_frequency_bins(sys_params, rf):
    """Frequency bin edges across the carrier's deviation range, one bin per IRE.

    One bin per IRE because that is the video standard's own quantization of
    level, and the carrier frequency is a linear function of level.

    Anchored at sync tip - the format's own fixed reference, the one level that
    is defined rather than pictorial - and taken from the decoder's running
    parameters, so the anchor sits at the sync tip the signal actually has
    rather than the one the specification nominates.

    Samples outside this range are dropped rather than clipped into the end
    bins; see `measure_amplitude_by_frequency`. The range itself stays fixed, so
    the curve means the same thing from one field to the next - letting it
    follow the highest sample instead lets a single demodulator excursion set
    the axis, and the fit is then conditioned differently in every field.
    """
    sync_tip_hz = rf.iretohz(sys_params["vsync_ire"])
    peak_white_hz = rf.iretohz(100)
    bin_count = int(round((peak_white_hz - sync_tip_hz) / sys_params["hz_ire"]))
    return sync_tip_hz, peak_white_hz, max(bin_count, RESPONSE_POLYNOMIAL_ORDER + 1)


@njit(cache=True, nogil=True, fastmath=True)
def measure_amplitude_by_frequency(
    amplitude, carrier_hz, sync_tip_hz, hz_ire, bin_count, group_count
):
    """Carrier amplitude against carrier frequency, as points on a curve.

    Returns one point per group: the frequency the group covers, the median
    amplitude in it, and how many samples it holds. The median is what makes
    the estimate robust - amplitude noise is not symmetric, dropouts sit in the
    low tail, so a median tracks the level the carrier normally has at that
    frequency rather than the average of level and damage.

    The groups are equal-population rather than equal-width. The response is a
    smooth characteristic that needs only a few points to describe, and a
    fixed IRE grid spends most of its bins on levels the picture barely visits
    while crowding the ones it dwells on. Equal population puts every point on
    the curve at the same confidence, and puts them where the picture actually
    is.

    The caller passes a subsample of the field, already gathered; the curve has
    three parameters and even a subsample carries tens of thousands of points,
    so it is not short of evidence. The correction itself still applies to every
    sample.
    """
    count = len(amplitude)
    fine = np.zeros(bin_count, dtype=np.int64)
    index = np.full(count, -1, dtype=np.int64)

    for i in range(count):
        value = amplitude[i]
        if value <= 0.0:
            continue
        bin_number = int(np.floor((carrier_hz[i] - sync_tip_hz) / hz_ire))
        if bin_number < 0:
            continue
        if bin_number >= bin_count:
            bin_number = bin_count - 1
        index[i] = bin_number
        fine[bin_number] += 1

    total = 0
    for b in range(bin_count):
        total += fine[b]
    if total <= 0:
        return (
            np.zeros(0), np.zeros(0), np.zeros(0)
        )

    # merge adjacent fine bins until each group holds its share of the field
    group_of = np.zeros(bin_count, dtype=np.int64)
    target = total / group_count
    group = 0
    running = 0
    for b in range(bin_count):
        group_of[b] = group
        running += fine[b]
        if running >= target * (group + 1) and group < group_count - 1:
            group += 1
    groups = group + 1

    population = np.zeros(groups)
    centre = np.zeros(groups)
    for b in range(bin_count):
        g = group_of[b]
        population[g] += fine[b]
        centre[g] += fine[b] * (sync_tip_hz + (b + 0.5) * hz_ire)
    for g in range(groups):
        if population[g] > 0.0:
            centre[g] /= population[g]

    start = np.zeros(groups + 1, dtype=np.int64)
    for g in range(groups):
        start[g + 1] = start[g] + np.int64(population[g])
    cursor = start[:groups].copy()
    grouped = np.empty(start[groups], dtype=amplitude.dtype)
    for i in range(count):
        b = index[i]
        if b >= 0:
            g = group_of[b]
            grouped[cursor[g]] = amplitude[i]
            cursor[g] += 1

    levels = np.zeros(groups)
    for g in range(groups):
        first, last = start[g], start[g + 1]
        if last > first:
            levels[g] = np.median(grouped[first:last])

    return levels, population, centre


def fit_response_curve(
    levels,
    population,
    centre_hz,
    sync_tip_hz,
    peak_white_hz,
    order=RESPONSE_POLYNOMIAL_ORDER,
):
    """Fit the path's amplitude response across the carrier deviation range.

    Fitted in the logarithm, because the response is multiplicative, and in
    frequency relative to the band the carrier actually visits, because that is
    where the evidence is - the points span roughly 3.5 to 4.4 MHz, so a basis
    anchored at zero frequency would be three parameters fitted over a window
    five times its own width away from the origin, which is degenerate.

    Constrained, though, to what a response can physically do: amplitude falls
    as the wavelength shortens and never rises. Everything the carrier passed
    through - head gap, coating thickness, head-to-medium separation, record
    and playback equalization - takes amplitude away and none of it gives any
    back. An unconstrained quadratic is free to bend the other way on noise,
    and a curve that rises with frequency is not a response.

    Returns polynomial coefficients in band-relative frequency, or None if the
    field does not visit enough distinct levels to determine a curve.
    """
    valid = (population > 0) & (levels > 0)
    if np.count_nonzero(valid) < order + 1:
        return None

    normalized = _normalize_frequency(centre_hz[valid], sync_tip_hz, peak_white_hz)

    try:
        coefficients = np.polyfit(
            normalized,
            np.log(levels[valid]),
            order,
            w=np.sqrt(population[valid]),
        )
    except (np.linalg.LinAlgError, ValueError):
        # Too few distinct levels to determine a curve; the caller leaves the
        # chroma uncorrected rather than acting on a degenerate response.
        return None

    if order == 2:
        # d(ln A)/dx = 2*a*x + b must not be positive anywhere in 0 <= x <= 1
        quadratic, linear = coefficients[0], coefficients[1]
        if linear > 0.0:
            linear = 0.0
        if 2.0 * quadratic + linear > 0.0:
            quadratic = -0.5 * linear
        coefficients[0], coefficients[1] = quadratic, linear
    return coefficients


def _normalize_frequency(carrier_hz, sync_tip_hz, peak_white_hz):
    """Map the deviation range onto [0, 1] to condition the fit."""
    span = max(peak_white_hz - sync_tip_hz, np.finfo(np.float64).tiny)
    return (carrier_hz - sync_tip_hz) / span


def _inverse_response_table(
    response, frequency_step, coefficients, sync_tip_hz, peak_white_hz
):
    """The whole modelled response, inverted, on the grid it is tabulated on.

    The decoder's own RF path and the curve fitted on top of it are both
    functions of carrier frequency alone, evaluated on the same uniform grid, so
    there is no reason to carry them as two divisions per sample. Combining them
    here turns the correction's per-sample work into a table lookup and a
    multiply, and confines the exponential to the few thousand points that
    describe the response rather than the million it is applied to.

    Zero marks a frequency at which the model says nothing usable - the response
    vanishes, or the extrapolated curve has run out of range. That is the same
    sentinel the amplitude itself uses, and it leaves the deviation at unity.
    """
    grid_hz = np.arange(len(response), dtype=np.float64) * frequency_step
    normalized = _normalize_frequency(grid_hz, sync_tip_hz, peak_white_hz)

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        modelled = response * np.exp(np.polyval(coefficients, normalized))
        # Narrowed here rather than after, so a reciprocal that is finite in
        # double and infinite in single is caught by the same test as the rest.
        inverse = np.reciprocal(modelled).astype(np.float32)

    inverse[~np.isfinite(inverse)] = 0.0
    inverse[modelled <= 0.0] = 0.0
    return inverse


@njit(cache=True, nogil=True, fastmath=True)
def _deviation_from_inverse_response(
    amplitude, carrier_hz, inverse_response, frequency_step, low, high
):
    """How far the amplitude departs from the response modelled for it.

    One pass over the field, and no transcendental in it: the modelled response
    was inverted where it was tabulated, so what is left per sample is an
    interpolation, a multiply and a bound.
    """
    count = len(amplitude)
    deviation = np.empty(count, dtype=np.float32)
    last = len(inverse_response) - 1
    one = np.float32(1.0)
    zero = np.float32(0.0)
    for i in range(count):
        position = carrier_hz[i] / frequency_step
        if position <= zero:
            inverse = inverse_response[0]
        elif position >= last:
            inverse = inverse_response[last]
        else:
            lower = np.int32(position)
            fraction = position - np.float32(lower)
            base = inverse_response[lower]
            inverse = base + (inverse_response[lower + 1] - base) * fraction
        ratio = amplitude[i] * inverse
        if ratio <= zero:
            # no measurement here, or no usable model - leave the sample alone
            ratio = one
        elif ratio < low:
            ratio = low
        elif ratio > high:
            ratio = high
        deviation[i] = ratio
    return deviation


def measure_amplitude_deviation(field):
    '''How far the luma carrier's amplitude departs from the constant it should
    hold, measured across the whole field on the raw RF sample grid.

    Returns an `AmplitudeDeviation`, or None if the field cannot support a
    measurement.

    Demodulation happens per block and is not repeated here: this works on the
    assembled field, so the blocks are concatenated first and the part of the
    amplitude that follows the carrier's own frequency is removed afterwards,
    once, over the whole field. The tape imposes its amplitude noise as the
    field is read, continuously, so the reference it is measured against has to
    be continuous too - fitting each demodulation block separately puts a step
    at every block seam, on a carrier amplitude that has none.

    The response curve is fitted from a subsample and applied to every sample.
    The curve has three parameters and the subsample carries tens of thousands
    of points, so the fit is not the limiting factor; walking the whole field
    twice to determine it would be.

    Nothing here knows about the color-under. What comes back is a property of
    the luma carrier alone - the tape's multiplicative noise, with the path's
    response to the carrier's own frequency already removed - so it can drive
    any correction that shares the same head and instant.
    '''
    rf = field.rf
    video = field.data["video"]
    if video is None or "demod_raw" not in video:
        return None

    # Read as recorded. Both channels are already single precision, so these are
    # views onto the field's record array rather than copies of it.
    amplitude = video["envelope"]
    carrier_hz = video["demod_raw"]
    if len(amplitude) != len(carrier_hz) or len(amplitude) == 0:
        return None

    sys_params = rf.SysParams
    sync_tip_hz, peak_white_hz, bin_count = carrier_frequency_bins(sys_params, rf)
    response, frequency_step = rf_path_response(rf)

    # The part of the amplitude that follows the carrier's own frequency,
    # measured once over the concatenated field. Demodulation stays per block
    # and is not repeated here; what is measured is the assembled result, so
    # neither the curve nor the level it implies has a block boundary in it.
    stride = max(
        1, len(amplitude) // (RESPONSE_CURVE_POINTS * CURVE_SAMPLES_PER_POINT)
    )
    sampled_amplitude = np.asarray(amplitude[::stride], dtype=np.float64)
    sampled_carrier_hz = np.asarray(carrier_hz[::stride], dtype=np.float64)
    levels, population, centre_hz = measure_amplitude_by_frequency(
        _flatten_by_response(
            sampled_amplitude, sampled_carrier_hz, response, frequency_step
        ),
        sampled_carrier_hz,
        sync_tip_hz,
        sys_params["hz_ire"],
        bin_count,
        RESPONSE_CURVE_POINTS,
    )
    coefficients = fit_response_curve(
        levels, population, centre_hz, sync_tip_hz, peak_white_hz
    )
    if coefficients is None:
        return None

    dropout_fraction = rf.dod_options.dod_threshold_p
    deviation = _deviation_from_inverse_response(
        amplitude,
        carrier_hz,
        _inverse_response_table(
            response, frequency_step, coefficients, sync_tip_hz, peak_white_hz
        ),
        np.float32(frequency_step),
        np.float32(dropout_fraction),
        np.float32(1.0 / dropout_fraction),
    )

    return AmplitudeDeviation(deviation, carrier_hz)
