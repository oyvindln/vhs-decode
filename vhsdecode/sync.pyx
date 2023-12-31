import cython
import numpy as np
cimport numpy as np
import math
import lddecode.utils as lddu
from libc.math cimport isnan, NAN

cdef Py_ssize_t NONE_INT = -1

cdef round_to_int(double a):
    return int(round(a))

cdef bint inrange(double a, double mi, double ma):
    return (a >= mi) & (a <= ma)

def pulse_qualitycheck(prev_pulse, pulse, int in_line_len):
    cdef double linelen
    cdef (double, double) exprange
    cdef bint inorder

    if prev_pulse[0] > 0 and pulse[0] > 0:
        exprange = (0.4, 0.6)
    elif prev_pulse[0] == 0 and pulse[0] == 0:
        exprange = (0.9, 1.1)
    else:  # transition to/from regular hsyncs can be .5 or 1H
        exprange = (0.4, 1.1)

    linelen = (pulse[1].start - prev_pulse[1].start) / in_line_len
    inorder = inrange(linelen, exprange[0], exprange[1])

    return inorder


cdef c_median(np.ndarray data):
    # TODO: For whatever reason the numba variant is massively faster than
    # calling np.median via cython.
    return lddu.nb_median(data)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bint is_out_of_range(double[:] data, double min, double max) nogil:
    """Check if data stays between min and max, returns fals if not."""
    cdef Py_ssize_t i
    for i in range(0, len(data)):
        if data[i] < min or data[i] > max:
            return True

    return False

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Py_ssize_t calczc_findfirst(double[:] data, double target, bint rising) nogil:
    """Find the index where data first crosses target, in the specified direction.
       returns NONE_INT if no crossing is found.
    """
    cdef Py_ssize_t i
    if rising:
        for i in range(0, len(data)):
            if data[i] >= target:
                return i

        return NONE_INT
    else:
        for i in range(0, len(data)):
            if data[i] <= target:
                return i

        return NONE_INT

cdef double calczc_do(double[:] data, Py_ssize_t _start_offset, double target, int edge=0, Py_ssize_t count=10) nogil:
    """Find the index where data first crosses target in the specified direction, and then try to estimate
    the exact crossing point between the samples.
    Will fail if the start point is already past the threshold in the requested direction.
    ."""
    cdef Py_ssize_t start_offset = max(1, _start_offset)
    cdef Py_ssize_t icount = count + 1

    if edge == 0:  # capture rising or falling edge
        if data[start_offset] < target:
            edge = 1
        else:
            edge = -1

    if edge == 1:
        if data[_start_offset] > target:
            return NAN
    elif edge == -1:
        if data[_start_offset] < target:
            return NAN

    cdef Py_ssize_t loc = calczc_findfirst(
        data[start_offset : start_offset + icount], target, edge == 1
    )

    # Need to use some proxy value instead of None here
    # if we want to avoid python interaction as there isn't an Option type in cython.
    if loc is NONE_INT:
        return NAN

    cdef Py_ssize_t x = start_offset + loc
    cdef double a = data[x - 1] - target
    cdef double b = data[x] - target
    cdef double y

    if b - a != 0:
        y = -a / (-a + b)
    else:
        # print(
        #     "RuntimeWarning: Div by zero prevented at lddecode/utils.calczc_do()", a, b
        # )
        y = 0

    return x - 1 + y


def calczc(double[:] data, Py_ssize_t _start_offset, double target, Py_ssize_t count=10, bint reverse=False, int edge=0):
    """ Calculate where data first crosses target in the direction specified by edge.
    edge:  -1 falling, 0 either, 1 rising
    count: How many samples to check
    reverse: If true, check in reverse direction
    """

    # NOTE:(oln) Not sure if we really need to specify start offset/count or if we could simply get by by
    # supplying the slice of data to look in.

    if reverse:
        # Instead of actually implementing this in reverse, use numpy to flip data
        rev_zc = calczc_do(data[_start_offset::-1], 0, target, edge, count)
        if isnan(rev_zc):
            return None

        return _start_offset - rev_zc

    res = calczc_do(data, _start_offset, target, edge, count)
    if isnan(res):
        return None
    return res

def valid_pulses_to_linelocs(
    validpulses,
    double line0loc,
    bint skip_detected,
    double meanlinelen,
    int linecount,
    double hsync_tolerance,
    int lastlineloc,
):
    """Goes through the list of detected sync pulses that seem to be valid,
    and maps the start locations to a line number and throws out ones that do not seem to match or are out of place.

    Args:
        validpulses ([type]): List of sync pulses
        line0loc (float): Start location of line 0
        skip_detected (bool): TODO
        meanlinelen (float): Average line length
        linecount (int): Number of lines in the field
        hsync_tolerance (float): How much a sync pulse can deviate from normal before being discarded.
        lastlineloc (float): Start location of the last line

    Returns:
        [type]: Two dicts, one containing a list of linelocs mapped to line numbers, and one containing the distance between the line start and expected line start.
    """

    cdef double lineloc
    cdef int rlineloc
    cdef double lineloc_distance

    # Lists to fill
    # TODO: Could probably use arrays here instead of converting to arrays later.
    linelocs_dict = {}
    linelocs_dist = {}
    for p in validpulses:
        # Calculate what line number the pulse corresponds closest too.
        lineloc = (p[1].start - line0loc) / meanlinelen
        rlineloc = round(lineloc)
        lineloc_distance = abs(lineloc - rlineloc)

        # TODO doc
        if skip_detected:
            lineloc_end = linecount - ((lastlineloc - p[1].start) / meanlinelen)
            rlineloc_end = round(lineloc_end)
            lineloc_end_distance = abs(lineloc_end - rlineloc_end)

            if p[0] == 0 and rlineloc > 23 and lineloc_end_distance < lineloc_distance:
                lineloc = lineloc_end
                rlineloc = rlineloc_end
                lineloc_distance = lineloc_end_distance

        # only record if it's closer to the (probable) beginning of the line
        if lineloc_distance > hsync_tolerance or (
            rlineloc in linelocs_dict and lineloc_distance > linelocs_dist[rlineloc]
        ):
            continue

        # also skip non-regular lines (non-hsync) that don't seem to be in valid order (p[2])
        # (or hsync lines in the vblank area)
        if rlineloc > 0 and not p[2]:
            if p[0] > 0 or (p[0] == 0 and rlineloc < 10):
                continue

        linelocs_dict[rlineloc] = p[1].start
        linelocs_dist[rlineloc] = lineloc_distance

    return linelocs_dict, linelocs_dist

def refine_linelocs_hsync(field, np.ndarray linebad, double hsync_threshold):
    """Refine the line start locations using horizontal sync data."""

    # Original used a copy here which resulted in a list.
    # Use an array instead for efficiency.
    cdef np.ndarray[np.float64_t, ndim=1] linelocs2 = np.asarray(field.linelocs1, dtype=np.float64)

    # Lookup these values here instead of doing it on every loop iteration.
    demod_05 = field.data["video"]["demod_05"]
    rf = field.rf
    cdef int normal_hsync_length = field.usectoinpx(rf.SysParams["hsyncPulseUS"])
    cdef int one_usec = rf.freq
    cdef float sample_rate_mhz = rf.freq
    cdef bint is_pal = rf.system == "PAL"
    cdef bint disable_right_hsync = rf.options.disable_right_hsync
    cdef double zc_threshold = rf.iretohz(rf.SysParams["vsync_ire"] / 2)
    cdef double ire_30 = rf.iretohz(30)
    cdef double ire_n_55 = rf.iretohz(-55)
    cdef double ire_110 = rf.iretohz(110)

    cdef bint right_cross_refined
    cdef double refined_from_right_lineloc = -1
    cdef double zc_fr
    cdef double porch_level
    cdef double prev_porch_level = -1
    cdef double sync_level
    cdef int ll1
    cdef int i
    cdef double[:] hsync_area

    # Make sure we've done a copy
    if type(field.linelocs1) == type(linelocs2):
        assert(field.linelocs1.dtype != linelocs2.dtype, "BUG! input to refine_linelocs_hsync was not copied!")

    for i in range(len(field.linelocs1)):
        # skip VSYNC lines, since they handle the pulses differently
        if inrange(i, 3, 6) or (is_pal and inrange(i, 1, 2)):
            linebad[i] = True
            continue

        # refine beginning of hsync

        # start looking 1 usec back
        ll1 = field.linelocs1[i] - one_usec
        # and locate the next time the half point between hsync and 0 is crossed.
        zc = calczc(
            demod_05,
            ll1,
            zc_threshold,
            reverse=False,
            count=one_usec * 2,
        )

        right_cross = None

        if not disable_right_hsync:
            right_cross = calczc(
                demod_05,
                ll1 + (normal_hsync_length) - one_usec,
                zc_threshold,
                reverse=False,
                count=round_to_int(one_usec * 3),
                edge=1,
            )
        right_cross_refined = False

        # If the crossing exists, we can check if the hsync pulse looks normal and
        # refine it.
        if zc is not None and not linebad[i]:
            linelocs2[i] = zc

            # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
            # TODO: Use correct values for NTSC/PAL here
            hsync_area = demod_05[
                round_to_int(zc - (one_usec * 0.75)) : round_to_int(zc + (one_usec * 3.5))
            ]
            back_porch = demod_05[
                round_to_int(zc + one_usec * 3.5) : round_to_int(zc + (one_usec * 8))
            ]
            if is_out_of_range(hsync_area, ire_n_55, ire_110): # or is_out_of_range(back_porch, ire_n_55, ire_110):
                # don't use the computed value here if it's bad
                linebad[i] = True
                linelocs2[i] = field.linelocs1[i]
            else:

                if np.amax(hsync_area) < ire_30:
                    porch_level = c_median(
                        demod_05[round_to_int(zc + (one_usec * 8)) : round_to_int(zc + (one_usec * 9))]
                    )
                else:
                    if prev_porch_level > 0:
                        porch_level = prev_porch_level
                    else:
                        porch_level = c_median(
                            demod_05[round_to_int(zc - (one_usec * 1.0)) : round_to_int(zc - (one_usec * 0.5))]
                        )
                sync_level = c_median(
                    demod_05[round_to_int(zc + (one_usec * 1)) : round_to_int(zc + (one_usec * 2.5))]
                )

                # Re-calculate the crossing point using the mid point between the measured sync
                # and porch levels
                zc2 = calczc(
                    demod_05,
                    ll1,
                    (porch_level + sync_level) / 2,
                    reverse=False,
                    count=400,
                )

                # any wild variation here indicates a failure
                if zc2 is not None and abs(zc2 - zc) < (one_usec / 2):
                    linelocs2[i] = zc2
                    prev_porch_level = porch_level
                else:
                    # Give up
                    # front_porch_level = c_median(
                    #     demod_05[int(zc - (one_usec * 1.0)) : int(zc - (one_usec * 0.5))]
                    # )

                    if prev_porch_level > 0:
                        # Try again with a earlier measurement porch.
                        zc2 = calczc(
                            demod_05,
                            ll1,
                            (prev_porch_level + sync_level) / 2,
                            reverse=False,
                            count=400,
                        )
                        if zc2 is not None and abs(zc2 - zc) < (one_usec / 2):
                            linelocs2[i] = zc2
                        else:
                            linebad[i] = True
                    else:
                        # Give up
                        linebad[i] = True
        else:
            linebad[i] = True

        # Check right cross
        if right_cross is not None:
            zc2 = None

            zc_fr = right_cross - normal_hsync_length

            # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
            # NOTE: This is more than hsync area, might wanna also check max levels of level in hsync
            hsync_area = demod_05[
                round_to_int(zc_fr - (one_usec * 0.75)) : round_to_int(zc_fr + (one_usec * 8))
            ]

            if not is_out_of_range(hsync_area, ire_n_55, ire_30):

                porch_level = c_median(
                    demod_05[round_to_int(zc_fr + normal_hsync_length + (one_usec * 1)) : round_to_int(zc_fr + normal_hsync_length + (one_usec * 2))]
                )

                sync_level = c_median(
                    demod_05[
                        round_to_int(zc_fr + (one_usec * 1)) : round_to_int(zc_fr + (one_usec * 2.5))
                    ]
                )

                # Re-calculate the crossing point using the mid point between the measured sync
                # and porch levels
                zc2 = calczc(
                    demod_05,
                    ll1 + normal_hsync_length - one_usec,
                    (porch_level + sync_level) / 2,
                    reverse=False,
                    count=400,
                )

                # any wild variation here indicates a failure
                if zc2 is not None and abs(zc2 - right_cross) < (one_usec / 2):
                    # TODO: Magic value here, this seem to give be approximately correct results
                    # but may not be ideal for all inputs.
                    # Value based on default sample rate so scale if it's different.
                    refined_from_right_lineloc = right_cross - normal_hsync_length + (2.25 * (sample_rate_mhz / 40.0))
                    # Don't use if it deviates too much which could indicate a false positive or non-standard hsync length.
                    if abs(refined_from_right_lineloc - linelocs2[i]) < (one_usec * 2):
                        right_cross = zc2
                        right_cross_refined = True
                        prev_porch_level = porch_level

        if linebad[i]:
            linelocs2[i] = field.linelocs1[
                i
            ]  # don't use the computed value here if it's bad

        if right_cross is not None and right_cross_refined:
            # If we get a good result from calculating hsync start from the
            # right side of the hsync pulse, we use that as it's less likely
            # to be messed up by overshoot.
            linebad[i] = False
            linelocs2[i] = refined_from_right_lineloc

    return linelocs2
