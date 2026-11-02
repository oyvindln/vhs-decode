import cython
import numpy as np
cimport numpy as np
from libc cimport stdint
import lddecode.utils as lddu

from libc.math cimport isnan, NAN, round, abs
from libc.stdlib cimport malloc, free

cdef Py_ssize_t NONE_INT = -1
cdef double NONE_DOUBLE = NAN

cdef inline double is_none_double(double a) nogil:
    return isnan(a)

cdef inline int round_to_int(double a) nogil:
    return <int> round(a)

cdef inline double c_abs(double a) nogil:
    return a * -1 if a < 0 else a

cdef bint inrange(double a, double mi, double ma) nogil:
    return (a >= mi) & (a <= ma)

@cython.cdivision
def pulse_qualitycheck(prev_pulse, pulse, int in_line_len):
    cdef double linelen
    cdef (double, double) exprange
    cdef bint inorder
    cdef int prev_pulse_type = prev_pulse[0]
    cdef int pulse_type = pulse[0]

    if prev_pulse_type > 0 and pulse_type > 0:
        exprange = (0.4, 0.6)
    elif prev_pulse_type == 0 and pulse_type == 0:
        exprange = (0.9, 1.1)
    else:  # transition to/from regular hsyncs can be .5 or 1H
        exprange = (0.4, 1.1)

    linelen = (pulse[1].start - prev_pulse[1].start) / <double> in_line_len
    inorder = inrange(linelen, exprange[0], exprange[1])

    return inorder

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision
cdef double c_median(double[::1] data) nogil:
    cdef Py_ssize_t data_len = len(data)
    cdef double result = 0
    cdef Py_ssize_t i
    
    for i in range(data_len):
        result += data[i]

    if data_len > 0:
        return result / data_len
    else:
        return 0

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double c_max(double[::1] data) nogil:
    cdef Py_ssize_t data_len = len(data)

    # TODO check
    cdef double result = NONE_DOUBLE

    for i in range(data_len):
        if data[i] > result:
            result = data[i]

    return result

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bint is_out_of_range(double[::1] data, double min, double max) nogil:
    """Check if data stays between min and max, returns fals if not."""
    cdef Py_ssize_t i
    cdef Py_ssize_t data_len = len(data)
    if data_len == 0:
        return True

    for i in range(data_len):
        if data[i] < min or data[i] > max:
            return True

    return False

@cython.wraparound(False)
@cython.boundscheck(False)
cdef Py_ssize_t calczc_findfirst(double[::1] data, double target, bint rising) nogil:
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

@cython.cdivision
cdef double calczc_do(double[::1] data, Py_ssize_t _start_offset, double target, Py_ssize_t count=10, int edge=0) nogil:
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
            return NONE_DOUBLE
    elif edge == -1:
        if data[_start_offset] < target:
            return NONE_DOUBLE

    cdef Py_ssize_t loc = calczc_findfirst(
        data[start_offset : start_offset + icount], target, edge == 1
    )

    # Need to use some proxy value instead of None here
    # if we want to avoid python interaction as there isn't an Option type in cython.
    if loc is NONE_INT:
        return NONE_DOUBLE

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[::1] copy_reversed(double[::1] arr):
    cdef Py_ssize_t arr_len = len(arr)
    cdef double[::1] reversed = np.empty((arr_len), dtype=np.float64)

    cdef Py_ssize_t i
    for i in range(arr_len):
        reversed[arr_len - i] = arr[i]

    return reversed

cdef double calczc(double[::1] data, Py_ssize_t _start_offset, double target, Py_ssize_t count=10, bint reverse=False, int edge=0):
    """ Calculate where data first crosses target in the direction specified by edge.
    edge:  -1 falling, 0 either, 1 rising
    count: How many samples to check
    reverse: If true, check in reverse direction
    """

    # NOTE:(oln) Not sure if we really need to specify start offset/count or if we could simply get by by
    # supplying the slice of data to look in.

    if reverse:
        # Instead of actually implementing this in reverse, flip data
        rev_zc = calczc_do(copy_reversed(data), 0, target, count, edge)

        if isnan(rev_zc):
            return NONE_DOUBLE

        return _start_offset - rev_zc

    return calczc_do(data, _start_offset, target, count, edge)

cdef struct s_sync_distance_input:
    double meanlinelen,
    double VSYNC_TOLERANCE_LINES,
    double hsync_start_line,
    double first_pulse,
    double second_pulse,
    double first_line,
    double second_line

cdef struct s_sync_distance_output:
    double distance_offset
    double hsync_loc
    int valid_locations

@cython.cdivision(True)
@cython.exceptval(check=False)
cdef inline void calc_sync_from_known_distances(
    s_sync_distance_input *sync_distance_input, 
    s_sync_distance_output *sync_distance_output
) nogil:
    sync_distance_output.distance_offset = 0
    sync_distance_output.hsync_loc = 0
    sync_distance_output.valid_locations = 0

    cdef double actual_lines
    cdef double expected_lines
    cdef double actual_lines_rounded

    # skip any pulses that are missing
    if sync_distance_input.first_pulse != -1 and sync_distance_input.second_pulse != -1 and sync_distance_input.meanlinelen != 0:
        actual_lines = (sync_distance_input.first_pulse - sync_distance_input.second_pulse) / sync_distance_input.meanlinelen
        expected_lines = sync_distance_input.first_line - sync_distance_input.second_line
        actual_lines_rounded = round_nearest_line_loc(actual_lines)

        if (
            actual_lines < expected_lines + sync_distance_input.VSYNC_TOLERANCE_LINES and 
            actual_lines > expected_lines - sync_distance_input.VSYNC_TOLERANCE_LINES
        ):
            sync_distance_output.distance_offset = actual_lines - expected_lines
            sync_distance_output.hsync_loc = sync_distance_input.second_pulse + sync_distance_input.meanlinelen * (sync_distance_input.hsync_start_line - sync_distance_input.second_line)
            sync_distance_output.valid_locations = 1
            # print("syncing on distance", expected_lines - actual_lines, sync_distance_input.first_line, sync_distance_input.second_line)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_first_hsync_loc(
    validpulses,
    double meanlinelen,
    bint is_ntsc,
    field_lines_in,
    int num_eq_pulses,
    int prev_first_field,
    double last_field_offset_lines,
    double prev_first_hsync_loc,
    double prev_hsync_diff,
    int field_order_confidence,
    double fallback_line0loc,
    int fallback_is_first_field,
    int fallback_is_first_field_confidence,
):
    """
    Returns: 
       * line0loc: Location of line 0 (last hsync pulse of the previous field)
       * first_hsync_loc: Location of the first hsync pulse (first line after the vblanking)
       * hsync_start_line: Line where the first hsync pulse is found
       * next_field: Location of the next field (last line + 1 after this field)
       * first_field: True if this is the first field
    """
    cdef Py_ssize_t i

    cdef double VSYNC_TOLERANCE_LINES = .5

    cdef Py_ssize_t field_order_lengths_len = 4
    cdef double *field_order_lengths = [-1, -1, -1, -1]
    cdef int *vblank_pulses = [-1, -1, -1, -1, -1, -1, -1, -1]
    cdef double *vblank_lines = [-1, -1, -1, -1, -1, -1, -1, -1]

    cdef int FIRST_VBLANK_EQ_1_START = 0
    cdef int FIRST_VBLANK_VSYNC_START = 1
    cdef int FIRST_VBLANK_VSYNC_END = 2
    cdef int FIRST_VBLANK_EQ_2_END = 3
    cdef int LAST_VBLANK_EQ_1_START = 4
    cdef int LAST_VBLANK_VSYNC_START = 5
    cdef int LAST_VBLANK_VSYNC_END = 6
    cdef int LAST_VBLANK_EQ_2_END = 7

    # ********************************************************************
    # Convert the incoming python objects to simple arrays for performance
    # ********************************************************************
    cdef Py_ssize_t validpulses_len = len(validpulses)
    cdef int *validpulses_type = <int *> malloc(validpulses_len * sizeof(int))
    cdef int *validpulses_start = <int *> malloc(validpulses_len * sizeof(int))
    cdef int *validpulses_valid = <int *> malloc(validpulses_len * sizeof(int))
    
    for i in range(validpulses_len):
        validpulses_type[i] = validpulses[i][0]
        validpulses_start[i] = validpulses[i][1].start
        validpulses_valid[i] = validpulses[i][2]

    cdef int field_lines_len = len(field_lines_in)
    cdef int *field_lines = <int *> malloc(field_lines_len * sizeof(int))
    for i in range(field_lines_len):
        field_lines[i] = field_lines_in[i]

    # **************************************************************************************
    # get the vblanking pulses and assign them to either the first or second vblanking group
    # **************************************************************************************
    cdef Py_ssize_t last_pulse = -1
    cdef int group = 0
    cdef int field_group = 0

    for i in range(validpulses_len):
        if last_pulse != -1 and validpulses_valid[i]:
            if (
                # move to the next vsync pulse group if we're currently on the first vsync interval
                group == 0 and

                # and the next vsync is close to the expected location of the next field
                validpulses_start[i] > validpulses_start[0] + field_lines[0] * meanlinelen
            ):
                group = 4
                field_group = 2

            # hsync -> [eq 1]
            if validpulses_type[last_pulse] == 0 and validpulses_type[i] > 0:
                vblank_pulses[0 + group] = validpulses_start[i]
                field_order_lengths[0 + field_group] = round_nearest_line_loc((validpulses_start[i] - validpulses_start[last_pulse]) / meanlinelen)

            # eq 1 -> [vsync]
            elif validpulses_type[last_pulse] == 1 and validpulses_type[i] == 2:
                vblank_pulses[1 + group] = validpulses_start[i]

            # [vsync] -> eq 2
            elif validpulses_type[last_pulse] == 2 and validpulses_type[i] == 3:
                vblank_pulses[2 + group] = validpulses_start[i]
                
            # [eq 2] -> hsync
            elif validpulses_type[last_pulse] > 0 and validpulses_type[i] == 0:
                vblank_pulses[3 + group] = validpulses_start[last_pulse]
                field_order_lengths[1 + field_group] = round_nearest_line_loc((validpulses_start[i] - validpulses_start[last_pulse]) / meanlinelen)

        last_pulse = i

    # ********************************************************************
    # determine if this is the first or second field based on pulse length
    # ********************************************************************
    cdef Py_ssize_t FIRST_HSYNC_LENGTH = 0
    cdef Py_ssize_t FIRST_EQPL2_LENGTH = 1
    cdef Py_ssize_t LAST_HSYNC_LENGTH = 2
    cdef Py_ssize_t LAST_EQPL2_LENGTH = 3

    cdef int first_field
    cdef int progressive_field = False

    cdef double *first_field_lengths = [-1, -1, -1, -1]
    cdef double *second_field_lengths = [-1, -1, -1, -1]
    cdef double *progressive_field_lengths = [-1, -1, -1, -1]

    if is_ntsc:
        first_field_lengths[FIRST_HSYNC_LENGTH] = 1
        first_field_lengths[FIRST_EQPL2_LENGTH] = .5
        first_field_lengths[LAST_HSYNC_LENGTH] = .5
        first_field_lengths[LAST_EQPL2_LENGTH] = 1

        second_field_lengths[FIRST_HSYNC_LENGTH] = .5
        second_field_lengths[FIRST_EQPL2_LENGTH] = 1
        second_field_lengths[LAST_HSYNC_LENGTH] = 1
        second_field_lengths[LAST_EQPL2_LENGTH] = .5

        # TODO: What is an expected progressive field order for NTSC?
        progressive_field_lengths[FIRST_HSYNC_LENGTH] = 1
        progressive_field_lengths[FIRST_EQPL2_LENGTH] = .5
        progressive_field_lengths[LAST_HSYNC_LENGTH] = 1
        progressive_field_lengths[LAST_EQPL2_LENGTH] = .5
    else:
        first_field_lengths[FIRST_HSYNC_LENGTH] = .5
        first_field_lengths[FIRST_EQPL2_LENGTH] = .5
        first_field_lengths[LAST_HSYNC_LENGTH] = 1
        first_field_lengths[LAST_EQPL2_LENGTH] = 1

        second_field_lengths[FIRST_HSYNC_LENGTH] = 1
        second_field_lengths[FIRST_EQPL2_LENGTH] = 1
        second_field_lengths[LAST_HSYNC_LENGTH] = .5
        second_field_lengths[LAST_EQPL2_LENGTH] = .5

        # TODO: What is an expected progressive field order for PAL?
        progressive_field_lengths[FIRST_HSYNC_LENGTH] = .5
        progressive_field_lengths[FIRST_EQPL2_LENGTH] = .5
        progressive_field_lengths[LAST_HSYNC_LENGTH] = .5
        progressive_field_lengths[LAST_EQPL2_LENGTH] = .5

    # measure the likelyhood of the field order based on the detected pulse widths vs. expected pulse widths, weigthed by the number of measurements
    cdef double interlaced_field_boundaries_consensus = 0
    cdef double interlaced_field_boundaries_detected = 0
    cdef double progressive_field_consensus = 0
    cdef double progressive_field_boundaries_detected = 0

    cdef double field_length
    for i in range(0, field_order_lengths_len):
        field_length = field_order_lengths[i]

        if field_length == first_field_lengths[i]:
            interlaced_field_boundaries_consensus += 1
            interlaced_field_boundaries_detected += 1

        if field_length == second_field_lengths[i]:
            interlaced_field_boundaries_detected += 1

        if field_length == progressive_field_lengths[i]:
            progressive_field_consensus += 1
            progressive_field_boundaries_detected += 1

        if field_length != -1:
            progressive_field_boundaries_detected += 1

    # guess the field order if no previous field exists
    if prev_first_field == -1:
        if interlaced_field_boundaries_detected == 0 or round(interlaced_field_boundaries_consensus / interlaced_field_boundaries_detected) == 1 or fallback_is_first_field == 1:
            first_field = True
        else:
            first_field = False
    # continue a previously established field order cadence, i.e. inverse of the previous field
    else:
        first_field = not prev_first_field
    
    # Override field cadence depending on confidence of field order detection
    # * Confidence depends on the proportion of `first_field = True` (y/n)
    # * Confidence is weighted by number of measurements (field_boundaries_detected / )
    # * Confidence is bounded between 0 and 100%
    # * If field_boundaries_detected = 0, confidence is undefined and we use the previously defined `first_field`

    cdef int first_field_confidence = 0
    cdef int second_field_confidence = 0
    cdef double interlaced_field_order_weighting = interlaced_field_boundaries_detected / field_order_lengths_len

    cdef int progressive_field_confidence = 0
    cdef double progressive_field_order_weighting = progressive_field_boundaries_detected / field_order_lengths_len
    if interlaced_field_boundaries_detected > 0:
        # when the fallback vsync is disabled, and there is no previous hsync location, lower min confidence to 50
        if (fallback_line0loc == -1 and prev_first_hsync_loc < 0):
            field_order_confidence = 50 if field_order_confidence > 50 else field_order_confidence

        # first try to match on interlaced fields
        first_field_confidence = round_to_int((interlaced_field_boundaries_consensus / interlaced_field_boundaries_detected) * interlaced_field_order_weighting * 100)
        second_field_confidence = round_to_int((interlaced_field_boundaries_detected - interlaced_field_boundaries_consensus) / interlaced_field_boundaries_detected * interlaced_field_order_weighting * 100)

        if first_field_confidence >= field_order_confidence and first_field_confidence > second_field_confidence:
            first_field = True
        elif second_field_confidence >= field_order_confidence and first_field_confidence < second_field_confidence:
            first_field = False
        
        # next, try to detect progressive fields
        # this should be true if we're 100% confident both fields are first fields
        if progressive_field_boundaries_detected > 0:
            progressive_field_confidence = round_to_int((progressive_field_boundaries_detected - progressive_field_consensus) / progressive_field_boundaries_detected * progressive_field_order_weighting * 100)
    
            if progressive_field_confidence == field_order_lengths_len:
                progressive_field = True

    # overrides previous first field, if fallback has more confidence
    if (
        fallback_is_first_field_confidence > first_field_confidence
        and fallback_is_first_field_confidence > second_field_confidence
    ):
        first_field = fallback_is_first_field == 1

    # ***********************************************************
    # calculate the expected line locations for each vblank pulse
    # ***********************************************************
    cdef double line0loc_line = 0
    cdef double vsync_section_lines = num_eq_pulses / 2.0
    cdef double hsync_start_line
    cdef double current_field_lines
    cdef double previous_field_lines

    if first_field:
        current_field_lengths = first_field_lengths
        previous_field_lines = field_lines[1]
        current_field_lines = field_lines[0]
    else:
        current_field_lengths = second_field_lengths
        previous_field_lines = field_lines[0]
        current_field_lines = field_lines[1]

    vblank_lines[FIRST_VBLANK_EQ_1_START] = line0loc_line + current_field_lengths[FIRST_HSYNC_LENGTH]
    vblank_lines[FIRST_VBLANK_VSYNC_START] = vblank_lines[FIRST_VBLANK_EQ_1_START] + vsync_section_lines
    vblank_lines[FIRST_VBLANK_VSYNC_END] = vblank_lines[FIRST_VBLANK_VSYNC_START] + vsync_section_lines
    vblank_lines[FIRST_VBLANK_EQ_2_END] = vblank_lines[FIRST_VBLANK_VSYNC_END] + vsync_section_lines - .5

    hsync_start_line = vblank_lines[FIRST_VBLANK_EQ_2_END] + current_field_lengths[FIRST_EQPL2_LENGTH]

    vblank_lines[LAST_VBLANK_EQ_1_START] = current_field_lines + current_field_lengths[LAST_HSYNC_LENGTH]
    vblank_lines[LAST_VBLANK_VSYNC_START] = vblank_lines[LAST_VBLANK_EQ_1_START] + vsync_section_lines
    vblank_lines[LAST_VBLANK_VSYNC_END] = vblank_lines[LAST_VBLANK_VSYNC_START] + vsync_section_lines
    vblank_lines[LAST_VBLANK_EQ_2_END] = vblank_lines[LAST_VBLANK_VSYNC_END] + vsync_section_lines - .5

    # **********************************************************************************
    # Use the vsync pulses and their expected lines to derive first hsync pulse location
    #   (i.e. pulse right after the first vblanking interval)
    # **********************************************************************************
    cdef s_sync_distance_input sync_distance_input
    sync_distance_input.meanlinelen = meanlinelen
    sync_distance_input.VSYNC_TOLERANCE_LINES = VSYNC_TOLERANCE_LINES
    sync_distance_input.hsync_start_line = hsync_start_line

    cdef s_sync_distance_output sync_distance_output

    cdef Py_ssize_t first_index
    cdef Py_ssize_t second_index

    # *************************************************************
    # check the vblanking area at the beginning for valid locations
    # *************************************************************
    cdef double first_vblank_first_hsync_loc = 0
    cdef int    first_vblank_valid_location_count = 0
    cdef double first_vblank_offset = 0
    cdef Py_ssize_t first_vblank_pulse_indexes_len = 4
    cdef Py_ssize_t *first_vblank_pulse_indexes = [
        FIRST_VBLANK_EQ_1_START,
        FIRST_VBLANK_VSYNC_START,
        FIRST_VBLANK_VSYNC_END,
        FIRST_VBLANK_EQ_2_END
    ]

    for first_index in range(0, first_vblank_pulse_indexes_len):
        for second_index in range(first_index+1, first_vblank_pulse_indexes_len):
            sync_distance_input.first_pulse = vblank_pulses[first_vblank_pulse_indexes[first_index]]
            sync_distance_input.second_pulse = vblank_pulses[first_vblank_pulse_indexes[second_index]]
            sync_distance_input.first_line = vblank_lines[first_vblank_pulse_indexes[first_index]]
            sync_distance_input.second_line = vblank_lines[first_vblank_pulse_indexes[second_index]]

            calc_sync_from_known_distances(
                &sync_distance_input,
                &sync_distance_output
            )

            first_vblank_offset += sync_distance_output.distance_offset
            first_vblank_first_hsync_loc += sync_distance_output.hsync_loc
            first_vblank_valid_location_count += sync_distance_output.valid_locations

    # *******************************************************
    # check the vblanking area at the end for valid locations
    # *******************************************************
    cdef double last_vblank_first_hsync_loc = 0
    cdef int   last_vblank_valid_location_count = 0
    cdef double last_vblank_offset = 0
    cdef Py_ssize_t last_vblank_pulse_indexes_len = 4
    cdef Py_ssize_t *last_vblank_pulse_indexes = [
        LAST_VBLANK_EQ_1_START,
        LAST_VBLANK_VSYNC_START,
        LAST_VBLANK_VSYNC_END,
        LAST_VBLANK_EQ_2_END
    ]

    for first_index in range(0, last_vblank_pulse_indexes_len):
        for second_index in range(first_index+1, last_vblank_pulse_indexes_len):
            sync_distance_input.first_pulse = vblank_pulses[last_vblank_pulse_indexes[first_index]]
            sync_distance_input.second_pulse = vblank_pulses[last_vblank_pulse_indexes[second_index]]
            sync_distance_input.first_line = vblank_lines[last_vblank_pulse_indexes[first_index]]
            sync_distance_input.second_line = vblank_lines[last_vblank_pulse_indexes[second_index]]

            calc_sync_from_known_distances(
                &sync_distance_input,
                &sync_distance_output
            )

            last_vblank_offset += sync_distance_output.distance_offset
            last_vblank_first_hsync_loc += sync_distance_output.hsync_loc
            last_vblank_valid_location_count += sync_distance_output.valid_locations

    cdef double first_hsync_loc = 0
    cdef int valid_location_count = 0
    cdef double offset = 0
    
    # ********************************************************
    # validate the distance between the two vblanking sections
    # ********************************************************
    cdef double first_vblank_hsync_estimate = first_vblank_first_hsync_loc / first_vblank_valid_location_count if first_vblank_valid_location_count != 0 else 0
    cdef double last_vblank_hsync_estimate = last_vblank_first_hsync_loc / last_vblank_valid_location_count if last_vblank_valid_location_count != 0 else 0
    
    # if both vblanks have estimated hsync start locations
    if (
        first_vblank_valid_location_count != 0 and 
        last_vblank_valid_location_count != 0 and
        # and the estimated starting locations are the same
        first_vblank_hsync_estimate < last_vblank_hsync_estimate + VSYNC_TOLERANCE_LINES * meanlinelen and
        first_vblank_hsync_estimate > last_vblank_hsync_estimate - VSYNC_TOLERANCE_LINES * meanlinelen
    ):
        # sync on both start and last vblanks

        first_hsync_loc = first_vblank_first_hsync_loc + last_vblank_first_hsync_loc
        valid_location_count = first_vblank_valid_location_count + last_vblank_valid_location_count
        offset = first_vblank_offset + last_vblank_offset
    
        # sync accross the two vblanks
        for first_index in range(0, first_vblank_pulse_indexes_len):
            for second_index in range(0, last_vblank_pulse_indexes_len):
                sync_distance_input.first_pulse = vblank_pulses[first_vblank_pulse_indexes[first_index]]
                sync_distance_input.second_pulse = vblank_pulses[last_vblank_pulse_indexes[second_index]]
                sync_distance_input.first_line = vblank_lines[first_vblank_pulse_indexes[first_index]]
                sync_distance_input.second_line = vblank_lines[last_vblank_pulse_indexes[second_index]]

                calc_sync_from_known_distances(
                    &sync_distance_input,
                    &sync_distance_output
                )

                offset += sync_distance_output.distance_offset
                first_hsync_loc += sync_distance_output.hsync_loc
                valid_location_count += sync_distance_output.valid_locations
    
        # print("using both vblank intervals")

    # otherwise, if fallback vsync is enabled, use that
    elif fallback_line0loc != -1:
        first_hsync_loc = fallback_line0loc + meanlinelen * hsync_start_line
        valid_location_count = 1
        offset = 0

        # print("using fallback")

    # otherwise sync on only one vblank
    # this will happen when on the very beginning or end of a recording
    elif (
        # sure about this vblank
        first_vblank_valid_location_count == 6 or 

        # or not synced yet and first vblank has more valid locations
        (prev_first_hsync_loc <= 0 and 
        first_vblank_valid_location_count != 0 and first_vblank_valid_location_count > last_vblank_valid_location_count)
    ):
        first_hsync_loc = first_vblank_first_hsync_loc
        valid_location_count = first_vblank_valid_location_count
        offset = first_vblank_offset
    
        # print("using first vblank interval")

    elif (
        # sure about this vblank
        last_vblank_valid_location_count == 6 or

        # or not synced yet and last vblank has more valid locations
        (prev_first_hsync_loc <= 0 and 
        last_vblank_valid_location_count != 0 and last_vblank_valid_location_count > first_vblank_valid_location_count)
    ):
        first_hsync_loc = last_vblank_first_hsync_loc
        valid_location_count = last_vblank_valid_location_count
        offset = last_vblank_offset
    
        # print("using last vblank interval")


    # ********************************************************************************
    # estimate the hsync location based on the previous valid field using read offsets
    # ********************************************************************************
    # TODO: not sure why this is, it should always be the previous field lines for all formats
    cdef double estimated_hsync_field_lines = previous_field_lines if is_ntsc else current_field_lines 

    cdef int estimated_hsync_loc = round_to_int(
        (
            last_field_offset_lines + 
            estimated_hsync_field_lines +
            prev_first_hsync_loc / meanlinelen # previous line location of last hsync
        ) * meanlinelen
    )

    cdef double estimated_hsync_with_offset
    cdef bint used_estimated_hsync = 0
    if (
        # if there are no valid sync distances
        valid_location_count == 0 and

        # and the previous hsync location is before the current field
        prev_first_hsync_loc > 0
    ):
        # previous field                  current field
        # |-------|-----------------------|-------|-----------------------|
        # offset--prev--------------------0-------curr--------------------total
        #         ^-------------------------------^

        # use the difference from the previous hsync if within .5 lines
        if prev_hsync_diff <= .5 and prev_hsync_diff >= -.5:
            # TODO: determine when to add or subtract the prev_hsync_diff
            #       maybe this can be based on difference in tape speed, add if slower, subtract if faster
            estimated_hsync_with_offset = estimated_hsync_loc + meanlinelen * prev_hsync_diff
        else:
            estimated_hsync_with_offset = estimated_hsync_loc
        
        # when estimated hsync is negative, just use the closest valid pulse or 0
        # we are not synced here, but continue to return a sync location to 
        # avoid dropping video and getting out of sync with audio
        if estimated_hsync_with_offset <= 0:
            estimated_hsync_with_offset = validpulses_start[0] if validpulses_len > 0 else 0

        first_hsync_loc += estimated_hsync_with_offset
        valid_location_count += 1
        used_estimated_hsync = 1

        # print("using estimated hsync")

        # validate the estimated hsync location with any existing vsync pulses
        #for i in range(0, len(pulse_indexes)):
        #    sync_distance_input.first_pulse = vblank_pulses[pulse_indexes[i]]
        #    sync_distance_input.second_pulse = estimated_hsync_with_offset
        #    sync_distance_input.first_line = vblank_lines[pulse_indexes[i]]
        #    sync_distance_input.second_line = hsync_start_line
        #
        #    calc_sync_from_known_distances(
        #        &sync_distance_input,
        #        &sync_distance_output
        #    )
        #
        #    offset += sync_distance_output.distance_offset
        #    first_hsync_loc += sync_distance_output.hsync_loc
        #    valid_location_count += sync_distance_output.valid_locations
        

    # use any sync pulses, if nothing is found
    #if valid_location_count == 0:
    #    for i in range(0, 4):
    #        sync_distance_input.first_pulse = vblank_pulses[i]
    #        sync_distance_input.second_pulse = vblank_pulses[i]
    #        sync_distance_input.first_line = vblank_lines[i]
    #        sync_distance_input.second_line = vblank_lines[i]
    #
    #        res = calc_sync_from_known_distances(
    #            &sync_distance_input,
    #            &sync_distance_output
    #        )
    #
    #        offset += sync_distance_output.distance_offset
    #        first_hsync_loc += sync_distance_output.hsync_loc
    #        valid_location_count += sync_distance_output.valid_locations

    # ******************************************************************************
    # Take the mean of the known vblanking locations to derive the first hsync pulse
    # ******************************************************************************
    cdef double hsync_offset = 0
    cdef int hsync_count = 0
    cdef double lineloc
    cdef double line0loc
    cdef int rlineloc
    cdef double next_field
    cdef int pulse_loc
    if valid_location_count > 0:
        offset /= valid_location_count
        first_hsync_loc = round((first_hsync_loc + offset) / valid_location_count)

        # don't change the previous distance if this is an estimated sync location
        # since we don't know what the actual current hsync is
        if not used_estimated_hsync:
            prev_hsync_diff = (first_hsync_loc - estimated_hsync_loc) / meanlinelen
        
        # ****************************************************************
        # Align estimated start with hsync pulses to prevent skipped lines
        # ****************************************************************
        for i in range(validpulses_len):
            if validpulses_type[i] != 0 or not validpulses_valid[i]:
                continue

            lineloc = (validpulses_start[i] - first_hsync_loc) / meanlinelen + hsync_start_line
            rlineloc = round_to_int(lineloc)

            if rlineloc > current_field_lines:
                break

            if rlineloc >= hsync_start_line:
                hsync_offset += first_hsync_loc + meanlinelen * (rlineloc - hsync_start_line) - validpulses_start[i]
                hsync_count += 1

        if hsync_count > 0:
            hsync_offset /= hsync_count
            first_hsync_loc -= hsync_offset

        line0loc = first_hsync_loc - meanlinelen * hsync_start_line
        next_field = first_hsync_loc + meanlinelen * (vblank_lines[LAST_VBLANK_EQ_1_START] - hsync_start_line)

        ## vsync debugging
        ## field order
        #print(
        #    "prev first field:", prev_first_field, 
        #    "curr first field:", 1 if first_field else 0,
        #    "field boundry consensus:", interlaced_field_boundaries_consensus,
        #    "field boundries detected:", interlaced_field_boundaries_detected, 
        #    "field boundries detected:", interlaced_field_boundaries_detected, 
        #    [x for x in field_order_lengths[:field_order_lengths_len]],
        #    "first field confidence:", first_field_confidence,
        #    "second field confidence:", second_field_confidence,
        #    "progressive field confidence:", progressive_field_confidence,
        #)
        #
        ## vsync pulses
        #print([x for x in vblank_lines[:8]])
        #print("actual vblank", [x for x in vblank_pulses[:8]])
        #
        ## calculated vs actual vsync pulses
        #for i in range(0, 8):
        #    vblank_lines[i] = round_to_int(first_hsync_loc + meanlinelen * (vblank_lines[i] - hsync_start_line))
        #print("calc vblank", [x for x in vblank_lines[:8]])
        #print("next field", next_field)
        #
        ## estimated hsync locations
        #print(
        #    "first vblank hsync",
        #    first_vblank_hsync_estimate,
        #    "last vblank hsync",
        #    last_vblank_hsync_estimate,
        #    "estimated hsync",
        #    estimated_hsync_loc,
        #    "fallback hsync",
        #    fallback_line0loc
        #)
        #print(
        #    "actual hsync loc",
        #    first_hsync_loc
        #)
        #print()

        free(validpulses_type)
        free(validpulses_start)
        free(validpulses_valid)
        free(field_lines)

        return line0loc, first_hsync_loc, hsync_start_line, next_field, first_field, progressive_field, prev_hsync_diff, np.asarray([x for x in vblank_pulses[:8]], dtype=np.int32)

    # no sync pulses found
    # print("no sync pulses found")

    free(validpulses_type)
    free(validpulses_start)
    free(validpulses_valid)
    free(field_lines)
    return None, None, hsync_start_line, None, first_field, progressive_field, prev_hsync_diff, np.asarray([x for x in vblank_pulses[:8]], dtype=np.int32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision
def valid_pulses_to_linelocs(
    validpulses_in,
    int reference_pulse,
    int reference_line,
    double meanlinelen,
    double hsync_tolerance,
    int proclines,
    double gap_detection_threshold,
):
    """Goes through the list of detected sync pulses that seem to be valid,
    and maps the start locations to a line number and throws out ones that do not seem to match or are out of place.

    Args:
        validpulses ([int]): List of sync pulses
        reference_pulse (int): Sample location of the reference pulse
        reference_line (int): Line that the reference pulse represents
        meanlinelen (double): Average line length
        hsync_tolerance (double): How much a sync pulse can deviate from normal before being discarded.
        proclines (int): Total number of lines to process
        gap_detection_threshold (double): Threshold to check for skipped hsync pulses

    Returns:
        * line_locations: locations for each field line where in the index is the line number and the value is the start of the pulse
        * line_location_errs: array of boolean indicating if the pulse was esimated from other near pulses
        * last_valid_line_location: the last valid pulse detected
    """

    # Lists to fill
    cdef double[::1] validpulses = np.sort(np.asarray(
        [
            p[1].start
            for p in validpulses_in
        ], dtype=np.double, order='c'
    ))
    cdef double[::1] line_locations = np.empty((proclines), dtype=np.double, order='c')
    cdef stdint.uint8_t[::1] line_location_errs = np.full(proclines, 0, dtype=np.uint8, order='c')

    cdef Py_ssize_t line_index
    cdef Py_ssize_t pulse_search_index
    
    # refine the line locations to pulses
    cdef double current_distance_from_pulse_to_line
    cdef double next_observed_distance_between_pulse_and_line
    cdef double smallest_distance_observed_from_pulse_to_line
    cdef double current_pulse_sample_location

    cdef Py_ssize_t current_pulse_index = 0
    cdef Py_ssize_t validpulses_len = len(validpulses)

    # This loop performs a best-fit to align the scan lines, which are expected to increment always
    # by around mean_line_len distance in samples, and the pulse locations in samples that were detected earlier
    # * Each line starts out by incrementing by the mean_line_length (estimated location)
    # * The inner loop searches for the nearest pulse at the estimated location within +- max_distance_between_pulse_and_line
    # * The closest pulse to the estimated location within the max_distance_between_pulse_and_line is assigned to the line
    #   * Each pulse is only ever assigned to a line once
    # * If there isn't a pulse within max_distance_between_pulse_and_line, then the line keeps the estimated location

    max_allowed_distance_between_pulse_and_line = meanlinelen / 1.5
    for line_index in range(0, proclines):
        # Start by setting this line's sample location to the expected location relative to the reference line
        line_locations[line_index] = reference_pulse + meanlinelen * (line_index - reference_line)

        # search for the closest pulse, pulse locations are assumed to be sorted
        if current_pulse_index < validpulses_len:
            # start the search using the distance between the current pulse and the expected line location
            current_distance_from_pulse_to_line = c_abs(validpulses[current_pulse_index] - line_locations[line_index])

            # start by setting this to the max allowed value so the loop will break if the next pulse is further away
            smallest_distance_observed_from_pulse_to_line = max_allowed_distance_between_pulse_and_line

            # reset the best fit variables
            next_observed_distance_between_pulse_and_line = -1
            current_pulse_sample_location = -1
            
            # start iteration at the pulse that hasn't been assigned yet
            pulse_search_index = current_pulse_index

            while pulse_search_index < validpulses_len - 1:
                if current_distance_from_pulse_to_line <= smallest_distance_observed_from_pulse_to_line:
                    smallest_distance_observed_from_pulse_to_line = current_distance_from_pulse_to_line

                    current_pulse_index = pulse_search_index
                    current_pulse_sample_location = validpulses[pulse_search_index]
    
                # peek ahead to the next pulse to measure the distance
                next_observed_distance_between_pulse_and_line = c_abs(validpulses[pulse_search_index+1] - line_locations[line_index])
                if next_observed_distance_between_pulse_and_line > current_distance_from_pulse_to_line:
                    # if the next pulse is greater than the current distance, we have already found the closest pulse
                    break
                else:
                    # if the distance is not greater, continue searching
                    current_distance_from_pulse_to_line = next_observed_distance_between_pulse_and_line
                    pulse_search_index += 1
            
            # if we found a pulse that was close enough (i.e. +- max_distance_between_pulse_and_line),
            # the set this line to the pulse's location, replacing the estimated location that was already set above
            # otherwise, keep the estimated location
            if current_pulse_sample_location != -1:
                # print(line_index, "using nearest pulse", current_pulse_sample_location)
                line_locations[line_index] = current_pulse_sample_location
                current_pulse_index += 1
            # else:
            #     # print(line_index, "using estimated pulse", line_locations[line_index])
            #     # fill any guessed lines with the previous field values
            #     line_location_errs[line_index] = 1

            # move on to the next line

    return line_locations, line_location_errs, current_pulse_sample_location # will be the last pulse that was assigned to a line at this point

# Rounds a line number to it's nearest line at intervals of .5 lines
@cython.cfunc
@cython.inline
@cython.cdivision(True)
cdef inline double round_nearest_line_loc(double line_number) nogil:
    return round(0.5 * round(line_number / 0.5) * 10) / 10.0

@cython.boundscheck(False)
def refine_linelocs_hsync(field, stdint.uint8_t[::1] linebad, double hsync_threshold):
    """Refine the line start locations using horizontal sync data."""

    # Original used a copy here which resulted in a list.
    # Use an array instead for efficiency.
    cdef double[::1] linelocs_original = np.asarray(field.linelocs1, dtype=np.float64)
    cdef double[::1] linelocs_refined = np.array(linelocs_original, dtype=np.float64, copy=True)

    # Lookup these values here instead of doing it on every loop iteration.
    cdef double[::1] demod_05 = np.array(field.data["video"]["demod_05"], dtype=np.float64, order='c')
    rf = field.rf
    cdef int normal_hsync_length = field.usectoinpx(rf.SysParams["hsyncPulseUS"])
    cdef int one_usec = rf.freq
    cdef float sample_rate_mhz = rf.freq
    cdef bint is_pal = rf.system == "PAL"
    cdef bint disable_right_hsync = rf.options.disable_right_hsync
    cdef double zc_threshold = hsync_threshold #rf.iretohz(rf.SysParams["vsync_ire"] / 2.0)
    cdef double ire_30 = rf.iretohz(30)
    cdef double ire_n_65 = rf.iretohz(-65)
    cdef double ire_110 = rf.iretohz(110)

    cdef bint right_cross_refined
    cdef double refined_from_right_lineloc = -1
    cdef double zc_fr
    cdef double porch_level
    cdef double prev_porch_level = -1
    cdef double sync_level
    cdef int ll1
    cdef Py_ssize_t i
    cdef double[::1] hsync_area
    cdef double[::1] back_porch
    cdef double zc
    cdef double zc2
    cdef double right_cross

    with nogil:
        for i in range(len(linelocs_original)):
            # skip VSYNC lines, since they handle the pulses differently
            if inrange(i, 3, 6) or (is_pal and inrange(i, 1, 2)):
                linebad[i] = True
                continue

            # refine beginning of hsync

            # start looking 1 usec back
            ll1 = round_to_int(linelocs_original[i]) - one_usec
            # and locate the next time the half point between hsync and 0 is crossed.
            zc = NONE_DOUBLE
            zc = calczc_do(
                demod_05,
                ll1,
                zc_threshold,
                count=one_usec * 2,
            )

            right_cross = NONE_DOUBLE
            if not disable_right_hsync:
                right_cross = calczc_do(
                    demod_05,
                    ll1 + (normal_hsync_length) - one_usec,
                    zc_threshold,
                    count=round_to_int(normal_hsync_length)*2,
                    edge=1,
                )
            right_cross_refined = False

            # If the crossing exists, we can check if the hsync pulse looks normal and
            # refine it.
            if not is_none_double(zc) and not linebad[i]:
                linelocs_refined[i] = zc

                # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
                # TODO: Use correct values for NTSC/PAL here
                hsync_area = demod_05[
                    round_to_int(zc - (sample_rate_mhz * 0.75)) : round_to_int(zc + (sample_rate_mhz * 3.5))
                ]
                back_porch = demod_05[
                    round_to_int(zc + sample_rate_mhz * 3.5) : round_to_int(zc + (sample_rate_mhz * 8))
                ]
                if is_out_of_range(hsync_area, ire_n_65, ire_110): # or is_out_of_range(back_porch, ire_n_65, ire_110):
                    # don't use the computed value here if it's bad
                    linebad[i] = True
                    linelocs_refined[i] = linelocs_original[i]
                else:

                    if c_max(hsync_area) < ire_30:
                        porch_level = c_median(
                            demod_05[round_to_int(zc + (sample_rate_mhz * 8)) : round_to_int(zc + (sample_rate_mhz * 9))]
                        )
                    else:
                        if prev_porch_level > 0:
                            porch_level = prev_porch_level
                        else:
                            porch_level = c_median(
                                demod_05[round_to_int(zc - (sample_rate_mhz * 1.0)) : round_to_int(zc - (sample_rate_mhz * 0.5))]
                            )
                    sync_level = c_median(
                        demod_05[round_to_int(zc + (sample_rate_mhz * 1)) : round_to_int(zc + (sample_rate_mhz * 2.5))]
                    )

                    # Re-calculate the crossing point using the mid point between the measured sync
                    # and porch levels
                    zc2 = calczc_do(
                        demod_05,
                        ll1,
                        (porch_level + sync_level) / 2.0,
                        count=400,
                    )

                    # any wild variation here indicates a failure
                    if not is_none_double(zc2) and c_abs(zc2 - zc) < (sample_rate_mhz / 2.0):
                        linelocs_refined[i] = zc2
                        prev_porch_level = porch_level
                    else:
                        # Give up
                        # front_porch_level = c_median(
                        #     demod_05[int(zc - (one_usec * 1.0)) : int(zc - (one_usec * 0.5))]
                        # )

                        if prev_porch_level > 0:
                            # Try again with a earlier measurement porch.
                            zc2 = calczc_do(
                                demod_05,
                                ll1,
                                (prev_porch_level + sync_level) / 2.0,
                                count=400,
                            )
                            if not is_none_double(zc2) and c_abs(zc2 - zc) < (sample_rate_mhz / 2.0):
                                linelocs_refined[i] = zc2
                            else:
                                linebad[i] = True
                        else:
                            # Give up
                            linebad[i] = True
            else:
                linebad[i] = True

            # Check right cross
            if not is_none_double(right_cross):
                zc2 = NONE_DOUBLE

                zc_fr = right_cross - normal_hsync_length

                # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
                # NOTE: This is more than hsync area, might wanna also check max levels of level in hsync
                hsync_area = demod_05[
                    round_to_int(zc_fr - (sample_rate_mhz * 0.75)) : round_to_int(zc_fr + (sample_rate_mhz * 8))
                ]

                if not is_out_of_range(hsync_area, ire_n_65, ire_30):

                    porch_level = c_median(
                        demod_05[round_to_int(zc_fr + normal_hsync_length + (sample_rate_mhz * 1)) : round_to_int(zc_fr + normal_hsync_length + (sample_rate_mhz * 2))]
                    )

                    sync_level = c_median(
                        demod_05[
                            round_to_int(zc_fr + (sample_rate_mhz * 1)) : round_to_int(zc_fr + (sample_rate_mhz * 2.5))
                        ]
                    )

                    # Re-calculate the crossing point using the mid point between the measured sync
                    # and porch levels
                    zc2 = calczc_do(
                        demod_05,
                        ll1 + normal_hsync_length - one_usec,
                        (porch_level + sync_level) / 2.0,
                        count=400,
                    )

                    # any wild variation here indicates a failure
                    if not is_none_double(zc2) and c_abs(zc2 - right_cross) < (sample_rate_mhz / 2.0):
                        # TODO: Magic value here, this seem to give be approximately correct results
                        # but may not be ideal for all inputs.
                        # Value based on default sample rate so scale if it's different.
                        refined_from_right_lineloc = right_cross - normal_hsync_length + (2.25 * (sample_rate_mhz / 40.0))
                        # Don't use if it deviates too much which could indicate a false positive or non-standard hsync length.
                        if c_abs(refined_from_right_lineloc - linelocs_refined[i]) < (sample_rate_mhz * 2):
                            right_cross = zc2
                            right_cross_refined = True
                            prev_porch_level = porch_level

            if linebad[i]:
                linelocs_refined[i] = linelocs_original[
                    i
                ]  # don't use the computed value here if it's bad

            if not is_none_double(right_cross) and right_cross_refined:
                # If we get a good result from calculating hsync start from the
                # right side of the hsync pulse, we use that as it's less likely
                # to be messed up by overshoot.
                linebad[i] = False
                linelocs_refined[i] = refined_from_right_lineloc

    return linelocs_refined
