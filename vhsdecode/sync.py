"""Mapping of detected sync pulses to line locations, and hsync refinement."""

import math

import numpy as np


def _c_round(a):
    """C round(): halves round away from zero, unlike Python's round()."""
    t = math.trunc(a)
    if abs(a - t) >= 0.5:
        t += 1 if a > 0 else -1
    return t


def _round_nearest_line_loc(line_number):
    """Round a line number to the nearest half line."""
    return _c_round(0.5 * _c_round(line_number / 0.5) * 10) / 10.0


def _mean(data):
    # Summed left to right; np.mean's pairwise summation would round differently.
    return float(np.add.accumulate(data)[-1]) / len(data) if len(data) else 0.0


def _is_out_of_range(data, low, high):
    """True if data is empty or leaves [low, high]."""
    return len(data) == 0 or bool(np.any((data < low) | (data > high)))


def _calczc(data, start_offset_in, target, count=10, edge=0):
    """Find where data first crosses target from start_offset on, interpolating between samples.

    edge: 1 rising, -1 falling, 0 whichever the start sample implies. Returns nan if the start
    sample is already past the target in the requested direction or no crossing is found within
    count samples.
    """
    start_offset = max(1, start_offset_in)

    if edge == 0:
        edge = 1 if data[start_offset] < target else -1

    if edge == 1:
        if data[start_offset_in] > target:
            return math.nan
    elif data[start_offset_in] < target:
        return math.nan

    window = data[start_offset : start_offset + count + 1]
    hits = np.nonzero(window >= target if edge == 1 else window <= target)[0]
    if len(hits) == 0:
        return math.nan

    x = start_offset + int(hits[0])
    a = data[x - 1] - target
    b = data[x] - target
    y = -a / (-a + b) if b - a != 0 else 0.0

    return x - 1 + y


def get_first_hsync_loc(
    validpulses,
    meanlinelen,
    is_ntsc,
    field_lines_in,
    num_eq_pulses,
    prev_first_field,
    last_field_offset_lines,
    prev_first_hsync_loc,
    prev_hsync_diff,
    field_order_confidence,
    fallback_line0loc,
    fallback_is_first_field,
    fallback_is_first_field_confidence,
):
    """
    Returns:
       * line0loc: Location of line 0 (last hsync pulse of the previous field)
       * first_hsync_loc: Location of the first hsync pulse (first line after the vblanking)
       * hsync_start_line: Line where the first hsync pulse is found
       * next_field: Location of the next field (last line + 1 after this field)
       * first_field: True if this is the first field
    """
    meanlinelen = float(meanlinelen)
    prev_first_field = int(prev_first_field)
    last_field_offset_lines = float(last_field_offset_lines)
    prev_first_hsync_loc = float(prev_first_hsync_loc)
    prev_hsync_diff = float(prev_hsync_diff)
    field_order_confidence = int(field_order_confidence)
    fallback_line0loc = float(fallback_line0loc)
    fallback_is_first_field = int(fallback_is_first_field)
    fallback_is_first_field_confidence = int(fallback_is_first_field_confidence)

    VSYNC_TOLERANCE_LINES = 0.5

    field_order_lengths = [-1.0, -1.0, -1.0, -1.0]
    vblank_pulses = [-1] * 8
    vblank_lines = [-1.0] * 8

    (
        FIRST_VBLANK_EQ_1_START,
        FIRST_VBLANK_VSYNC_START,
        FIRST_VBLANK_VSYNC_END,
        FIRST_VBLANK_EQ_2_END,
        LAST_VBLANK_EQ_1_START,
        LAST_VBLANK_VSYNC_START,
        LAST_VBLANK_VSYNC_END,
        LAST_VBLANK_EQ_2_END,
    ) = range(8)

    # Pulse starts and field line counts are used as integers, truncating any fraction.
    validpulses_type = [int(p[0]) for p in validpulses]
    validpulses_start = [int(p[1].start) for p in validpulses]
    validpulses_valid = [int(p[2]) for p in validpulses]
    field_lines = [int(x) for x in field_lines_in]

    # **************************************************************************************
    # get the vblanking pulses and assign them to either the first or second vblanking group
    # **************************************************************************************
    last_pulse = -1
    group = 0
    field_group = 0

    for i in range(len(validpulses)):
        if last_pulse != -1 and validpulses_valid[i]:
            if (
                # move to the next vsync pulse group if we're currently on the first vsync interval
                group == 0
                # and the next vsync is close to the expected location of the next field
                and validpulses_start[i] > validpulses_start[0] + field_lines[0] * meanlinelen
            ):
                group = 4
                field_group = 2

            # hsync -> [eq 1]
            if validpulses_type[last_pulse] == 0 and validpulses_type[i] > 0:
                vblank_pulses[0 + group] = validpulses_start[i]
                field_order_lengths[0 + field_group] = _round_nearest_line_loc(
                    (validpulses_start[i] - validpulses_start[last_pulse]) / meanlinelen
                )

            # eq 1 -> [vsync]
            elif validpulses_type[last_pulse] == 1 and validpulses_type[i] == 2:
                vblank_pulses[1 + group] = validpulses_start[i]

            # [vsync] -> eq 2
            elif validpulses_type[last_pulse] == 2 and validpulses_type[i] == 3:
                vblank_pulses[2 + group] = validpulses_start[i]

            # [eq 2] -> hsync
            elif validpulses_type[last_pulse] > 0 and validpulses_type[i] == 0:
                vblank_pulses[3 + group] = validpulses_start[last_pulse]
                field_order_lengths[1 + field_group] = _round_nearest_line_loc(
                    (validpulses_start[i] - validpulses_start[last_pulse]) / meanlinelen
                )

        last_pulse = i

    # ********************************************************************
    # determine if this is the first or second field based on pulse length
    # ********************************************************************
    FIRST_HSYNC_LENGTH, FIRST_EQPL2_LENGTH, LAST_HSYNC_LENGTH, LAST_EQPL2_LENGTH = range(4)

    progressive_field = False

    if is_ntsc:
        first_field_lengths = [1.0, 0.5, 0.5, 1.0]
        second_field_lengths = [0.5, 1.0, 1.0, 0.5]
        # TODO: What is an expected progressive field order for NTSC?
        progressive_field_lengths = [1.0, 0.5, 1.0, 0.5]
    else:
        first_field_lengths = [0.5, 0.5, 1.0, 1.0]
        second_field_lengths = [1.0, 1.0, 0.5, 0.5]
        # TODO: What is an expected progressive field order for PAL?
        progressive_field_lengths = [0.5, 0.5, 0.5, 0.5]

    # measure the likelyhood of the field order based on the detected pulse widths vs. expected pulse widths, weigthed by the number of measurements
    interlaced_field_boundaries_consensus = 0.0
    interlaced_field_boundaries_detected = 0.0
    progressive_field_consensus = 0.0
    progressive_field_boundaries_detected = 0.0

    for i in range(4):
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
        first_field = (
            interlaced_field_boundaries_detected == 0
            or _c_round(interlaced_field_boundaries_consensus / interlaced_field_boundaries_detected) == 1
            or fallback_is_first_field == 1
        )
    # continue a previously established field order cadence, i.e. inverse of the previous field
    else:
        first_field = not prev_first_field

    # Override field cadence depending on confidence of field order detection
    # * Confidence depends on the proportion of `first_field = True` (y/n)
    # * Confidence is weighted by number of measurements (field_boundaries_detected / )
    # * Confidence is bounded between 0 and 100%
    # * If field_boundaries_detected = 0, confidence is undefined and we use the previously defined `first_field`

    first_field_confidence = 0
    second_field_confidence = 0
    interlaced_field_order_weighting = interlaced_field_boundaries_detected / 4

    progressive_field_confidence = 0
    progressive_field_order_weighting = progressive_field_boundaries_detected / 4
    if interlaced_field_boundaries_detected > 0:
        # when the fallback vsync is disabled, and there is no previous hsync location, lower min confidence to 50
        if fallback_line0loc == -1 and prev_first_hsync_loc < 0:
            field_order_confidence = 50 if field_order_confidence > 50 else field_order_confidence

        # first try to match on interlaced fields
        first_field_confidence = _c_round(
            (interlaced_field_boundaries_consensus / interlaced_field_boundaries_detected)
            * interlaced_field_order_weighting
            * 100
        )
        second_field_confidence = _c_round(
            (interlaced_field_boundaries_detected - interlaced_field_boundaries_consensus)
            / interlaced_field_boundaries_detected
            * interlaced_field_order_weighting
            * 100
        )

        if first_field_confidence >= field_order_confidence and first_field_confidence > second_field_confidence:
            first_field = True
        elif second_field_confidence >= field_order_confidence and first_field_confidence < second_field_confidence:
            first_field = False

        # next, try to detect progressive fields
        # this should be true if we're 100% confident both fields are first fields
        if progressive_field_boundaries_detected > 0:
            progressive_field_confidence = _c_round(
                (progressive_field_boundaries_detected - progressive_field_consensus)
                / progressive_field_boundaries_detected
                * progressive_field_order_weighting
                * 100
            )

            if progressive_field_confidence == 4:
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
    line0loc_line = 0.0
    vsync_section_lines = num_eq_pulses / 2.0

    if first_field:
        current_field_lengths = first_field_lengths
        previous_field_lines = float(field_lines[1])
        current_field_lines = float(field_lines[0])
    else:
        current_field_lengths = second_field_lengths
        previous_field_lines = float(field_lines[0])
        current_field_lines = float(field_lines[1])

    vblank_lines[FIRST_VBLANK_EQ_1_START] = line0loc_line + current_field_lengths[FIRST_HSYNC_LENGTH]
    vblank_lines[FIRST_VBLANK_VSYNC_START] = vblank_lines[FIRST_VBLANK_EQ_1_START] + vsync_section_lines
    vblank_lines[FIRST_VBLANK_VSYNC_END] = vblank_lines[FIRST_VBLANK_VSYNC_START] + vsync_section_lines
    vblank_lines[FIRST_VBLANK_EQ_2_END] = vblank_lines[FIRST_VBLANK_VSYNC_END] + vsync_section_lines - 0.5

    hsync_start_line = vblank_lines[FIRST_VBLANK_EQ_2_END] + current_field_lengths[FIRST_EQPL2_LENGTH]

    vblank_lines[LAST_VBLANK_EQ_1_START] = current_field_lines + current_field_lengths[LAST_HSYNC_LENGTH]
    vblank_lines[LAST_VBLANK_VSYNC_START] = vblank_lines[LAST_VBLANK_EQ_1_START] + vsync_section_lines
    vblank_lines[LAST_VBLANK_VSYNC_END] = vblank_lines[LAST_VBLANK_VSYNC_START] + vsync_section_lines
    vblank_lines[LAST_VBLANK_EQ_2_END] = vblank_lines[LAST_VBLANK_VSYNC_END] + vsync_section_lines - 0.5

    # **********************************************************************************
    # Use the vsync pulses and their expected lines to derive first hsync pulse location
    #   (i.e. pulse right after the first vblanking interval)
    # **********************************************************************************
    def sync_from_known_distances(first_index, second_index):
        """(distance offset, hsync location, 1) if the two pulses are the expected distance apart."""
        first_pulse = float(vblank_pulses[first_index])
        second_pulse = float(vblank_pulses[second_index])
        first_line = vblank_lines[first_index]
        second_line = vblank_lines[second_index]

        # skip any pulses that are missing
        if first_pulse != -1 and second_pulse != -1 and meanlinelen != 0:
            actual_lines = (first_pulse - second_pulse) / meanlinelen
            expected_lines = first_line - second_line

            if (
                actual_lines < expected_lines + VSYNC_TOLERANCE_LINES
                and actual_lines > expected_lines - VSYNC_TOLERANCE_LINES
            ):
                return (
                    actual_lines - expected_lines,
                    second_pulse + meanlinelen * (hsync_start_line - second_line),
                    1,
                )
        return 0.0, 0.0, 0

    first_vblank_pulse_indexes = [
        FIRST_VBLANK_EQ_1_START,
        FIRST_VBLANK_VSYNC_START,
        FIRST_VBLANK_VSYNC_END,
        FIRST_VBLANK_EQ_2_END,
    ]
    last_vblank_pulse_indexes = [
        LAST_VBLANK_EQ_1_START,
        LAST_VBLANK_VSYNC_START,
        LAST_VBLANK_VSYNC_END,
        LAST_VBLANK_EQ_2_END,
    ]

    # *************************************************************
    # check the vblanking area at the beginning for valid locations
    # *************************************************************
    first_vblank_first_hsync_loc = 0.0
    first_vblank_valid_location_count = 0
    first_vblank_offset = 0.0

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            distance_offset, hsync_loc, valid_locations = sync_from_known_distances(
                first_vblank_pulse_indexes[first_index], first_vblank_pulse_indexes[second_index]
            )
            first_vblank_offset += distance_offset
            first_vblank_first_hsync_loc += hsync_loc
            first_vblank_valid_location_count += valid_locations

    # *******************************************************
    # check the vblanking area at the end for valid locations
    # *******************************************************
    last_vblank_first_hsync_loc = 0.0
    last_vblank_valid_location_count = 0
    last_vblank_offset = 0.0

    for first_index in range(4):
        for second_index in range(first_index + 1, 4):
            distance_offset, hsync_loc, valid_locations = sync_from_known_distances(
                last_vblank_pulse_indexes[first_index], last_vblank_pulse_indexes[second_index]
            )
            last_vblank_offset += distance_offset
            last_vblank_first_hsync_loc += hsync_loc
            last_vblank_valid_location_count += valid_locations

    first_hsync_loc = 0.0
    valid_location_count = 0
    offset = 0.0

    # ********************************************************
    # validate the distance between the two vblanking sections
    # ********************************************************
    first_vblank_hsync_estimate = (
        first_vblank_first_hsync_loc / first_vblank_valid_location_count
        if first_vblank_valid_location_count != 0
        else 0
    )
    last_vblank_hsync_estimate = (
        last_vblank_first_hsync_loc / last_vblank_valid_location_count
        if last_vblank_valid_location_count != 0
        else 0
    )

    # if both vblanks have estimated hsync start locations
    if (
        first_vblank_valid_location_count != 0
        and last_vblank_valid_location_count != 0
        # and the estimated starting locations are the same
        and first_vblank_hsync_estimate < last_vblank_hsync_estimate + VSYNC_TOLERANCE_LINES * meanlinelen
        and first_vblank_hsync_estimate > last_vblank_hsync_estimate - VSYNC_TOLERANCE_LINES * meanlinelen
    ):
        # sync on both start and last vblanks
        first_hsync_loc = first_vblank_first_hsync_loc + last_vblank_first_hsync_loc
        valid_location_count = first_vblank_valid_location_count + last_vblank_valid_location_count
        offset = first_vblank_offset + last_vblank_offset

        # sync accross the two vblanks
        for first_index in range(4):
            for second_index in range(4):
                distance_offset, hsync_loc, valid_locations = sync_from_known_distances(
                    first_vblank_pulse_indexes[first_index], last_vblank_pulse_indexes[second_index]
                )
                offset += distance_offset
                first_hsync_loc += hsync_loc
                valid_location_count += valid_locations

    # otherwise, if fallback vsync is enabled, use that
    elif fallback_line0loc != -1:
        first_hsync_loc = fallback_line0loc + meanlinelen * hsync_start_line
        valid_location_count = 1
        offset = 0.0

    # otherwise sync on only one vblank
    # this will happen when on the very beginning or end of a recording
    elif (
        # sure about this vblank
        first_vblank_valid_location_count == 6
        # or not synced yet and first vblank has more valid locations
        or (
            prev_first_hsync_loc <= 0
            and first_vblank_valid_location_count != 0
            and first_vblank_valid_location_count > last_vblank_valid_location_count
        )
    ):
        first_hsync_loc = first_vblank_first_hsync_loc
        valid_location_count = first_vblank_valid_location_count
        offset = first_vblank_offset

    elif (
        # sure about this vblank
        last_vblank_valid_location_count == 6
        # or not synced yet and last vblank has more valid locations
        or (
            prev_first_hsync_loc <= 0
            and last_vblank_valid_location_count != 0
            and last_vblank_valid_location_count > first_vblank_valid_location_count
        )
    ):
        first_hsync_loc = last_vblank_first_hsync_loc
        valid_location_count = last_vblank_valid_location_count
        offset = last_vblank_offset

    # ********************************************************************************
    # estimate the hsync location based on the previous valid field using read offsets
    # ********************************************************************************
    # TODO: not sure why this is, it should always be the previous field lines for all formats
    estimated_hsync_field_lines = previous_field_lines if is_ntsc else current_field_lines

    estimated_hsync_loc = _c_round(
        (
            last_field_offset_lines
            + estimated_hsync_field_lines
            + prev_first_hsync_loc / meanlinelen  # previous line location of last hsync
        )
        * meanlinelen
    )

    used_estimated_hsync = False
    if (
        # if there are no valid sync distances
        valid_location_count == 0
        # and the previous hsync location is before the current field
        and prev_first_hsync_loc > 0
    ):
        # previous field                  current field
        # |-------|-----------------------|-------|-----------------------|
        # offset--prev--------------------0-------curr--------------------total
        #         ^-------------------------------^

        # use the difference from the previous hsync if within .5 lines
        if prev_hsync_diff <= 0.5 and prev_hsync_diff >= -0.5:
            # TODO: determine when to add or subtract the prev_hsync_diff
            #       maybe this can be based on difference in tape speed, add if slower, subtract if faster
            estimated_hsync_with_offset = estimated_hsync_loc + meanlinelen * prev_hsync_diff
        else:
            estimated_hsync_with_offset = float(estimated_hsync_loc)

        # when estimated hsync is negative, just use the closest valid pulse or 0
        # we are not synced here, but continue to return a sync location to
        # avoid dropping video and getting out of sync with audio
        if estimated_hsync_with_offset <= 0:
            estimated_hsync_with_offset = float(validpulses_start[0]) if len(validpulses) > 0 else 0.0

        first_hsync_loc += estimated_hsync_with_offset
        valid_location_count += 1
        used_estimated_hsync = True

    # ******************************************************************************
    # Take the mean of the known vblanking locations to derive the first hsync pulse
    # ******************************************************************************
    vblank_pulses = np.asarray(vblank_pulses, dtype=np.int32)

    if valid_location_count > 0:
        offset /= valid_location_count
        first_hsync_loc = float(_c_round((first_hsync_loc + offset) / valid_location_count))

        # don't change the previous distance if this is an estimated sync location
        # since we don't know what the actual current hsync is
        if not used_estimated_hsync:
            prev_hsync_diff = (first_hsync_loc - estimated_hsync_loc) / meanlinelen

        # ****************************************************************
        # Align estimated start with hsync pulses to prevent skipped lines
        # ****************************************************************
        hsync_offset = 0.0
        hsync_count = 0
        for i in range(len(validpulses)):
            if validpulses_type[i] != 0 or not validpulses_valid[i]:
                continue

            lineloc = (validpulses_start[i] - first_hsync_loc) / meanlinelen + hsync_start_line
            rlineloc = _c_round(lineloc)

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

        return (
            line0loc,
            first_hsync_loc,
            hsync_start_line,
            next_field,
            int(first_field),
            int(progressive_field),
            prev_hsync_diff,
            vblank_pulses,
        )

    # no sync pulses found
    return None, None, hsync_start_line, None, int(first_field), int(progressive_field), prev_hsync_diff, vblank_pulses


def valid_pulses_to_linelocs(
    validpulses_in,
    reference_pulse,
    reference_line,
    meanlinelen,
    hsync_tolerance,
    proclines,
    gap_detection_threshold,
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
    # The reference pulse and line are used as whole numbers, truncating any fraction.
    reference_pulse = int(reference_pulse)
    reference_line = int(reference_line)
    meanlinelen = float(meanlinelen)
    proclines = int(proclines)

    validpulses = np.sort(np.asarray([p[1].start for p in validpulses_in], dtype=np.double))
    line_locations = np.empty(proclines, dtype=np.double)
    line_location_errs = np.zeros(proclines, dtype=np.uint8)

    current_pulse_index = 0
    validpulses_len = len(validpulses)
    current_pulse_sample_location = -1.0

    # This loop performs a best-fit to align the scan lines, which are expected to increment always
    # by around mean_line_len distance in samples, and the pulse locations in samples that were detected earlier
    # * Each line starts out by incrementing by the mean_line_length (estimated location)
    # * The inner loop searches for the nearest pulse at the estimated location within +- max_distance_between_pulse_and_line
    # * The closest pulse to the estimated location within the max_distance_between_pulse_and_line is assigned to the line
    #   * Each pulse is only ever assigned to a line once
    # * If there isn't a pulse within max_distance_between_pulse_and_line, then the line keeps the estimated location

    max_allowed_distance_between_pulse_and_line = meanlinelen / 1.5
    for line_index in range(proclines):
        # Start by setting this line's sample location to the expected location relative to the reference line
        line_locations[line_index] = reference_pulse + meanlinelen * (line_index - reference_line)

        # search for the closest pulse, pulse locations are assumed to be sorted
        if current_pulse_index < validpulses_len:
            # start the search using the distance between the current pulse and the expected line location
            current_distance_from_pulse_to_line = abs(validpulses[current_pulse_index] - line_locations[line_index])

            # start by setting this to the max allowed value so the loop will break if the next pulse is further away
            smallest_distance_observed_from_pulse_to_line = max_allowed_distance_between_pulse_and_line

            # reset the best fit variables
            current_pulse_sample_location = -1.0

            # start iteration at the pulse that hasn't been assigned yet
            pulse_search_index = current_pulse_index

            while pulse_search_index < validpulses_len - 1:
                if current_distance_from_pulse_to_line <= smallest_distance_observed_from_pulse_to_line:
                    smallest_distance_observed_from_pulse_to_line = current_distance_from_pulse_to_line

                    current_pulse_index = pulse_search_index
                    current_pulse_sample_location = validpulses[pulse_search_index]

                # peek ahead to the next pulse to measure the distance
                next_observed_distance_between_pulse_and_line = abs(
                    validpulses[pulse_search_index + 1] - line_locations[line_index]
                )
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
                line_locations[line_index] = current_pulse_sample_location
                current_pulse_index += 1

    return line_locations, line_location_errs, current_pulse_sample_location


def refine_linelocs_hsync(field, linebad, hsync_threshold):
    """Refine the line start locations using horizontal sync data. Marks unusable lines in linebad."""
    linelocs_original = np.asarray(field.linelocs1, dtype=np.float64)
    linelocs_refined = linelocs_original.copy()

    demod_05 = np.asarray(field.data["video"]["demod_05"], dtype=np.float64)
    rf = field.rf
    # The sample rate and hsync length are used as whole sample counts, truncating any fraction,
    # except for the magic right-edge offset which was tuned at single precision.
    normal_hsync_length = int(field.usectoinpx(rf.SysParams["hsyncPulseUS"]))
    one_usec = int(rf.freq)
    sample_rate_mhz = float(np.float32(rf.freq))
    is_pal = rf.system == "PAL"
    disable_right_hsync = rf.options.disable_right_hsync
    zc_threshold = float(hsync_threshold)
    ire_30 = rf.iretohz(30)
    ire_n_65 = rf.iretohz(-65)
    ire_110 = rf.iretohz(110)

    prev_porch_level = -1.0

    for i in range(len(linelocs_original)):
        # skip VSYNC lines, since they handle the pulses differently
        if 3 <= i <= 6 or (is_pal and 1 <= i <= 2):
            linebad[i] = True
            continue

        # refine beginning of hsync

        # start looking 1 usec back
        ll1 = _c_round(linelocs_original[i]) - one_usec
        # and locate the next time the half point between hsync and 0 is crossed.
        zc = _calczc(demod_05, ll1, zc_threshold, count=one_usec * 2)

        right_cross = math.nan
        if not disable_right_hsync:
            right_cross = _calczc(
                demod_05,
                ll1 + normal_hsync_length - one_usec,
                zc_threshold,
                count=normal_hsync_length * 2,
                edge=1,
            )
        right_cross_refined = False

        # If the crossing exists, we can check if the hsync pulse looks normal and
        # refine it.
        if not math.isnan(zc) and not linebad[i]:
            linelocs_refined[i] = zc

            # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
            # TODO: Use correct values for NTSC/PAL here
            hsync_area = demod_05[_c_round(zc - (one_usec * 0.75)) : _c_round(zc + (one_usec * 3.5))]
            if _is_out_of_range(hsync_area, ire_n_65, ire_110):
                # don't use the computed value here if it's bad
                linebad[i] = True
                linelocs_refined[i] = linelocs_original[i]
            else:
                if prev_porch_level > 0:
                    porch_level = prev_porch_level
                else:
                    porch_level = _mean(
                        demod_05[_c_round(zc - (one_usec * 1.0)) : _c_round(zc - (one_usec * 0.5))]
                    )
                sync_level = _mean(demod_05[_c_round(zc + (one_usec * 1)) : _c_round(zc + (one_usec * 2.5))])

                # Re-calculate the crossing point using the mid point between the measured sync
                # and porch levels
                zc2 = _calczc(demod_05, ll1, (porch_level + sync_level) / 2.0, count=400)

                # any wild variation here indicates a failure
                if not math.isnan(zc2) and abs(zc2 - zc) < (one_usec / 2.0):
                    linelocs_refined[i] = zc2
                    prev_porch_level = porch_level
                else:
                    # Give up
                    linebad[i] = True
        else:
            linebad[i] = True

        # Check right cross
        if not math.isnan(right_cross):
            zc_fr = right_cross - normal_hsync_length

            # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
            # NOTE: This is more than hsync area, might wanna also check max levels of level in hsync
            hsync_area = demod_05[_c_round(zc_fr - (one_usec * 0.75)) : _c_round(zc_fr + (one_usec * 8))]

            if not _is_out_of_range(hsync_area, ire_n_65, ire_30):
                porch_level = _mean(
                    demod_05[
                        _c_round(zc_fr + normal_hsync_length + (one_usec * 1)) : _c_round(
                            zc_fr + normal_hsync_length + (one_usec * 2)
                        )
                    ]
                )

                sync_level = _mean(
                    demod_05[_c_round(zc_fr + (one_usec * 1)) : _c_round(zc_fr + (one_usec * 2.5))]
                )

                # Re-calculate the crossing point using the mid point between the measured sync
                # and porch levels
                zc2 = _calczc(
                    demod_05,
                    ll1 + normal_hsync_length - one_usec,
                    (porch_level + sync_level) / 2.0,
                    count=400,
                )

                # any wild variation here indicates a failure
                if not math.isnan(zc2) and abs(zc2 - right_cross) < (one_usec / 2.0):
                    # TODO: Magic value here, this seem to give be approximately correct results
                    # but may not be ideal for all inputs.
                    # Value based on default sample rate so scale if it's different.
                    refined_from_right_lineloc = right_cross - normal_hsync_length + (2.25 * (sample_rate_mhz / 40.0))
                    # Don't use if it deviates too much which could indicate a false positive or non-standard hsync length.
                    if abs(refined_from_right_lineloc - linelocs_refined[i]) < (one_usec * 2):
                        right_cross_refined = True
                        prev_porch_level = porch_level

        if linebad[i]:
            linelocs_refined[i] = linelocs_original[i]  # don't use the computed value here if it's bad

        if right_cross_refined:
            # If we get a good result from calculating hsync start from the
            # right side of the hsync pulse, we use that as it's less likely
            # to be messed up by overshoot.
            linebad[i] = False
            linelocs_refined[i] = refined_from_right_lineloc

    return linelocs_refined
