import math
from collections import namedtuple
import numpy as np
import lddecode.utils as lddu
import lddecode.core as ldd
from vhsdecode.utils import get_line
import vhsdecode.utils as utils
import scipy.signal as sps


from numba import njit


@njit(cache=True)
def needs_recheck(sum_0: float, sum_1: float):
    """Check if the magnitude difference between the sums is larger than the set threshold
    If it's small, we want to make sure we re-check on the next field.
    """
    # Trigger a re-check on next field if the magnitude difference is smaller than this value.
    RECHECK_THRESHOLD = 1.2
    return True if max(sum_0, sum_1) / min(sum_0, sum_1) < RECHECK_THRESHOLD else False


@njit(cache=True)
def chroma_to_u16(chroma):
    """Scale the chroma output array to a 16-bit value for output."""
    S16_ABS_MAX = 32767

    # Disabled for now as it's misleading.
    # if np.max(chroma) > S16_ABS_MAX:
    #     ldd.logger.warning("Chroma signal clipping.")
    return (chroma + S16_ABS_MAX).astype(np.uint16)


@njit(cache=True, nogil=True)
def acc(chroma, burst_abs_ref, burststart, burstend, linelength, lines):
    """Scale chroma according to the level of the color burst on each line."""

    output = np.zeros(chroma.size, dtype=np.double)
    for linenumber in range(16, lines):
        linestart = linelength * linenumber
        lineend = linestart + linelength
        line = chroma[linestart:lineend]
        output[linestart:lineend] = acc_line(line, burst_abs_ref, burststart, burstend)

    return output


@njit(cache=True)
def acc_line(chroma, burst_abs_ref, burststart, burstend):
    """Scale chroma according to the level of the color burst the line."""
    output = np.zeros(chroma.size, dtype=np.double)

    line = chroma
    burst_abs_mean = lddu.rms(line[burststart:burstend])
    # np.sqrt(np.mean(np.square(line[burststart:burstend])))
    #    burst_abs_mean = np.mean(np.abs(line[burststart:burstend]))
    scale = burst_abs_ref / burst_abs_mean if burst_abs_mean != 0 else 1
    output = line * scale

    return output


@njit(cache=True, nogil=True)
def comb_c_pal(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 2H,
    and one advanced by 2H
    line by line. VCRs do this to reduce crosstalk.
    Helps chroma stability on LP tapes in particular.
    """

    # TODO: Compensate for PAL quarter cycle offset
    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        adv2h = data2[(line_num + 2) * line_len : (line_num + 3) * line_len]
        delayed2h = data2[(line_num - 2) * line_len : (line_num - 1) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/4 and advanced 1/4.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - (delayed2h) - adv2h
        ) / 4
    return data


@njit(cache=True)
def comb_c_ntsc(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 1H,
    line by line. VCRs do this to reduce crosstalk.
    """

    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        delayed1h = data2[(line_num - 1) * line_len : (line_num) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/3.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - (delayed1h)
        ) / 3
    return data


@njit(cache=True, nogil=True)
def upconvert_chroma(
    chroma,
    lineoffset,
    linesout,
    outwidth,
    chroma_heterodyne,
    phase_rotation,
    starting_phase,
):
    uphet = np.zeros(len(chroma), dtype=np.double)
    if phase_rotation == 0:
        # Track 1 - for PAL, phase doesn't change.
        start = lineoffset
        end = lineoffset + (outwidth * linesout)
        heterodyne = chroma_heterodyne[0][start:end]
        c = chroma[start:end]
        # Mixing the chroma signal with a signal at the frequency of colour under + fsc gives us
        # a signal with frequencies at the difference and sum, the difference is what we want as
        # it's at the right frequency.
        uphet[start:end] = heterodyne * c

    else:
        #        rotation = [(0,0),(90,-270),(180,-180),(270,-90)]
        # Track 2 - needs phase rotation or the chroma will be inverted.
        phase = starting_phase
        for linenumber in range(lineoffset, linesout + lineoffset):
            linestart = (linenumber - lineoffset) * outwidth
            lineend = linestart + outwidth

            heterodyne = chroma_heterodyne[phase][linestart:lineend]

            c = chroma[linestart:lineend]

            line = heterodyne * c

            uphet[linestart:lineend] = line

            phase = (phase + phase_rotation) % 4
    return uphet


@njit(cache=True)
def burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea):
    for line in range(lineoffset, linesout + lineoffset):
        linestart = (line - lineoffset) * outwidth
        lineend = linestart + outwidth

        chroma[linestart + burstarea[1] + 5 : lineend] *= 2

    return chroma


def demod_chroma_filt(data, filter, blocklen, notch, do_notch=None, move=10):
    out_chroma = utils.filter_simple(data[:blocklen], filter)

    if do_notch is not None and do_notch:
        out_chroma = sps.filtfilt(
            notch[0],
            notch[1],
            out_chroma,
        )

    # Move chroma to compensate for Y filter delay.
    # value needs tweaking, ideally it should be calculated if possible.
    # TODO: Not sure if we need this after hilbert filter change, needs check.
    out_chroma = np.roll(out_chroma, move)
    # crude DC offset removal
    out_chroma -= np.mean(out_chroma)
    return out_chroma


def process_chroma(
    field,
    track_phase,
    disable_deemph=False,
    disable_comb=False,
    disable_tracking_cafc=False,
):
    # Run TBC/downscale on chroma (if new field, else uses cache)
    if (
        field.rf.field_number != field.rf.chroma_last_field
        or field.rf.chroma_last_field == -1
    ):
        chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")
        field.rf.chroma_last_field = field.rf.field_number

        # If chroma AFC is enabled
        if field.rf.do_cafc:
            # it does the chroma filtering AFTER the TBC
            chroma = demod_chroma_filt(
                chroma,
                field.rf.chroma_afc.get_chroma_bandpass(),
                len(chroma),
                field.rf.Filters["FVideoNotch"],
                field.rf.notch,
                move=0,
            )

            if not disable_tracking_cafc:
                spec, meas, offset, cphase = field.rf.chroma_afc.freqOffset(chroma)
                ldd.logger.debug(
                    "Chroma under AFC: %.02f kHz, Offset (long term): %.02f Hz, Phase: %.02f deg"
                    % (meas / 1e3, offset, cphase * 360 / (2 * np.pi))
                )

        field.rf.chroma_tbc_buffer = chroma
    else:
        chroma = field.rf.chroma_tbc_buffer

    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    burst_area_init = get_burst_area(field)
    burstarea = burst_area_init[0] - 5, burst_area_init[1] + 10

    # narrow_filtered = utils.filter_simple(chroma, field.rf.Filters["FBurstNarrow"])

    # for line_num in range(16, linesout - 2):
    # lstart = line_num * outwidth
    # lend = (line_num + 1) * outwidth
    #
    # chroma[lstart:lend][burstarea[0]:burstarea[1]] = narrow_filtered[lstart:lend][burstarea[0]:burstarea[1]] * 2

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.color_system == "NTSC" and not disable_deemph:
        chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    # Track 2 is rotated ccw in both NTSC and PAL for VHS
    # u-matic has no phase rotation.
    phase_rotation = -1 if track_phase is not None else 0
    # What phase we start on. (Needed for NTSC to get the color phase correct)
    starting_phase = 0

    if track_phase is not None and field.rf.field_number % 2 == track_phase:
        if field.rf.color_system == "PAL" or field.rf.color_system == "MPAL":
            # For PAL, track 1 has no rotation.
            phase_rotation = 0
        elif field.rf.color_system == "NTSC":
            # For NTSC, track 1 rotates cw
            phase_rotation = 1
            starting_phase = 1
        else:
            raise Exception("Unknown video system!", field.rf.color_system)

    uphet = upconvert_chroma(
        chroma,
        lineoffset,
        linesout,
        outwidth,
        field.rf.chroma_afc.getChromaHet()
        if (field.rf.do_cafc and not disable_tracking_cafc)
        else field.rf.chroma_heterodyne,
        phase_rotation,
        starting_phase,
    )

    # Filter out unwanted frequencies from the final chroma signal.
    # Mixing the signals will produce waves at the difference and sum of the
    # frequencies. We only want the difference wave which is at the correct color
    # carrier frequency here.
    # We do however want to be careful to avoid filtering out too much of the sideband.
    uphet = utils.filter_simple(uphet, field.rf.Filters["FChromaFinal"])

    # Basic comb filter for NTSC to calm the color a little.
    if not disable_comb:
        if field.rf.color_system == "NTSC":
            uphet = comb_c_ntsc(uphet, outwidth)
        else:
            uphet = comb_c_pal(uphet, outwidth)

    # Final automatic chroma gain.
    uphet = acc(
        uphet,
        field.rf.SysParams["burst_abs_ref"],
        burstarea[0],
        burstarea[1],
        outwidth,
        linesout,
    )

    return uphet


def check_increment_field_no(rf):
    """Increment field number if the raw data location moved significantly since the last call"""
    raw_loc = rf.decoder.readloc / rf.decoder.bytes_per_field

    if rf.last_raw_loc is None:
        rf.last_raw_loc = raw_loc

    if raw_loc > rf.last_raw_loc:
        rf.field_number += 1
    else:
        ldd.logger.debug("Raw data loc didn't advance.")

    return raw_loc


def decode_chroma_vhs(field, rotation=True):
    """Do track detection if needed and upconvert the chroma signal"""
    rf = field.rf

    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = check_increment_field_no(rf)

    if rotation:
        # If we moved significantly more than the length of one field, re-check phase
        # as we may have skipped fields.
        if raw_loc - rf.last_raw_loc > 1.3:
            if rf.detect_track:
                ldd.logger.info("Possibly skipped a track, re-checking phase..")
                rf.needs_detect = True

        if rf.detect_track and rf.needs_detect or rf.recheck_phase:
            rf.track_phase, rf.needs_detect = field.try_detect_track()

    uphet = process_chroma(
        field,
        rf.track_phase if rotation else None,
        disable_comb=rf.options.disable_comb,
        disable_tracking_cafc=False,
    )
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def decode_chroma_umatic(field):
    """Do track detection if needed and upconvert the chroma signal"""
    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = check_increment_field_no(field.rf)

    uphet = process_chroma(
        field, None, True, field.rf.options.disable_comb, disable_tracking_cafc=False
    )
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    field.rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def decode_chroma_betamax(field):
    """Do track detection if needed and upconvert the chroma signal"""
    rf = field.rf

    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = check_increment_field_no(rf)

    # If we moved significantly more than the length of one field, re-check phase
    # as we may have skipped fields.
    if raw_loc - rf.last_raw_loc > 1.3:
        if rf.detect_track:
            ldd.logger.info("Possibly skipped a track, re-checking phase..")
            rf.needs_detect = True

    if rf.detect_track and rf.needs_detect or rf.recheck_phase:
        rf.track_phase, rf.needs_detect = field.try_detect_track()

    uphet = process_chroma(
        field,
        None,
        disable_comb=rf.options.disable_comb,
        disable_tracking_cafc=False,
    )
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def decode_chroma_video8(field):
    """Do track detection if needed and upconvert the chroma signal"""
    rf = field.rf

    ldd.logger.info(
        "Track detection and phase inversion not implemented for video8 yet!"
    )

    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = check_increment_field_no(rf)

    # If we moved significantly more than the length of one field, re-check phase
    # as we may have skipped fields.
    if raw_loc - rf.last_raw_loc > 1.3:
        if rf.detect_track:
            ldd.logger.info("Possibly skipped a track, re-checking phase..")
            rf.needs_detect = True

    if rf.detect_track and rf.needs_detect or rf.recheck_phase:
        rf.track_phase, rf.needs_detect = field.try_detect_track()

    uphet = process_chroma(
        field,
        None,
        disable_comb=rf.options.disable_comb,
        disable_tracking_cafc=False,
    )
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def get_burst_area(field):
    return (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])),
    )


def mean_of_burst_sums(chroma_data, line_length, lines, burst_start, burst_end):
    """Sum the burst areas of two and two lines together, and return the mean of these sums."""
    IGNORED_LINES = 16

    burst_sums = []

    # We ignore the top and bottom 16 lines. The top will typically not have a color burst, and
    # the bottom 16 may be after or at the head switch where the phase rotation will be different.
    start_line = IGNORED_LINES
    end_line = lines - IGNORED_LINES

    for line_number in range(start_line, end_line, 2):
        burst_a = get_line(chroma_data, line_length, line_number)[burst_start:burst_end]
        burst_b = get_line(chroma_data, line_length, line_number + 1)[
            burst_start:burst_end
        ]

        # Use the absolute of the sums to differences cancelling out.
        mean_dev = np.mean(abs(burst_a + burst_b))

        burst_sums.append(mean_dev)

    mean_burst_sum = np.nanmean(burst_sums)
    return mean_burst_sum


@njit(cache=True, nogil=True)
def detect_burst_pal(
    chroma_data, sine_wave, cosine_wave, burst_area, line_length, lines
):
    """Decode the burst of most lines to see if we have a valid PAL color burst."""

    # Ignore the first and last 16 lines of the field.
    # first ones contain sync and often doesn't have color burst,
    # while the last lines of the field will contain the head switch and may be distorted.
    IGNORED_LINES = 16
    line_data = []
    burst_norm = np.full(lines, np.nan)

    # Decode the burst vectors on each line and try to get an average of the burst amplitude.
    for linenumber in range(IGNORED_LINES, lines - IGNORED_LINES):
        info = detect_burst_pal_line(
            chroma_data, sine_wave, cosine_wave, burst_area, line_length, linenumber
        )
        line_data.append(info)
        burst_norm[linenumber] = info.burst_norm

    burst_mean = np.nanmean(burst_norm[IGNORED_LINES : lines - IGNORED_LINES])

    return line_data, burst_mean


LineInfo = namedtuple("LineInfo", "linenum, bp, bq, vsw, burst_norm")


@njit(cache=True, nogil=True)
def detect_burst_pal_line(
    chroma_data, sine, cosine, burst_area, line_length, line_number
):
    """Detect burst function ported from the C++ chroma decoder (palcolour.cpp)

    Tries to decode the PAL chroma vectors from the line's color burst
    """
    empty_line = np.zeros_like(chroma_data[0:line_length])
    num_lines = chroma_data.size / line_length

    # Use an empty line if we try to access outside the field.
    def line_or_empty(line):
        return (
            get_line(chroma_data, line_length, line)
            if line >= 0 and line < num_lines
            else empty_line
        )

    in0 = line_or_empty(line_number)
    in1 = line_or_empty(line_number - 1)
    in2 = line_or_empty(line_number + 1)
    in3 = line_or_empty(line_number - 2)
    in4 = line_or_empty(line_number + 2)
    bp = 0
    bq = 0
    bpo = 0
    bqo = 0

    # (Comment from palcolor.cpp)
    # Find absolute burst phase relative to the reference carrier by
    # product detection.
    #
    # To avoid hue-shifts on alternate lines, the phase is determined by
    # averaging the phase on the current-line with the average of two
    # other lines, one above and one below the current line.
    #
    # For PAL we use the next-but-one line above and below (in the field),
    # which will have the same V-switch phase as the current-line (and 180
    # degree change of phase), and we also analyse the average (bpo/bqo
    # 'old') of the line immediately above and below, which have the
    # opposite V-switch phase (and a 90 degree subcarrier phase shift).
    for i in range(burst_area[0], burst_area[1]):
        bp += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * sine[i]
        bq += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * cosine[i]
        bpo += ((in2[i] - in1[i]) / 2.0) * sine[i]
        bqo += ((in2[i] - in1[i]) / 2.0) * cosine[i]

    # (Comment from palcolor.cpp)
    # Normalise the sums above
    burst_length = burst_area[1] - burst_area[0]

    bp /= burst_length
    bq /= burst_length
    bpo /= burst_length
    bqo /= burst_length

    # (Comment from palcolor.cpp)
    # Detect the V-switch state on this line.
    # I forget exactly why this works, but it's essentially comparing the
    # vector magnitude /difference/ between the phases of the burst on the
    # present line and previous line to the magnitude of the burst. This
    # may effectively be a dot-product operation...

    line_bp = 0
    line_bq = 0
    line_vsw = -1
    line_burst_norm = 0

    if ((bp - bpo) * (bp - bpo) + (bq - bqo) * (bq - bqo)) < (bp * bp + bq * bq) * 2:
        line_vsw = 1

    # (Comment from palcolor.cpp)
    # Average the burst phase to get -U (reference) phase out -- burst
    # phase is (-U +/-V). bp and bq will be of the order of 1000.
    line_bp = (bp - bqo) / 2
    line_bq = (bq + bpo) / 2

    # (Comment from palcolor.cpp)
    # Normalise the magnitude of the bp/bq vector to 1.
    # Kill colour if burst too weak.
    # XXX magic number 130000 !!! check!
    burst_norm = max(math.sqrt(line_bp * line_bp + line_bq * line_bq), 10000.0 / 128)
    line_burst_norm = burst_norm
    line_bp /= burst_norm
    line_bq /= burst_norm

    return LineInfo(line_number, line_bp, line_bq, line_vsw, line_burst_norm)


@njit(cache=True, nogil=True)
def detect_burst_ntsc(
    chroma_data, sine_wave, cosine_wave, burst_area, line_length, lines
):
    """Check the phase of the color burst."""

    # Ignore the first and last 16 lines of the field.
    # first ones contain sync and often doesn't have color burst,
    # while the last lines of the field will contain the head switch and may be distorted.
    IGNORED_LINES = 16
    odd_i_acc = 0
    even_i_acc = 0

    for linenumber in range(IGNORED_LINES, lines - IGNORED_LINES):
        bi, _, _ = detect_burst_ntsc_line(
            chroma_data, sine_wave, cosine_wave, burst_area, line_length, linenumber
        )
        #        line_data.append((bi, bq, linenumber))
        if linenumber % 2 == 0:
            even_i_acc += bi
        else:
            odd_i_acc += bi

    num_lines = lines - (IGNORED_LINES * 2)

    return even_i_acc / num_lines, odd_i_acc / num_lines


@njit(cache=True)
def detect_burst_ntsc_line(
    chroma_data, sine, cosine, burst_area, line_length, line_number
):
    bi = 0
    bq = 0
    # TODO:
    sine = sine[burst_area[0] :]
    cosine = cosine[burst_area[0] :]
    line = get_line(chroma_data, line_length, line_number)
    for i in range(burst_area[0], burst_area[1]):
        bi += line[i] * sine[i]
        bq += line[i] * cosine[i]

    burst_length = burst_area[1] - burst_area[0]

    bi /= burst_length
    bq /= burst_length

    burst_norm = max(math.sqrt(bi * bi + bq * bq), 130000.0 / 128)
    bi /= burst_norm
    bq /= burst_norm
    return bi, bq, burst_norm


def get_field_phase_id(field):
    """Try to determine which of the 4 NTSC phase cycles the field is.
    For tapes the result seem to not be cyclical at all, not sure if that's normal
    or if something is off.
    The most relevant thing is which lines the burst phase is positive or negative on.
    TODO: Current code does not give the correct result!!!!
    """
    burst_area = get_burst_area(field)

    sine_wave = field.rf.fsc_wave
    cosine_wave = field.rf.fsc_cos_wave

    # Try to detect the average burst phase of odd and even lines.
    even, odd = detect_burst_ntsc(
        field.uphet_temp,
        sine_wave,
        cosine_wave,
        burst_area,
        field.outlinelen,
        field.outlinecount,
    )

    # This map is based on (first field, field14)
    map4 = {
        (True, True): 1,
        (False, False): 2,
        (True, False): 3,
        (False, True): 4,
    }

    phase_id = (
        map4[(field.isFirstField, even < odd)] if field.isFirstField is not None else 0
    )

    # ldd.logger.info("Field: %i, Odd I %f , Even I %f, phase id %i, field first %i",
    #                field.rf.field_number, even, odd, phase_id, field.isFirstField)

    return phase_id


# Phase comprensation stuff - needs rework.
# def phase_shift(data, angle):
#     return np.fft.irfft(np.fft.rfft(data) * np.exp(1.0j * angle), len(data)).real


def get_burstarea(field):
    return (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])),
    )


def log_track_phase(track_phase, phase0_mean, phase1_mean, assumed_phase):
    ldd.logger.info("Phase previously set: %i", track_phase)
    ldd.logger.info("phase0 mean: %.02f", phase0_mean)
    ldd.logger.info("phase1 mean: %.02f", phase1_mean)
    ldd.logger.info("assumed_phase: %d", assumed_phase)


def try_detect_track_vhs_pal(field):
    """Try to detect what video track we are on.

    VHS tapes have two tracks with different azimuth that alternate and are read by alternating
    heads on the video drum. The phase of the color heterodyne varies depending on what track is
    being read from to avoid chroma crosstalk.
    Additionally, most tapes are recorded with a luma half-shift which shifts the fm-encoded
    luma frequencies slightly depending on the track to avoid luma crosstalk.
    """
    ldd.logger.debug("Trying to detect PAL track phase...")
    burst_area = get_burstarea(field)

    # Upconvert chroma twice, once for each possible track phase
    uphet = [process_chroma(field, 0, True, True), process_chroma(field, 1, True, True)]

    sine_wave = field.rf.fsc_wave
    cosine_wave = field.rf.fsc_cos_wave

    # Try to decode the color burst from each of the upconverted chroma signals
    phase = list()
    for ix, uph in enumerate(uphet):
        phase.append(
            detect_burst_pal(
                uph,
                sine_wave,
                cosine_wave,
                burst_area,
                field.outlinelen,
                field.outlinecount,
            )
        )

    # We use the one where the phase of the chroma vectors make the most sense.
    phase0_mean, phase1_mean = phase[0][1], phase[1][1]
    assumed_phase = int(phase0_mean < phase1_mean)

    log_track_phase(field.rf.track_phase, phase0_mean, phase1_mean, assumed_phase)

    return assumed_phase, needs_recheck(phase0_mean, phase1_mean)


def try_detect_track_vhs_ntsc(field):
    """Try to detect which track the current field was read from.
    returns 0 or 1 depending on detected track phase.

    We use the fact that the color burst in NTSC is inverted on every line, so
    in a perfect signal, the burst from one line and the previous one should cancel
    each other out when summed together. When upconverting with the wrong phase rotation,
    the bursts will have the same phase instead, and thus the mean absolute
    sum will be much higher. This seem to give a reasonably good guess, but could probably
    be improved.
    """
    ldd.logger.debug("Trying to detect NTSC track phase ...")
    burst_area = get_burstarea(field)

    # Upconvert chroma twice, once for each possible track phase
    uphet = [process_chroma(field, 0, True), process_chroma(field, 1, True)]

    # Look at the bursts from each upconversion and see which one looks most
    # normal.
    burst_mean_sum = list()
    for ix, uph in enumerate(uphet):
        burst_mean_sum.append(
            mean_of_burst_sums(
                uph, field.outlinelen, field.outlinecount, burst_area[0], burst_area[1]
            )
        )

    burst_mean_sum_0, burst_mean_sum_1 = burst_mean_sum[0], burst_mean_sum[1]

    assumed_phase = int(burst_mean_sum_1 < burst_mean_sum_0)

    log_track_phase(
        field.rf.track_phase, burst_mean_sum_0, burst_mean_sum_1, assumed_phase
    )

    return assumed_phase, needs_recheck(burst_mean_sum_0, burst_mean_sum_1)
