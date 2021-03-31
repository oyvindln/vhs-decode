import math
import numpy as np
import scipy.signal as sps
import copy

import itertools

import lddecode.core as ldd
import lddecode.utils as lddu
from lddecode.utils import unwrap_hilbert, inrange
import vhsdecode.utils as utils
from vhsdecode.utils import get_line

import vhsdecode.formats as vhs_formats
from vhsdecode.addons.FMdeemph import FMDeEmphasisB
from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.addons.resync import DCrestore

from numba import njit

# Use PyFFTW's faster FFT implementation if available
try:
    import pyfftw.interfaces.numpy_fft as npfft
    import pyfftw.interfaces

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ImportError:
    import numpy.fft as npfft


def chroma_to_u16(chroma):
    """Scale the chroma output array to a 16-bit value for output."""
    S16_ABS_MAX = 32767

    if np.max(chroma) > S16_ABS_MAX or abs(np.min(chroma)) > S16_ABS_MAX:
        ldd.logger.warning("Chroma signal clipping.")
    return np.uint16(chroma + S16_ABS_MAX)


@njit
def acc(chroma, burst_abs_ref, burststart, burstend, linelength, lines):
    """Scale chroma according to the level of the color burst on each line."""

    output = np.zeros(chroma.size, dtype=np.double)
    for linenumber in range(16, lines):
        linestart = linelength * linenumber
        lineend = linestart + linelength
        line = chroma[linestart:lineend]
        output[linestart:lineend] = acc_line(line, burst_abs_ref, burststart, burstend)

    return output


@njit
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


def getpulses_override(field):
    """Find sync pulses in the demodulated video sigal

    NOTE: TEMPORARY override until an override for the value itself is added upstream.
    """
    # pass one using standard levels

    # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
    pulse_hz_min = field.rf.iretohz(field.rf.SysParams["vsync_ire"] - 10)
    pulse_hz_max = field.rf.iretohz(field.rf.SysParams["vsync_ire"] / 2)

    pulses = lddu.findpulses(
        field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
    )

    if len(pulses) == 0:
        # can't do anything about this
        return pulses

    # determine sync pulses from vsync
    vsync_locs = []
    vsync_means = []

    for i, p in enumerate(pulses):
        if p.len > field.usectoinpx(10):
            vsync_locs.append(i)
            vsync_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + field.rf.freq) : int(
                            p.start + p.len - field.rf.freq
                        )
                    ]
                )
            )

    if len(vsync_means) == 0:
        return None

    synclevel = np.median(vsync_means)

    if np.abs(field.rf.hztoire(synclevel) - field.rf.SysParams["vsync_ire"]) < 5:
        # sync level is close enough to use
        return pulses

    if vsync_locs is None or not len(vsync_locs):
        return None

    # Now compute black level and try again

    # take the eq pulses before and after vsync
    r1 = range(vsync_locs[0] - 5, vsync_locs[0])
    r2 = range(vsync_locs[-1] + 1, vsync_locs[-1] + 6)

    black_means = []

    for i in itertools.chain(r1, r2):
        if i < 0 or i >= len(pulses):
            continue

        p = pulses[i]
        if inrange(p.len, field.rf.freq * 0.75, field.rf.freq * 3):
            black_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + (field.rf.freq * 5)) : int(
                            p.start + (field.rf.freq * 20)
                        )
                    ]
                )
            )

    blacklevel = np.median(black_means)

    pulse_hz_min = synclevel - (field.rf.SysParams["hz_ire"] * 10)
    pulse_hz_max = (blacklevel + synclevel) / 2

    return lddu.findpulses(field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)


@njit
def comb_c_pal(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 2H,
    and one advanced by 2H
    line by line. VCRs do this to reduce crosstalk.
    """

    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        adv2h = data2[(line_num + 2) * line_len : (line_num + 3) * line_len]
        delayed2h = data2[(line_num - 2) * line_len : (line_num - 1) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/4 and advanced 1/4.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice) - (delayed2h) - adv2h
        ) / 3
    return data


@njit
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


@njit
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
        mixed = heterodyne * c

        uphet[start:end] = mixed

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


@njit
def burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea):
    for line in range(lineoffset, linesout + lineoffset):
        linestart = (line - lineoffset) * outwidth
        lineend = linestart + outwidth

        chroma[linestart + burstarea[1] + 5 : lineend] *= 2

    return chroma


def process_chroma(field, track_phase, disable_deemph=False):
    # Run TBC/downscale on chroma.
    chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")

    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    burstarea = (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0]) - 5),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])) + 10,
    )

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.system == "NTSC" and not disable_deemph:
        chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    # Track 2 is rotated ccw in both NTSC and PAL for VHS
    # u-matic has no phase rotation.
    phase_rotation = -1 if track_phase is not None else 0
    # What phase we start on. (Needed for NTSC to get the color phase correct)
    starting_phase = 0

    if track_phase is not None and field.rf.field_number % 2 == track_phase:
        if field.rf.system == "PAL":
            # For PAL, track 1 has no rotation.
            phase_rotation = 0
        elif field.rf.system == "NTSC":
            # For NTSC, track 1 rotates cw
            phase_rotation = 1
            starting_phase = 1
        else:
            raise Exception("Unknown video system!", field.rf.system)

    uphet = upconvert_chroma(
        chroma,
        lineoffset,
        linesout,
        outwidth,
        field.rf.chroma_heterodyne,
        phase_rotation,
        starting_phase,
    )

    # uphet = comb_c_pal(uphet,outwidth)

    # Filter out unwanted frequencies from the final chroma signal.
    # Mixing the signals will produce waves at the difference and sum of the
    # frequencies. We only want the difference wave which is at the correct color
    # carrier frequency here.
    # We do however want to be careful to avoid filtering out too much of the sideband.
    uphet = utils.filter_simple(uphet, field.rf.Filters["FChromaFinal"])

    # Basic comb filter for NTSC to calm the color a little.
    if field.rf.system == "NTSC":
        uphet = comb_c_ntsc(uphet, outwidth)
    #    else:
    #        uphet = comb_c_pal(uphet, outwidth)

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


def decode_chroma_vhs(field):
    """Do track detection if needed and upconvert the chroma signal"""
    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = field.rf.decoder.readloc / field.rf.decoder.bytes_per_field

    check_increment_field_no(field.rf)

    # If we moved significantly more than the length of one field, re-check phase
    # as we may have skipped fields.
    if raw_loc - field.rf.last_raw_loc > 2.0:
        if field.rf.detect_track:
            ldd.logger.info("Possibly skipped track, re-checking phase..")
            field.rf.needs_detect

    if field.rf.detect_track and field.rf.needs_detect:
        field.rf.track_phase = field.try_detect_track()
        field.rf.needs_detect = False

    uphet = process_chroma(field, field.rf.track_phase)
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    field.rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def decode_chroma_umatic(field):
    """Do track detection if needed and upconvert the chroma signal"""
    # Use field number based on raw data position
    # This may not be 100% accurate, so we may want to add some more logic to
    # make sure we re-check the phase occasionally.
    raw_loc = field.rf.decoder.readloc / field.rf.decoder.bytes_per_field

    check_increment_field_no(field.rf)

    uphet = process_chroma(field, None, True)
    field.uphet_temp = uphet
    # Store previous raw location so we can detect if we moved in the next call.
    field.rf.last_raw_loc = raw_loc
    return chroma_to_u16(uphet)


def get_burst_area(field):
    return (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])),
    )


class LineInfo:
    """Helper class to store line burst info for PAL."""

    def __init__(self, num):
        self.linenum = num
        self.bp = 0
        self.bq = 0
        self.vsw = -1
        self.burst_norm = 0

    def __str__(self):
        return "<num: %s, bp: %s, bq: %s, vsw: %s, burst_norm: %s>" % (
            self.linenum,
            self.bp,
            self.bq,
            self.vsw,
            self.burst_norm,
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
    line = LineInfo(line_number)
    if ((bp - bpo) * (bp - bpo) + (bq - bqo) * (bq - bqo)) < (bp * bp + bq * bq) * 2:
        line.vsw = 1

    # (Comment from palcolor.cpp)
    # Average the burst phase to get -U (reference) phase out -- burst
    # phase is (-U +/-V). bp and bq will be of the order of 1000.
    line.bp = (bp - bqo) / 2
    line.bq = (bq + bpo) / 2

    # (Comment from palcolor.cpp)
    # Normalise the magnitude of the bp/bq vector to 1.
    # Kill colour if burst too weak.
    # XXX magic number 130000 !!! check!
    burst_norm = max(math.sqrt(line.bp * line.bp + line.bq * line.bq), 130000.0 / 128)
    line.burst_norm = burst_norm
    line.bp /= burst_norm
    line.bq /= burst_norm

    return line


@njit
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


@njit
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

    phase_id = map4[(field.isFirstField, even < odd)]

    # ldd.logger.info("Field: %i, Odd I %f , Even I %f, phase id %i, field first %i",
    #                field.rf.field_number, even, odd, phase_id, field.isFirstField)

    return phase_id


def find_crossings(data, threshold):
    """Find where the data crosses the set threshold."""

    # We do this by constructing array where positions above
    # the threshold are marked as true, other sfalse,
    # and use diff to mark where the value changes.
    crossings = np.diff(data < threshold)
    # TODO: See if we can avoid reduntantly looking for both up and
    # down crossing when we just need one of them.
    return crossings


def find_crossings_dir(data, threshold, look_for_down):
    """Find where the data crosses the set threshold
    the look_for_down parameters determines if the crossings returned are down
    or up crossings.
    ."""
    crossings = find_crossings(data, threshold)
    crossings_pos = np.argwhere(crossings)[:, 0]
    if len(crossings_pos) <= 0:
        return []
    first_cross = crossings_pos[0]
    if first_cross >= len(data):
        return []
    first_crossing_is_down = data[first_cross] > data[first_cross + 1]
    if first_crossing_is_down == look_for_down:
        return crossings_pos[::2]
    else:
        return crossings_pos[1::2]


def combine_to_dropouts(crossings_down, crossings_up, merge_threshold):
    """Combine arrays of up and down crossings, and merge ones with small gaps between them.
    Intended to be used where up and down crossing levels are different, the two lists will not
    always alternate or have the same length.
    Returns a list of start/end tuples.
    """
    used = []

    # TODO: Fix when ending on dropout

    cr_up = iter(crossings_up)
    last_u = 0
    # Loop through crossings and combine
    # TODO: Doing this via a loop is probably not ideal in python,
    # we may want to look for a way to more directly generate a list of down/up crossings
    # with hysteresis.
    for d in crossings_down:
        if d < last_u:
            continue

        # If the distance between two dropouts is very small, we merge them.
        if d - last_u < merge_threshold and len(used) > 0:
            # Pop the last added dropout and use it's starting point
            # as the start of the merged one.
            last = used.pop()
            d = last[0]

        for u in cr_up:
            if u > d:
                used.append((d, u))
                last_u = u
                break

    return used


def detect_dropouts_rf(field):
    """Look for dropouts in the input data, based on rf envelope amplitude.
    Uses either an percentage of the frame average rf level, or an absolute value.
    TODO: A more advanced algorithm with hysteresis etc.
    """
    env = field.data["video"]["envelope"]
    threshold_p = field.rf.dod_threshold_p
    threshold_abs = field.rf.dod_threshold_a
    hysteresis = field.rf.dod_hysteresis

    threshold = 0.0
    if threshold_abs is not None:
        threshold = threshold_abs
    else:
        # Generate a threshold based on the field envelope average.
        # This may not be ideal on a field with a lot of droputs,
        # so we may want to use statistics of the previous averages
        # to avoid the threshold ending too low.
        field_average = np.mean(field.data["video"]["envelope"])
        threshold = field_average * threshold_p

    errlist = []

    crossings_down = find_crossings_dir(env, threshold, True)
    crossings_up = find_crossings_dir(env, threshold * hysteresis, False)

    if (
        len(crossings_down) > 0
        and len(crossings_up) > 0
        and crossings_down[0] > crossings_up[0]
        and env[0] < threshold
    ):
        # Handle if we start on a dropout by adding a zero at the start since we won't have any
        # down crossing for it in the data.
        crossings_down = np.concatenate((np.array([0]), crossings_down), axis=None)

    errlist = combine_to_dropouts(
        crossings_down, crossings_up, vhs_formats.DOD_MERGE_THRESHOLD
    )

    # Drop very short dropouts that were not merged.
    # We do this after mergin to avoid removing short consecutive dropouts that
    # could be merged.
    errlist = list(filter(lambda s: s[1] - s[0] > vhs_formats.DOD_MIN_LENGTH, errlist))

    rv_lines = []
    rv_starts = []
    rv_ends = []

    # Convert to tbc positions.
    dropouts = dropout_errlist_to_tbc(field, errlist)
    for r in dropouts:
        rv_lines.append(r[0] - 1)
        rv_starts.append(int(r[1]))
        rv_ends.append(int(r[2]))

    return rv_lines, rv_starts, rv_ends


def dropout_errlist_to_tbc(field, errlist):
    """Convert data from raw data coordinates to tbc coordinates, and splits up
    multi-line dropouts.
    """
    dropouts = []

    if len(errlist) == 0:
        return dropouts

    # Now convert the above errlist into TBC locations
    errlistc = errlist.copy()

    lineoffset = -field.lineoffset

    # Remove dropouts occuring before the start of the frame so they don't
    # cause the rest to be skipped
    curerr = errlistc.pop(0)
    while len(errlistc) > 0 and curerr[0] < field.linelocs[field.lineoffset]:
        curerr = errlistc.pop(0)

    # TODO: This could be reworked to be a bit cleaner and more performant.

    for line in range(field.lineoffset, field.linecount + field.lineoffset):
        while curerr is not None and inrange(
            curerr[0], field.linelocs[line], field.linelocs[line + 1]
        ):
            start_rf_linepos = curerr[0] - field.linelocs[line]
            start_linepos = start_rf_linepos / (
                field.linelocs[line + 1] - field.linelocs[line]
            )
            start_linepos = int(start_linepos * field.outlinelen)

            end_rf_linepos = curerr[1] - field.linelocs[line]
            end_linepos = end_rf_linepos / (
                field.linelocs[line + 1] - field.linelocs[line]
            )
            end_linepos = int(np.round(end_linepos * field.outlinelen))

            first_line = line + 1 + lineoffset

            # If the dropout spans multiple lines, we need to split it up into one for each line.
            if end_linepos > field.outlinelen:
                num_lines = end_linepos // field.outlinelen

                # First line.
                dropouts.append((first_line, start_linepos, field.outlinelen))
                # Full lines in the middle.
                for n in range(num_lines - 1):
                    dropouts.append((first_line + n + 1, 0, field.outlinelen))
                # leftover on last line.
                dropouts.append(
                    (
                        first_line + (num_lines),
                        0,
                        np.remainder(end_linepos, field.outlinelen),
                    )
                )
            else:
                dropouts.append((first_line, start_linepos, end_linepos))

            if len(errlistc):
                curerr = errlistc.pop(0)
            else:
                curerr = None

    return dropouts


# Phase comprensation stuff - needs rework.
# def phase_shift(data, angle):
#     return np.fft.irfft(np.fft.rfft(data) * np.exp(1.0j * angle), len(data)).real


def check_increment_field_no(rf):
    """Increment field number if the raw data location moved significantly since the last call"""
    raw_loc = rf.decoder.readloc / rf.decoder.bytes_per_field

    if rf.last_raw_loc is None:
        rf.last_raw_loc = raw_loc

    if raw_loc > rf.last_raw_loc:
        rf.field_number += 1
    else:
        ldd.logger.info("Raw data loc didn't advance.")


class FieldPALVHS(ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALVHS, self).__init__(*args, **kwargs)

    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as standard vhs does not use have a pilot burst."""
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALVHS, self).downscale(
            final, *args, **kwargs
        )
        dschroma = decode_chroma_vhs(self)

        return (dsout, dschroma), dsaudio, dsefm

    def calc_burstmedian(self):
        # Set this to a constant value for now to avoid the comb filter messing with chroma levels.
        return 1.0

    def try_detect_track(self):
        """Try to detect what video track we are on.

        VHS tapes have two tracks with different azimuth that alternate and are read by alternating
        heads on the video drum. The phase of the color heterodyne varies depending on what track is
        being read from to avoid chroma crosstalk.
        Additionally, most tapes are recorded with a luma half-shift which shifts the fm-encoded
        luma frequencies slightly depending on the track to avoid luma crosstalk.
        """
        ldd.logger.info("Trying to detect track phase...")
        burst_area = (
            math.floor(self.usectooutpx(self.rf.SysParams["colorBurstUS"][0])),
            math.ceil(self.usectooutpx(self.rf.SysParams["colorBurstUS"][1])),
        )

        # Upconvert chroma twice, once for each possible track phase
        uphet = [process_chroma(self, 0), process_chroma(self, 1)]

        sine_wave = self.rf.fsc_wave
        cosine_wave = self.rf.fsc_cos_wave

        # Try to decode the color burst from each of the upconverted chroma signals
        phase0, phase0_mean = detect_burst_pal(
            uphet[0],
            sine_wave,
            cosine_wave,
            burst_area,
            self.outlinelen,
            self.outlinecount,
        )
        phase1, phase1_mean = detect_burst_pal(
            uphet[1],
            sine_wave,
            cosine_wave,
            burst_area,
            self.outlinelen,
            self.outlinecount,
        )

        # We use the one where the phase of the chroma vectors make the most sense.
        assumed_phase = int(phase0_mean < phase1_mean)

        ldd.logger.info("Phase previously set: %i", self.rf.track_phase)
        ldd.logger.info("phase0 mean: %d", phase0_mean)
        ldd.logger.info("phase1 mean: %d", phase1_mean)
        ldd.logger.info("assumed_phase: %d", assumed_phase)

        return assumed_phase

    def determine_field_number(self):
        """Workaround to shut down phase id mismatch warnings, the actual code
        doesn't work properly with the vhs output at the moment."""
        return 1 + (self.rf.field_number % 8)

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as tapes have large variations in line pos
        Due to e.g head switch.
        compute errors based off the second derivative - if it exceeds 1 something's wrong,
        and if 4 really wrong...
        """
        return baserr

    def dropout_detect(self):
        return detect_dropouts_rf(self)


class FieldPALUMatic(ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALUMatic, self).__init__(*args, **kwargs)

    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as regular-band u-matic does not have a pilot burst."""
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALUMatic, self).downscale(
            final, *args, **kwargs
        )
        dschroma = decode_chroma_umatic(self)

        return (dsout, dschroma), dsaudio, dsefm

    def calc_burstmedian(self):
        # Set this to a constant value for now to avoid the comb filter messing with chroma levels.
        return 1.0

    def determine_field_number(self):
        """Workaround to shut down phase id mismatch warnings, the actual code
        doesn't work properly with the vhs output at the moment."""
        return 1 + (self.rf.field_number % 8)

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as tapes have large variations in line pos
        Due to e.g head switch.
        compute errors based off the second derivative - if it exceeds 1 something's wrong,
        and if 4 really wrong...
        """
        return baserr

    def dropout_detect(self):
        return detect_dropouts_rf(self)


class FieldNTSCVHS(ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCVHS, self).__init__(*args, **kwargs)
        self.fieldPhaseID = 0

    def refine_linelocs_burst(self, linelocs=None):
        """Override this as it's LD specific
        At some point in the future we could maybe use the burst location to improve hsync accuracy,
        but ignore it for now.
        """
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        return linelocs

    def calc_burstmedian(self):
        # Set this to a constant value for now to avoid the comb filter messing with chroma levels.
        return 1.0

    def try_detect_track(self):
        """Try to detect which track the current field was read from.
        returns 0 or 1 depending on detected track phase.

        We use the fact that the color burst in NTSC is inverted on every line, so
        in a perfect signal, the burst from one line and the previous one should cancel
        each other out when summed together. When upconverting with the wrong phase rotation,
        the bursts will have the same phase instead, and thus the mean absolute
        sum will be much higher. This seem to give a reasonably good guess, but could probably
        be improved.
        """
        ldd.logger.info("Trying to detect track phase...")
        burst_area = (
            math.floor(self.usectooutpx(self.rf.SysParams["colorBurstUS"][0])),
            math.ceil(self.usectooutpx(self.rf.SysParams["colorBurstUS"][1])),
        )

        # Upconvert chroma twice, once for each possible track phase
        uphet = [process_chroma(self, 0, True), process_chroma(self, 1, True)]

        # Look at the bursts from each upconversion and see which one looks most
        # normal.
        burst_mean_sum_0 = mean_of_burst_sums(
            uphet[0], self.outlinelen, self.outlinecount, burst_area[0], burst_area[1]
        )

        burst_mean_sum_1 = mean_of_burst_sums(
            uphet[1], self.outlinelen, self.outlinecount, burst_area[0], burst_area[1]
        )

        assumed_phase = int(burst_mean_sum_1 < burst_mean_sum_0)

        ldd.logger.info("burst mean sum 0: %f", burst_mean_sum_0)
        ldd.logger.info("burst mean sum 1: %f", burst_mean_sum_1)
        ldd.logger.info("assumed phase: %i", assumed_phase)

        return assumed_phase

    def downscale(self, linesoffset=0, final=False, *args, **kwargs):
        """Downscale the channels and upconvert chroma to standard color carrier frequency."""
        dsout, dsaudio, dsefm = super(FieldNTSCVHS, self).downscale(
            linesoffset, final, *args, **kwargs
        )

        dschroma = decode_chroma_vhs(self)

        self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, dschroma), dsaudio, dsefm

    def dropout_detect(self):
        return detect_dropouts_rf(self)

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as line starts can vary widely."""
        return baserr


class FieldNTSCUMatic(ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCUMatic, self).__init__(*args, **kwargs)
        self.fieldPhaseID = 0

    def refine_linelocs_burst(self, linelocs=None):
        """Override this as it's LD specific
        At some point in the future we could maybe use the burst location to improve hsync accuracy,
        but ignore it for now.
        """
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        return linelocs

    def calc_burstmedian(self):
        # Set this to a constant value for now to avoid the comb filter messing with chroma levels.
        return 1.0

    def downscale(self, linesoffset=0, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldNTSCUMatic, self).downscale(
            linesoffset, final, *args, **kwargs
        )
        dschroma = decode_chroma_umatic(self)

        self.fieldPhaseID = self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, dschroma), dsaudio, dsefm

    def dropout_detect(self):
        return detect_dropouts_rf(self)

    def getpulses(self):
        """Find sync pulses in the demodulated video sigal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as line starts can vary widely."""
        return baserr


# Superclass to override laserdisc-specific parts of ld-decode with stuff that works for VHS
#
# We do this simply by using inheritance and overriding functions. This results in some redundant
# work that is later overridden, but avoids altering any ld-decode code to ease merging back in
# later as the ld-decode is in flux at the moment.
class VHSDecode(ldd.LDdecode):
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        logger,
        system="NTSC",
        tape_format="VHS",
        doDOD=True,
        threads=1,
        inputfreq=40,
        level_adjust=0.2,
        rf_options={},
        extra_options={},
    ):
        super(VHSDecode, self).__init__(
            fname_in,
            fname_out,
            freader,
            logger,
            analog_audio=False,
            system=system,
            doDOD=doDOD,
            threads=threads,
            extra_options=extra_options,
        )
        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust
        # Overwrite the rf decoder with the VHS-altered one
        self.rf = VHSRFDecode(
            system=system,
            tape_format=tape_format,
            inputfreq=inputfreq,
            rf_options=rf_options,
        )
        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        if system == "PAL":
            if tape_format == "UMATIC":
                self.FieldClass = FieldPALUMatic
            else:
                self.FieldClass = FieldPALVHS
        elif system == "NTSC":
            if tape_format == "UMATIC":
                self.FieldClass = FieldNTSCUMatic
            else:
                self.FieldClass = FieldNTSCVHS
        else:
            raise Exception("Unknown video system!", system)

        self.demodcache = VTRDemodCache(
            self.rf,
            self.infile,
            self.freader,
            num_worker_threads=self.numthreads,
            cvbs_decode=extra_options["cvbs"],
        )

        if fname_out is not None:
            self.outfile_chroma = open(fname_out + "_chroma.tbc", "wb")
        else:
            self.outfile_chroma = None

    # Override to avoid NaN in JSON.
    def calcsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        signal = np.mean(data)
        noise = np.std(data)

        # Make sure signal is positive so we don't try to do log on a negative value.
        if signal < 0.0:
            ldd.logger.info(
                "WARNING: Negative mean for SNR, changing to absolute value."
            )
            signal = abs(signal)
        if noise == 0:
            return 0
        return 20 * np.log10(signal / noise)

    def calcpsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        #        signal = np.mean(data)
        noise = np.std(data)
        if noise == 0:
            return 0
        return 20 * np.log10(100 / noise)

    def buildmetadata(self, f):
        if math.isnan(f.burstmedian):
            f.burstmedian = 0.0
        return super(VHSDecode, self).buildmetadata(f)

    # For laserdisc this decodes frame numbers from VBI metadata, but there won't be such a thing on
    # VHS, so just skip it.
    def decodeFrameNumber(self, f1, f2):
        return None

    # Again ignored for tapes
    def checkMTF(self, field, pfield=None):
        return True

    def writeout(self, dataset):
        f, fi, (picturey, picturec), audio, efm = dataset

        fi["audioSamples"] = 0
        self.fieldinfo.append(fi)

        self.outfile_video.write(picturey)
        self.outfile_chroma.write(picturec)
        self.fields_written += 1

    def close(self):
        setattr(self, "outfile_chroma", None)
        super(VHSDecode, self).close()

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None

    def build_json(self, f):
        try:
            jout = super(VHSDecode, self).build_json(f)
            black = jout["videoParameters"]["black16bIre"]
            white = jout["videoParameters"]["white16bIre"]

            jout["videoParameters"]["black16bIre"] = black * (1 - self.level_adjust)
            jout["videoParameters"]["white16bIre"] = white * (1 + self.level_adjust)
            return jout
        except TypeError as e:
            print("Cannot build json: %s" % e)
            return None


class VTRDemodCache(ldd.DemodCache):
    def __init__(
        self,
        rf,
        infile,
        loader,
        cachesize=256,
        num_worker_threads=1,
        MTF_tolerance=0.05,
        cvbs_decode=False,
    ):

        self.cvbs_decode = cvbs_decode

        super(VTRDemodCache, self).__init__(
            rf,
            infile,
            loader,
            cachesize,
            -1 if cvbs_decode else num_worker_threads,
            MTF_tolerance,
        )

    def worker(self, pipein):
        while True:
            ispiped = False
            if pipein.poll():
                item = pipein.recv()
                ispiped = True
            else:
                item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, target_MTF = item[1:]

                output = {}

                if "fft" not in block:
                    output["fft"] = npfft.fft(block["rawinput"])
                    fftdata = output["fft"]
                else:
                    fftdata = block["fft"]

                if (
                    "demod" not in block
                    or np.abs(block["MTF"] - target_MTF) > self.MTF_tolerance
                ):
                    if not self.cvbs_decode:
                        # RF decode
                        output["demod"] = self.rf.demodblock(
                            fftdata=fftdata, mtf_level=target_MTF, cut=True
                        )
                    else:
                        # CVBS decode
                        output["demod"] = self.rf.cvbsblock(
                            fftdata=fftdata, mtf_level=target_MTF, cut=True
                        )

                    output["MTF"] = target_MTF
                    self.q_out.put((blocknum, output))

            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])

            if not ispiped:
                self.q_in.task_done()


class VHSRFDecode(ldd.RFDecode):
    def __init__(self, inputfreq=40, system="NTSC", tape_format="VHS", rf_options={}):

        # First init the rf decoder normally.
        super(VHSRFDecode, self).__init__(
            inputfreq, system, decode_analog_audio=False, has_analog_audio=False
        )

        # controls the sharpness EQ gain
        self.sharpness_level = (
            rf_options.get("sharpness", vhs_formats.DEFAULT_SHARPNESS) / 100
        )

        self.dod_threshold_p = rf_options.get(
            "dod_threshold_p", vhs_formats.DEFAULT_THRESHOLD_P_DDD
        )
        self.dod_threshold_a = rf_options.get("dod_threshold_a", None)
        self.dod_hysteresis = rf_options.get(
            "dod_hysteresis", vhs_formats.DEFAULT_HYSTERESIS
        )
        self.chroma_trap = rf_options.get("chroma_trap", False)
        track_phase = rf_options.get("track_phase", None)
        high_boost = rf_options.get("high_boost", None)
        self.notch = rf_options.get("notch", None)
        self.notch_q = rf_options.get("notch_q", 10.0)

        if track_phase is None:
            self.track_phase = 0
            self.detect_track = True
            self.needs_detect = True
        elif track_phase == 0 or track_phase == 1:
            self.track_phase = track_phase
            self.detect_track = False
            self.needs_detect = False
        else:
            raise Exception("Track phase can only be 0, 1 or None")
        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        # Then we override the laserdisc parameters with VHS ones.
        if system == "PAL":
            if tape_format == "UMATIC":
                self.SysParams = copy.deepcopy(vhs_formats.SysParams_PAL_UMATIC)
                self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_PAL_UMATIC)
            else:
                # Give the decoder it's separate own full copy to be on the safe side.
                self.SysParams = copy.deepcopy(vhs_formats.SysParams_PAL_VHS)
                self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_PAL_VHS)
        elif system == "NTSC":
            if tape_format == "UMATIC":
                self.SysParams = copy.deepcopy(vhs_formats.SysParams_NTSC_UMATIC)
                self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_NTSC_UMATIC)
            else:
                self.SysParams = copy.deepcopy(vhs_formats.SysParams_NTSC_VHS)
                self.DecoderParams = copy.deepcopy(vhs_formats.RFParams_NTSC_VHS)
        else:
            raise Exception("Unknown video system! ", system)

        # Lastly we re-create the filters with the new parameters.
        self.computevideofilters()

        cc = self.DecoderParams["color_under_carrier"] / 1000000

        DP = self.DecoderParams

        self.high_boost = high_boost if high_boost is not None else DP["boost_bpf_mult"]

        self.Filters["RFVideoRaw"] = lddu.filtfft(
            sps.butter(
                DP["video_bpf_order"],
                [
                    DP["video_bpf_low"] / self.freq_hz_half,
                    DP["video_bpf_high"] / self.freq_hz_half,
                ],
                btype="bandpass",
            ),
            self.blocklen,
        )

        self.Filters["EnvLowPass"] = sps.butter(
            1, [1.0 / self.freq_half], btype="lowpass"
        )

        # Filter for rf before demodulating.
        y_fm = lddu.filtfft(
            sps.butter(
                DP["video_bpf_order"],
                [
                    DP["video_bpf_low"] / self.freq_hz_half,
                    DP["video_bpf_high"] / self.freq_hz_half,
                ],
                btype="bandpass",
            ),
            self.blocklen,
        )

        y_fm_lowpass = lddu.filtfft(
            sps.butter(
                DP["video_lpf_extra_order"],
                [DP["video_lpf_extra"] / self.freq_hz_half],
                btype="lowpass",
            ),
            self.blocklen,
        )

        y_fm_highpass = lddu.filtfft(
            sps.butter(
                DP["video_hpf_extra_order"],
                [DP["video_hpf_extra"] / self.freq_hz_half],
                btype="highpass",
            ),
            self.blocklen,
        )

        self.Filters["RFVideo"] = y_fm * y_fm_lowpass * y_fm_highpass

        self.Filters["RFTop"] = sps.butter(
            1,
            [
                DP["boost_bpf_low"] / self.freq_hz_half,
                DP["boost_bpf_high"] / self.freq_hz_half,
            ],
            btype="bandpass",
            output="sos",
        )

        # Video (luma) main de-emphasis
        #        db3, da3 = FMDeEmphasis(self.freq_hz, tau=DP["deemph_tau"]).get()
        db, da = FMDeEmphasisB(self.freq_hz, DP["deemph_gain"], DP["deemph_mid"]).get()

        #        da3, db3 = gen_high_shelf(260000 / 1.0e6, 14, 1 / 2, inputfreq)

        if False:
            import matplotlib.pyplot as plt

            corner_freq = 1 / (math.pi * 2 * DP["deemph_tau"])

            db2, da2 = FMDeEmphasisB(
                self.freq_hz, DP["deemph_gain"], DP["deemph_mid"] + 50000
            ).get()
            db3, da3 = FMDeEmphasisB(
                self.freq_hz, DP["deemph_gain"], DP["deemph_mid"] - 50000
            ).get()
            self.Filters["FVideo2"] = (
                lddu.filtfft((db2, da2), self.blocklen) * self.Filters["Fvideo_lpf"]
            )
            self.Filters["FVideo3"] = (
                lddu.filtfft((db3, da3), self.blocklen) * self.Filters["Fvideo_lpf"]
            )

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

            w1, h1 = sps.freqz(db, da, fs=self.freq_hz)
            w2, h2 = sps.freqz(db2, da2, fs=self.freq_hz)
            w3, h3 = sps.freqz(db3, da3, fs=self.freq_hz)
            # VHS eyeballed freqs.
            test_arr = np.array(
                [
                    [
                        0.04,
                        0.05,
                        0.07,
                        0.1,
                        corner_freq / 1e6,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.7,
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                    [
                        0.4,
                        0.6,
                        1.2,
                        2.2,
                        3,
                        5.25,
                        7.5,
                        9.2,
                        10.5,
                        11.75,
                        12.75,
                        13.5,
                        13.8,
                        13.9,
                        14,
                    ],
                ]
            )
            # print(test_arr[0])
            test_arr[0] *= 1000000.0
            test_arr[1] *= -1
            #            test_arr[0::] *= 1e6

            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.plot(test_arr[0], test_arr[1], color="#000000")
            ax1.plot(w1, 20 * np.log10(h1))
            ax2.plot(test_arr[0], test_arr[1], color="#000000")
            ax2.plot(w2, 20 * np.log10(h2))
            ax3.plot(test_arr[0], test_arr[1], color="#000000")
            ax3.plot(w3, 20 * np.log10(h3))
            ax4.plot(test_arr[0], test_arr[1])
            ax1.axhline(-3)
            ax2.axhline(-3)
            ax3.axhline(-3)
            ax1.axhline(-7)
            ax2.axhline(-7)
            ax3.axhline(-7)
            ax1.axvline(corner_freq)
            ax2.axvline(corner_freq)
            ax3.axvline(corner_freq)
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            # ax2.plot(data[:2048])
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
            #            exit(0)

        self.Filters["FEnvPost"] = sps.butter(
            1, [700000 / self.freq_hz_half], btype="lowpass", output="sos"
        )

        self.Filters["Fdeemp"] = lddu.filtfft((db, da), self.blocklen)
        self.Filters["FVideo"] = self.Filters["Fvideo_lpf"] * self.Filters["Fdeemp"]
        SF = self.Filters
        SF["FVideo05"] = SF["Fvideo_lpf"] * SF["Fdeemp"] * SF["F05"]

        # Filter to pick out color-under chroma component.
        # filter at about twice the carrier. (This seems to be similar to what VCRs do)
        # TODO: Needs tweaking
        # Note: order will be doubled since we use filtfilt.
        chroma_lowpass = sps.butter(
            2,
            [50000 / self.freq_hz_half, DP["chroma_bpf_upper"] / self.freq_hz_half],
            btype="bandpass",
            output="sos",
        )
        self.Filters["FVideoBurst"] = chroma_lowpass

        if self.notch is not None:
            self.Filters["FVideoNotch"] = sps.iirnotch(
                self.notch / self.freq_half, self.notch_q
            )
            self.Filters["FVideoNotchF"] = lddu.filtfft(
                self.Filters["FVideoNotch"], self.blocklen
            )

        # The following filters are for post-TBC:
        # The output sample rate is at approx 4fsc
        fsc_mhz = self.SysParams["fsc_mhz"]
        out_sample_rate_mhz = fsc_mhz * 4
        out_frequency_half = out_sample_rate_mhz / 2
        het_freq = fsc_mhz + cc
        fieldlen = self.SysParams["outlinelen"] * max(self.SysParams["field_lines"])

        # Final band-pass filter for chroma output.
        # Mostly to filter out the higher-frequency wave that results from signal mixing.
        # Needs tweaking.
        # Note: order will be doubled since we use filtfilt.
        chroma_bandpass_final = sps.butter(
            1,
            [
                (fsc_mhz - 0.64) / out_frequency_half,
                (fsc_mhz + 0.34) / out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )
        self.Filters["FChromaFinal"] = chroma_bandpass_final

        # Bandpass filter to select heterodyne frequency from the mixed fsc and color carrier signal
        het_filter = sps.butter(
            1,
            [
                (het_freq - 0.001) / out_frequency_half,
                (het_freq + 0.001) / out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )
        samples = np.arange(fieldlen)

        # As this is done on the tbced signal, we need the sampling frequency of that,
        # which is 4fsc for NTSC and approx. 4 fsc for PAL.
        # TODO: Correct frequency for pal?
        cc_wave_scale = cc / out_sample_rate_mhz
        self.cc_ratio = cc_wave_scale
        # 0 phase downconverted color under carrier wave
        self.cc_wave = np.sin(2 * np.pi * cc_wave_scale * samples)
        # +90 deg and so on phase wave for track2 phase rotation
        cc_wave_90 = np.sin((2 * np.pi * cc_wave_scale * samples) + (np.pi / 2))  #
        cc_wave_180 = np.sin((2 * np.pi * cc_wave_scale * samples) + np.pi)
        cc_wave_270 = np.sin(
            (2 * np.pi * cc_wave_scale * samples) + np.pi + (np.pi / 2)
        )

        # Standard frequency color carrier wave.
        self.fsc_wave = utils.gen_wave_at_frequency(
            fsc_mhz, out_sample_rate_mhz, fieldlen
        )
        self.fsc_cos_wave = utils.gen_wave_at_frequency(
            fsc_mhz, out_sample_rate_mhz, fieldlen, np.cos
        )

        # Heterodyne wave
        # We combine the color carrier with a wave with a frequency of the
        # subcarrier + the downconverted chroma carrier to get the original
        # color wave back.
        self.chroma_heterodyne = np.array(
            [
                sps.sosfiltfilt(het_filter, self.cc_wave * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_90 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_180 * self.fsc_wave),
                sps.sosfiltfilt(het_filter, cc_wave_270 * self.fsc_wave),
            ]
        )

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024
        self.demods = 0

        # sharpness filter / video EQ
        iir_eq_loband = utils.firdes_highpass(
            self.freq_hz,
            DP["video_eq"]["loband"]["corner"],
            DP["video_eq"]["loband"]["transition"],
            DP["video_eq"]["loband"]["order_limit"],
        )

        self.videoEQFilter = {
            0: utils.FiltersClass(iir_eq_loband[0], iir_eq_loband[1], self.freq_hz),
            # 1: utils.FiltersClass(iir_eq_hiband[0], iir_eq_hiband[1], self.freq_hz),
        }

        self.chromaTrap = ChromaSepClass(self.freq_hz, self.SysParams["fsc_mhz"])

        #FMDeEmphasis(self.freq_hz, tau=DP["deemph_tau"]).get()
        self.DCrestore = DCrestore(self.freq_hz, self.SysParams, self.blocklen, self.iretohz)


    def computedelays(self, mtf_level=0):
        """Override computedelays
        It's normally used for dropout compensation, but the dropout compensation implementation
        in ld-decode assumes composite color. This function is called even if it's disabled, and
        seems to break with the VHS setup, so we disable it by overriding it for now.
        """
        # Set these to 0 for now, the metrics calculations look for them.
        self.delays = {}
        self.delays["video_sync"] = 0
        self.delays["video_white"] = 0

    # It enhances the upper band of the video signal
    def video_EQ(self, demod):
        overlap = 10  # how many samples the edge distortion produces
        ha = self.videoEQFilter[0].filtfilt(demod)
        hb = self.videoEQFilter[0].lfilt(demod[:overlap])
        hc = np.concatenate(
            (hb[:overlap], ha[overlap:])
        )  # edge distortion compensation, needs check
        hf = np.multiply(self.DecoderParams["video_eq"]["loband"]["gain"], hc)

        gain = self.sharpness_level
        result = np.multiply(np.add(np.roll(np.multiply(gain, hf), 0), demod), 1)

        return result

    def demodblock(self, data=None, mtf_level=0, fftdata=None, cut=False):
        rv = {}

        if fftdata is not None:
            indata_fft = fftdata
        elif data is not None:
            indata_fft = npfft.fft(data[: self.blocklen])
        else:
            raise Exception("demodblock called without raw or FFT data")

        if data is None:
            data = npfft.ifft(indata_fft).real

        if self.notch is not None:
            indata_fft = indata_fft * self.Filters["FVideoNotchF"]

        raw_filtered = npfft.ifft(
            indata_fft * self.Filters["RFVideoRaw"] * self.Filters["hilbert"]
        ).real

        # Calculate an evelope with signal strength using absolute of hilbert transform.
        # Roll this a bit to compensate for filter delay, value eyballed for now.
        raw_env = np.roll(np.abs(raw_filtered), 4)
        env = utils.filter_simple(raw_env, self.Filters["FEnvPost"])
        env_mean = np.mean(env)

        # Applies RF filters
        indata_fft_filt = indata_fft * self.Filters["RFVideo"]
        data_filtered = npfft.ifft(indata_fft_filt)
        # Boost high frequencies in areas where the signal is weak to reduce missed zero crossings
        # on sharp transitions. Using filtfilt to avoid phase issues.
        high_part = utils.filter_simple(data_filtered, self.Filters["RFTop"]) * (
            (env_mean * 0.9) / env
        )
        indata_fft_filt += npfft.fft(high_part * self.high_boost)

        hilbert = npfft.ifft(indata_fft_filt * self.Filters["hilbert"])

        # FM demodulator
        demod = unwrap_hilbert(hilbert, self.freq_hz).real

        if self.chroma_trap:
            # applies the Subcarrier trap
            demod = self.chromaTrap.work(demod)

        # Disabled if sharpness level is zero (default).
        if self.sharpness_level > 0:
            # applies the video EQ
            demod = self.video_EQ(demod)

        # applies main deemphasis filter
        demod_fft = npfft.rfft(demod)
        out_video = npfft.irfft(
            demod_fft * self.Filters["FVideo"][0 : (self.blocklen // 2) + 1]
        ).real

        out_video05 = npfft.irfft(
            demod_fft * self.Filters["FVideo05"][0 : (self.blocklen // 2) + 1]
        ).real
        out_video05 = np.roll(out_video05, -self.Filters["F05_offset"])

        #self.DCrestore.work(luma05)
        #luma05 = self.DCrestore.compensate_sync(luma05)
        #luma = self.DCrestore.compensate_sync(luma)

        # Filter out the color-under signal from the raw data.
        out_chroma = utils.filter_simple(
            data[: self.blocklen], self.Filters["FVideoBurst"]
        )

        if self.notch is not None:
            out_chroma = sps.filtfilt(
                self.Filters["FVideoNotch"][0],
                self.Filters["FVideoNotch"][1],
                out_chroma,
            )

        # Move chroma to compensate for Y filter delay.
        # value needs tweaking, ideally it should be calculated if possible.
        # TODO: Not sure if we need this after hilbert filter change, needs check.
        out_chroma = np.roll(out_chroma, 10)
        # crude DC offset removal
        out_chroma = out_chroma - np.mean(
            out_chroma[self.blockcut : -self.blockcut_end]
        )

        if False:
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots()

            # out_video2 = np.fft.irfft(
            #    demod_fft * self.Filters["FVideo2"][0 : (self.blocklen // 2) + 1]
            # ).real
            # out_video3 = np.fft.irfft(
            #    demod_fft * self.Filters["FVideo3"][0 : (self.blocklen // 2) + 1]
            # ).real
            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.axhline(self.iretohz(0))
            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
            ax1.axhline(self.iretohz(7.5))
            ax1.axhline(self.iretohz(100))
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            ax1.plot(out_video[:2048])
            # ax2.plot(out_video2[:2048])
            # ax3.plot(out_video3[:2048])
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
            exit(0)

        # demod_burst is a bit misleading, but keeping the naming for compatability.
        video_out = np.rec.array(
            [out_video, demod, out_video05, out_chroma, env, data],
            names=["demod", "demod_raw", "demod_05", "demod_burst", "envelope", "raw"],
        )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        return rv
