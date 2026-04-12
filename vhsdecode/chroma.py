import math
import numpy as np
import lddecode.utils as lddu
import lddecode.core as ldd
import scipy.signal as sps
from vhsdecode.rust_utils import sosfiltfilt_rust

from numba import njit


@njit(cache=True, nogil=True)
def chroma_to_u16(chroma):
    """Scale the chroma output array to a 16-bit value for output."""
    S16_ABS_MAX = 32767

    # Disabled for now as it's misleading.
    # if np.max(chroma) > S16_ABS_MAX:
    #     ldd.logger.warning("Chroma signal clipping.")
    return (chroma + S16_ABS_MAX).astype(np.uint16)


@njit(cache=True, nogil=True)
def acc(chroma, burst_abs_ref, burststart, burstend, linelength, lines, burst_detected_line):
    """Scale chroma according to the level of the color burst on each line."""
    STARTING_LINE = int(16)
    assert lines > STARTING_LINE

    output = np.zeros(chroma.size, dtype=np.double)
    mean_burst_accumulator = 0
    for linenumber in range(16, lines):
        linestart = linelength * linenumber
        lineend = linestart + linelength

        if linenumber < burst_detected_line:
            # color killer active for this line
            output[linestart:lineend] = 0
        else:
            line = chroma[linestart:lineend]
            acced, rms = acc_line(line, burst_abs_ref, burststart, burstend)
            output[linestart:lineend] = acced
            mean_burst_accumulator += rms

    return output, mean_burst_accumulator / (lines - STARTING_LINE)


@njit(cache=True, nogil=True)
def acc_line(chroma, burst_abs_ref, burststart, burstend):
    """Scale chroma according to the level of the color burst the line."""
    output = np.zeros(chroma.size, dtype=np.double)

    line = chroma
    burst_abs_mean = lddu.rms(line[burststart:burstend])
    # np.sqrt(np.mean(np.square(line[burststart:burstend])))
    #    burst_abs_mean = np.mean(np.abs(line[burststart:burstend]))
    scale = burst_abs_ref / burst_abs_mean if burst_abs_mean != 0 else 1
    output = line * scale

    return output, burst_abs_mean


@njit(cache=True, nogil=True)
def comb_c_pal(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 2H,
    and one advanced by 2H
    line by line. VCRs do this to reduce crosstalk.
    Helps chroma stability on LP tapes in particular.
    (VCRs only adds delayed by 1h instead)
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


@njit(cache=True, nogil=True)
def comb_c_ntsc(data, line_len):
    """Very basic comb filter, adds the signal together with a signal delayed by 1H,
    and one advanced by 1h
    line by line. VCRs do this to reduce crosstalk.
    (VCRs only adds delayed by 1h instead)
    """

    data2 = data.copy()
    numlines = len(data) // line_len
    for line_num in range(16, numlines - 2):
        advanced1h = data2[(line_num + 1) * line_len : (line_num + 2) * line_len]
        delayed1h = data2[(line_num - 1) * line_len : (line_num) * line_len]
        line_slice = data[line_num * line_len : (line_num + 1) * line_len]
        # Let the delayed signal contribute 1/3.
        # Could probably make the filtering configurable later.
        data[line_num * line_len : (line_num + 1) * line_len] = (
            (line_slice * 2) - advanced1h - delayed1h
        ) / 4
    return data


@njit(cache=True, nogil=True, fastmath=True)
def _demod_burst(burst, burst_start, burst_len, burst_sin, burst_cos):
    I = 0
    Q = 0

    for i in range(burst_len):
        I += burst[i] * burst_cos[i + burst_start]
        Q += burst[i] * burst_sin[i + burst_start]

    burst_magnitude = np.hypot(I, Q)
    burst_phase = np.arctan2(Q, I)
    burst_phase_deg = np.degrees(burst_phase) % 360

    return burst_phase_deg, burst_magnitude, I, Q


def _get_upconverted_burst(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    current_phase,
    burstarea,
    burst_sin,
    burst_cos,
    linenumber,
    lineoffset,
    outwidth,
):
    burst_padding = burstarea[1] - burstarea[0]  # pad the area being filtered
    burst_start = max(
        0, (linenumber - lineoffset) * outwidth + burstarea[0] - burst_padding
    )
    burst_end = min(len(chroma), burst_start + burstarea[1] + burst_padding)

    burst = (
        chroma_heterodyne[current_phase][burst_start:burst_end]
        * chroma[burst_start:burst_end]
    )

    # filter out noise so only the color burst is present
    filtered_padded = sosfiltfilt_rust(chroma_filter, burst)
    filtered = filtered_padded[burst_padding:-burst_padding]

    burst_len = len(filtered)

    return _demod_burst(
        filtered, burst_start + burst_padding, burst_len, burst_sin, burst_cos
    )


def _get_phase_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_starting_index,
    burstarea,
    burst_sin,
    burst_cos,
    lineoffset,
    outwidth,
    last_line,
    detect_chroma_track_phase,
    rotation_check_start_line,
    track_change_threshold,
    color_system
):
    do_phase_rotation_check = (
        detect_chroma_track_phase
        and chroma_rotation is not None
        and chroma_heterodyne is not None
    )

    phase_sequence = []

    if chroma_rotation_starting_index is None:
        # first field
        chroma_rotation_starting_index = 0
        chroma_rotation_index = 0

    if chroma_rotation:
        # color under format that uses a phase rotated heterodyne to down convert the composite chroma
        chroma_rotation_index = chroma_rotation_starting_index
        track_rotation = chroma_rotation[chroma_rotation_index]
    else:
        # format that uses a fixed heterodyne phase, or does not rotate
        chroma_rotation_index = 0
        track_rotation = chroma_rotation_starting_index
    """
    "...a signal that represents phase zero with respect to the chroma signal phase 
    +90°, +180°, +270° etc. or a phase 0°, -90°. —180°. —270°. etc., 
    depending upon which head is on the tape at the particular time.

    The direction of phase rotation, being related to which head is on the tape at a given time,
    can be determined and preset by sensing whether the PG (pulse generator) pulse is positive-going or negative-going."
     - https://archive.org/details/rca-vcr-1-red-book-w-cover/page/n25/mode/2up?q=phase

    See also: https://archive.org/details/video-technical-guide/page/1-9/mode/2up?q=phase

    The phase rotation switch is determined at record time depending on which video head is on the tape.
    This rotation switch can occur in the middle of a line, causing a small phase artifact
    TODO: It may be possible to detect where this happens on the line and correct the phase issue mid-line
          Possibly a 2D aware detection could be used to determine where the color phase is rotated +-90 degrees relative to the lines above and below
    """

    current_phase = 0
    use_next_phase = False
    for linenumber in range(lineoffset, last_line):
        if use_next_phase:
            # reuse the calculated phase from the previous iteration
            current_phase = next_phase
            current_burst_phase = next_burst_phase
            current_burst_I = next_burst_I
            current_burst_Q = next_burst_Q
            current_burst_magnitude = next_burst_magnitude

            use_next_phase = False
        else:
            current_phase = (current_phase + track_rotation) % 4
            (
                current_burst_phase,
                current_burst_magnitude,
                current_burst_I,
                current_burst_Q,
            ) = _get_upconverted_burst(
                chroma,
                chroma_heterodyne,
                chroma_filter,
                current_phase,
                burstarea,
                burst_sin,
                burst_cos,
                linenumber,
                lineoffset,
                outwidth,
            )

        # check if the track has rotated around the head switching area
        if (
            do_phase_rotation_check
            and linenumber >= rotation_check_start_line
            and linenumber < last_line - 1
        ):
            # get the next burst using the phase rotation for the current track
            next_phase = (current_phase + track_rotation) % 4
            next_burst_phase, next_burst_magnitude, next_burst_I, next_burst_Q = (
                _get_upconverted_burst(
                    chroma,
                    chroma_heterodyne,
                    chroma_filter,
                    next_phase,
                    burstarea,
                    burst_sin,
                    burst_cos,
                    linenumber + 1,
                    lineoffset,
                    outwidth,
                )
            )

            if color_system == "NTSC":
                # check one line back
                comparison_burst = current_burst_phase
            elif color_system == "PAL":
                # check two lines back
                comparison_burst = phase_sequence[-1][2]

            phase_delta_quadrant = abs(
                (next_burst_phase - comparison_burst + 180) % 360 - 180
            )
            if phase_delta_quadrant > track_change_threshold:
                # burst is more in phase than out of phase, flip rotation so it remains out of phase
                chroma_rotation_index = (chroma_rotation_index + 1) % 2
                track_rotation = chroma_rotation[chroma_rotation_index]
            else:
                use_next_phase = True

        phase_sequence.append(
            (
                linenumber,
                current_phase,
                current_burst_phase,
                current_burst_magnitude,
                current_burst_I,
                current_burst_Q,
            )
        )

    if chroma_rotation and chroma_rotation_index == chroma_rotation_starting_index:
        # rotate the phase for the next field, if rotation was not detected
        chroma_rotation_index = (chroma_rotation_index + 1) % 2

    return chroma_rotation_index, phase_sequence


def get_phase_rotation_sequence(
    chroma,
    chroma_heterodyne,
    chroma_filter,
    chroma_rotation,
    chroma_rotation_index,
    lineoffset,
    linesout,
    outwidth,
    burstarea,
    burst_sin,
    burst_cos,
    detect_chroma_track_phase,
    rotation_check_start_line,
    enable_color_killer,
    prev_burst_detected_line,
    color_system,
):
    # Detects the correct color-under heterodyne starting phase and rotation direction
    # Additional for NTSC, this function calculates the color burst average for burst-locked TBC later on
    track_change_threshold = 90
    burst_check_skip_lines = 16

    # TODO Expose as option, possible this needs to be relative to the sync pulse and level detection
    burst_magnitude_threshold = 2.5e4

    end = linesout + lineoffset

    chroma_rotation_index, phase_sequence = _get_phase_sequence(
        chroma,
        chroma_heterodyne,
        chroma_filter,
        chroma_rotation,
        chroma_rotation_index,
        burstarea,
        burst_sin,
        burst_cos,
        lineoffset,
        outwidth,
        end,
        detect_chroma_track_phase,
        rotation_check_start_line,
        track_change_threshold,
        color_system
    )

    burst_check_start = burst_check_skip_lines
    burst_check_end = end - burst_check_skip_lines
    burst_detected_line = 0 # color enabled by default

    if chroma_rotation:
        # detect relative phase difference between lines
        delta_0 = 0
        delta_90 = 0
        delta_180 = 0
        delta_270 = 0

        for i in range(1, len(phase_sequence)):
            _, _, previous_burst_phase, _, _, _ = phase_sequence[i - 1]
            line_number, _, current_burst_phase, _, _, _ = phase_sequence[i]

            if line_number > burst_check_start and line_number < burst_check_end:
                delta = (current_burst_phase - previous_burst_phase) % 360
                bucket = int((delta + 45) // 90) % 4

                if bucket == 0:
                    delta_0 += 1
                elif bucket == 1:
                    delta_90 += 1
                elif bucket == 2:
                    delta_180 += 1
                else:
                    delta_270 += 1

        if color_system == "NTSC":
            # if the bursts are out of phase with each other, the track was miss-detected, flip phase and recalculate sequence
            flip_track_phase = delta_0 < delta_180
        else:  # color_system in ("PAL", "PAL_M", "NLINHA", "MESECAM")
            # each line should alternate phase, if there are repeated sequences of phase, recalculate
            alt1 = delta_90 + delta_270
            alt2 = delta_0 + delta_180

            # choose whichever pattern dominates
            flip_track_phase = alt1 < alt2
    else:
        # no difference between track phases, do not flip
        flip_track_phase = False

    if flip_track_phase:
        # recalculate with the corrected track rotation
        chroma_rotation_index, phase_sequence = _get_phase_sequence(
            chroma,
            chroma_heterodyne,
            chroma_filter,
            chroma_rotation,
            chroma_rotation_index,
            burstarea,
            burst_sin,
            burst_cos,
            lineoffset,
            outwidth,
            end,
            detect_chroma_track_phase,
            rotation_check_start_line,
            track_change_threshold,
            color_system
        )

    if color_system == "NTSC":
        # find the phase of the color burst for the entire field
        I_total = 0
        Q_total = 0
        avg_count = 0
        magnitude_avg = 0
        for line_number, _, _, magnitude, I, Q in phase_sequence:
            if line_number > burst_check_start and line_number < burst_check_end:
                if magnitude != 0:
                    I_total += I / magnitude
                    Q_total += Q / magnitude
                    avg_count += 1
                    magnitude_avg += magnitude

                    if enable_color_killer:
                        # find the first line that might have a valid burst if the previous field had the burst disabled
                        # broadcasters would sometime turn on the burst mid-field, so attempt to detect that transition here
                        if (
                            prev_burst_detected_line == -1 # previous field had color killer activated
                            and burst_detected_line == 0 and magnitude > burst_magnitude_threshold # first burst that exceeds threshold
                        ):
                            # first burst that exceeds threshold
                            # color killer will be active until this line, then it deactivates
                            # it is only reactivated after an entire field is without color (below)
                            burst_detected_line = line_number

        magnitude_avg /= avg_count
        burst_phase_avg = np.degrees(np.arctan2(Q_total, I_total)) % 360

        if enable_color_killer:
            if magnitude_avg < burst_magnitude_threshold:
                # (re)activate color killer for the entire field
                burst_detected_line = -1

    elif color_system in ("MPAL", "NLINHA"):
        burst_phase_avg = None
    else:
        burst_phase_avg = None

    return chroma_rotation_index, phase_sequence, burst_phase_avg, burst_detected_line


@njit(cache=True, nogil=True, fastmath=True)
def upconvert_chroma(
    chroma,
    uphet,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    chroma_heterodyne,
):
    for linenumber, current_phase, _, _, _, _ in phase_rotation_sequence:
        linestart = (linenumber - lineoffset) * outwidth
        lineend = linestart + outwidth

        heterodyne = chroma_heterodyne[current_phase][linestart:lineend]
        c = chroma[linestart:lineend]
        uphet[linestart:lineend] = c * heterodyne


@njit(cache=True, nogil=True, fastmath=True)
def upconvert_chroma_phase_comp(
    chroma,
    uphet,
    lineoffset,
    outwidth,
    phase_rotation_sequence,
    color_under_carrier_fs,
    fsc,
    target_phase
):
    deg2rad_scale = np.pi / 180.0
    pi_over_two = np.pi / 2.0

    het_mhz = color_under_carrier_fs / 1e6
    het_coefficient = pi_over_two * (1.0 + het_mhz / fsc)

    target_phase_rad = target_phase * deg2rad_scale

    # check if there are other adjustments needed after this phase adjust
    # chroma decoder seems to be correcting an amplitude problem

    for linenumber, phase_rotation, burst_phase, _, _, _ in phase_rotation_sequence:
        linestart = (linenumber - lineoffset) * outwidth
        lineend = linestart + outwidth

        theta = het_coefficient * linestart + (
            phase_rotation * pi_over_two # heterodyne rotation
            + target_phase_rad + burst_phase * deg2rad_scale # phase offset relative to line
        )

        for i in range(linestart, lineend):
            uphet[i] = chroma[i] * -math.cos(theta)
            theta += het_coefficient


@njit(cache=True, nogil=True)
def burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea):
    for line in range(lineoffset, linesout + lineoffset):
        linestart = (line - lineoffset) * outwidth
        lineend = linestart + outwidth

        chroma[linestart + burstarea[1] + 5 : lineend] *= 2

    return chroma


@njit(cache=True, nogil=True, fastmath=True)
def shift_chroma_and_remove_dc(out_chroma, move):
    out_chroma = np.roll(out_chroma, move)
    # crude DC offset removal
    out_chroma -= np.mean(out_chroma)
    return out_chroma


def demod_chroma_filt(
    data, filter, blocklen, notch, do_notch=None, move=10, audio_notch=None
):
    out_chroma = sosfiltfilt_rust(filter, data[:blocklen])

    if audio_notch is not None:
        out_chroma = sps.filtfilt(
            audio_notch[0],
            audio_notch[1],
            out_chroma,
        )

    if do_notch is not None and do_notch:
        out_chroma = sps.filtfilt(
            notch[0],
            notch[1],
            out_chroma,
        )

    # Move chroma to compensate for Y filter delay.
    # value needs tweaking, ideally it should be calculated if possible.
    # TODO: Not sure if we need this after hilbert filter change, needs check.
    return shift_chroma_and_remove_dc(out_chroma, move)


def decode_chroma_phase_rotation(
    field,
    disable_tracking_cafc=False,
    chroma_rotation=None,
    detect_chroma_track_phase=False,
):
    chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")

    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    burst_area_init = get_burst_area(field)
    rotation_check_start_line = lineoffset + linesout - 16

    burstarea = burst_area_init[0] - 5, burst_area_init[1] + 10

    # Rotation per track
    # VHS PAL:      Track1 0,   Track2 -90
    # VHS NTSC:     Track1 +90, Track2 -90
    # Betamax PAL:  None - uses frequency offset instead
    # Betamax NTSC: Track1 180, Track2 0
    # Video8 PAL:   Track1 0,   Track2 -90
    # Video8 NTSC:  Track1 0,   Track2 180

    chroma_heterodyne = (
        field.rf.chroma_afc.getChromaHet()
        if (field.rf.do_cafc and not disable_tracking_cafc)
        else field.rf.chroma_heterodyne
    )

    prev_burst_detected_line = 0
    if field.prevfield is not None:
        prev_burst_detected_line = field.prevfield.burst_detected_line

    track_phase, phase_sequence, burst_phase_avg, burst_detected_line = get_phase_rotation_sequence(
        chroma,
        chroma_heterodyne,
        field.rf.Filters["FChromaFinal"],
        chroma_rotation,
        field.rf.track_phase, # index for chroma rotation, and static if there is no chroma rotation
        lineoffset,
        linesout,
        outwidth,
        burstarea,
        field.rf.fsc_wave,
        field.rf.fsc_cos_wave,
        detect_chroma_track_phase,
        rotation_check_start_line, # check for track phase rotation around the headswitching area (bottom of field)
        field.rf.options.enable_color_killer,
        prev_burst_detected_line,
        field.rf.color_system,
    )

    return track_phase, phase_sequence, burst_phase_avg, burst_detected_line

ntsc_color_framing_phase_shift = 33
ntsc_color_framing_map = {
    (1, 0): (1, 0 - ntsc_color_framing_phase_shift),
    (0, 1): (2, 180 - ntsc_color_framing_phase_shift),
    (1, 1): (3, 180 - ntsc_color_framing_phase_shift),
    (0, 0): (4, 0 - ntsc_color_framing_phase_shift),
}

def process_chroma(
    field,
    disable_deemph=False,
    disable_comb=False,
    disable_tracking_cafc=False,
    do_chroma_deemphasis=False,
):
    lineoffset = field.lineoffset + 1
    linesout = field.outlinecount
    outwidth = field.outlinelen

    uphet = np.zeros((linesout * outwidth), dtype=np.float32)
    if field.burst_detected_line == -1:
        # skip chroma if the color killer is active for the whole field
        return uphet

    # Run TBC/downscale on chroma (if new field, else uses cache)
    # Cached if chroma process is run multiple times on one field due to track detection.
    if field.chroma_tbc_buffer is None:
        chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst")

        # If chroma AFC is enabled
        if field.rf.do_cafc:
            # it does the chroma filtering AFTER the TBC
            chroma = demod_chroma_filt(
                chroma,
                field.rf.chroma_afc.get_chroma_bandpass(),
                len(chroma),
                field.rf.Filters["FVideoNotch"],
                field.rf.notch,
                move=(int(10 * (field.rf.sys_params["outfreq"] / 40))),
                audio_notch=field.rf.Filters.get("FChromaAudioNotch", None),
            )

            if not disable_tracking_cafc:
                spec, meas, offset, cphase = field.rf.chroma_afc.freqOffset(chroma)
                ldd.logger.debug(
                    "Chroma under AFC: %.02f kHz, Offset (long term): %.02f Hz, Phase: %.02f deg"
                    % (meas / 1e3, offset, cphase * 360 / (2 * np.pi))
                )

        field.rf.chroma_tbc_buffer = chroma
        field.chroma_tbc_buffer = chroma
    else:
        chroma = field.chroma_tbc_buffer

    burst_area_init = get_burst_area(field)
    burstarea = burst_area_init[0] - 5, burst_area_init[1] + 10

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.color_system == "NTSC":
        if not disable_deemph:
            chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    if (
        field.rf.color_system == "NTSC"
        and not field.rf.options.disable_phase_correction
    ):
        # regenerates the color framing
        field.fieldPhaseID, target_phase = ntsc_color_framing_map[
            (field.isFirstField, (field.field_number // 2) % 2)
        ]

        upconvert_chroma_phase_comp(
            chroma,
            uphet,
            lineoffset,
            outwidth,
            field.phase_sequence,
            field.rf.chroma_afc.color_under,
            field.rf.chroma_afc.fsc_mhz,
            target_phase
        )
    else:
        chroma_heterodyne = (
            field.rf.chroma_afc.getChromaHet()
            if (field.rf.do_cafc and not disable_tracking_cafc)
            else field.rf.chroma_heterodyne
        )
    
        upconvert_chroma(
            chroma,
            uphet,
            lineoffset,
            outwidth,
            field.phase_sequence,
            chroma_heterodyne
        )

    # Filter out unwanted frequencies from the final chroma signal.
    # Mixing the signals will produce waves at the difference and sum of the
    # frequencies. We only want the difference wave which is at the correct color
    # carrier frequency here.
    # We do however want to be careful to avoid filtering out too much of the sideband.
    uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)

    # FFT filter way to use a supergauss filter to more sharply cut out the upper harmonic
    # This may be a better approach but slows down things a bit much so not using for now
    # orig_len = len(uphet)
    # uphet = np_fft.irfft(np_fft.rfft(uphet) * field.rf.Filters["FChromaFinal"], n=orig_len)

    if do_chroma_deemphasis:
        b, a = field.rf.Filters["chroma_deemphasis"]
        uphet = sps.lfilter(b, a, uphet)

    # Basic comb filter for NTSC to calm the color a little.
    if not disable_comb:
        if field.rf.color_system == "NTSC":
            uphet = comb_c_ntsc(uphet, outwidth)
        else:
            uphet = comb_c_pal(uphet, outwidth)

    # Final automatic chroma gain.
    uphet, mean_rms = acc(
        uphet,
        field.rf.SysParams["burst_abs_ref"],
        burstarea[0],
        burstarea[1],
        outwidth,
        linesout,
        field.burst_detected_line
    )

    field.rf.field_averages.chroma_level.push(mean_rms)

    return uphet


def decode_chroma(field, do_chroma_deemphasis=False):
    if field.rf.options.write_chroma:
        """Do track detection if needed and upconvert the chroma signal"""
        field.chroma_tbc_buffer = None

        uphet = process_chroma(
            field,
            disable_comb=field.rf.options.disable_comb,
            disable_tracking_cafc=False,
            do_chroma_deemphasis=do_chroma_deemphasis,
        )
        field.uphet_temp = uphet
        # Release to avoid keeping this im memory - should do this in a cleaner manner.
        field.chroma_tbc_buffer = None
        return chroma_to_u16(uphet)

    return None


def get_burst_area(field):
    return (
        math.floor(field.usectooutpx(field.rf.SysParams["colorBurstUS"][0])),
        math.ceil(field.usectooutpx(field.rf.SysParams["colorBurstUS"][1])),
    )
