"""Chroma utilities shared by the QAM and SECAM paths, and the
top-level per-field orchestrators that tie the two together."""

import math

import numpy as np
import scipy.signal as sps
from numba import njit

import lddecode.core as ldd
from vhsdecode.rust_utils import sosfiltfilt_rust
from vhsdecode.chroma.qam import (
    burst_deemphasis,
    chroma_automatic_gain,
    chroma_transient_improvement,
    comb_c_ntsc,
    comb_c_pal,
    filter_chroma_fft,
    get_burst_area,
    ntsc_color_framing_map,
    upconvert_chroma,
    upconvert_chroma_phase_comp,
)
from vhsdecode.chroma.secam import (
    _process_chroma_secam_method1,
    measure_secam_under_carrier_offset,
)


@njit(cache=True, nogil=True, fastmath=True)
def chroma_to_u16(chroma):
    """
    Scale the chroma output array to a 16-bit value for output.
    """
    S16_ABS_MAX = 32767.0
    N = len(chroma)

    out = np.empty(N, dtype=np.uint16)

    for i in range(N):
        out[i] = np.uint16(chroma[i] + S16_ABS_MAX)
        
    return out


@njit(cache=True, nogil=True, fastmath=False)
def shift_chroma_and_remove_dc(out_chroma, move):
    n = len(out_chroma)
    move %= n
    
    mean_acc = 0

    # save wrapped values
    tmp = np.empty(move, dtype=out_chroma.dtype)

    for i in range(move):
        tmp[i] = out_chroma[n - move + i]

    # single pass shift
    for i in range(n - move - 1, -1, -1):
        mean_acc += out_chroma[i]
        out_chroma[i + move] = out_chroma[i]

    # small wrap-around copy
    for i in range(move):
        mean_acc += tmp[i]
        out_chroma[i] = tmp[i]

    mean_acc /= n

    # crude DC offset removal
    for i in range(n):
        out_chroma[i] -= mean_acc


def chroma_color_under_filter(
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
    shift_chroma_and_remove_dc(out_chroma, move)

    return out_chroma


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

    if field.burst_detected_line == -1:
        # skip chroma if the color killer is active for the whole field
        return np.zeros((linesout * outwidth), dtype=np.float32)
    
    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        field.fieldPhaseID, target_phase = ntsc_color_framing_map[
            (field.isFirstField, (field.field_number // 2) % 2)
        ]
        chroma_shift_direction = 1 if target_phase else -1
    else:
        chroma_shift_direction = 0

    # Run TBC/downscale on chroma (if new field, else uses cache)
    # Cached if chroma process is run multiple times on one field due to track detection.
    if field.chroma_tbc_buffer is None:
        # shift the chroma to reverse group delay caused by the color under heterodyne filter
        # this is dependent on color framing, and is disabled if color framing is disabled
        # TODO: shift amount may need tuning / needs validation
        chroma_subcarrier_delay_cycles = field.rf.SysParams['fsc_mhz'] * 1e6 / (2.0 * np.pi * field.rf.DecoderParams["color_under_carrier"])
        chroma_subcarrier_delay_samples = chroma_subcarrier_delay_cycles * 4
        chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst", shift=chroma_subcarrier_delay_samples * chroma_shift_direction)

        # If chroma AFC is enabled
        if field.rf.do_cafc:
            # it does the chroma filtering AFTER the TBC
            chroma = chroma_color_under_filter(
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

        if (
            field.rf.color_system == "MESECAM"
            and field.rf.options.secam_carrier_servo
        ):
            # Measure the rest carrier pair on the late back porch,
            # 3.7 to 0.3 us before active video starts.
            active_start_px = field.usectooutpx(field.rf.SysParams["activeVideoUS"][0])
            porch_window = (int(active_start_px) - 65, int(active_start_px) - 5)

            carrier_offset = measure_secam_under_carrier_offset(
                chroma,
                linesout,
                outwidth,
                porch_window,
                field.rf.chroma_afc.true_samp_rate,
                field.rf.DecoderParams["color_under_carrier"],
            )
            if carrier_offset is not None:
                field.rf.secam_servo_avg.push(carrier_offset)
                ldd.logger.debug(
                    "SECAM carrier servo: measured offset %.02f Hz" % carrier_offset
                )

        field.rf.chroma_tbc_buffer = chroma
        field.chroma_tbc_buffer = chroma
    else:
        chroma = field.chroma_tbc_buffer

    burstarea = get_burst_area(field)

    if field.rf.color_system == "SECAM":
        # Method 1 restores the chroma block by phase multiplication rather
        # than by mixing against a heterodyne, so it skips the shared
        # up-conversion path below entirely.
        return _process_chroma_secam_method1(
            field, chroma, linesout, outwidth, burstarea
        )

    # For NTSC, the color burst amplitude is doubled when recording, so we have to undo that.
    if field.rf.color_system == "NTSC":
        if not disable_deemph:
            chroma = burst_deemphasis(chroma, lineoffset, linesout, outwidth, burstarea)

    if (
        not field.rf.options.disable_phase_correction
        and field.rf.color_system == "NTSC"
    ):
        target_phase_even = target_phase
        target_phase_odd = target_phase

        # TODO: PAL color framing is disabled for now.
        #       need to find a reliable way to detect if this is field 1,2 vs 3,4
        # if field.rf.color_system == "PAL":
        #     line_6_burst_present = field.phase_sequence[4 + lineoffset][3] > field.burst_magnitude_avg / 3
        #     field.fieldPhaseID, target_phase_even, target_phase_odd = pal_color_framing_map[
        #         (field.isFirstField, line_6_burst_present, (field.field_number // 4) % 2)
        #     ]

        # this uses the burst measurements to interpolate the correct phase of the color under heterodyne
        # phase issues are corrected continiously for each sample using a linear spline interpolated from the burst measurements
        # the mixing is performed on the upsampled signal to avoid aliasing introduced from the up-heterodyne mixing product
        upconvert_chroma_phase_comp(
            chroma, # modifies this in place
            lineoffset,
            outwidth,
            field.phase_sequence,
            field.rf.DecoderParams["color_under_carrier"],
            field.rf.SysParams["fsc_mhz"] * 1e6,
            target_phase_even,
            target_phase_odd,
        )
        uphet = chroma
    else:
        if field.rf.chroma_afc.conversion_lo is not None:
            # Explicit conversion LO (ME-SECAM): trim it by the smoothed
            # measured carrier offset (cancelling the recording VCR's
            # converter crystal error), and keep the heterodyne phase
            # continuous across fields.
            lo_trim = 0.0
            # Holds either live servo measurements or a seeded/fixed trim
            # (secam_lo_trim); with the servo disabled and no seed it's empty.
            if field.rf.secam_servo_avg.has_values():
                # Quantize so measurement noise doesn't dither the LO.
                lo_trim = np.clip(
                    round(field.rf.secam_servo_avg.pull() / 10.0) * 10.0,
                    -10e3,
                    10e3,
                )
            field.rf.chroma_afc.updateConversion(
                lo_trim, field.field_number * linesout * outwidth
            )
            chroma_heterodyne = field.rf.chroma_afc.getChromaHet()
        else:
            chroma_heterodyne = (
                field.rf.chroma_afc.getChromaHet()
                if (field.rf.do_cafc and not disable_tracking_cafc)
                else field.rf.chroma_heterodyne
            )

        uphet = np.zeros((linesout * outwidth), dtype=np.float32)
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
    if field.rf.color_system == "MESECAM":
        # The restored SECAM FM block is anchored at conversion_lo -
        # color_under (4.328125 MHz), not fsc, so the fsc-anchored FFT mask
        # sits ~106 kHz high on it and loses the tight top edge that
        # suppresses high-side FM splatter from saturated transitions. Keep
        # the block-anchored Butterworth here.
        uphet = sosfiltfilt_rust(field.rf.Filters["FChromaFinal"], uphet)
    else:
        uphet = filter_chroma_fft(
            uphet,
            field.rf.SysParams["fsc_mhz"] * 1e6,
            field.rf.DecoderParams["color_under_carrier"],
            1.3e6, # lower chroma bandwidth (roughly this for PAL / NTSC)
            80.0   # heterodyne up-mixing attenuation
        )

    if do_chroma_deemphasis:
        b, a = field.rf.Filters["chroma_deemphasis"]
        uphet = sps.lfilter(b, a, uphet)

    # Basic comb filter for NTSC to calm the color a little.
    if not disable_comb:
        if field.rf.color_system == "NTSC":
            uphet = comb_c_ntsc(uphet, outwidth)
        else:
            uphet = comb_c_pal(uphet, outwidth)

    # Chroma AGC
    mean_rms, chroma_noise_floor = chroma_automatic_gain(
        uphet,
        field.rf.SysParams["burst_abs_ref"],
        field.phase_sequence,
        field.burst_detected_line,
        math.floor(field.usectooutpx(field.rf.SysParams["hsyncPulseUS"]))
    )

    field.rf.field_averages.chroma_level.push(mean_rms)

    if field.rf.options.cti_mix != 0:
        chroma_transient_improvement(
            uphet,
            lineoffset * outwidth,
            outwidth,
            chroma_noise_floor,
            field.rf.options.cti_width,
            field.rf.options.cti_mix,
        )

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
