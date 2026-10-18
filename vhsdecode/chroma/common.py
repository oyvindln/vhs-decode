"""Chroma utilities shared by every colour-under system: output scaling,
the colour-under filter, the chroma TBC step, and the per-field wrapper
that runs a colour system's chroma processing."""

import numpy as np
import scipy.signal as sps
from numba import njit

import lddecode.core as ldd
from vhsdecode.rust_utils import sosfiltfilt_rust


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


def tbc_chroma(field, shift_direction=0, disable_tracking_cafc=False):
    """Run TBC/downscale on the colour-under chroma, then the chroma AFC
    band-pass and carrier tracking if chroma AFC is enabled.

    shift_direction (-1, 0 or 1) shifts the chroma to reverse the group
    delay of the colour-under heterodyne filter.
    """
    # Cached if chroma process is run multiple times on one field due to track detection.
    if field.chroma_tbc_buffer is not None:
        return field.chroma_tbc_buffer

    # shift the chroma to reverse group delay caused by the color under heterodyne filter
    # this is dependent on color framing, and is disabled if color framing is disabled
    # TODO: shift amount may need tuning / needs validation
    chroma_subcarrier_delay_cycles = field.rf.SysParams['fsc_mhz'] * 1e6 / (2.0 * np.pi * field.rf.DecoderParams["color_under_carrier"])
    chroma_subcarrier_delay_samples = chroma_subcarrier_delay_cycles * 4
    chroma, _, _ = ldd.Field.downscale(field, channel="demod_burst", shift=chroma_subcarrier_delay_samples * shift_direction)

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

    field.rf.chroma_tbc_buffer = chroma
    field.chroma_tbc_buffer = chroma
    return chroma


def decode_chroma(field, process_chroma, **kwargs):
    """Run a colour system's process_chroma on the field and scale the result
    for output. Returns None if chroma output is disabled."""
    if not field.rf.options.write_chroma:
        return None

    field.chroma_tbc_buffer = None

    if field.burst_detected_line == -1:
        # skip chroma if the color killer is active for the whole field
        uphet = np.zeros((field.outlinecount * field.outlinelen), dtype=np.float32)
    else:
        uphet = process_chroma(field, **kwargs)

    field.uphet_temp = uphet
    # Release to avoid keeping this im memory - should do this in a cleaner manner.
    field.chroma_tbc_buffer = None
    return chroma_to_u16(uphet)
