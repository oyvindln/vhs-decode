"""Module containing parameters for VHS and SVHS"""

# Frequency of SECAM color component carriers.
# https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
# Not sure if these should be left here or not
SECAM_FOR = 4406250
SECAM_FOB = 4250000


def fill_rfparams_vhs_shared(rfparams):
    """Fill in parameters that are shared between systems for VHS"""

    # PAL and NTSC uses the same main de-emphasis
    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    rfparams["deemph_mid"] = 260000
    rfparams["deemph_gain"] = 14

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    rfparams["nonlinear_highpass_freq"] = 600000
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = -20000


def fill_rfparams_svhs_shared(rfparams):
    """Fill in parameters that are shared between systems for Super VHS
    SVHS uses the same luma frequencies for NTSC and PAL
    """
    # 5.4-7.0 ± 0.1 mhz
    rfparams["video_bpf_low"] = 2000000
    rfparams["video_bpf_high"] = 8980000

    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 1

    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 9210000
    rfparams["video_lpf_extra_order"] = 3

    rfparams["video_hpf_extra"] = 1520000
    rfparams["video_hpf_extra_order"] = 3

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 6500000
    rfparams["video_lpf_order"] = 1

    rfparams["boost_bpf_low"] = 7000000
    rfparams["boost_bpf_high"] = 8400000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = 1.1


def get_rfparams_pal_vhs(rfparams_pal):
    """Get RF params for PAL VHS"""

    RFParams_PAL_VHS = {**rfparams_pal}

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    RFParams_PAL_VHS["video_bpf_low"] = 2500000
    RFParams_PAL_VHS["video_bpf_high"] = 5680000
    # Band-pass filter order.
    # Order may be fine as is.
    RFParams_PAL_VHS["video_bpf_order"] = 1
    # Sharper upper cutoff to get rid of high-frequency junk.
    RFParams_PAL_VHS["video_lpf_extra"] = 6010000
    RFParams_PAL_VHS["video_lpf_extra_order"] = 3

    RFParams_PAL_VHS["video_hpf_extra"] = 1520000
    RFParams_PAL_VHS["video_hpf_extra_order"] = 1

    # Low-pass filter on Y after demodulation
    RFParams_PAL_VHS["video_lpf_freq"] = 3500000
    RFParams_PAL_VHS["video_lpf_order"] = 1

    # PAL color under carrier is 40H + 1953
    RFParams_PAL_VHS["color_under_carrier"] = ((625 * 25) * 40) + 1953

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    RFParams_PAL_VHS["chroma_bpf_upper"] = 1200000

    # Video EQ after FM demod (PAL VHS)
    RFParams_PAL_VHS["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    # Video Y FM de-emphasis (1.25~1.35µs)
    RFParams_PAL_VHS["deemph_tau"] = 1.30e-6

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    # RFParams_PAL_VHS["deemph_mid"] = 260000
    # RFParams_PAL_VHS["deemph_gain"] = 14

    # Filter to pull out high frequencies for high frequency boost
    # This should cover the area around reference white.
    # Used to reduce streaks due to amplitude loss on phase change around
    # sharp transitions.
    RFParams_PAL_VHS["boost_bpf_low"] = 4200000
    RFParams_PAL_VHS["boost_bpf_high"] = 5600000
    # Multiplier for the boosted signal to add in.
    RFParams_PAL_VHS["boost_bpf_mult"] = 2

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    # RFParams_PAL_VHS["nonlinear_highpass_freq"] = 600000
    # RFParams_PAL_VHS["nonlinear_highpass_limit_h"] = 5000
    # RFParams_PAL_VHS["nonlinear_highpass_limit_l"] = -20000

    fill_rfparams_vhs_shared(RFParams_PAL_VHS)

    return RFParams_PAL_VHS


def get_sysparams_pal_vhs(sysparams_pal):
    """Get system params for PAL VHS"""

    SysParams_PAL_VHS = {**sysparams_pal}
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_PAL_VHS["hz_ire"] = 1e6 / (100 + (-SysParams_PAL_VHS["vsync_ire"]))

    # 0 IRE level after demodulation
    SysParams_PAL_VHS["ire0"] = 4.8e6 - (SysParams_PAL_VHS["hz_ire"] * 100)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    # This has to be low enough to avoid clipping, so we have to
    # tell the chroma decoder to boost it by a bit afterwards.
    SysParams_PAL_VHS["burst_abs_ref"] = 5000

    return SysParams_PAL_VHS


def get_rfparams_pal_svhs(sysparams_pal):
    """Get RF params for PAL SVHS"""
    # Super-VHS

    RFParams_PAL_SVHS = get_rfparams_pal_vhs(sysparams_pal)

    fill_rfparams_svhs_shared(RFParams_PAL_SVHS)

    RFParams_PAL_SVHS["deemph_mid"] = 260000
    RFParams_PAL_SVHS["deemph_gain"] = 10

    RFParams_PAL_SVHS["nonlinear_highpass_freq"] = 500000
    RFParams_PAL_SVHS["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_SVHS["nonlinear_highpass_limit_l"] = -250000

    # Main deemph and chroma is the same as for normal VHS

    return RFParams_PAL_SVHS


def get_sysparams_pal_svhs(sysparams_pal):
    SysParams_PAL_SVHS = get_sysparams_pal_vhs(sysparams_pal)

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_PAL_SVHS["hz_ire"] = 1.6e6 / (100 + (-SysParams_PAL_SVHS["vsync_ire"]))

    # 0 IRE level after demodulation
    SysParams_PAL_SVHS["ire0"] = 7e6 - (SysParams_PAL_SVHS["hz_ire"] * 100)

    return SysParams_PAL_SVHS


def get_rfparams_ntsc_vhs(rfparams_ntsc):
    """Get RF params for NTSC VHS"""

    RFParams_NTSC_VHS = {**rfparams_ntsc}

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    RFParams_NTSC_VHS["video_bpf_low"] = 2600000
    RFParams_NTSC_VHS["video_bpf_high"] = 5300000

    RFParams_NTSC_VHS["video_bpf_order"] = 1

    RFParams_NTSC_VHS["video_lpf_extra"] = 6080000
    RFParams_NTSC_VHS["video_lpf_extra_order"] = 3

    RFParams_NTSC_VHS["video_hpf_extra"] = 1300000
    RFParams_NTSC_VHS["video_hpf_extra_order"] = 2

    # Low-pass filter on Y after demodulation
    RFParams_NTSC_VHS["video_lpf_freq"] = 3000000

    # Order may be fine as is.
    RFParams_NTSC_VHS["video_lpf_order"] = 1

    # NTSC color under carrier is 40H
    RFParams_NTSC_VHS["color_under_carrier"] = (525 * (30 / 1.001)) * 40

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    RFParams_NTSC_VHS["chroma_bpf_upper"] = 1400000

    RFParams_NTSC_VHS["luma_carrier"] = 455.0 * ((525 * (30 / 1.001)) / 2.0)

    # Video EQ after FM demod (NTSC VHS)
    RFParams_NTSC_VHS["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 4},
    }

    # Video Y FM de-emphasis (1.25~1.35µs)
    RFParams_NTSC_VHS["deemph_tau"] = 1.30e-6

    RFParams_NTSC_VHS["boost_bpf_low"] = 4100000
    RFParams_NTSC_VHS["boost_bpf_high"] = 5000000
    RFParams_NTSC_VHS["boost_bpf_mult"] = 1

    fill_rfparams_vhs_shared(RFParams_NTSC_VHS)

    return RFParams_NTSC_VHS


def get_sysparams_ntsc_vhs(sysparams_ntsc):
    SysParams_NTSC_VHS = {**sysparams_ntsc}

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_VHS["hz_ire"] = 1e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_VHS["ire0"] = 4.4e6 - (SysParams_NTSC_VHS["hz_ire"] * 100)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    SysParams_NTSC_VHS["burst_abs_ref"] = 3200

    return SysParams_NTSC_VHS


def get_rfparams_ntsc_svhs(rfparams_ntsc):
    RFParams_NTSC_SVHS = get_rfparams_ntsc_vhs(rfparams_ntsc)

    # PAL and NTSC use much of the same values for SVHS.
    fill_rfparams_svhs_shared(RFParams_NTSC_SVHS)

    return RFParams_NTSC_SVHS


def get_sysparams_ntsc_svhs(sysparams_ntsc):

    SysParams_NTSC_SVHS = get_sysparams_ntsc_vhs(sysparams_ntsc)

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_SVHS["hz_ire"] = 1.6e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_SVHS["ire0"] = 7e6 - (SysParams_NTSC_SVHS["hz_ire"] * 100)

    return SysParams_NTSC_SVHS


def get_rfparams_mpal_vhs(rfparams_ntsc):
    params = get_rfparams_ntsc_vhs(rfparams_ntsc)
    # Same as NTSC other than color carrier
    params["color_under_carrier"] = 631.337e3

    return params


def get_sysparams_mpal_vhs(sysparams_ntsc):
    from lddecode.core import calclinelen

    params = get_sysparams_ntsc_vhs(sysparams_ntsc)
    # PAL-M sysparams override (From JVC Video technical guide)
    # Slightly different frequency from NTSC
    params["fsc_mhz"] = 3.575611888111
    params["fieldPhases"] = 8

    # Should be the same as NTSC in practice
    params["line_period"] = 1 / (params["fsc_mhz"] / (909 / 4.0))
    params["activeVideoUS"] = (9.45, params["line_period"] - 1.0)
    # SysParams_NTSC["FPS"] = 1000000 / (525 * params["line_period"])

    params["outlinelen"] = calclinelen(params, 4, "fsc_mhz")
    params["outfreq"] = 4 * params["fsc_mhz"]
    params["burst_abs_ref"] = 5000

    return params


def get_rfparams_mesecam_vhs(rfparams_pal):
    params = get_rfparams_pal_vhs(rfparams_pal)

    # Average of the two carriers specified, need to check if this works correctly
    # and calculate exact value
    params["color_under_carrier"] = (654300 + 810500) / 2

    return params


def get_sysparams_mesecam_vhs(sysparams_pal):
    """Get system params for MESECAM VHS"""

    # This will be the same as PAL other than chroma
    sysparams = get_sysparams_pal_vhs(sysparams_pal)

    # FSC specified in Panasonic NV-F55/95 handbook, need to check if this is correct
    # This differs from normal SECAM, possibly it's done like this to put the upconverted
    # subcarriers at the correct frequencies.
    # TODO: Needs testing
    # This is one of the secam freqs - not sure if it's
    # correct to use this or if PAL one should be used instead.
    sysparams["fsc_mhz"] = 4.406

    return sysparams


def get_rfparams_secam_vhs(rfparams_pal):
    params = get_rfparams_pal_vhs(rfparams_pal)

    # SECAM carriers on tape
    # TODO: need accurate freq.
    SECAM_FCB_TAPE = SECAM_FOB / 4
    SECAM_FCR_TAPE = SECAM_FOR / 4

    # Average of the two carriers specified, need to check if this works correctly
    # and calculate exact value
    params["color_under_carrier"] = (SECAM_FCB_TAPE + SECAM_FCR_TAPE) / 2

    return params


def get_sysparams_secam_vhs(sysparams_pal):
    """Get system params for SECAM VHS"""
    # from lddecode.core import calclinelen

    # This will be the same as PAL other than chroma
    sysparams = get_sysparams_pal_vhs(sysparams_pal)

    # Using pal sample rate for output for now.

    # Using average of carriers as FSC, may or may not be correct.
    # sysparams["fsc_mhz"] = ((SECAM_FOR + SECAM_FOB) / 2) / 1e6

    # sysparams["outlinelen"] = calclinelen(sysparams, 4, "fsc_mhz")
    # sysparams["outfreq"] = 4 * sysparams["fsc_mhz"]

    return sysparams
