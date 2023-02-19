"""Module containing parameters for video8 and hi8"""


def fill_rfparams_video8_shared(rfparams):
    """Fill in parameters that are shared between systems for VHS"""

    # sync 4.2 mhz
    # peak white 5.4 mhz

    # ch1 is 1/2 fh higher than ch2

    # PAL and NTSC uses the same main de-emphasis
    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value
    # video8 uses same time constant as vhs, but lower 'x' value
    rfparams["deemph_mid"] = 260000
    rfparams["deemph_gain"] = 14

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    rfparams["nonlinear_highpass_freq"] = 600000
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = -20000

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    rfparams["video_bpf_low"] = 2100000
    rfparams["video_bpf_high"] = 6300000
    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 1
    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 6310000
    rfparams["video_lpf_extra_order"] = 3

    rfparams["video_hpf_extra"] = 1520000
    rfparams["video_hpf_extra_order"] = 1

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 3500000
    rfparams["video_lpf_order"] = 1

    # Video Y FM de-emphasis (1.25~1.35µs)
    rfparams["deemph_tau"] = 1.30e-6

    # Filter to pull out high frequencies for high frequency boost
    # This should cover the area around reference white.
    # Used to reduce streaks due to amplitude loss on phase change around
    # sharp transitions.
    rfparams["boost_bpf_low"] = 5200000
    rfparams["boost_bpf_high"] = 5700000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = 0


def fill_rfparams_hi8_shared(rfparams):
    """Fill in parameters that are shared between systems for VHS"""

    # sync 5.7 mhz
    # peak white 7.7 mhz

    # no half-shift?

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

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    rfparams["video_bpf_low"] = 2100000
    rfparams["video_bpf_high"] = 8300000
    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 1
    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 8810000
    rfparams["video_lpf_extra_order"] = 3

    rfparams["video_hpf_extra"] = 1520000
    rfparams["video_hpf_extra_order"] = 1

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 5500000
    rfparams["video_lpf_order"] = 1

    # Video Y FM de-emphasis (1.25~1.35µs)
    rfparams["deemph_tau"] = 1.30e-6

    # Filter to pull out high frequencies for high frequency boost
    # This should cover the area around reference white.
    # Used to reduce streaks due to amplitude loss on phase change around
    # sharp transitions.
    rfparams["boost_bpf_low"] = 7200000
    rfparams["boost_bpf_high"] = 7800000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = 0

    # Video EQ after FM demod (PAL VHS)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }


def fill_chroma_params_pal(rfparams):
    # PAL color under carrier is 46.875 fH
    rfparams["color_under_carrier"] = (625 * 25) * 46.875


def fill_chroma_params_ntsc(rfparams):
    # NTSC color under carrier is 47.25 fH
    rfparams["color_under_carrier"] = (525 * (30 / 1.001)) * 47.25


def fill_sysparams_video8_shared(sysparams):
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    sysparams["hz_ire"] = 1.2e6 / (100 + (-sysparams["vsync_ire"]))

    # 0 IRE level after demodulation
    sysparams["ire0"] = 5.2e6 - (sysparams["hz_ire"] * 100)


def fill_sysparams_hi8_shared(sysparams):
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    sysparams["hz_ire"] = 2e6 / (100 + (-sysparams["vsync_ire"]))

    # 0 IRE level after demodulation
    sysparams["ire0"] = 7.7e6 - (sysparams["hz_ire"] * 100)


def get_rfparams_pal_video8(rfparams_pal):
    """Get RF params for PAL video8"""

    rfparams = {**rfparams_pal}

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    rfparams["chroma_bpf_upper"] = 1200000

    # Video EQ after FM demod (PAL VHS)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    fill_chroma_params_pal(rfparams)
    fill_rfparams_video8_shared(rfparams)

    return rfparams


def get_sysparams_pal_video8(sysparams_pal):
    sysparams = {**sysparams_pal}

    fill_sysparams_video8_shared(sysparams_pal)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    sysparams["burst_abs_ref"] = 1750

    return sysparams


def get_rfparams_ntsc_video8(rfparams_ntsc):
    """Get RF params for PAL video8"""

    rfparams = {**rfparams_ntsc}

    fill_chroma_params_ntsc(rfparams)

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    rfparams["chroma_bpf_upper"] = 1200000

    # Video EQ after FM demod (PAL VHS)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    fill_rfparams_video8_shared(rfparams)

    return rfparams


def get_sysparams_ntsc_video8(sysparams_ntsc):
    sysparams = {**sysparams_ntsc}

    fill_sysparams_video8_shared(sysparams)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    sysparams["burst_abs_ref"] = 1750

    return sysparams


def get_rfparams_pal_hi8(rfparams_pal):
    rfparams = {**rfparams_pal}

    fill_chroma_params_pal(rfparams)
    fill_rfparams_hi8_shared(rfparams)

    return rfparams


def get_rfparams_ntsc_hi8(rfparams_ntsc):
    """Get RF params for PAL video8"""

    rfparams = {**rfparams_ntsc}

    fill_chroma_params_ntsc(rfparams)

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    rfparams["chroma_bpf_upper"] = 1200000

    fill_rfparams_hi8_shared(rfparams)

    return rfparams


def get_sysparams_ntsc_hi8(sysparams_ntsc):
    sysparams = {**sysparams_ntsc}

    fill_sysparams_hi8_shared(sysparams)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    sysparams["burst_abs_ref"] = 1750

    return sysparams
