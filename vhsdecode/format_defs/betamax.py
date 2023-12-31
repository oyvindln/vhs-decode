def get_rfparams_pal_betamax(rfparams_pal):
    """Get RF params for PAL VHS"""

    rfparams = {**rfparams_pal}

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking, this is a bit random as of now.
    rfparams["video_bpf_low"] = 1900000
    rfparams["video_bpf_high"] = 6100000
    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 2
    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 6550000
    rfparams["video_lpf_extra_order"] = 2

    rfparams["video_hpf_extra"] = 1320000
    rfparams["video_hpf_extra_order"] = 2

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 3500000
    rfparams["video_lpf_order"] = 1

    # PAL color under carrier is ??
    # TODO: Need exact freq
    # PAL Betamax uses different freq on track A/B to achieve 90 degree offset between tracks
    # See Television UK June 1980
    # https://worldradiohistory.com/UK/Practical/Television/80s/Television-Servicing-UK-1980-06.pdf
    trackA_freq = 685546.88
    trackB_freq = 689453.12
    # rfparams["color_under_carrier"] = (trackA_freq, trackB_freq, (trackA_freq + trackB_freq) / 2)
    rfparams["color_under_carrier"] = (trackA_freq + trackB_freq) / 2

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    rfparams["chroma_bpf_upper"] = 1300000

    # Video EQ after FM demod (PAL VHS)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    # Video Y FM de-emphasis (1.25~1.35µs)
    rfparams["deemph_tau"] = 1.30e-6

    # Filter to pull out high frequencies for high frequency boost
    # This should cover the area around reference white.
    # Used to reduce streaks due to amplitude loss on phase change around
    # sharp transitions.
    rfparams["boost_bpf_low"] = 4450000
    rfparams["boost_bpf_high"] = 5700000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = 1

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    # rfparams["deemph_mid"] = 330000
    # rfparams["deemph_gain"] = 13.5
    rfparams["deemph_mid"] = 330000
    rfparams["deemph_gain"] = 12.5

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    rfparams["nonlinear_highpass_freq"] = 662300
    rfparams["nonlinear_bandpass_upper"] = 4.5e6
    rfparams["nonlinear_exp_scaling"] = 0.25
    rfparams["nonlinear_scaling_1"] = 0.7
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = -20000

    # Phase rotation applied to chroma signal on each hsync for each track.
    # Not used on PAL Betamax
    rfparams["chroma_rotation"] = [0, 0]

    return rfparams


def get_sysparams_pal_betamax(sysparams_pal):
    """Get system params for PAL Betamax"""

    sysparams = {**sysparams_pal}
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    sysparams["hz_ire"] = 1.4e6 / (100 + (-sysparams["vsync_ire"]))

    # 0 IRE level after demodulation
    sysparams["ire0"] = 5.2e6 - (sysparams["hz_ire"] * 100)

    # Beta black level 3.8, tip 5.2 acc to telev. mag 1983 09
    # Also half-shift (fl/2) between a/b track

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    # This has to be low enough to avoid clipping, so we have to
    # tell the chroma decoder to boost it by a bit afterwards.
    sysparams["burst_abs_ref"] = 5000

    sysparams["nonlinear_deviation"] = sysparams["hz_ire"] * (
        100 + (-sysparams["vsync_ire"])
    )

    return sysparams


def _fill_rfparams_ntsc_betamax_common(rfparams):
    """Get RF params for NTSC Betamax"""

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking, this is a bit random as of now.
    rfparams["video_bpf_low"] = 1600000
    rfparams["video_bpf_high"] = 5380000
    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 1
    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 5910000
    rfparams["video_lpf_extra_order"] = 3

    rfparams["video_hpf_extra"] = 1320000
    rfparams["video_hpf_extra_order"] = 1

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 3000000
    rfparams["video_lpf_order"] = 1

    # NTSC betamax uses phase inversion on every other field on one track rather than
    # different frequencies like on PAL
    rfparams["color_under_carrier"] = 43.75 * (525 * (30 / 1.001))

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    rfparams["chroma_bpf_upper"] = 1400000

    # Video EQ after FM demod (PAL VHS)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    # Video Y FM de-emphasis (1.25~1.35µs)
    rfparams["deemph_tau"] = 1.30e-6

    # Filter to pull out high frequencies for high frequency boost
    # This should cover the area around reference white.
    # Used to reduce streaks due to amplitude loss on phase change around
    # sharp transitions.
    rfparams["boost_bpf_low"] = 4300000
    rfparams["boost_bpf_high"] = 5600000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = 1

    """Fill in parameters that are shared between systems for betamax"""

    # Not sure if this is the same between PAL/NTSC

    # Temporary video emphasis filter constants
    # Needs to be verified
    # NOT CORRECT!!
    rfparams["deemph_mid"] = 250000
    rfparams["deemph_gain"] = 12.5

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    rfparams["nonlinear_highpass_freq"] = 600000
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = -20000

    # Phase rotation applied (in 90 degree / Pi/2 radian steps) to chroma signal on each hsync for each track.
    rfparams["chroma_rotation"] = [2, 0]


def get_rfparams_ntsc_betamax(rfparams_ntsc):
    rfparams = {**rfparams_ntsc}
    _fill_rfparams_ntsc_betamax_common(rfparams)

    return rfparams


def get_rfparams_ntsc_betamax_hifi(rfparams_ntsc):
    rfparams = {**rfparams_ntsc}
    _fill_rfparams_ntsc_betamax_common(rfparams)

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking, this is a bit random as of now.
    rfparams["video_bpf_low"] = 2000000
    rfparams["video_bpf_high"] = 5880000
    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = 1
    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 6010000
    rfparams["video_lpf_extra_order"] = 3

    rfparams["video_hpf_extra"] = 1520000
    rfparams["video_hpf_extra_order"] = 1

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 3000000
    rfparams["video_lpf_order"] = 1

    return rfparams


def get_sysparams_ntsc_betamax(sysparams_ntsc):
    """Get system params for NTSC VHS"""

    # Need to check f it's the same as PAL or not

    sysparams = {**sysparams_ntsc}
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    sysparams["hz_ire"] = 1.2e6 / (100 + (-sysparams["vsync_ire"]))

    # 0 IRE level after demodulation
    sysparams["ire0"] = 4.8e6 - (sysparams["hz_ire"] * 100)

    # Beta black level 3.8, tip 5.2 acc to telev. mag 1983 09
    # Also half-shift (fl/2) between a/b track

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    # This has to be low enough to avoid clipping, so we have to
    # tell the chroma decoder to boost it by a bit afterwards.
    sysparams["burst_abs_ref"] = 4000

    return sysparams


def get_sysparams_ntsc_betamax_hifi(sysparams_ntsc):
    """Get system params for NTSC VHS"""

    # Need to check f it's the same as PAL or not

    sysparams = {**sysparams_ntsc}
    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    sysparams["hz_ire"] = 1.4e6 / (100 + (-sysparams["vsync_ire"]))

    # 0 IRE level after demodulation
    sysparams["ire0"] = 5.2e6 - (sysparams["hz_ire"] * 100)

    # Beta black level 3.8, tip 5.2 acc to telev. mag 1983 09
    # Also half-shift (fl/2) between a/b track

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    # This has to be low enough to avoid clipping, so we have to
    # tell the chroma decoder to boost it by a bit afterwards.
    sysparams["burst_abs_ref"] = 4000

    return sysparams
