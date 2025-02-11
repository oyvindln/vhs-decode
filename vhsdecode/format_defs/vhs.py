"""Module containing parameters for VHS and SVHS"""

PAL_ROTATION = [0, -1]
NTSC_ROTATION = [-1, 1]


def get_svhs_linear_subdeemphasis_filter(r0: float, r1: float, c1: float, r2: float, c2: float) -> tuple:
    """ Calculates the filter parameters for the linear part of the S-VHS sub-de-emphasis """

    # The calculation is based on this circuit. Values differ between SP/LP
    # and also between devices
    #
    #     │╲   Out             C1
    # In──┤+╲   │ ______   ┌───┤├───┐    ______  C2
    #     │  >──┴─┤ R0 ├───┤ ______ ├─┬──┤ R2 ├──┤├──▷ GND
    #  ┌──┤-╱     ‾‾‾‾‾‾   └─┤ R1 ├─┘ │  ‾‾‾‾‾‾
    #  │  │╱                 ‾‾‾‾‾‾   │
    #  └──────────────────────────────┘

    b2 = 1
    b1 = c2*r0+c1*r1+c2*r1+c2*r2
    b0 = c1*c2*r0*r1+c1*c2*r1*r2
    a2 = 1
    a1 = c1*r1+c2*r2
    a0 = c1*c2*r1*r2
    return ((b0, b1, b2), (a0, a1, a2))


def fill_rfparams_vhs_shared(rfparams: dict, tape_speed: int = 0) -> None:
    """Fill in parameters that are shared between systems for VHS"""

    # PAL and NTSC uses the same main de-emphasis
    # Temporary video emphasis filter constants
    # The values are calculated based on the VHS spec IEC 774-1 (1994), page 67
    # A first-order shelf filter is given with these parameters:
    #  RC time constant: 1.3µs
    #  resistor divider ratio: 4:1 (=> gain factor 5)

    rfparams["deemph_mid"] = 273755.82  # sqrt(gain_factor)/(2*pi*r*c)
    rfparams["deemph_gain"] = 13.9794  # 20*log10(gain_factor)
    rfparams["deemph_q"] = (
        0.462088186  # 1/sqrt(sqrt(gain_factor) + 1/sqrt(gain_factor) + 2)
    )

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    rfparams["nonlinear_highpass_freq"] = 820000
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = -20000

    rfparams["nonlinear_scaling_1"] = 0.1
    rfparams["nonlinear_exp_scaling"] = 0.12
    rfparams["use_sub_deemphasis"] = [False, True, True, True][tape_speed]


def fill_rfparams_svhs_shared(rfparams: dict, tape_speed: int = 0) -> None:
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

    rfparams["video_hpf_extra"] = 1720000
    rfparams["video_hpf_extra_order"] = 3

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 7500000
    rfparams["video_lpf_order"] = 6
    rfparams["video_lpf_supergauss"] = True

    if tape_speed == 0:
      # ab, aa = get_svhs_linear_subdeemphasis_filter(r0=470, r1=1000, c1= 82e-12, r2=3300, c2=270e-12) # panasonic
      # ab, aa = get_svhs_linear_subdeemphasis_filter(r0=330, r1= 560, c1=150e-12, r2=1800, c2=470e-12) # sony
        ab, aa = get_svhs_linear_subdeemphasis_filter(r0=370, r1= 560, c1= 82e-12, r2=1500, c2=680e-12) # optimized to recording on JVC HR-S7611
    else:
        ab, aa = get_svhs_linear_subdeemphasis_filter(r0=  0, r1=1500, c1=220e-12, r2=3300, c2=270e-12) # panasonic LP

    rfparams["video_custom_luma_filters"] = [
         {"type": "coefficients", "analog": True, "b": ab, "a": aa },
    ]

    rfparams["boost_bpf_low"] = 7000000
    rfparams["boost_bpf_high"] = 8400000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = None

    # Use linear ramp to boost RF
    rfparams["boost_rf_linear_0"] = 0.5
    rfparams["boost_rf_linear_20"] = 100
    rfparams["boost_rf_linear_double"] = False

    # SVHS uses the emphasis curve from VHS + an additional sub-emphasis filter
    # The latter isn't properly implemented yet but
    # adjusting the corner frequency here makes it look a bit closer
    # than just using the values for VHS, it needs to be properly
    # sorted though.
    # rfparams["deemph_mid"] = 335000 # the optimal value of this parameter is currently dependant on the recording devices
    # rfparams["deemph_gain"] = 14

    rfparams["nonlinear_highpass_freq"] = 320000
    rfparams["nonlinear_amp_lpf_freq"] = 590000
    rfparams["nonlinear_exp_scaling"] = 0.23
    rfparams["nonlinear_scaling_1"] = None
    rfparams["nonlinear_scaling_2"] = 0.72
    rfparams["nonlinear_logistic_mid"] = 0.2
    rfparams["nonlinear_logistic_rate"] = 14.0
    rfparams["use_sub_deemphasis"] = True


def get_rfparams_pal_vhs(rfparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get RF params for PAL VHS"""

    RFParams_PAL_VHS = {**rfparams_pal}

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    RFParams_PAL_VHS["video_bpf_low"] = 1300000
    RFParams_PAL_VHS["video_bpf_high"] = 5780000
    # Band-pass filter order.
    # Order may be fine as is.
    RFParams_PAL_VHS["video_bpf_order"] = None
    # Sharper upper cutoff to get rid of high-frequency junk.
    RFParams_PAL_VHS["video_lpf_extra"] = 5910000
    RFParams_PAL_VHS["video_lpf_extra_order"] = 20

    RFParams_PAL_VHS["video_bpf_supergauss"] = False

    RFParams_PAL_VHS["video_hpf_extra"] = 1320000
    RFParams_PAL_VHS["video_hpf_extra_order"] = 12

    # Low-pass filter on Y after demodulation
    RFParams_PAL_VHS["video_lpf_freq"] = 3400000
    RFParams_PAL_VHS["video_lpf_order"] = 6

    # PAL color under carrier is 40H + 1953
    RFParams_PAL_VHS["color_under_carrier"] = ((625 * 25) * 40) + 1953

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    # For vhs decks it's typically a bit more than 2x cc
    RFParams_PAL_VHS["chroma_bpf_upper"] = 1300000

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
    RFParams_PAL_VHS["boost_bpf_mult"] = None

    # Use linear ramp to boost RF
    # Lower number attenuates lower freqs more giving a "softer" look with less ringing but potentially less detail
    RFParams_PAL_VHS["boost_rf_linear_0"] = 0
    # This param doesn't really seem to matter.
    RFParams_PAL_VHS["boost_rf_linear_20"] = 1
    # Double up ramp filter to more closely mimic VHS EQ
    RFParams_PAL_VHS["boost_rf_linear_double"] = False

    RFParams_PAL_VHS["start_rf_linear"] = RFParams_PAL_VHS["color_under_carrier"]

    RFParams_PAL_VHS["video_rf_peak_freq"] = 4700000
    RFParams_PAL_VHS["video_rf_peak_gain"] = 4
    RFParams_PAL_VHS["video_rf_peak_bandwidth"] = 1.5e7

    # Parameters for high-pass filter used for non-linear deemphasis, these are
    # probably not correct.
    # RFParams_PAL_VHS["nonlinear_highpass_freq"] = 600000
    # RFParams_PAL_VHS["nonlinear_highpass_limit_h"] = 5000
    # RFParams_PAL_VHS["nonlinear_highpass_limit_l"] = -20000

    RFParams_PAL_VHS["chroma_rotation"] = PAL_ROTATION

    # Frequency of fm audio channels - used for notch filter
    # TODO: sync between hifi and vhs decode.
    RFParams_PAL_VHS["fm_audio_channel_0_freq"] = 1400000
    RFParams_PAL_VHS["fm_audio_channel_1_freq"] = 1800000

    fill_rfparams_vhs_shared(RFParams_PAL_VHS, tape_speed)

    return RFParams_PAL_VHS


def get_sysparams_pal_vhs(sysparams_pal: dict, tape_speed: int = 0) -> dict:
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


def get_rfparams_pal_svhs(sysparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get RF params for PAL SVHS"""
    # Super-VHS

    RFParams_PAL_SVHS = get_rfparams_pal_vhs(sysparams_pal, tape_speed)

    fill_rfparams_svhs_shared(RFParams_PAL_SVHS, tape_speed)

    # RFParams_PAL_SVHS["nonlinear_highpass_freq"] = 500000
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


def get_rfparams_ntsc_vhs(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    """Get RF params for NTSC VHS"""

    RFParams_NTSC_VHS = {**rfparams_ntsc}

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    RFParams_NTSC_VHS["video_bpf_low"] = 1000000
    RFParams_NTSC_VHS["video_bpf_high"] = 5400000

    RFParams_NTSC_VHS["video_bpf_order"] = 6
    RFParams_NTSC_VHS["video_bpf_supergauss"] = True

    RFParams_NTSC_VHS["video_lpf_extra"] = 5350000
    RFParams_NTSC_VHS["video_lpf_extra_order"] = 25

    RFParams_NTSC_VHS["video_hpf_extra"] = 500000
    RFParams_NTSC_VHS["video_hpf_extra_order"] = 20

    # Low-pass filter on Y after demodulation
    RFParams_NTSC_VHS["video_lpf_supergauss"] = True
    RFParams_NTSC_VHS["video_lpf_freq"] = 6600000
    RFParams_NTSC_VHS["video_lpf_order"] = 9

    # NTSC color under carrier is 40H
    RFParams_NTSC_VHS["color_under_carrier"] = (525 * (30 / 1.001)) * 40

    RFParams_NTSC_VHS["chroma_rotation"] = NTSC_ROTATION

    # Upper frequency of bandpass to filter out chroma from the rf signal.
    RFParams_NTSC_VHS["chroma_bpf_upper"] = 1200000

    RFParams_NTSC_VHS["luma_carrier"] = 455.0 * ((525 * (30 / 1.001)) / 2.0)

    # Video EQ after FM demod (NTSC VHS)
    RFParams_NTSC_VHS["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 4},
    }

    # Video Y FM de-emphasis (1.25~1.35µs)
    RFParams_NTSC_VHS["deemph_tau"] = 1.30e-6

    RFParams_NTSC_VHS["boost_bpf_low"] = 4500000
    RFParams_NTSC_VHS["boost_bpf_high"] = 5400000
    RFParams_NTSC_VHS["boost_bpf_mult"] = None

    # Use linear ramp to boost RF
    # Lower number attenuates lower freqs more giving a "softer" look with less ringing but potentially less detail
    RFParams_NTSC_VHS["boost_rf_linear_0"] = 0.21
    # This param doesn't really seem to matter.
    RFParams_NTSC_VHS["boost_rf_linear_20"] = 8
    # Double up ramp filter to more closely mimic VHS EQ
    RFParams_NTSC_VHS["boost_rf_linear_double"] = True

    # Frequency of fm audio channels - used for notch filter
    RFParams_NTSC_VHS["fm_audio_channel_0_freq"] = 1300000
    RFParams_NTSC_VHS["fm_audio_channel_1_freq"] = 1700000

    fill_rfparams_vhs_shared(RFParams_NTSC_VHS, tape_speed)

    return RFParams_NTSC_VHS


def get_sysparams_ntsc_vhs(sysparams_ntsc: dict, tape_speed: int = 0) -> dict:
    SysParams_NTSC_VHS = {**sysparams_ntsc}

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_VHS["hz_ire"] = 1e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_VHS["ire0"] = 4.4e6 - (SysParams_NTSC_VHS["hz_ire"] * 100)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
    SysParams_NTSC_VHS["burst_abs_ref"] = 3200

    return SysParams_NTSC_VHS


def get_rfparams_ntsc_svhs(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    RFParams_NTSC_SVHS = get_rfparams_ntsc_vhs(rfparams_ntsc, tape_speed)

    # PAL and NTSC use much of the same values for SVHS.
    fill_rfparams_svhs_shared(RFParams_NTSC_SVHS, tape_speed)

    return RFParams_NTSC_SVHS


def get_sysparams_ntsc_svhs(sysparams_ntsc):
    SysParams_NTSC_SVHS = get_sysparams_ntsc_vhs(sysparams_ntsc)

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_SVHS["hz_ire"] = 1.6e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_SVHS["ire0"] = 7e6 - (SysParams_NTSC_SVHS["hz_ire"] * 100)

    return SysParams_NTSC_SVHS


def get_rfparams_mpal_vhs(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    params = get_rfparams_ntsc_vhs(rfparams_ntsc, tape_speed)
    # Same as NTSC other than color carrier
    params["color_under_carrier"] = 631.337e3
    params["chroma_rotation"] = PAL_ROTATION

    return params


def get_sysparams_mpal_vhs(sysparams_ntsc: dict, tape_speed: int = 0) -> dict:
    from lddecode.core import calclinelen

    params = get_sysparams_ntsc_vhs(sysparams_ntsc, tape_speed)
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


def get_rfparams_mesecam_vhs(rfparams_pal: dict, tape_speed: int = 0) -> dict:
    params = get_rfparams_pal_vhs(rfparams_pal, tape_speed)

    # Average of the two carriers specified, need to check if this works correctly
    # and calculate exact value
    params["color_under_carrier"] = (654300 + 810500) / 2
    params["chroma_rotation"] = None

    return params


def get_sysparams_mesecam_vhs(sysparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get system params for MESECAM VHS"""

    # This will be the same as PAL other than chroma
    sysparams = get_sysparams_pal_vhs(sysparams_pal, tape_speed)

    # FSC specified in Panasonic NV-F55/95 handbook, need to check if this is correct
    # This differs from normal SECAM, possibly it's done like this to put the upconverted
    # subcarriers at the correct frequencies.
    # TODO: Needs testing
    sysparams["fsc_mhz"] = 4.406

    return sysparams
