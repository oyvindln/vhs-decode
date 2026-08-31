"""Module containing parameters for VHS and SVHS"""

PAL_ROTATION = [0, -1]
NTSC_ROTATION = [-1, 1]


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

    rfparams["nonlinear_scaling_1"] = [0.5, 0.1, 0.1, 0.1][tape_speed]
    rfparams["nonlinear_exp_scaling"] = [0.06, 0.12, 0.12, 0.12][tape_speed]
    rfparams["use_sub_deemphasis"] = [False, True, True, True][tape_speed]

    # Make sure these exist in the dict so they can be overridden
    rfparams["video_rf_peak_freq"] = None
    rfparams["video_rf_peak_gain"] = None
    rfparams["video_rf_peak_bandwidth"] = None


def fill_rfparams_svhs_shared(rfparams: dict) -> None:
    """Fill in parameters that are shared between systems for Super VHS
    SVHS uses the same luma frequencies for NTSC and PAL
    """
    # 5.4-7.0 ± 0.1 mhz
    # for standard SHVS white clip at 210% of sync tip, which would be 5.4 * (1.6 * 2.1), approx 8.76
    rfparams["video_bpf_low"] = 2000000
    rfparams["video_bpf_high"] = 8980000

    # Band-pass filter order.
    # Order may be fine as is.
    rfparams["video_bpf_order"] = None
    rfparams["video_bpf_supergauss"] = False

    # Sharper upper cutoff to get rid of high-frequency junk.
    rfparams["video_lpf_extra"] = 9100000
    rfparams["video_lpf_extra_order"] = 25

    rfparams["video_hpf_extra"] = 1300000
    rfparams["video_hpf_extra_order"] = 20

    # Low-pass filter on Y after demodulation
    rfparams["video_lpf_freq"] = 7500000
    rfparams["video_lpf_order"] = 6
    rfparams["video_lpf_supergauss"] = True

    rfparams["video_custom_luma_filters"] = [
        {"type": "file", "filename": "svhs-sp-linear-subdeemphasis"},
        {"type": "highshelf", "gain": 4.0, "midfreq": 2000000, "q": 0.4967045},
    ]

    rfparams["boost_bpf_low"] = 6700000
    rfparams["boost_bpf_high"] = 8000000
    # Multiplier for the boosted signal to add in.
    rfparams["boost_bpf_mult"] = None

    # Use linear ramp to boost RF
    rfparams["boost_rf_linear_0"] = 1
    rfparams["boost_rf_linear_20"] = 4
    rfparams["boost_rf_linear_double"] = False
    rfparams["start_rf_linear"] = rfparams["color_under_carrier"]

    # SVHS uses the emphasis curve from VHS + an additional sub-emphasis filter
    # The latter isn't properly implemented yet but
    # adjusting the corner frequency here makes it look a bit closer
    # than just using the values for VHS, it needs to be properly
    # sorted though.
    # rfparams["deemph_mid"] = 335000 # the optimal value of this parameter is currently dependant on the recording devices
    # rfparams["deemph_gain"] = 14

    rfparams["video_rf_peak_freq"] = 6200000
    rfparams["video_rf_peak_gain"] = 4
    rfparams["video_rf_peak_bandwidth"] = 1.6e7

    rfparams["nonlinear_highpass_freq"] = 320000
    rfparams["nonlinear_amp_lpf_freq"] = 590000
    rfparams["nonlinear_exp_scaling"] = 0.23
    rfparams["nonlinear_scaling_1"] = None
    rfparams["nonlinear_scaling_2"] = 0.72
    rfparams["nonlinear_logistic_mid"] = 0.2
    rfparams["nonlinear_logistic_rate"] = 14.0
    rfparams["use_sub_deemphasis"] = True


def get_rfparams_pal_svhs_et(rfparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get RF params for PAL SVHS"""
    # Super-VHS ET

    # use svhs as a starting point but adjust upper lpf as ET mode
    # has lower white clip
    rfparams = get_rfparams_pal_svhs(rfparams_pal)
    rfparams["video_lpf_extra"] = 8450000

    # TODO: add extra deemph

    return rfparams


def get_rfparams_pal_vhs(rfparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get RF params for PAL VHS"""

    RFParams_PAL_VHS = {**rfparams_pal}

    fill_rfparams_vhs_shared(RFParams_PAL_VHS, tape_speed)

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
    # The HR-D565 technical manual specifies a lpf with corner 3.3 mhz
    RFParams_PAL_VHS["video_lpf_freq"] = 3400000
    RFParams_PAL_VHS["video_lpf_order"] = 6

    # PAL color under carrier is 40H + 1953
    # Recorded video track width, micrometers. JVC VTG82063 gives PAL LP as
    # 0.024 mm at 11.70 mm/s against SP's 23.39; the exact ratio is 24.51.
    # PAL VHS has no EP speed, so LP repeats for any higher index.
    RFParams_PAL_VHS["video_track_width"] = [49.0, 24.5, 24.5][min(tape_speed, 2)]

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

    RFParams_PAL_VHS["video_rf_peak_freq"] = 4300000
    RFParams_PAL_VHS["video_rf_peak_gain"] = 4
    RFParams_PAL_VHS["video_rf_peak_bandwidth"] = 1.0e7

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


def get_rfparams_pal_svhs(sysparams_pal):
    """Get RF params for PAL SVHS"""
    # Super-VHS

    RFParams_PAL_SVHS = get_rfparams_pal_vhs(sysparams_pal)

    fill_rfparams_svhs_shared(RFParams_PAL_SVHS)

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

    # One track has an offset of f_h/2
    # SysParams_PAL_SVHS["track_ire0_offset"] = [7812.5, 0]

    return SysParams_PAL_SVHS


def get_sysparams_pal_vhshq(sysparams_pal: dict, tape_speed: int) -> dict:
    SysParams_PAL_VHSHQ = get_sysparams_pal_vhs(sysparams_pal)

    # One track has an offset of f_h/2
    SysParams_PAL_VHSHQ["track_ire0_offset"] = [7812.5, 0]

    return SysParams_PAL_VHSHQ


def get_rfparams_ntsc_vhs(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    """Get RF params for NTSC VHS"""

    RFParams_NTSC_VHS = {**rfparams_ntsc}

    fill_rfparams_vhs_shared(RFParams_NTSC_VHS, tape_speed)

    # Band-pass filter for Video rf.
    # TODO: Needs tweaking
    RFParams_NTSC_VHS["video_bpf_low"] = 500000
    RFParams_NTSC_VHS["video_bpf_high"] = [6500000, 6500000, 6400000][tape_speed]

    RFParams_NTSC_VHS["video_bpf_order"] = 8
    RFParams_NTSC_VHS["video_bpf_supergauss"] = False

    RFParams_NTSC_VHS["video_lpf_extra"] = [6000000, 5900000, 5800000][tape_speed]
    RFParams_NTSC_VHS["video_lpf_extra_order"] = 25

    RFParams_NTSC_VHS["video_hpf_extra"] = 1200000
    RFParams_NTSC_VHS["video_hpf_extra_order"] = 20

    # Low-pass filter on Y after demodulation
    RFParams_NTSC_VHS["video_lpf_supergauss"] = True
    RFParams_NTSC_VHS["video_lpf_freq"] = 6600000
    RFParams_NTSC_VHS["video_lpf_order"] = 9

    # Recorded video track width, micrometers. JVC VTG82063 section 1: 0.058 mm
    # at SP, 0.019 mm at EP. The drum turns at one rate whatever the tape speed,
    # so the pitch follows the tape speed and EP is 58 * 11.12 / 33.35 = 19.34.
    RFParams_NTSC_VHS["video_track_width"] = [58.0, 29.0, 19.3][tape_speed]

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
    RFParams_NTSC_VHS["boost_rf_linear_0"] = 0
    # This param doesn't really seem to matter.
    RFParams_NTSC_VHS["boost_rf_linear_20"] = 1
    # Double up ramp filter to more closely mimic VHS EQ
    RFParams_NTSC_VHS["boost_rf_linear_double"] = False
    RFParams_NTSC_VHS["start_rf_linear"] = RFParams_NTSC_VHS["color_under_carrier"]

    RFParams_NTSC_VHS["video_rf_peak_freq"] = 3900000
    RFParams_NTSC_VHS["video_rf_peak_gain"] = 4
    RFParams_NTSC_VHS["video_rf_peak_bandwidth"] = 1.0e7

    # Frequency of fm audio channels - used for notch filter
    RFParams_NTSC_VHS["fm_audio_channel_0_freq"] = 1300000
    RFParams_NTSC_VHS["fm_audio_channel_1_freq"] = 1700000

    return RFParams_NTSC_VHS


def get_sysparams_ntsc_vhs(sysparams_ntsc: dict, tape_speed: int = 0) -> dict:
    SysParams_NTSC_VHS = {**sysparams_ntsc}

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_VHS["hz_ire"] = 1e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_VHS["ire0"] = 4.4e6 - (SysParams_NTSC_VHS["hz_ire"] * 100)

    # Mean absolute value of color burst for Automatic Chroma Control.
    # The value is eyeballed to give ok chroma level as of now, needs to be tweaked.

    # TODO: LP value is doubled as a workaround for burst emp not being used in LP mode
    # this should be fixed properly by disabling burst deemp in LP mode
    burst_abs_ref = 5730
    SysParams_NTSC_VHS["burst_abs_ref"] = [burst_abs_ref, burst_abs_ref / 2, burst_abs_ref][tape_speed]

    return SysParams_NTSC_VHS


def get_rfparams_ntsc_svhs(rfparams_ntsc):
    RFParams_NTSC_SVHS = get_rfparams_ntsc_vhs(rfparams_ntsc)

    # PAL and NTSC use much of the same values for SVHS.
    fill_rfparams_svhs_shared(RFParams_NTSC_SVHS)

    return RFParams_NTSC_SVHS


def get_rfparams_ntsc_svhs_et(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    """Get RF params for PAL SVHS"""
    # Super-VHS ET

    # use svhs as a starting point but adjust upper lpf as ET mode
    # has lower white clip
    rfparams = get_rfparams_ntsc_svhs(rfparams_ntsc)
    rfparams["video_lpf_extra"] = 8450000

    # TODO: add extra deemph

    return rfparams


def get_sysparams_ntsc_svhs(sysparams_ntsc):
    SysParams_NTSC_SVHS = get_sysparams_ntsc_vhs(sysparams_ntsc)

    # frequency/ire IRE change pr frequency (Is this labeled correctly?)
    SysParams_NTSC_SVHS["hz_ire"] = 1.6e6 / 140

    # 0 IRE level after demodulation
    SysParams_NTSC_SVHS["ire0"] = 7e6 - (SysParams_NTSC_SVHS["hz_ire"] * 100)

    # One track has an offset of f_h/2
    # TODO: Test
    # SysParams_NTSC_SVHS["track_ire0_offset"] = [0, 7867]

    return SysParams_NTSC_SVHS


def get_sysparams_ntsc_vhshq(sysparams_ntsc: dict, tape_speed: int = 0) -> dict:
    SysParams_NTSC_VHSHQ = get_sysparams_ntsc_vhs(sysparams_ntsc, tape_speed)

    # One track has an offset of f_h/2
    # SysParams_NTSC_VHSHQ["track_ire0_offset"] = [0, 7867]

    return SysParams_NTSC_VHSHQ


def get_rfparams_palm_vhs(rfparams_ntsc: dict, tape_speed: int = 0) -> dict:
    params = get_rfparams_ntsc_vhs(rfparams_ntsc, tape_speed)
    # Same as NTSC other than color carrier
    params["color_under_carrier"] = 631.337e3
    params["chroma_rotation"] = PAL_ROTATION

    return params


def get_sysparams_palm_vhs(sysparams_ntsc: dict, tape_speed: int = 0) -> dict:
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

    # ME-SECAM records the SECAM FM chroma block the same way as PAL colour-under
    # (IEC 60774-1 Annex E / 6.3.1) but without carrier phase rotation and with the
    # APC disabled: an inverting mix against the PAL converter LO of
    # fsc + (40 * fh + 1953) = 4433618.75 + 626953.125 = 5060571.875 Hz.
    # That puts the rest carriers on tape at:
    #   D'r: 5060571.875 - 4406250 = 654321.875 Hz
    #   D'b: 5060571.875 - 4250000 = 810571.875 Hz
    # color_under_carrier is only used for band-pass centering; the up-conversion
    # mixes against chroma_conversion_lo so the restored carriers land back on the
    # studio frequencies (ITU-R BT.470: foR 4406250 Hz, foB 4250000 Hz).
    params["color_under_carrier"] = (654321.875 + 810571.875) / 2
    params["chroma_conversion_lo"] = 5060571.875
    params["chroma_rotation"] = None

    return params


def get_sysparams_mesecam_vhs(sysparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get system params for MESECAM VHS"""

    # Same as PAL, including fsc_mhz, so the TBC output stays at standard PAL 4fsc
    # and stays consistent with outlinelen (which is derived from the PAL fsc before
    # this function runs). The restored SECAM subcarriers are placed by
    # chroma_conversion_lo rather than by fsc + color under carrier.
    return get_sysparams_pal_vhs(sysparams_pal, tape_speed)


def get_rfparams_secam_vhs(rfparams_pal: dict, tape_speed: int = 0) -> dict:
    params = get_rfparams_pal_vhs(rfparams_pal, tape_speed)

    # SECAM method 1 (IEC 60774-1 6.4.1 / annex E figure E1): the studio SECAM
    # chroma block is separated with a band-pass centred on 4.32 MHz (-3 dB at
    # +-800 kHz) and counted down by 4 in a digital divider, so the rest
    # carriers on tape are exactly foB/4 = 1062500 Hz and foR/4 = 1101562.5 Hz
    # and the FM deviations are also divided by 4 (max ~+-126.5 kHz).
    # Playback restores the block by multiplying the carrier phase by 4 again;
    # there is no conversion crystal anywhere in the chain, so unlike ME-SECAM
    # there is no LO error to servo out, and tape timebase error scales the
    # deviation together with the carrier so it self-corrects (JVC video
    # technical guide 3.1.4).
    params["color_under_carrier"] = (1062500.0 + 1101562.5) / 2  # 1082031.25
    params["chroma_rotation"] = None
    params["chroma_carrier_mult"] = 4
    # The under carriers sit much higher than PAL colour-under, so raise the
    # take-off low edge to keep head-switch thump and low-frequency junk out
    # of the phase measurement. The upper edge (chroma_bpf_upper, 1.3 MHz)
    # stays below the luma FM lower sideband area and the 1.4 MHz FM audio
    # carrier.
    params["chroma_bpf_lower"] = 350000

    return params


def get_sysparams_secam_vhs(sysparams_pal: dict, tape_speed: int = 0) -> dict:
    """Get system params for SECAM (method 1) VHS"""

    # Same as PAL/ME-SECAM: the TBC output stays at standard PAL 4fsc. The
    # x4 phase multiplication puts the restored subcarriers on the studio
    # frequencies (ITU-R BT.470: foR 4406250 Hz, foB 4250000 Hz) by itself.
    return get_sysparams_pal_vhs(sysparams_pal, tape_speed)
