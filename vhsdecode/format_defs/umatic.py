def get_rfparams_pal_umatic(RFParams_PAL):
    RFParams_PAL_UMATIC = {**RFParams_PAL}
    # UMATIC section
    # These need tweaking.
    RFParams_PAL_UMATIC["video_bpf_low"] = 1400000
    RFParams_PAL_UMATIC["video_bpf_high"] = 7000000
    RFParams_PAL_UMATIC["video_bpf_order"] = None
    RFParams_PAL_UMATIC["video_lpf_extra"] = 6800000
    RFParams_PAL_UMATIC["video_lpf_extra_order"] = 14
    RFParams_PAL_UMATIC["video_hpf_extra"] = 1400000
    RFParams_PAL_UMATIC["video_hpf_extra_order"] = 14
    RFParams_PAL_UMATIC["video_lpf_freq"] = 4200000
    RFParams_PAL_UMATIC["video_lpf_order"] = 6
    # 685546 ± 200
    RFParams_PAL_UMATIC["color_under_carrier"] = (625 * 25) * (351 / 8)
    RFParams_PAL_UMATIC["chroma_bpf_upper"] = 1500000

    # Video EQ after FM demod (PAL UMATIC) (based on NTSC one, needs tweak)
    RFParams_PAL_UMATIC["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis (550 ~ 650ns)
    RFParams_PAL_UMATIC["deemph_tau"] = 600e-9

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    RFParams_PAL_UMATIC["deemph_mid"] = 500000
    RFParams_PAL_UMATIC["deemph_gain"] = 10.8

    # Use linear ramp to boost RF
    RFParams_PAL_UMATIC["boost_rf_linear_0"] = 0.5
    RFParams_PAL_UMATIC["boost_rf_linear_20"] = 3

    # This has not really been stress-tested due to lack of crummy umatic samples.
    RFParams_PAL_UMATIC["boost_bpf_low"] = 5000000
    RFParams_PAL_UMATIC["boost_bpf_high"] = 5800000
    RFParams_PAL_UMATIC["boost_bpf_mult"] = 1

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_UMATIC["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_UMATIC["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_UMATIC["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_UMATIC


def get_sysparams_pal_umatic(sysparams_pal):
    SysParams_PAL_UMATIC = {**sysparams_pal}

    # PAL and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    SysParams_PAL_UMATIC["ire0"] = 4257143
    SysParams_PAL_UMATIC["hz_ire"] = 1600000 / 140.0
    SysParams_PAL_UMATIC["burst_abs_ref"] = 5000

    return SysParams_PAL_UMATIC


def get_rfparams_pal_umatic_hi(RFParams_PAL):
    RFParams_PAL_UMATIC = {**RFParams_PAL}
    # UMATIC section
    # These need tweaking.
    RFParams_PAL_UMATIC["video_bpf_low"] = 2500000
    RFParams_PAL_UMATIC["video_bpf_high"] = 7000000
    RFParams_PAL_UMATIC["video_bpf_order"] = 1
    RFParams_PAL_UMATIC["video_lpf_extra"] = 7200000
    RFParams_PAL_UMATIC["video_lpf_extra_order"] = 3
    RFParams_PAL_UMATIC["video_hpf_extra"] = 1500000
    RFParams_PAL_UMATIC["video_hpf_extra_order"] = 8
    RFParams_PAL_UMATIC["video_lpf_freq"] = 4200000
    RFParams_PAL_UMATIC["video_lpf_order"] = 6
    # 923828 ± x00
    RFParams_PAL_UMATIC["color_under_carrier"] = 923828  # (625 * 25) * (351 / 8)
    RFParams_PAL_UMATIC["chroma_bpf_upper"] = 1300000

    # Video EQ after FM demod (PAL UMATIC) (based on NTSC one, needs tweak)
    RFParams_PAL_UMATIC["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis (795 ~ 805ns)
    RFParams_PAL_UMATIC["deemph_tau"] = 800e-9

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    RFParams_PAL_UMATIC["deemph_mid"] = 500000
    RFParams_PAL_UMATIC["deemph_gain"] = 10.8

    # This has not really been stress-tested due to lack of crummy umatic samples.
    RFParams_PAL_UMATIC["boost_bpf_low"] = 6000000
    RFParams_PAL_UMATIC["boost_bpf_high"] = 6800000
    RFParams_PAL_UMATIC["boost_bpf_mult"] = 0

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_UMATIC["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_UMATIC["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_UMATIC["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_UMATIC


def get_sysparams_pal_umatic_hi(sysparams_pal):
    SysParams_PAL_UMATIC = {**sysparams_pal}

    # PAL and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    SysParams_PAL_UMATIC["hz_ire"] = 1600000 / 140.0
    SysParams_PAL_UMATIC["ire0"] = 6.4e6 - (SysParams_PAL_UMATIC["hz_ire"] * 100)
    SysParams_PAL_UMATIC["burst_abs_ref"] = 5000

    return SysParams_PAL_UMATIC


# def get_rfparams_umatic_hi(RFParams_PAL):
#     RFParams_PAL_UMATIC_HI["video_bpf_low"] = 0
#     RFParams_PAL_UMATIC_HI["video_bpf_high"] = 0
#     RFParams_PAL_UMATIC_HI["video_bpf_order"] = 0
#     RFParams_PAL_UMATIC_HI["video_lpf_extra"] = 0
#     RFParams_PAL_UMATIC_HI["video_lpf_extra_order"] = 0
#     RFParams_PAL_UMATIC_HI["video_hpf_extra"] = 0
#     RFParams_PAL_UMATIC_HI["video_hpf_extra_order"] = 0
#     RFParams_PAL_UMATIC_HI["video_lpf_freq"] = 0
#     RFParams_PAL_UMATIC_HI["video_lpf_order"] = 0
#     RFParams_PAL_UMATIC_HI["color_under_carrier"] = 983803
#     RFParams_PAL_UMATIC_HI["chroma_bpf_upper"] = 0


def get_rfparams_ntsc_umatic(rfparams_ntsc):
    rfparams = {**rfparams_ntsc}

    rfparams["video_bpf_low"] = 1400000
    rfparams["video_bpf_high"] = 6500000
    rfparams["video_bpf_order"] = None
    rfparams["video_lpf_extra"] = 7000000
    rfparams["video_lpf_extra_order"] = 16
    rfparams["video_hpf_extra"] = 1400000
    rfparams["video_hpf_extra_order"] = 14
    rfparams["video_lpf_freq"] = 4000000
    rfparams["video_lpf_order"] = 6
    #  688374 ± 200
    # (525 * (30 / 1.001)) * (175/4)
    rfparams["color_under_carrier"] = (525 * (30 / 1.001)) * (175 / 4)
    rfparams["chroma_bpf_upper"] = 1500000

    # Video EQ after FM demod (NTSC UMATIC) (needs tweak)
    rfparams["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
    }

    # Video Y FM de-emphasis (550 ~ 650ns)
    rfparams["deemph_tau"] = 600e-9

    rfparams["deemph_mid"] = 500000
    rfparams["deemph_gain"] = 10.8

    # Use linear ramp to boost RF
    rfparams["boost_rf_linear_0"] = 0
    rfparams["boost_rf_linear_20"] = 1
    rfparams["boost_rf_linear_double"] = False

    # This has not really been stress-tested due to lack of crummy umatic samples.
    rfparams["boost_bpf_low"] = 5000000
    rfparams["boost_bpf_high"] = 5800000
    rfparams["boost_bpf_mult"] = None

    rfparams["video_rf_peak_freq"] = 4600000
    rfparams["video_rf_peak_gain"] = 4
    rfparams["video_rf_peak_bandwidth"] = 1.6e7

    # Needs to be tweaked, just using some random values for now.
    rfparams["nonlinear_highpass_freq"] = 1000000
    rfparams["nonlinear_highpass_limit_h"] = 5000
    rfparams["nonlinear_highpass_limit_l"] = 20000

    return rfparams


def get_sysparams_ntsc_umatic(sysparams_ntsc):
    SysParams_NTSC_UMATIC = {**sysparams_ntsc}

    SysParams_NTSC_UMATIC["ire0"] = 4257143
    SysParams_NTSC_UMATIC["hz_ire"] = 1600000 / 140.0
    SysParams_NTSC_UMATIC["burst_abs_ref"] = 2500

    return SysParams_NTSC_UMATIC
