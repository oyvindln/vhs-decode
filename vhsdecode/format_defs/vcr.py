def get_rfparams_pal_vcr(RFParams_PAL):
    RFParams_PAL_VCR = {**RFParams_PAL}
    # VCR section
    # These need tweaking.
    RFParams_PAL_VCR["video_bpf_low"] = 1400000
    RFParams_PAL_VCR["video_bpf_high"] = 5000000
    RFParams_PAL_VCR["video_bpf_order"] = 2
    RFParams_PAL_VCR["video_lpf_extra"] = 5500000
    RFParams_PAL_VCR["video_lpf_extra_order"] = 3
    RFParams_PAL_VCR["video_hpf_extra"] = 1200000
    RFParams_PAL_VCR["video_hpf_extra_order"] = 2
    RFParams_PAL_VCR["video_lpf_freq"] = 3000000
    RFParams_PAL_VCR["video_lpf_order"] = 1

    RFParams_PAL_VCR["color_under_carrier"] = 562500
    RFParams_PAL_VCR["chroma_bpf_upper"] = 1200000

    # Video EQ after FM demod (PAL VCR) (needs tweak)
    RFParams_PAL_VCR["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis ()
    # TODO: calc
    RFParams_PAL_VCR["deemph_tau"] = 520e-9

    # Temporary video emphasis filter constants
    # TODO: Not correct, need to find the correct setup.
    RFParams_PAL_VCR["deemph_mid"] = 600000
    RFParams_PAL_VCR["deemph_gain"] = 16  # 6.8
    #

    # This has not really been stress-tested due to lack of crummy VCR samples.
    RFParams_PAL_VCR["boost_bpf_low"] = 4000000
    RFParams_PAL_VCR["boost_bpf_high"] = 4500000
    RFParams_PAL_VCR["boost_bpf_mult"] = 1

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_VCR["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_VCR["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_VCR["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_VCR


def get_sysparams_pal_vcr(sysparams_pal):
    SysParams_PAL_VCR = {**sysparams_pal}

    # PAL and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    # SysParams_PAL_VCR["ire0"] = 4257143
    # SysParams_PAL_VCR["hz_ire"] = 1600000 / 140.0
    # SysParams_PAL_VCR["burst_abs_ref"] = 5000
    SysParams_PAL_VCR["hz_ire"] = 1400000 / (100 + -SysParams_PAL_VCR["vsync_ire"])
    SysParams_PAL_VCR["ire0"] = 3.0e6 + (
        SysParams_PAL_VCR["hz_ire"] * -SysParams_PAL_VCR["vsync_ire"]
    )
    SysParams_PAL_VCR["burst_abs_ref"] = 4000

    return SysParams_PAL_VCR


def get_rfparams_pal_vcr_lp(RFParams_PAL):
    RFParams_PAL_VCR = {**RFParams_PAL}
    # VCR section
    # These need tweaking.
    RFParams_PAL_VCR["video_bpf_low"] = 1400000
    RFParams_PAL_VCR["video_bpf_high"] = 5500000
    RFParams_PAL_VCR["video_bpf_order"] = 2
    RFParams_PAL_VCR["video_lpf_extra"] = 6000000
    RFParams_PAL_VCR["video_lpf_extra_order"] = 2
    RFParams_PAL_VCR["video_hpf_extra"] = 1200000
    RFParams_PAL_VCR["video_hpf_extra_order"] = 2
    RFParams_PAL_VCR["video_lpf_freq"] = 3000000
    RFParams_PAL_VCR["video_lpf_order"] = 1

    RFParams_PAL_VCR["color_under_carrier"] = 562500
    RFParams_PAL_VCR["chroma_bpf_upper"] = 1200000

    # Video EQ after FM demod (PAL VCR) (needs tweak)
    RFParams_PAL_VCR["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis ()
    # TODO: calc
    RFParams_PAL_VCR["deemph_tau"] = 520e-9

    # Temporary video emphasis filter constants
    # TODO: Not correct, need to find the correct setup.
    RFParams_PAL_VCR["deemph_mid"] = 600000
    RFParams_PAL_VCR["deemph_gain"] = 6.8
    #

    # This has not really been stress-tested due to lack of crummy VCR samples.
    RFParams_PAL_VCR["boost_bpf_low"] = 4400000
    RFParams_PAL_VCR["boost_bpf_high"] = 5000000
    RFParams_PAL_VCR["boost_bpf_mult"] = 1

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_VCR["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_VCR["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_VCR["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_VCR


def get_sysparams_pal_vcr_lp(sysparams_pal):
    SysParams_PAL_VCR = {**sysparams_pal}

    # PAL and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    # SysParams_PAL_VCR["ire0"] = 4257143
    # SysParams_PAL_VCR["hz_ire"] = 1600000 / 140.0
    # SysParams_PAL_VCR["burst_abs_ref"] = 5000
    SysParams_PAL_VCR["hz_ire"] = 1500000 / (100 + -SysParams_PAL_VCR["vsync_ire"])
    SysParams_PAL_VCR["ire0"] = 3.3e6 + (
        SysParams_PAL_VCR["hz_ire"] * -SysParams_PAL_VCR["vsync_ire"]
    )
    SysParams_PAL_VCR["burst_abs_ref"] = 4000

    return SysParams_PAL_VCR
