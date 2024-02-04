def get_rfparams_ntsc_typec(RFParams_NTSC):
    RFParams_NTSC_TYPEC = {**RFParams_NTSC}
    # TYPEC section
    # These need tweaking.
    RFParams_NTSC_TYPEC["video_bpf_low"] = 2500000
    RFParams_NTSC_TYPEC["video_bpf_high"] = 12000000
    RFParams_NTSC_TYPEC["video_bpf_order"] = 1
    RFParams_NTSC_TYPEC["video_lpf_extra"] = 14000000
    RFParams_NTSC_TYPEC["video_lpf_extra_order"] = 3
    RFParams_NTSC_TYPEC["video_hpf_extra"] = 1500000
    RFParams_NTSC_TYPEC["video_hpf_extra_order"] = 1
    RFParams_NTSC_TYPEC["video_lpf_freq"] = 4200000
    RFParams_NTSC_TYPEC["video_lpf_order"] = 6
    # 923828 ± x00
    RFParams_NTSC_TYPEC["color_under_carrier"] = 3.58e6  # TODO set to fsc properly
    RFParams_NTSC_TYPEC["chroma_bpf_upper"] = 200000

    # Video EQ after FM demod (NTSC TYPEC) (based on NTSC one, needs tweak)
    RFParams_NTSC_TYPEC["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis 1
    RFParams_NTSC_TYPEC["deemph_tau"] = 240e-9
    # RFParams_NTSC_TYPEC["deemph_tau"] = 600e-9

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    RFParams_NTSC_TYPEC["deemph_mid"] = 420000
    RFParams_NTSC_TYPEC["deemph_gain"] = 7.35

    # This has not really been stress-tested due to lack of crummy TYPEC samples.
    RFParams_NTSC_TYPEC["boost_bpf_low"] = 9500000
    RFParams_NTSC_TYPEC["boost_bpf_high"] = 10000000
    RFParams_NTSC_TYPEC["boost_bpf_mult"] = 0

    # Needs to be tweaked, just using some random values for now.
    RFParams_NTSC_TYPEC["nonlinear_highpass_freq"] = 1000000
    RFParams_NTSC_TYPEC["nonlinear_highpass_limit_h"] = 5000
    RFParams_NTSC_TYPEC["nonlinear_highpass_limit_l"] = -20000

    return RFParams_NTSC_TYPEC


def get_sysparams_ntsc_typec(sysparams_NTSC):
    SysParams_NTSC_TYPEC = {**sysparams_NTSC}

    # NTSC and NTSC "regular-band" use the same frequencies, but
    # not sure if NTSC sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    SysParams_NTSC_TYPEC["hz_ire"] = 2940000 / 140.0
    SysParams_NTSC_TYPEC["ire0"] = 7.9e6
    SysParams_NTSC_TYPEC["burst_abs_ref"] = 5000

    # Sync tip = 7.06 mhz
    # Peak white = 10.00 mhz

    return SysParams_NTSC_TYPEC


def get_rfparams_pal_typec(RFParams_PAL):
    RFParams_PAL_TYPEC = {**RFParams_PAL}
    # TYPEC section
    # These need tweaking.
    RFParams_PAL_TYPEC["video_bpf_low"] = 1500000
    RFParams_PAL_TYPEC["video_bpf_high"] = 12000000
    RFParams_PAL_TYPEC["video_bpf_order"] = 1
    RFParams_PAL_TYPEC["video_lpf_extra"] = 14000000
    RFParams_PAL_TYPEC["video_lpf_extra_order"] = 3
    RFParams_PAL_TYPEC["video_hpf_extra"] = 500000
    RFParams_PAL_TYPEC["video_hpf_extra_order"] = 1
    RFParams_PAL_TYPEC["video_lpf_freq"] = 5200000
    RFParams_PAL_TYPEC["video_lpf_order"] = 6
    # 923828 ± x00
    RFParams_PAL_TYPEC["color_under_carrier"] = 4.43e6  # TODO set to fsc properly
    RFParams_PAL_TYPEC["chroma_bpf_upper"] = 200000

    # Video EQ after FM demod (NTSC TYPEC) (based on NTSC one, needs tweak)
    RFParams_PAL_TYPEC["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis 1
    RFParams_PAL_TYPEC["deemph_tau"] = 180e-9
    # RFParams_PAL_TYPEC["deemph_tau2"] = 610e-9

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    RFParams_PAL_TYPEC["deemph_mid"] = 478500
    RFParams_PAL_TYPEC["deemph_gain"] = 9.53

    # This has not really been stress-tested due to lack of crummy TYPEC samples.
    RFParams_PAL_TYPEC["boost_bpf_low"] = 9500000
    RFParams_PAL_TYPEC["boost_bpf_high"] = 10000000
    RFParams_PAL_TYPEC["boost_bpf_mult"] = 0

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_TYPEC["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_TYPEC["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_TYPEC["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_TYPEC


def get_sysparams_pal_typec(sysparams_PAL):
    SysParams_PAL_TYPEC = {**sysparams_PAL}

    # NTSC and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    SysParams_PAL_TYPEC["hz_ire"] = 1740000 / 143.0
    SysParams_PAL_TYPEC["ire0"] = 7.68e6
    SysParams_PAL_TYPEC["burst_abs_ref"] = 5000

    # Sync tip = 7.16 mhz
    # Peak white = 8.90 mhz

    return SysParams_PAL_TYPEC

def get_rfparams_ntsc_typeb(RFParams_NTSC):
    # Freqs seem shared with typec so re-using those for now.
    return get_rfparams_ntsc_typec(RFParams_NTSC)


def get_sysparams_ntsc_typeb(sysparams_NTSC):
    # Freqs seem shared with typec so re-using those for now.
    return get_sysparams_ntsc_typec(sysparams_NTSC)


def get_rfparams_pal_typeb(RFParams_PAL):
    RFParams_PAL_TYPEB = {**RFParams_PAL}
    # TYPEB section
    # These need tweaking.
    RFParams_PAL_TYPEB["video_bpf_low"] = 1500000
    RFParams_PAL_TYPEB["video_bpf_high"] = 10000000
    RFParams_PAL_TYPEB["video_bpf_order"] = 1
    RFParams_PAL_TYPEB["video_lpf_extra"] = 14000000
    RFParams_PAL_TYPEB["video_lpf_extra_order"] = 3
    RFParams_PAL_TYPEB["video_hpf_extra"] = 500000
    RFParams_PAL_TYPEB["video_hpf_extra_order"] = 1
    RFParams_PAL_TYPEB["video_lpf_freq"] = 5200000
    RFParams_PAL_TYPEB["video_lpf_order"] = 6
    RFParams_PAL_TYPEB["color_under_carrier"] = 4.43e6  # TODO set to fsc properly
    RFParams_PAL_TYPEB["chroma_bpf_upper"] = 200000

    # Video EQ after FM demod (NTSC TYPEB) (based on NTSC one, needs tweak)
    RFParams_PAL_TYPEB["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis 1
    RFParams_PAL_TYPEB["deemph_tau"] = 240e-9
    # RFParams_PAL_TYPEB["deemph_tau2"] = 600e-9

    # Deemphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # it's eyeballed based on graph and output.
    # TODO: copied typec values, do simluation using tau values for typec
    RFParams_PAL_TYPEB["deemph_mid"] = 420000
    RFParams_PAL_TYPEB["deemph_gain"] = 7.35

    # This has not really been stress-tested due to lack of crummy TYPEB samples.
    RFParams_PAL_TYPEB["boost_bpf_low"] = 9500000
    RFParams_PAL_TYPEB["boost_bpf_high"] = 10000000
    RFParams_PAL_TYPEB["boost_bpf_mult"] = 0

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_TYPEB["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_TYPEB["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_TYPEB["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_TYPEB


def get_sysparams_pal_typeb(sysparams_PAL):
    SysParams_PAL_TYPEB = {**sysparams_PAL}

    # NTSC and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    SysParams_PAL_TYPEB["hz_ire"] = (8.9e6-7.4e6) / 143.0
    SysParams_PAL_TYPEB["ire0"] = 7.40e6
    SysParams_PAL_TYPEB["burst_abs_ref"] = 5000

    # Sync tip = 6.76 mhz
    # Peak white = 8.90 mhz

    return SysParams_PAL_TYPEB
