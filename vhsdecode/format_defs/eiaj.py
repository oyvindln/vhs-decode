def get_rfparams_pal_eiaj(RFParams_PAL):
    RFParams_PAL_EIAJ = {**RFParams_PAL}
    # EIAJ section
    # These need tweaking.
    RFParams_PAL_EIAJ["video_bpf_low"] = 1500000
    RFParams_PAL_EIAJ["video_bpf_high"] = 6700000
    RFParams_PAL_EIAJ["video_bpf_order"] = 1
    RFParams_PAL_EIAJ["video_lpf_extra"] = 6900000
    RFParams_PAL_EIAJ["video_lpf_extra_order"] = 3
    RFParams_PAL_EIAJ["video_hpf_extra"] = 1500000
    RFParams_PAL_EIAJ["video_hpf_extra_order"] = 1
    RFParams_PAL_EIAJ["video_lpf_freq"] = 3000000
    RFParams_PAL_EIAJ["video_lpf_order"] = 1
    # 685546 ± 200
    RFParams_PAL_EIAJ["color_under_carrier"] = (625 * 25) * (351 / 8)
    RFParams_PAL_EIAJ["chroma_bpf_upper"] = 1300000

    # Video EQ after FM demod (PAL EIAJ) (based on NTSC one, needs tweak)
    RFParams_PAL_EIAJ["video_eq"] = {
        "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 1},
    }

    # Video Y FM de-emphasis ()
    # Seems to be around 215-220 ns according to AV-3670 schematics?
    # Eyeballed approximation based on circuit simulation.
    RFParams_PAL_EIAJ["deemph_tau"] = 520e-9

    # Temporary video emphasis filter constants
    # Ideally we would calculate this based on tau and 'x' value, for now
    # Parameters eyeballed based circuit simluation of AV-3670 by ifb.
    RFParams_PAL_EIAJ["deemph_mid"] = 360000
    RFParams_PAL_EIAJ["deemph_gain"] = 10

    # This has not really been stress-tested due to lack of crummy EIAJ samples.
    RFParams_PAL_EIAJ["boost_bpf_low"] = 5400000
    RFParams_PAL_EIAJ["boost_bpf_high"] = 6000000
    RFParams_PAL_EIAJ["boost_bpf_mult"] = 0

    # Needs to be tweaked, just using some random values for now.
    RFParams_PAL_EIAJ["nonlinear_highpass_freq"] = 1000000
    RFParams_PAL_EIAJ["nonlinear_highpass_limit_h"] = 5000
    RFParams_PAL_EIAJ["nonlinear_highpass_limit_l"] = -20000

    return RFParams_PAL_EIAJ


def get_sysparams_pal_eiaj(sysparams_pal):
    SysParams_PAL_EIAJ = {**sysparams_pal}

    # PAL and NTSC "regular-band" use the same frequencies, but
    # not sure if PAL sync being -43 and ntsc being -40 makes
    # a difference on these parameters.
    # SysParams_PAL_EIAJ["ire0"] = 4257143
    # SysParams_PAL_EIAJ["hz_ire"] = 1600000 / 140.0
    # SysParams_PAL_EIAJ["burst_abs_ref"] = 5000
    SysParams_PAL_EIAJ["hz_ire"] = 1800000 / (100 + -SysParams_PAL_EIAJ["vsync_ire"])
    SysParams_PAL_EIAJ["ire0"] = 3.8e6 + (
        SysParams_PAL_EIAJ["hz_ire"] * -SysParams_PAL_EIAJ["vsync_ire"]
    )
    SysParams_PAL_EIAJ["burst_abs_ref"] = 1000

    return SysParams_PAL_EIAJ
