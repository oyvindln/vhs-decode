from lddecode.core import (
    FilterParams_PAL,
    FilterParams_NTSC,
    SysParams_PAL,
    SysParams_NTSC,
)

# Default thresholds for rf dropout detection.
DEFAULT_THRESHOLD_P_DDD = 0.18
DEFAULT_THRESHOLD_P_CXADC = 0.35
DEFAULT_HYSTERESIS = 1.25
# Merge dropouts if they there is less than this number of samples between them.
DOD_MERGE_THRESHOLD = 30
DOD_MIN_LENGTH = 10
DEFAULT_SHARPNESS = 0
# lddecode uses 0.5 - upping helps decode some tapes with bad vsync.
EQ_PULSE_TOLERANCE = 0.9
MAX_WOW = 1.06

SysParams_NTSC["analog_audio"] = False
SysParams_PAL["analog_audio"] = False

TAPE_SPEEDS = {"sp": 0, "lp": 1, "ep": 2, "slp": 2, "vp": 3}


def parse_tape_speed(tape_speed: str) -> int:
    return TAPE_SPEEDS.get(tape_speed.lower(), 0)


def get_sys_params_405():
    """Get sys params for 405 line 25 fps system based on PAL parameters

    Returns:
        dict: dict of params for 405-line 25 fps.
    """

    sys_params_405 = {**SysParams_PAL}
    sys_params_405["frame_lines"] = 405
    sys_params_405["field_lines"] = (202, 203)
    sys_params_405["line_period"] = 98.765
    # start, length of active video.
    sys_params_405["activeVideoUS"] = (16.5, 98.765 - 1.75)
    sys_params_405["firstFieldH"] = (1, 0.0)  # TODO: find out if correct
    # "blacksnr_slice": (22, 12, 50),
    # TODO: This causes some odd behaviour, need to fix
    # sys_params_405["numPulses"] = 8  # number of equalization pulses per section
    sys_params_405["hsyncPulseUS"] = 9.0
    sys_params_405["eqPulseUS"] = 10.0
    sys_params_405["vsyncPulseUS"] = 40.0

    return sys_params_405


def get_sys_params_819():
    sys_params_819 = {**SysParams_PAL}
    # TODO: Do we need to distinquish between system E (Brance)
    # and and F (Belgium)?

    # See https://web.archive.org/web/20060924213330/http://www.pembers.freeserve.co.uk/World-TV-Standards/Line-Standards.html#Vertical

    sys_params_819["frame_lines"] = 819
    sys_params_819["field_lines"] = (409, 410)
    sys_params_819["line_period"] = 48.8
    # start, length of active video.
    sys_params_819["activeVideoUS"] = (
        8.5,
        48.8 - 1.75,
    )  # TODO: These are just some approximations
    sys_params_819["firstFieldH"] = (
        -0.5,
        -0.5,
    )  # TODO: find out if correct - test if -0.5, -0.5 works
    # "blacksnr_slice": (22, 12, 50),
    # TODO this causes rolling on sample, need to fix
    # sys_params_819["numPulses"] = 1  # or should possibly be 0 - 7 for System F
    sys_params_819["hsyncPulseUS"] = 2.5
    sys_params_819["eqPulseUS"] = 2.5
    # For the system E half line field pulse - not sure if it technically counts as a vsync pulse
    # System F is a bit different
    sys_params_819["vsyncPulseUS"] = sys_params_819["line_period"] / 2
    # TODO: May or may not have +50 mv pedestal

    return sys_params_819


def is_color_under(tape_format: str) -> bool:
    return tape_format not in ["TYPEC", "TYPEB", "QUADRUPLEX", "VHD"]


def parent_system(system: str) -> str:
    """Returns 'PAL' for 625 line systems and 405 line, and 'NTSC' for 525-line systems"""
    if system == "PAL_M" or system == "NLINHA":
        parent_system = "NTSC"
    elif system == "MESECAM" or system == "SECAM" or system == "405" or system == "819":
        parent_system = "PAL"
    else:
        parent_system = system
    return parent_system


def get_cvbs_params(system: str) -> tuple:
    from vhsdecode.format_defs import cvbs

    if parent_system(system) == "PAL":
        return cvbs.get_sys_params_pal(SysParams_PAL), cvbs.get_filter_params_pal(FilterParams_PAL)
    elif parent_system(system) == "NTSC":
        return cvbs.get_sys_params_ntsc(SysParams_NTSC), cvbs.get_filter_params_pal(
            FilterParams_NTSC
        )
    else:
        raise Exception("Unknown video system! ", system)


def get_format_params(system: str, tape_format: str, tape_speed: int, logger) -> tuple:
    """Get format parameters based on video system and tape format.
    Will raise an exception if the system is not one of PAL, NTSC and PAL_M
    """
    # We base the parameters off the original laserdisc ones and override the ones
    # we need.

    # TODO: This needs to be reworked and made cleaner.

    assert tape_speed <= 4, "Tape speed was > 4!"

    if system == "PAL":
        if tape_format == "UMATIC":
            from vhsdecode.format_defs.umatic import (
                get_rfparams_pal_umatic,
                get_sysparams_pal_umatic,
            )

            return get_sysparams_pal_umatic(SysParams_PAL), get_rfparams_pal_umatic(
                FilterParams_PAL
            )
        if tape_format == "UMATIC_HI":
            from vhsdecode.format_defs.umatic import (
                get_rfparams_pal_umatic_hi,
                get_sysparams_pal_umatic_hi,
            )

            return get_sysparams_pal_umatic_hi(SysParams_PAL), get_rfparams_pal_umatic_hi(
                FilterParams_PAL
            )
        elif tape_format == "UMATIC_SP":
            from vhsdecode.format_defs.umatic import (
                get_rfparams_pal_umatic_sp,
                get_sysparams_pal_umatic_sp,
            )

            return get_sysparams_pal_umatic_sp(SysParams_PAL), get_rfparams_pal_umatic_sp(
                FilterParams_PAL
            )
        elif tape_format == "BETAMAX":
            from vhsdecode.format_defs.betamax import (
                get_rfparams_pal_betamax,
                get_sysparams_pal_betamax,
            )

            return get_sysparams_pal_betamax(SysParams_PAL), get_rfparams_pal_betamax(
                FilterParams_PAL
            )
        elif tape_format == "VHSHQ":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_pal_vhs,
                get_sysparams_pal_vhshq,
            )

            return get_sysparams_pal_vhshq(SysParams_PAL, tape_speed), get_rfparams_pal_vhs(
                FilterParams_PAL, tape_speed
            )
        elif tape_format == "SVHS":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_pal_svhs,
                get_sysparams_pal_svhs,
            )

            return get_sysparams_pal_svhs(SysParams_PAL), get_rfparams_pal_svhs(FilterParams_PAL)
        elif tape_format == "SVHS_ET":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_pal_svhs_et,
                get_sysparams_pal_svhs,
            )

            return get_sysparams_pal_svhs(SysParams_PAL), get_rfparams_pal_svhs_et(FilterParams_PAL)
        elif tape_format == "VIDEO8":
            from vhsdecode.format_defs.video8 import (
                get_rfparams_pal_video8,
                get_sysparams_pal_video8,
            )

            return get_sysparams_pal_video8(SysParams_PAL), get_rfparams_pal_video8(
                FilterParams_PAL
            )
        elif tape_format == "HI8":
            from vhsdecode.format_defs.video8 import (
                get_rfparams_pal_hi8,
                get_sysparams_pal_hi8,
            )

            return get_sysparams_pal_hi8(SysParams_PAL), get_rfparams_pal_hi8(
                FilterParams_PAL, tape_speed
            )
        elif tape_format == "EIAJ":
            from vhsdecode.format_defs.eiaj import (
                get_rfparams_pal_eiaj,
                get_sysparams_pal_eiaj,
            )

            return get_sysparams_pal_eiaj(SysParams_PAL), get_rfparams_pal_eiaj(FilterParams_PAL)
        elif tape_format == "QUADRUPLEX":
            from vhsdecode.format_defs.typec import (
                get_rfparams_pal_quad,
                get_sysparams_pal_quad,
            )

            return get_sysparams_pal_quad(SysParams_PAL), get_rfparams_pal_quad(FilterParams_PAL)
        elif tape_format == "TYPEB":
            from vhsdecode.format_defs.typec import (
                get_rfparams_pal_typeb,
                get_sysparams_pal_typeb,
            )

            return get_sysparams_pal_typeb(SysParams_PAL), get_rfparams_pal_typeb(FilterParams_PAL)
        elif tape_format == "TYPEC":
            from vhsdecode.format_defs.typec import (
                get_rfparams_pal_typec,
                get_sysparams_pal_typec,
            )

            return get_sysparams_pal_typec(SysParams_PAL), get_rfparams_pal_typec(FilterParams_PAL)
        elif tape_format == "VCR":
            from vhsdecode.format_defs.vcr import (
                get_rfparams_pal_vcr,
                get_sysparams_pal_vcr,
            )

            return get_sysparams_pal_vcr(SysParams_PAL), get_rfparams_pal_vcr(FilterParams_PAL)
        elif tape_format == "VCR_LP":
            from vhsdecode.format_defs.vcr import (
                get_rfparams_pal_vcr_lp,
                get_sysparams_pal_vcr_lp,
            )

            return get_sysparams_pal_vcr_lp(SysParams_PAL), get_rfparams_pal_vcr_lp(
                FilterParams_PAL
            )
        elif tape_format == "VIDEO2000":
            from vhsdecode.format_defs.video2000 import (
                get_rfparams_pal_video2000,
                get_sysparams_pal_video2000,
            )

            return get_sysparams_pal_video2000(SysParams_PAL), get_rfparams_pal_video2000(
                FilterParams_PAL
            )
        else:
            if tape_format != "VHS":
                logger.warning('Tape format "%s" not supported for PAL yet', tape_format)
            from vhsdecode.format_defs.vhs import (
                get_rfparams_pal_vhs,
                get_sysparams_pal_vhs,
            )

            return get_sysparams_pal_vhs(SysParams_PAL, tape_speed), get_rfparams_pal_vhs(
                FilterParams_PAL, tape_speed
            )
    elif system == "NTSC":
        if tape_format == "UMATIC":
            from vhsdecode.format_defs.umatic import (
                get_rfparams_ntsc_umatic,
                get_sysparams_ntsc_umatic,
            )

            return get_sysparams_ntsc_umatic(SysParams_NTSC), get_rfparams_ntsc_umatic(
                FilterParams_NTSC
            )
        elif tape_format == "VHSHQ":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_ntsc_vhs,
                get_sysparams_ntsc_vhshq,
            )

            return get_sysparams_ntsc_vhshq(SysParams_NTSC, tape_speed), get_rfparams_ntsc_vhs(
                FilterParams_NTSC, tape_speed
            )
        elif tape_format == "SVHS":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_ntsc_svhs,
                get_sysparams_ntsc_svhs,
            )

            return get_sysparams_ntsc_svhs(SysParams_NTSC), get_rfparams_ntsc_svhs(
                FilterParams_NTSC
            )
        elif tape_format == "SVHS_ET":
            from vhsdecode.format_defs.vhs import (
                get_rfparams_ntsc_svhs_et,
                get_sysparams_ntsc_svhs,
            )

            return get_sysparams_ntsc_svhs(SysParams_NTSC), get_rfparams_ntsc_svhs_et(
                FilterParams_NTSC
            )
        elif tape_format == "BETAMAX":
            from vhsdecode.format_defs.betamax import (
                get_rfparams_ntsc_betamax,
                get_sysparams_ntsc_betamax,
            )

            return get_sysparams_ntsc_betamax(SysParams_NTSC), get_rfparams_ntsc_betamax(
                FilterParams_NTSC
            )
        elif tape_format == "BETAMAX_HIFI":
            from vhsdecode.format_defs.betamax import (
                get_rfparams_ntsc_betamax_hifi,
                get_sysparams_ntsc_betamax_hifi,
            )

            return get_sysparams_ntsc_betamax_hifi(SysParams_NTSC), get_rfparams_ntsc_betamax_hifi(
                FilterParams_NTSC
            )
        elif tape_format == "SUPERBETA":
            from vhsdecode.format_defs.betamax import (
                get_rfparams_ntsc_super_betamax,
                get_sysparams_ntsc_super_betamax,
            )

            return get_sysparams_ntsc_super_betamax(
                SysParams_NTSC
            ), get_rfparams_ntsc_super_betamax(FilterParams_NTSC)
        elif tape_format == "VIDEO8":
            from vhsdecode.format_defs.video8 import (
                get_rfparams_ntsc_video8,
                get_sysparams_ntsc_video8,
            )

            return get_sysparams_ntsc_video8(SysParams_NTSC), get_rfparams_ntsc_video8(
                FilterParams_NTSC
            )
        elif tape_format == "HI8":
            from vhsdecode.format_defs.video8 import (
                get_rfparams_ntsc_hi8,
                get_sysparams_ntsc_hi8,
            )

            return get_sysparams_ntsc_hi8(SysParams_NTSC), get_rfparams_ntsc_hi8(
                FilterParams_NTSC, tape_speed
            )
        elif tape_format == "TYPEB":
            from vhsdecode.format_defs.typec import (
                get_rfparams_ntsc_typeb,
                get_sysparams_ntsc_typeb,
            )

            return get_sysparams_ntsc_typeb(SysParams_NTSC), get_rfparams_ntsc_typeb(
                FilterParams_NTSC
            )
        elif tape_format == "TYPEC":
            from vhsdecode.format_defs.typec import (
                get_rfparams_ntsc_typec,
                get_sysparams_ntsc_typec,
            )

            return get_sysparams_ntsc_typec(SysParams_NTSC), get_rfparams_ntsc_typec(
                FilterParams_NTSC
            )
        elif tape_format == "VHD":
            from vhsdecode.format_defs.vhd import (
                get_rfparams_ntsc_vhd,
                get_sysparams_ntsc_vhd,
            )

            return get_sysparams_ntsc_vhd(SysParams_NTSC), get_rfparams_ntsc_vhd(FilterParams_NTSC)
        else:
            if tape_format != "VHS":
                logger.warning('Tape format "%s" not supported for NTSC yet', tape_format)
            from vhsdecode.format_defs.vhs import (
                get_rfparams_ntsc_vhs,
                get_sysparams_ntsc_vhs,
            )

            return get_sysparams_ntsc_vhs(SysParams_NTSC, tape_speed), get_rfparams_ntsc_vhs(
                FilterParams_NTSC, tape_speed
            )
    elif system == "PAL_M":
        if tape_format != "VHS":
            logger.warning('Tape format "%s" not supported for PAL-M yet', tape_format)
        from vhsdecode.format_defs.vhs import (
            get_rfparams_palm_vhs,
            get_sysparams_palm_vhs,
        )

        return get_sysparams_palm_vhs(SysParams_NTSC, tape_speed), get_rfparams_palm_vhs(
            FilterParams_NTSC, tape_speed
        )
    elif system == "NLINHA":
        # NLINHA - hacky unofficial system used on old brazilian vcrs that
        # uses PAL-M color but NTSC carrier freq and rotation.
        if tape_format != "VHS":
            logger.warning('Tape format "%s" not supported for NLINHA yet', tape_format)
        from vhsdecode.format_defs.vhs import (
            get_rfparams_ntsc_vhs,
            get_sysparams_palm_vhs,
        )

        return get_sysparams_palm_vhs(SysParams_NTSC, tape_speed), get_rfparams_ntsc_vhs(
            FilterParams_NTSC, tape_speed
        )
    elif system == "MESECAM":
        if tape_format != "VHS":
            logger.warning('Tape format "%s" not supported for MESECAM yet', tape_format)
        from vhsdecode.format_defs.vhs import (
            get_rfparams_mesecam_vhs,
            get_sysparams_mesecam_vhs,
        )

        return get_sysparams_mesecam_vhs(SysParams_PAL, tape_speed), get_rfparams_mesecam_vhs(
            FilterParams_PAL, tape_speed
        )
    elif system == "SECAM":
        if tape_format != "VHS":
            logger.warning('Tape format "%s" not supported for SECAM yet', tape_format)
        from vhsdecode.format_defs.vhs import (
            get_rfparams_secam_vhs,
            get_sysparams_secam_vhs,
        )

        return get_sysparams_secam_vhs(SysParams_PAL, tape_speed), get_rfparams_secam_vhs(
            FilterParams_PAL, tape_speed
        )
    elif system == "405":
        if tape_format == "BETAMAX":
            from vhsdecode.format_defs.betamax import (
                get_rfparams_405line_betamax,
                get_sysparams_405line_betamax,
            )

            sys_params = get_sys_params_405()

            return get_sysparams_405line_betamax(sys_params), get_rfparams_405line_betamax(
                FilterParams_PAL
            )
        else:
            raise Exception("Tape format not supported for 405 line yet", tape_format)
    elif system == "819":
        if tape_format == "QUADRUPLEX":
            from vhsdecode.format_defs.typec import (
                get_rfparams_819line_quad,
                get_sysparams_819line_quad,
            )

            sys_params = get_sys_params_819()

            return get_sysparams_819line_quad(sys_params), get_rfparams_819line_quad(
                FilterParams_PAL
            )
        else:
            raise Exception("Tape format not supported for 819 line yet", tape_format)
    else:
        raise Exception("Unknown video system! ", system)
