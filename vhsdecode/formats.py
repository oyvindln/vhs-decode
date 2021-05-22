import copy
from lddecode.core import (
    RFParams_PAL,
    RFParams_NTSC,
    SysParams_PAL,
    SysParams_NTSC,
    calclinelen,
)

# We base the parameters off the original laserdisc ones and override the ones
# we need.
# NOTE: (This requires python 3.5+)
RFParams_PAL_VHS = {**RFParams_PAL}
RFParams_NTSC_VHS = {**RFParams_NTSC}

RFParams_PAL_UMATIC = {**RFParams_PAL}
RFParams_PAL_UMATIC_HI = {**RFParams_PAL}
RFParams_NTSC_UMATIC = {**RFParams_NTSC}
# Add SP/hi-band etc later

# Tape-specific paramaters that differs from the laserdisc analogues
# VHS PAL section

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
RFParams_PAL_VHS["video_lpf_freq"] = 3200000
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
RFParams_PAL_VHS["deemph_mid"] = 260000
RFParams_PAL_VHS["deemph_gain"] = 14

# Filter to pull out high frequencies for high frequency boost
# This should cover the area around reference white.
# Used to reduce streaks due to amplitude loss on phase change around
# sharp transitions.
RFParams_PAL_VHS["boost_bpf_low"] = 4200000
RFParams_PAL_VHS["boost_bpf_high"] = 5600000
# Multiplier for the boosted signal to add in.
RFParams_PAL_VHS["boost_bpf_mult"] = 2

# NTSC VHS section

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

RFParams_NTSC_VHS["deemph_mid"] = RFParams_PAL_VHS["deemph_mid"]
RFParams_NTSC_VHS["deemph_gain"] = RFParams_PAL_VHS["deemph_gain"]

RFParams_NTSC_VHS["boost_bpf_low"] = 4100000
RFParams_NTSC_VHS["boost_bpf_high"] = 5000000
RFParams_NTSC_VHS["boost_bpf_mult"] = 1

# PAL-M VHS section
RFParams_MPAL_VHS = copy.deepcopy(RFParams_NTSC_VHS)
RFParams_MPAL_VHS["color_under_carrier"] = 631.337e3

# UMATIC section
# These need tweaking.
RFParams_PAL_UMATIC["video_bpf_low"] = 2500000
RFParams_PAL_UMATIC["video_bpf_high"] = 6700000
RFParams_PAL_UMATIC["video_bpf_order"] = 1
RFParams_PAL_UMATIC["video_lpf_extra"] = 6900000
RFParams_PAL_UMATIC["video_lpf_extra_order"] = 3
RFParams_PAL_UMATIC["video_hpf_extra"] = 1500000
RFParams_PAL_UMATIC["video_hpf_extra_order"] = 1
RFParams_PAL_UMATIC["video_lpf_freq"] = 4200000
RFParams_PAL_UMATIC["video_lpf_order"] = 2
# 685546 ± 200
RFParams_PAL_UMATIC["color_under_carrier"] = (625 * 25) * (351 / 8)
RFParams_PAL_UMATIC["chroma_bpf_upper"] = 1300000

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

# This has not really been stress-tested due to lack of crummy umatic samples.
RFParams_PAL_UMATIC["boost_bpf_low"] = 5000000
RFParams_PAL_UMATIC["boost_bpf_high"] = 5800000
RFParams_PAL_UMATIC["boost_bpf_mult"] = 1

RFParams_PAL_UMATIC_HI["video_bpf_low"] = 0
RFParams_PAL_UMATIC_HI["video_bpf_high"] = 0
RFParams_PAL_UMATIC_HI["video_bpf_order"] = 0
RFParams_PAL_UMATIC_HI["video_lpf_extra"] = 0
RFParams_PAL_UMATIC_HI["video_lpf_extra_order"] = 0
RFParams_PAL_UMATIC_HI["video_hpf_extra"] = 0
RFParams_PAL_UMATIC_HI["video_hpf_extra_order"] = 0
RFParams_PAL_UMATIC_HI["video_lpf_freq"] = 0
RFParams_PAL_UMATIC_HI["video_lpf_order"] = 0
RFParams_PAL_UMATIC_HI["color_under_carrier"] = 983803
RFParams_PAL_UMATIC_HI["chroma_bpf_upper"] = 0

RFParams_NTSC_UMATIC["video_bpf_low"] = 2500000
RFParams_NTSC_UMATIC["video_bpf_high"] = 6500000
RFParams_NTSC_UMATIC["video_bpf_order"] = 1
RFParams_NTSC_UMATIC["video_lpf_extra"] = 6900000
RFParams_NTSC_UMATIC["video_lpf_extra_order"] = 3
RFParams_NTSC_UMATIC["video_hpf_extra"] = 1200000
RFParams_NTSC_UMATIC["video_hpf_extra_order"] = 1
RFParams_NTSC_UMATIC["video_lpf_freq"] = 4000000
RFParams_NTSC_UMATIC["video_lpf_order"] = 1
#  688374 ± 200
# (525 * (30 / 1.001)) * (175/4)
RFParams_NTSC_UMATIC["color_under_carrier"] = (525 * (30 / 1.001)) * (175 / 4)
RFParams_NTSC_UMATIC["chroma_bpf_upper"] = 1500000

# Video EQ after FM demod (NTSC UMATIC) (needs tweak)
RFParams_NTSC_UMATIC["video_eq"] = {
    "loband": {"corner": 2.62e6, "transition": 500e3, "order_limit": 20, "gain": 2},
}

# Video Y FM de-emphasis (550 ~ 650ns)
RFParams_NTSC_UMATIC["deemph_tau"] = 600e-9

RFParams_NTSC_UMATIC["deemph_mid"] = 500000
RFParams_NTSC_UMATIC["deemph_gain"] = 10.8

# This has not really been stress-tested due to lack of crummy umatic samples.
RFParams_NTSC_UMATIC["boost_bpf_low"] = 5000000
RFParams_NTSC_UMATIC["boost_bpf_high"] = 5800000
RFParams_NTSC_UMATIC["boost_bpf_mult"] = 1

SysParams_PAL_VHS = {**SysParams_PAL}
SysParams_NTSC_VHS = {**SysParams_NTSC}
SysParams_PAL_UMATIC = {**SysParams_PAL}
SysParams_NTSC_UMATIC = {**SysParams_NTSC}

# 0 IRE level after demodulation
SysParams_PAL_VHS["ire0"] = 4100000

# frequency/ire IRE change pr frequency (Is this labeled correctly?)
SysParams_PAL_VHS["hz_ire"] = 700000 / 100.0

# Top/white point defined by the standard 4.8 MHz
SysParams_PAL_VHS["max_ire"] = 100

# Mean absolute value of color burst for Automatic Chroma Control.
# The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
# This has to be low enough to avoid clipping, so we have to
# tell the chroma decoder to boost it by a bit afterwards.
SysParams_PAL_VHS["burst_abs_ref"] = 2500

# 0 IRE level after demodulation
SysParams_NTSC_VHS["ire0"] = 3685000

# frequency/ire IRE change pr frequency (Is this labeled correctly?)
SysParams_NTSC_VHS["hz_ire"] = 715000 / 100.0

# Top/white point defined by the standard
SysParams_NTSC_VHS["max_ire"] = 100

# Mean absolute value of color burst for Automatic Chroma Control.
# The value is eyeballed to give ok chroma level as of now, needs to be tweaked.
SysParams_NTSC_VHS["burst_abs_ref"] = 1750

# PAL-M sysparams override (From JVC Video technical guide)
SysParams_MPAL_VHS = copy.deepcopy(SysParams_NTSC_VHS)
SysParams_MPAL_VHS["fsc_mhz"] = 3.575611888111
SysParams_MPAL_VHS["fieldPhases"] = 8

## Should be the same as NTSC in practice
SysParams_MPAL_VHS["line_period"] = 1 / (SysParams_MPAL_VHS["fsc_mhz"] / (909 / 4.0))
SysParams_MPAL_VHS["activeVideoUS"] = (9.45, SysParams_MPAL_VHS["line_period"] - 1.0)
# SysParams_NTSC["FPS"] = 1000000 / (525 * SysParams_MPAL_VHS["line_period"])

SysParams_MPAL_VHS["outlinelen"] = calclinelen(SysParams_MPAL_VHS, 4, "fsc_mhz")
SysParams_MPAL_VHS["outfreq"] = 4 * SysParams_MPAL_VHS["fsc_mhz"]
SysParams_MPAL_VHS["burst_abs_ref"] = 3500

# PAL and NTSC "regular-band" use the same frequencies, but
# not sure if PAL sync being -43 and ntsc being -40 makes
# a difference on these parameters.
SysParams_PAL_UMATIC["ire0"] = 4257143
SysParams_PAL_UMATIC["hz_ire"] = 1600000 / 140.0
SysParams_PAL_UMATIC["burst_abs_ref"] = 2750

SysParams_NTSC_UMATIC["ire0"] = 4257143
SysParams_NTSC_UMATIC["hz_ire"] = 1600000 / 140.0
SysParams_NTSC_UMATIC["burst_abs_ref"] = 2750

# SysParams_PAL['outlinelen'] = calclinelen(SysParams_PAL, 4, 'fsc_mhz')
# SysParams_PAL['outlinelen_pilot'] = calclinelen(SysParams_PAL, 4, 'pilot_mhz')
# SysParams_PAL['vsync_ire'] = -.3 * (100 / .7)

# TODO: SECAM

# Default thresholds for rf dropout detection.
DEFAULT_THRESHOLD_P_DDD = 0.12
DEFAULT_THRESHOLD_P_CXADC = 0.3
DEFAULT_HYSTERESIS = 1.25
# Merge dropouts if they there is less than this number of samples between them.
DOD_MERGE_THRESHOLD = 30
DOD_MIN_LENGTH = 10
DEFAULT_SHARPNESS = 0
BLANK_LENGTH_THRESHOLD = 9
# lddecode uses 0.5 - upping helps decode some tapes with bad vsync.
EQ_PULSE_TOLERANCE = 0.7
