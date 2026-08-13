"""Colour-under chroma processing.

Public entry points are re-exported here so external code can keep
importing from vhsdecode.chroma.
"""

from vhsdecode.chroma.qam import (
    SECAM_IDENT_MIN_CONFIDENCE,
    SECAM_M1_SEPARATION_RANGE,
    SECAM_M1_UNDER_PAIR_CENTER,
    SecamParityFlywheel,
    chroma_color_under_filter,
    decode_chroma,
    decode_chroma_phase_rotation,
    fit_secam_line_alternation,
    measure_secam_under_carrier_offset,
    regenerate_secam_blanking,
    secam_bell_gain,
    upconvert_secam_method1,
)
