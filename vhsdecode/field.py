import numpy as np

import lddecode.core as ldd
import lddecode.utils as lddu
from lddecode.utils import inrange
from lddecode.utils import hz_to_output_array

import vhsdecode.sync as sync
import vhsdecode.formats as formats
from vhsdecode.doc import detect_dropouts_rf
from vhsdecode.addons.resync import Pulse
from vhsdecode.chroma import (
    decode_chroma_simple,
    decode_chroma,
    get_field_phase_id,
    try_detect_track_vhs_pal,
    try_detect_track_ntsc,
    try_detect_track_betamax_pal,
)

from vhsdecode.debug_plot import plot_data_and_pulses

NO_PULSES_FOUND = 1

# def ynr(data, hpfdata, line_len):
#     """Dumb vcr-line ynr
#     """

#     numlines = len(data) // line_len
#     hpfdata = np.clip(hpfdata, -7000, 7000)
#     for line_num in range(16, numlines - 2):
#         delayed1h = hpfdata[(line_num - 1) * line_len : (line_num) * line_len]
#         line_slice = hpfdata[line_num * line_len : (line_num + 1) * line_len]
#         adv1h = hpfdata[(line_num + 1) * line_len : (line_num + 2) * line_len]
#         # Let the delayed signal contribute 1/3.
#         # Could probably make the filtering configurable later.
#         data[line_num * line_len : (line_num + 1) * line_len] -= line_slice
#         data[line_num * line_len : (line_num + 1) * line_len] += (((delayed1h + line_slice + adv1h) / 3) - line_slice)
#     return data


# Can't use numba here due to clip being a recent addition.
# @njit(cache=True)
def y_comb(data, line_len, limit):
    """Basic Y comb filter, essentially just blending a line with it's neighbours, limited to some maximum
    Utilized for Betamax, VHS LP etc as the half-shift in those formats helps put crosstalk on opposite phase on
    adjecent lines
    """

    diffb = data - np.roll(data, -line_len)
    difff = data - np.roll(data, line_len)

    data -= np.clip(diffb + difff, -limit, limit) / 2

    return data


def field_class_from_formats(system: str, tape_format: str):
    field_class = None
    if system == "PAL":
        if tape_format in ["UMATIC", "UMATIC_HI", "EIAJ", "VCR", "VCR_LP"]:
            # These use simple chroma downconversion and filters.
            field_class = FieldPALUMatic
        elif (
            tape_format == "TYPEC"
            or tape_format == "TYPEB"
            or tape_format == "QUADRUPLEX"
        ):
            field_class = FieldPALTypeC
        elif tape_format == "SVHS" or tape_format == "SVHS_ET":
            field_class = FieldPALSVHS
        elif tape_format == "BETAMAX":
            field_class = FieldPALBetamax
        elif tape_format == "VIDEO8" or tape_format == "HI8":
            field_class = FieldPALVideo8
        else:
            if (
                tape_format != "VHS"
                and tape_format != "VHSHQ"
                and tape_format != "VIDEO2000"
            ):
                ldd.logger.info(
                    "Tape format unimplemented for PAL, using VHS field class."
                )
            field_class = FieldPALVHS
    elif system == "NTSC":
        if tape_format == "UMATIC":
            field_class = FieldNTSCUMatic
        elif tape_format == "TYPEC" or tape_format == "TYPEB":
            field_class = FieldNTSCTypeC
        elif tape_format == "SVHS" or tape_format == "SVHS_ET":
            field_class = FieldNTSCSVHS
        elif (
            tape_format == "BETAMAX"
            or tape_format == "BETAMAX_HIFI"
            or tape_format == "SUPERBETA"
        ):
            field_class = FieldNTSCBetamax
        elif tape_format == "VIDEO8" or tape_format == "HI8":
            field_class = FieldNTSCVideo8
        else:
            if tape_format != "VHS" and tape_format != "VHSHQ":
                ldd.logger.info(
                    "Tape format unimplemented for NTSC, using VHS field class."
                )
            field_class = FieldNTSCVHS
    elif (system == "MPAL" or system == "NLINHA") and tape_format == "VHS":
        field_class = FieldMPALVHS
    elif system == "MESECAM" and tape_format == "VHS":
        field_class = FieldMESECAMVHS
    elif system == "405":
        if tape_format == "BETAMAX":
            field_class = FieldPALTypeC
        else:
            raise Exception("405 line not implemented for format!", format)
    elif system == "819":
        if tape_format == "QUADRUPLEX":
            field_class = FieldPALTypeC
        else:
            raise Exception("819 line not implemented for format!", format)

    if not field_class:
        raise Exception("Unknown video system!", system)

    return field_class


P_HSYNC, P_EQPL1, P_VSYNC, P_EQPL2, P_EQPL, P_OTHER_S, P_OTHER_L = range(7)
P_NAME = ["HSYNC", "EQPL1", "VSYNC", "EQPL2", "EQPL", "OTHER_S", "OTHER_L"]


def print_output_order(n, done, pulses):
    nums = map(lambda p: P_NAME[p[0]], pulses)
    print("n:", n, " ", done, " ", list(nums))


def print_output_types(pulses):
    nums = map(lambda p: (P_NAME[p[0]], p[1]), pulses)
    print(list(nums))


def _len_to_type(pulse, lt_hsync, lt_eq, lt_vsync):
    if inrange(pulse.len, *lt_hsync):
        return P_HSYNC
    elif inrange(pulse.len, *lt_eq):
        return P_EQPL
    elif inrange(pulse.len, *lt_vsync):
        return P_VSYNC
    elif pulse.len < lt_hsync[0]:
        print("outside", pulse.len)
        return P_OTHER_S
    else:
        return P_OTHER_L


def _to_type_list(raw_pulses, lt_hsync, lt_eq, lt_vsync):
    return list(map(lambda p: _len_to_type(p, lt_hsync, lt_eq, lt_vsync), raw_pulses))


def _add_type_to_pulses(raw_pulses, lt_hsync, lt_eq, lt_vsync):
    return list(
        map(lambda p: (_len_to_type(p, lt_hsync, lt_eq, lt_vsync), p), raw_pulses)
    )


def _to_seq(type_list, num_pulses, skip_bad=True):
    cur_type = None
    output = list()
    for pulse_type in type_list:
        if not skip_bad or pulse_type <= P_EQPL:
            if pulse_type == cur_type:
                cur = output[-1]
                output[-1] = (cur[0], cur[1] + 1)
            else:
                output.append((pulse_type, 1))
                cur_type = pulse_type

    return output


def _is_valid_seq(type_list, num_pulses):
    return len(type_list) >= 4 and type_list[1:3] == [
        (P_EQPL, num_pulses),
        (P_VSYNC, num_pulses),
        (P_EQPL, num_pulses),
    ]


def debug_plot_line0_fallback(
    demod_05, long_pulses, valid_pulses, raw_pulses, line_0, last_lineloc
):
    if True:
        # len(validpulses) > 300:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(demod_05)

        for raw_pulse in long_pulses:
            ax2.axvline(raw_pulse.start, color="#910000")
            ax2.axvline(raw_pulse.start + raw_pulse.len, color="#090909")

        if line_0 is not None:
            ax2.axvline(line_0, color="#00ff00")

        if last_lineloc is not None:
            ax2.axvline(last_lineloc, color="#0000ff")

        for raw_pulse in raw_pulses:
            ax3.axvline(raw_pulse.start, color="#910000")
            ax3.axvline(raw_pulse.start + raw_pulse.len, color="#090909")

        # print(valid_pulses)

        for valid_pulse in valid_pulses:
            ax3.axvline(valid_pulse.start, color="#ff0000")
            ax3.axvline(valid_pulse.start + valid_pulse.len, color="#0000ff")

        # for valid_pulse in long_pulses:
        #     color = (
        #         "#FF0000"
        #         if valid_pulse[0] == 2
        #         else "#00FF00"
        #         if valid_pulse[0] == 1
        #         else "#0F0F0F"
        #     )
        #     ax3.axvline(valid_pulse[1][0], color=color)
        #     ax3.axvline(valid_pulse[1][0] + valid_pulse[1][1], color="#009900")

        plt.show()


def get_line0_fallback(
    valid_pulses,
    raw_pulses,
    demod_05,
    lt_vsync,
    linelen,
    num_eq_pulses,
    frame_lines,
    relaxed=False,
    expected_line0=None,
    expected_first_field=None,
):
    """
    Try a more primitive way of locating line 0 if the normal approach fails.
    This doesn't actually fine line 0, rather it locates the approx position of the last vsync before vertical blanking
    as the later code is designed to work off of that.
    It is searched for in this order:
     -Find the start of long vsync pulses
     -Find the end of long vsync pulses
     -Find the end of short-distance eq pulses
     -Find the start of short-distance eq pulses
     -Just look for the first "long" pulse that could be start of vsync pulses in
      e.g a 240p/280p signal (that is, a pulse that is at least vsync pulse length.)
    """
    DEBUG_PLOT = False
    DEBUG_PRINT = False

    PULSE_START = 0
    PULSE_LEN = 1

    filtered_pulses = [raw_pulses[0]]
    i = 1
    while i < (len(raw_pulses) - 2):
        if (raw_pulses[i + 1].start - raw_pulses[i].start) > 0.45 * linelen and (
            raw_pulses[i].start - raw_pulses[i - 1].start
        ) > 0.45 * linelen:
            # normal case: pulse starts are at least 0.45 lines apart
            filtered_pulses.append(raw_pulses[i])
        else:
            # either pulse i or i+1 is wrong
            dis12 = (raw_pulses[i].start - raw_pulses[i - 1].start) / linelen
            dis13 = (raw_pulses[i + 1].start - raw_pulses[i - 1].start) / linelen
            dis24 = (raw_pulses[i + 2].start - raw_pulses[i].start) / linelen
            dis34 = (raw_pulses[i + 2].start - raw_pulses[i + 1].start) / linelen
            ddis12 = min(abs(dis12 - 0.5), abs(dis12 - 1.0))
            ddis13 = min(abs(dis13 - 0.5), abs(dis13 - 1.0))
            ddis24 = min(abs(dis24 - 0.5), abs(dis24 - 1.0))
            ddis34 = min(abs(dis34 - 0.5), abs(dis34 - 1.0))

            if ddis13 + ddis34 < ddis12 + ddis24:
                filtered_pulses.append(raw_pulses[i + 1])
            else:
                filtered_pulses.append(raw_pulses[i])
            i += 1
        i += 1
    while i < len(raw_pulses):
        filtered_pulses.append(raw_pulses[i])
        i += 1

    line_0 = None
    line_0_backup = None

    first_field = -1
    first_field_confidence = -1

    SHORT_PULSE_MAX = 0.2 * linelen
    LONG_PULSE_MIN = 0.35 * linelen

    if DEBUG_PRINT:
        print(f"get_line0_fallback called. Raw pulses: {len(raw_pulses)}, Filtered: {len(filtered_pulses)} (filtering logic skipped for debug print)")

    # First try: Find end of long sync pulses
    i = 15
    while line_0 is None and i < (len(filtered_pulses) - 2):
        disPPspp = (
            filtered_pulses[i - 1].start - filtered_pulses[i - 2].start
        ) / linelen
        dispPSpp = (filtered_pulses[i].start - filtered_pulses[i - 1].start) / linelen
        disppSPp = (filtered_pulses[i + 1].start - filtered_pulses[i].start) / linelen
        disppsPP = (
            filtered_pulses[i + 2].start - filtered_pulses[i + 1].start
        ) / linelen
        if (
            abs(disPPspp - 0.5) < 0.06
            and abs(dispPSpp - 0.5) < 0.06
            and abs(disppSPp - 0.5) < 0.06
            and abs(disppsPP - 0.5) < 0.06
            and filtered_pulses[i - 2].len > LONG_PULSE_MIN
            and filtered_pulses[i - 1].len > LONG_PULSE_MIN
            and filtered_pulses[i].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 2].len < SHORT_PULSE_MAX
        ):
            # we measure the distance of three pulses in previous field to start of long pulses
            # to check if this is first or second field
            # as there may be broken sync pulses we scan backwards
            measured_linelen = (disPPspp + dispPSpp + disppSPp + disppsPP) * (
                linelen / 2
            )
            line_offset = None
            # count "half lines" for detecting top/bottom field:
            half_lines = 0
            j = i
            while j < i + 9:
                dis = (
                    filtered_pulses[j + 1].start - filtered_pulses[j].start
                ) / linelen
                if (
                    abs(dis - 0.5) < 0.06
                    and filtered_pulses[j].len < SHORT_PULSE_MAX
                    and filtered_pulses[j + 1].len < SHORT_PULSE_MAX
                ):
                    half_lines += 1
                elif (
                    abs(dis - 1.0) < 0.06
                    and filtered_pulses[j].len < SHORT_PULSE_MAX
                    and filtered_pulses[j + 1].len < SHORT_PULSE_MAX
                ):
                    break
                else:
                    half_lines = 0
                    break
                j += 1
            if half_lines == 4 and frame_lines == 625:
                first_field = 0
                first_field_confidence = 100
                line_offset = 5.0
            elif half_lines == 5:
                if frame_lines == 625:
                    first_field = 1
                    first_field_confidence = 100
                    line_offset = 4.5
                else:
                    first_field = 0
                    first_field_confidence = 100
                    line_offset = 5.5
            elif half_lines == 6 and frame_lines == 525:
                first_field = 1
                line_offset = 6.0

            # if we couldn't detect field type based on half lines check phase
            if line_offset is None:
                phase_cnt = [0, 0, 0]
                for d in range(15, min(i, 30) + 1):
                    pp = np.array(
                        [
                            (
                                filtered_pulses[i - 2].start
                                - filtered_pulses[i - d].start
                            )
                            / measured_linelen,
                            (
                                filtered_pulses[i - 2].start
                                - filtered_pulses[i - d + 1].start
                            )
                            / measured_linelen,
                            (
                                filtered_pulses[i - 2].start
                                - filtered_pulses[i - d + 2].start
                            )
                            / measured_linelen,
                        ]
                    )
                    # PAL:  for start of first field all values should be 1, for second field all should be 0
                    # NTSC: for start of first field all values should be 0, for second field all should be 1
                    pps = np.sum(np.mod(np.round(pp * 2), 2))
                    if pps == 0:
                        phase_cnt[0] += 1
                    elif pps == 3:
                        phase_cnt[1] += 1
                    else:
                        phase_cnt[2] += 1
                    if sum(phase_cnt[0:2]) >= 5:
                        break
                phase = np.argmax(phase_cnt)
                if phase == 0:
                    # we need to differ between 625 and 525 line
                    if frame_lines == 625:
                        first_field = 0
                        line_offset = 5.0
                    else:
                        first_field = 1
                        line_offset = 6.0
                    first_field_confidence = phase_cnt[0] * 100 // sum(phase_cnt)
                elif phase == 1:
                    # we need to differ between 625 and 525 line
                    if frame_lines == 625:
                        first_field = 1
                        line_offset = 4.5
                    else:
                        first_field = 0
                        line_offset = 5.5
                    first_field_confidence = phase_cnt[1] * 100 // sum(phase_cnt)

            if line_offset is not None:
                # in case we cannot find a matching pulse, we can still use this prediction
                line_0_est = (
                    filtered_pulses[i - 2].start - line_offset * measured_linelen
                )
                if DEBUG_PRINT:
                    print(
                        f"1. End of long sync pulses, pred: {line_0_est}, First-Field: {first_field}, confidence: {first_field_confidence}"
                    )
                if line_0_backup is None:
                    line_0_backup = line_0_est
                    first_field_backup = first_field
                    first_field_confidence_backup = first_field_confidence
                # find pulse
                for j in range(max(0, i - 20), i) if relaxed else range(max(0, i - 16), i - 4):
                    if abs(filtered_pulses[j].start - line_0_est) / linelen < 0.08:
                        line_0 = filtered_pulses[j].start
                        break
        i += 1

    # Second try: Find beginng of long sync pulses
    i = 10
    while (line_0 is None or line_0 > (linelen * (frame_lines - 1) / 2)) and i < (
        len(filtered_pulses) - 2
    ):
        disPPspp = (
            filtered_pulses[i - 1].start - filtered_pulses[i - 2].start
        ) / linelen
        dispPSpp = (filtered_pulses[i].start - filtered_pulses[i - 1].start) / linelen
        disppSPp = (filtered_pulses[i + 1].start - filtered_pulses[i].start) / linelen
        disppsPP = (
            filtered_pulses[i + 2].start - filtered_pulses[i + 1].start
        ) / linelen

        if (
            abs(disPPspp - 0.5) < 0.06
            and abs(dispPSpp - 0.5) < 0.06
            and abs(disppSPp - 0.5) < 0.06
            and abs(disppsPP - 0.5) < 0.06
            and filtered_pulses[i - 2].len < SHORT_PULSE_MAX
            and filtered_pulses[i - 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i].len > LONG_PULSE_MIN
            and filtered_pulses[i + 1].len > LONG_PULSE_MIN
            and filtered_pulses[i + 2].len > LONG_PULSE_MIN
        ):
            # we measure the distance of three pulses in previous field to start of long pulses
            # to check if this is first or second field
            # as there may be broken syncs we scan backwards
            measured_linelen = (disPPspp + dispPSpp + disppSPp + disppsPP) * (
                linelen / 2
            )
            line_offset = None
            phase_cnt = [0, 0, 0]
            for d in range(10, min(i, 25) + 1):
                pp = np.array(
                    [
                        (filtered_pulses[i - 2].start - filtered_pulses[i - d].start)
                        / measured_linelen,
                        (
                            filtered_pulses[i - 2].start
                            - filtered_pulses[i - d + 1].start
                        )
                        / measured_linelen,
                        (
                            filtered_pulses[i - 2].start
                            - filtered_pulses[i - d + 2].start
                        )
                        / measured_linelen,
                    ]
                )
                # for start of first field all values should be 0, for second field all should be 1
                pps = np.sum(np.mod(np.round(pp * 2), 2))
                if pps == 0:
                    phase_cnt[0] += 1
                elif pps == 3:
                    phase_cnt[1] += 1
                else:
                    phase_cnt[2] += 1
                if sum(phase_cnt[0:2]) >= 5:
                    break
            phase = np.argmax(phase_cnt)
            if phase == 0:
                # we need to differ between 625 and 525 line
                line_offset = 2.0 if frame_lines == 625 else 3.0
                _first_field = 1
                _first_field_confidence = phase_cnt[0] * 100 // sum(phase_cnt)
            elif phase == 1:
                line_offset = 2.5 if frame_lines == 625 else 2.5
                _first_field = 0
                _first_field_confidence = phase_cnt[1] * 100 // sum(phase_cnt)

            if line_offset is not None:
                # in case we cannot find a matching pulse, we can still use this prediction
                line_0_est = (
                    filtered_pulses[i - 2].start - line_offset * measured_linelen
                )
                if DEBUG_PRINT:
                    print(
                        f"2. Begin of long sync pulses, pred: {line_0_est}, First-Field: {_first_field}, confidence: {_first_field_confidence}"
                    )
                if line_0_backup is None:
                    line_0_backup = line_0_est
                    first_field_backup = _first_field
                    first_field_confidence_backup = _first_field_confidence
                # find pulse
                for j in range(max(0, i - 15), i) if relaxed else range(max(0, i - 10), i - 3):
                    if abs(filtered_pulses[j].start - line_0_est) / linelen < 0.08:
                        if (
                            line_0 != filtered_pulses[j].start
                            or _first_field_confidence > first_field_confidence
                        ):
                            first_field = _first_field
                            first_field_confidence = first_field_confidence
                        line_0 = filtered_pulses[j].start
                        break
        i += 1

    # Third try: Find end of blanking
    i = 10
    while (line_0 is None or line_0 > (linelen * (frame_lines - 1) / 2)) and i < (
        len(filtered_pulses) - 2
    ):
        disPpspp = (
            filtered_pulses[i - 2].start - filtered_pulses[i - 3].start
        ) / linelen
        disPPspp = (
            filtered_pulses[i - 1].start - filtered_pulses[i - 2].start
        ) / linelen
        dispPSpp = (filtered_pulses[i].start - filtered_pulses[i - 1].start) / linelen
        disppSPp = (filtered_pulses[i + 1].start - filtered_pulses[i].start) / linelen
        disppsPP = (
            filtered_pulses[i + 2].start - filtered_pulses[i + 1].start
        ) / linelen

        if DEBUG_PRINT and i < 60:
             print(f"Try 3 scan i={i}: Pps={disPpspp:.3f} PPs={disPPspp:.3f} pSP={dispPSpp:.3f} ppS={disppSPp:.3f} pps={disppsPP:.3f}")
             print(f"  lens: {filtered_pulses[i - 2].len:.1f}, {filtered_pulses[i - 1].len:.1f}, {filtered_pulses[i].len:.1f}")

        # Relaxed check: ignore the first interval (disPpspp) to handle dropouts better
        check_strict = (
            abs(disPpspp - 0.5) < 0.06
            and abs(disPPspp - 0.5) < 0.06
            and abs(dispPSpp - 0.5) < 0.06
            and abs(disppSPp - 1.0) < 0.06
            and abs(disppsPP - 1.0) < 0.06
        )
        check_relaxed = (
            # abs(disPpspp - 0.5) < 0.06 and 
            abs(disPPspp - 0.5) < 0.06
            and abs(dispPSpp - 0.5) < 0.06
            and abs(disppSPp - 1.0) < 0.06
            and abs(disppsPP - 1.0) < 0.06
        )

        if (
            (check_relaxed if relaxed else check_strict)
            and filtered_pulses[i - 2].len < SHORT_PULSE_MAX
            and filtered_pulses[i - 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 2].len < SHORT_PULSE_MAX
        ):
            # we measure the distance of three pulses in previous field to start of long pulses
            # to check if this is first or second field
            # as there may be broken sync pulses we scan backwards
            measured_linelen = (disPPspp + dispPSpp + disppSPp + disppsPP) * (
                linelen / 3.0
            )
            line_offset = None
            eq_pulse_len = (
                filtered_pulses[i - 2].len + filtered_pulses[i - 1].len
            ) / 2.0
            hsync_pulse_len = (
                filtered_pulses[i + 1].len + filtered_pulses[i + 2].len
            ) / 2.0

            if hsync_pulse_len / eq_pulse_len > 1.75:
                if filtered_pulses[i].len < eq_pulse_len * 1.25:
                    if frame_lines == 625:
                        line_offset = 7.0
                    else:
                        line_offset = 8.0
                    _first_field = 0
                    _first_field_confidence = (
                        80 if filtered_pulses[i].len < eq_pulse_len * 1.1 else 60
                    )
                elif filtered_pulses[i].len > hsync_pulse_len * 0.75:
                    if frame_lines == 625:
                        line_offset = 7.0
                    else:
                        line_offset = 9.0
                    _first_field = 1
                    _first_field_confidence = (
                        80 if filtered_pulses[i].len > hsync_pulse_len * 0.9 else 60
                    )
                    
            if line_offset is not None:
                # in case we cannot find a matching pulse, we can still use this prediction
                line_0_est = (
                    filtered_pulses[i - 2].start - line_offset * measured_linelen
                )
                if DEBUG_PRINT:
                    print(
                        f"3. End of blanking, pred: {line_0_est}, First-Field: {_first_field}, confidence: {_first_field_confidence}"
                    )
                if line_0_backup is None:
                    line_0_backup = line_0_est
                    first_field_backup = _first_field
                    first_field_confidence_backup = _first_field_confidence
                # find pulse
                if DEBUG_PRINT:
                     print(f"Searching for pulse near {line_0_est} (range {max(0, i - 25)} to {i})")
                for j in range(max(0, i - 25), i) if relaxed else range(max(0, i - 20), i - 4):
                    diff = abs(filtered_pulses[j].start - line_0_est) / linelen
                    if DEBUG_PRINT:
                        print(f"    Checking pulse {j}: start={filtered_pulses[j].start}, est={line_0_est}, diff_lines={diff:.4f}")

                    if diff < 0.08:
                        if (
                            line_0 != filtered_pulses[j].start
                            or _first_field_confidence > first_field_confidence
                        ):
                            first_field = _first_field
                            first_field_confidence = first_field_confidence
                        line_0 = filtered_pulses[j].start
                        break
        i += 1

    # Fourth try: Find beginning of blanking
    i = 2
    while (line_0 is None or line_0 > (linelen * (frame_lines - 1) / 2)) and i < (
        len(filtered_pulses) - 3
    ):
        disPPspp = (
            filtered_pulses[i - 1].start - filtered_pulses[i - 2].start
        ) / linelen
        dispPSpp = (filtered_pulses[i].start - filtered_pulses[i - 1].start) / linelen
        disppSPp = (filtered_pulses[i + 1].start - filtered_pulses[i].start) / linelen
        disppsPP = (
            filtered_pulses[i + 2].start - filtered_pulses[i + 1].start
        ) / linelen
        disppspP = (
            filtered_pulses[i + 3].start - filtered_pulses[i + 2].start
        ) / linelen

        # Relaxed check: ignore the last interval (disppspP)
        check_strict = (
            abs(disPPspp - 1.0) < 0.06
            and abs(dispPSpp - 1.0) < 0.06
            and abs(disppSPp - 0.5) < 0.06
            and abs(disppsPP - 0.5) < 0.06
            and abs(disppspP - 0.5) < 0.06
        )
        check_relaxed = (
            abs(disPPspp - 1.0) < 0.06
            and abs(dispPSpp - 1.0) < 0.06
            and abs(disppSPp - 0.5) < 0.06
            and abs(disppsPP - 0.5) < 0.06
            # and abs(disppspP - 0.5) < 0.06
        )

        if (
            (check_relaxed if relaxed else check_strict)
            and filtered_pulses[i - 2].len < SHORT_PULSE_MAX
            and filtered_pulses[i - 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 1].len < SHORT_PULSE_MAX
            and filtered_pulses[i + 2].len < SHORT_PULSE_MAX
        ):
            hsync_pulse_len = (
                filtered_pulses[i - 2].len + filtered_pulses[i - 1].len
            ) / 2.0
            eq_pulse_len = (
                filtered_pulses[i + 1].len + filtered_pulses[i + 2].len
            ) / 2.0

            if hsync_pulse_len / eq_pulse_len > 1.75:
                if filtered_pulses[i].len < eq_pulse_len * 1.25:
                    _first_field_confidence = (
                        60 if filtered_pulses[i].len < eq_pulse_len * 1.1 else 40
                    )
                    if (
                        line_0 != filtered_pulses[i - 1].start
                        or _first_field_confidence > first_field_confidence
                    ):
                        first_field_confidence = _first_field_confidence
                        if frame_lines == 625:
                            first_field = 0
                        else:
                            first_field = 1
                    line_0 = filtered_pulses[i - 1].start
                elif filtered_pulses[i].len > hsync_pulse_len * 0.75:
                    _first_field_confidence = (
                        60 if filtered_pulses[i].len > hsync_pulse_len * 0.9 else 40
                    )
                    if (
                        line_0 != filtered_pulses[i].start
                        or _first_field_confidence > first_field_confidence
                    ):
                        first_field_confidence = _first_field_confidence
                        if frame_lines == 625:
                            first_field = 0
                        else:
                            first_field = 1
                    line_0 = filtered_pulses[i].start
                if DEBUG_PRINT:
                    print(
                        f"4. Start of blanking (pulses): {line_0}, First-Field: {first_field}, confidence: {_first_field_confidence}"
                    )

            # the pulse duration was not clear, we need to check contents
            # the interval between the first pulses half a line apart is either active or not
            if line_0 is None:
                lineP_avg = np.mean(
                    demod_05[
                        filtered_pulses[i - 1].start
                        + filtered_pulses[i - 1].len
                        + 40 : filtered_pulses[i].start
                        - 40
                    ]
                )
                lineP_std = np.std(
                    demod_05[
                        filtered_pulses[i - 1].start
                        + filtered_pulses[i - 1].len
                        + 40 : filtered_pulses[i].start
                        - 40
                    ]
                )
                lineI_avg = np.mean(
                    demod_05[
                        filtered_pulses[i].start
                        + filtered_pulses[i].len
                        + 40 : filtered_pulses[i + 1].start
                        - 40
                    ]
                )
                lineI_std = np.std(
                    demod_05[
                        filtered_pulses[i].start
                        + filtered_pulses[i].len
                        + 40 : filtered_pulses[i + 1].start
                        - 40
                    ]
                )
                lineN_avg = np.mean(
                    demod_05[
                        filtered_pulses[i + 1].start
                        + filtered_pulses[i + 1].len
                        + 40 : filtered_pulses[i + 2].start
                        - 40
                    ]
                )
                lineN_std = np.std(
                    demod_05[
                        filtered_pulses[i + 1].start
                        + filtered_pulses[i + 1].len
                        + 40 : filtered_pulses[i + 2].start
                        - 40
                    ]
                )

                if (
                    abs(lineP_avg - lineI_avg) / (lineP_avg + lineI_avg) < 0.05
                    and abs(lineP_avg - lineN_avg) / (lineP_avg + lineN_avg) > 0.15
                    and abs(lineP_std - lineI_std) * 2 < abs(lineP_std - lineN_std)
                ):
                    if (
                        line_0 != filtered_pulses[i - 1].start
                        or 20 > first_field_confidence
                    ):
                        if frame_lines == 625:
                            first_field = 0
                        else:
                            _first_field = 1
                        first_field_confidence = 20
                    line_0 = filtered_pulses[i - 1].start
                elif (
                    abs(lineP_avg - lineI_avg) / (lineP_avg + lineI_avg) > 0.15
                    and abs(lineP_avg - lineN_avg) / (lineP_avg + lineN_avg) > 0.15
                    and lineI_std * 2 > lineP_std
                ):
                    if (
                        line_0 != filtered_pulses[i].start
                        or 20 > first_field_confidence
                    ):
                        if frame_lines == 625:
                            first_field = 0
                        else:
                            first_field = 1
                        first_field_confidence = 20
                    line_0 = filtered_pulses[i].start
                if DEBUG_PRINT:
                    print(
                        f"4. Start of blanking (content): {line_0}, First-Field: {first_field}, confidence: {first_field_confidence}"
                    )
        i += 1

    if (
        line_0 is not None
        and line_0 > (linelen * (frame_lines - 1) / 2)
        and line_0_backup is not None
        and line_0_backup < (line_0 - (linelen * (frame_lines - 5) / 2))
    ):
        ldd.logger.info(
            "WARNING, line0 hsync not found for current field, but vsync area found, using predicted position, result may be garbled."
        )
        line_0 = line_0_backup
        first_field = first_field_backup
        first_field_confidence = first_field_confidence_backup - 20
    elif line_0 is not None and line_0 > (linelen * (frame_lines - 1) / 2):
        # Check if we have a backup that is valid (within first half of frame)
        if relaxed and line_0_backup is not None and line_0_backup < (linelen * (frame_lines - 1) / 2):
             ldd.logger.info("Switching to backup line0 estimation as primary is out of range.")
             line_0 = line_0_backup
             first_field = first_field_backup
             first_field_confidence = first_field_confidence_backup - 20
        else:
            if expected_line0 is None:
                ldd.logger.info(
                    "WARNING, line0 hsync not found for current field, probably skipping one field."
                )

    if line_0 is None and line_0_backup is not None:
        ldd.logger.info(
            "WARNING, line0 hsync not found in entire block, but vsync area found, using predicted position, result may be garbled."
        )
        line_0 = line_0_backup
        first_field = first_field_backup
        first_field_confidence = first_field_confidence_backup - 20

    if (line_0 is None or line_0 > (linelen * (frame_lines - 1) / 2)) and expected_line0 is not None:
        limit = (linelen * (frame_lines - 1) / 2)
        if expected_line0 < limit and expected_line0 > -5 * linelen:
            if DEBUG_PRINT:
                print(f"Attempting to use predicted line0 from previous field: {expected_line0}")
            best_p = None
            min_diff = 1000000
            # Search range: Only snap to a pulse if it is very close to the prediction (0.7 lines).
            # A wider range (e.g. 10 lines) causes it to snap to the wrong pulse (e.g. adjacent HSYNC/EQ)
            # when the correct VSYNC pulse is missing due to dropout.
            # search_range = 10.0 * linelen # Original search range
            search_range = 0.7 * linelen
            for p in filtered_pulses:
                diff = abs(p.start - expected_line0)
                if diff < search_range:
                    if diff < min_diff:
                        min_diff = diff
                        best_p = p
            if best_p:
                if DEBUG_PRINT:
                    print(f"Found pulse near prediction: {best_p.start} (diff {min_diff/linelen:.2f} lines)")
                line_0 = best_p.start
                if expected_first_field is not None:
                    first_field = expected_first_field
                    first_field_confidence = 50
            elif relaxed and expected_line0 > 0:
                if DEBUG_PRINT:
                    print(f"No pulse found near prediction, forcing expected location: {expected_line0}")
                line_0 = expected_line0
                if expected_first_field is not None:
                    first_field = expected_first_field
                    first_field_confidence = 40
            else:
                if DEBUG_PRINT:
                    print("Prediction available but no matching pulse found and relaxed mode disabled.")
                if line_0 is not None and line_0 > (linelen * (frame_lines - 1) / 2):
                    ldd.logger.info(
                        "WARNING, line0 hsync not found for current field, probably skipping one field."
                    )


    if line_0 is not None:
        if DEBUG_PLOT:
            long_pulses = list(
                filter(
                    lambda p: inrange(p[PULSE_LEN], lt_vsync[0], lt_vsync[1] * 10),
                    raw_pulses,
                )
            )
            debug_plot_line0_fallback(
                demod_05, long_pulses, filtered_pulses, raw_pulses, line_0, None
            )
        return line_0, None, True, first_field, first_field_confidence

    # 5th try: just find the last hsync in front of a long block

    # TODO: get max len from field.
    long_pulses = list(
        filter(
            lambda p: inrange(p[PULSE_LEN], lt_vsync[0], lt_vsync[1] * 10), raw_pulses
        )
    )

    if long_pulses:
        # Offset from start of first vsync to first line
        # NOTE: Not technically to first line but to the loc that would be expected for getLine0.
        # may need tweaking..

        first_long_pulse_pos = long_pulses[0][PULSE_START]

        # TODO: Optimize this
        # TODO: This will not give the correct result if the last hsync is damaged somehow, need
        # to add some compensation for that case.
        # Look for the last vsync before the vsync area as that is what
        # the other functions want.
        for p in valid_pulses:
            if p[1][PULSE_START] > first_long_pulse_pos:
                break
            if p[0] == P_HSYNC:
                line_0 = p[1][PULSE_START]

        if line_0 is None:
            ldd.logger.info(
                "WARNING, line0 hsync not found, guessing something, result may be garbled."
            )
            line_0 = first_long_pulse_pos - (3 * linelen)

        offset = num_eq_pulses * linelen

        # If we see exactly 2 groups of 3 long pulses, assume that we are dealing with a 240p/288p signal and
        # use the second group as loc of last line
        # TODO: we also have examples where vsync is one very long pulse, need to sort that too here.
        # TODO: Needs to be validated properly on 240p/288p input
        last_lineloc = (
            long_pulses[3][PULSE_START] - offset
            if len(long_pulses) == 6
            and long_pulses[3][PULSE_START] - long_pulses[2][PULSE_START]
            > (lt_vsync[1] * 10)
            else None
        )
        if DEBUG_PLOT:
            debug_plot_line0_fallback(
                demod_05, long_pulses, filtered_pulses, raw_pulses, line_0, last_lineloc
            )
        return line_0, last_lineloc, True, -1, -1
    else:
        if DEBUG_PLOT:
            debug_plot_line0_fallback(
                demod_05, long_pulses, filtered_pulses, raw_pulses, None, None
            )
        return None, None, None, None, None


def _run_vblank_state_machine(raw_pulses, line_timings, num_pulses, in_line_len):
    """Look though raw_pulses for a set valid vertical sync pulse series.
    num_pulses_half: number of equalization pulses per section / 2
    """
    done = False
    num_pulses_half = num_pulses / 2

    vsyncs = []  # VSYNC area (first broad pulse->first EQ after broad pulses)

    validpulses = []
    vsync_start = None

    # state_end tracks the earliest expected phase transition...
    state_end = 0
    # ... and state length is set by the phase transition to set above (in H)
    state_length = None

    lt_hsync = line_timings["hsync"]
    lt_eq = line_timings["eq"]
    lt_vsync = line_timings["vsync"]

    # state order: HSYNC -> EQPUL1 -> VSYNC -> EQPUL2 -> HSYNC
    HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

    # test_list = _to_seq(_to_type_list(raw_pulses, lt_hsync, lt_eq, lt_vsync), num_pulses)
    # print_output_types(test_list)
    # print("Is valid: ", _is_valid_seq(test_list, num_pulses))

    for p in raw_pulses:
        spulse = None

        state = validpulses[-1][0] if len(validpulses) > 0 else -1

        if state == -1:
            # First valid pulse must be a regular HSYNC
            if inrange(p.len, *lt_hsync):
                spulse = (HSYNC, p)
        elif state == HSYNC:
            # HSYNC can transition to EQPUL/pre-vsync at the end of a field
            if inrange(p.len, *lt_hsync):
                spulse = (HSYNC, p)
            elif inrange(p.len, *lt_eq):
                spulse = (EQPL1, p)
                state_length = num_pulses_half
            elif inrange(p.len, *lt_vsync):
                # should not happen(tm)
                vsync_start = len(validpulses) - 1
                spulse = (VSYNC, p)
        elif state == EQPL1:
            if inrange(p.len, *lt_eq):
                spulse = (EQPL1, p)
            elif inrange(p.len, *lt_vsync):
                # len(validpulses)-1 before appending adds index to first VSYNC pulse
                vsync_start = len(validpulses) - 1
                spulse = (VSYNC, p)
                state_length = num_pulses_half
            elif inrange(p.len, *lt_hsync):
                # previous state transition was likely in error!
                spulse = (HSYNC, p)
        elif state == VSYNC:
            if inrange(p.len, *lt_eq):
                # len(validpulses)-1 before appending adds index to first EQ pulse
                vsyncs.append((vsync_start, len(validpulses) - 1))
                spulse = (EQPL2, p)
                state_length = num_pulses_half
            elif inrange(p.len, *lt_vsync):
                spulse = (VSYNC, p)
            elif p.start > state_end and inrange(p.len, *lt_hsync):
                spulse = (HSYNC, p)
        elif state == EQPL2:
            if inrange(p.len, *lt_eq):
                spulse = (EQPL2, p)
            elif inrange(p.len, *lt_hsync):
                spulse = (HSYNC, p)
                done = True

        if spulse is not None and spulse[0] != state:
            if spulse[1].start < state_end:
                spulse = None
            elif state_length:
                state_end = spulse[1].start + ((state_length - 0.1) * in_line_len)
                state_length = None

        # Quality check
        if spulse is not None:
            good = (
                sync.pulse_qualitycheck(validpulses[-1], spulse, in_line_len)
                if len(validpulses)
                else False
            )

            validpulses.append((spulse[0], spulse[1], good))

        if done:
            return done, validpulses

    return done, validpulses


class FieldShared:
    def process(self):
        if self.prevfield:
            if self.readloc > self.prevfield.readloc:
                self.field_number = self.prevfield.field_number + 1
            else:
                self.field_number = self.prevfield.field_number
                ldd.logger.debug("readloc loc didn't advance.")
        else:
            self.field_number = 0

        super(FieldShared, self).process()

        # TODO: DO this in a cleaner manner.
        if self.rf.color_system == "405":
            self.linecount = 203 if self.isFirstField else 202
        elif self.rf.color_system == "819":
            self.linecount = 410 if self.isFirstField else 409

    def hz_to_output(self, input):
        if type(input) is np.ndarray:
            if self.rf.options.export_raw_tbc:
                return input.astype(np.single)
            else:
                ire0 = self.rf.DecoderParams["ire0"]
                if (
                    self.rf.options.ire0_adjust
                    and input.size == self.outlinecount * self.outlinelen
                ):
                    blank_levels = np.empty(self.outlinecount)
                    for i in range(0, self.outlinecount):
                        blank_levels[i] = np.median(
                            input[
                                i * self.outlinelen
                                + self.ire0_backporch[0] : i * self.outlinelen
                                + self.ire0_backporch[1]
                            ]
                        )
                    blank_levels = np.sort(blank_levels)
                    ire0 = np.mean(
                        blank_levels[
                            int(self.outlinecount / 3) : int(self.outlinecount * 2 / 3)
                        ]
                    )
                    ldd.logger.debug("calculated ire0: %.02f", ire0)
                return hz_to_output_array(
                    input,
                    ire0
                    + self.rf.DecoderParams["track_ire0_offset"][
                        self.rf.track_phase ^ (self.field_number % 2)
                    ],
                    self.rf.DecoderParams["hz_ire"],
                    self.rf.SysParams["outputZero"],
                    self.rf.DecoderParams["vsync_ire"],
                    self.out_scale,
                )

        # This is reached when it's called with a single value to scale an ire value to an output from build_json

        if self.rf.options.export_raw_tbc:
            return np.single(input)

        # Since this is just used for converting a value for the whole file don't do the track compensation here.
        reduced = (input - self.rf.DecoderParams["ire0"]) / self.rf.DecoderParams[
            "hz_ire"
        ]
        reduced -= self.rf.DecoderParams["vsync_ire"]

        return np.uint16(
            np.clip(
                (reduced * self.out_scale) + self.rf.SysParams["outputZero"], 0, 65535
            )
            + 0.5
        )

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldShared, self).downscale(
            final=False, *args, **kwargs
        )

        # hpf = utils.filter_simple(dsout, self.rf.Filters["NLHighPass"])
        # dsout = ynr(dsout, hpf, self.outlinelen)
        y_comb_value = self.rf.options.y_comb
        if y_comb_value != 0:
            dsout = y_comb(dsout, self.outlinelen, y_comb_value)

        if final:
            dsout = self.hz_to_output(dsout)
            self.dspicture = dsout

        return dsout, dsaudio, dsefm

    def _get_line0_fallback(self, valid_pulses):
        expected_line0 = None
        expected_first_field = None

        if hasattr(self.rf, "prev_first_hsync_readloc") and self.rf.prev_first_hsync_readloc != -1:
            prev_abs = self.rf.prev_first_hsync_readloc + self.rf.prev_first_hsync_loc
            lines_per_field = self.rf.SysParams["frame_lines"] / 2.0
            target_abs = prev_abs + (lines_per_field * self.meanlinelen)
            # Target VSYNC area approx 8 lines before active video (Start of VSYNC block)
            expected_line0_abs = target_abs - (8.0 * self.meanlinelen)
            expected_line0 = expected_line0_abs - self.readloc

            if hasattr(self.rf, "prev_first_field") and self.rf.prev_first_field != -1:
                expected_first_field = 1 - self.rf.prev_first_field

        res = get_line0_fallback(
            valid_pulses,
            self.rawpulses,
            self.data["video"]["demod_05"],
            self.lt_vsync,
            self.inlinelen,
            self.rf.SysParams["numPulses"],
            self.rf.SysParams["frame_lines"],
            relaxed=self.rf.options.relaxed_line0,
            expected_line0=expected_line0,
            expected_first_field=expected_first_field,
        )
        # Not needed after this.
        del self.lt_vsync
        return res

    def pulse_qualitycheck(self, prev_pulse, pulse):
        return sync.pulse_qualitycheck(prev_pulse, pulse, self.inlinelen)

    def run_vblank_state_machine(self, pulses, LT):
        """Determines if a pulse set is a valid vblank by running a state machine"""
        a = _run_vblank_state_machine(
            pulses, LT, self.rf.SysParams["numPulses"], self.inlinelen
        )
        return a

    def refinepulses(self):
        LT = self.get_timings()
        lt_hsync = LT["hsync"]
        lt_eq = LT["eq"]
        self.lt_vsync = LT["vsync"]

        HSYNC, EQPL1, VSYNC, EQPL2 = range(4)

        i = 0

        # print("lt_hsync: ", lt_hsync, " lt_eq: ", lt_eq, " lt_vsync: ", lt_vsync)

        # Pulse = namedtuple("Pulse", "start len")
        valid_pulses = []
        num_vblanks = 0

        # test_list = _to_seq(_to_type_list(self.rawpulses, lt_hsync, lt_eq, lt_vsync), self.rf.SysParams["numPulses"])
        # print_output_types(test_list)

        while i < len(self.rawpulses):
            curpulse = self.rawpulses[i]
            if inrange(curpulse.len, *lt_hsync):
                good = (
                    self.pulse_qualitycheck(valid_pulses[-1], (0, curpulse))
                    if len(valid_pulses)
                    else False
                )
                valid_pulses.append((HSYNC, curpulse, good))
                i += 1
            elif inrange(curpulse.len, lt_hsync[1], lt_hsync[1] * 3):
                # If the pulse is longer than expected, we could have ended up detecting the back
                # porch as sync.
                # try to move a bit lower to see if we hit a hsync.
                data = self.data["video"]["demod_05"][
                    curpulse.start : curpulse.start + curpulse.len
                ]
                threshold = self.rf.iretohz(self.rf.hztoire(data[0]) - 10)
                pulses = self.rf.resync.findpulses(data, threshold)
                if len(pulses):
                    newpulse = Pulse(curpulse.start + pulses[0].start, pulses[0].len)
                    self.rawpulses[i] = newpulse
                    curpulse = newpulse
                else:
                    # spulse = (HSYNC, self.rawpulses[i], False)
                    i += 1
            elif (
                i > 2
                and inrange(self.rawpulses[i].len, *lt_eq)
                and (len(valid_pulses) and valid_pulses[-1][0] == HSYNC)
            ):
                done, vblank_pulses = self.run_vblank_state_machine(
                    self.rawpulses[i - 2 : i + 24], LT
                )
                # print_output_order(i, done, vblank_pulses)
                if done:
                    [valid_pulses.append(p) for p in vblank_pulses[2:]]
                    i += len(vblank_pulses) - 2
                    num_vblanks += 1
                else:
                    # spulse = (HSYNC, self.rawpulses[i], False)
                    i += 1
            else:
                # spulse = (HSYNC, self.rawpulses[i], False)
                i += 1

        return valid_pulses  # , num_vblanks

    def _try_get_pulses(self, check_levels):
        self.rawpulses = self.rf.resync.get_pulses(self, check_levels)

        if (
            self.rawpulses is None
            or
            # when no pulses are found and there has not been a previous sync location and fallback vsync is not enabled
            (
                len(self.rawpulses) == 0
                and (
                    not hasattr(self.rf, "prev_first_hsync_loc")
                    or self.rf.options.fallback_vsync
                )
            )
        ):
            return NO_PULSES_FOUND

        self.validpulses = validpulses = self.refinepulses()
        meanlinelen = self.computeLineLen(validpulses)
        self.meanlinelen = meanlinelen

        # fill in empty values, when decoding starts
        if not hasattr(self.rf, "prev_first_hsync_loc"):
            self.rf.prev_first_hsync_readloc = -1
            self.rf.prev_first_hsync_loc = -1
            self.rf.prev_first_hsync_diff = -1

            if hasattr(self.prevfield, "isFirstField"):
                self.rf.prev_first_field = 1 if self.prevfield.isFirstField else 0
            else:
                self.rf.prev_first_field = -1

            if hasattr(self.prevfield, "isProgressiveField"):
                self.rf.prev_progressive_field = (
                    1 if self.prevfield.isProgressiveField else 0
                )
            else:
                self.rf.prev_progressive_field = -1

        # calculate in terms of lines to prevent integer overflow when seeking ahead large amounts
        if self.rf.prev_first_hsync_readloc != -1:
            prev_first_hsync_offset_lines = (
                self.rf.prev_first_hsync_readloc - self.readloc
            ) / meanlinelen
        else:
            prev_first_hsync_offset_lines = 0

        fallback_line0loc = None
        if (
            self.rf.options.fallback_vsync
            and hasattr(self, "lt_vsync")
            and self.lt_vsync is not None
        ):
            (
                fallback_line0loc,
                _,
                _,
                fallback_is_first_field,
                fallback_is_first_field_confidence,
            ) = self._get_line0_fallback(validpulses)

        if fallback_line0loc is None:
            fallback_line0loc = -1
            fallback_is_first_field = -1
            fallback_is_first_field_confidence = -1

        # find the location of the first hsync pulse (first line of video after the vsync pulses)
        # this function relies on the pulse type (hsync, vsync, eq pulse) being accurate in validpulses
        (
            line0loc,
            self.first_hsync_loc,
            self.first_hsync_loc_line,
            self.vblank_next,
            self.isFirstField,
            self.isProgressiveField,
            prev_hsync_diff,
            vblank_pulses,
        ) = sync.get_first_hsync_loc(
            validpulses,
            meanlinelen,
            1 if self.rf.system == "NTSC" else 0,
            self.rf.SysParams["field_lines"],
            self.rf.SysParams["numPulses"],
            self.rf.prev_first_field,
            prev_first_hsync_offset_lines,
            self.rf.prev_first_hsync_loc,
            self.rf.prev_first_hsync_diff,
            self.rf.options.field_order_confidence,
            fallback_line0loc,
            fallback_is_first_field,
            fallback_is_first_field_confidence,
        )

        # save the current hsync pulse location to the previous hsync pulse
        if self.first_hsync_loc is not None:
            self.rf.prev_first_hsync_readloc = self.readloc
            self.rf.prev_first_hsync_loc = self.first_hsync_loc
            self.rf.prev_first_hsync_diff = prev_hsync_diff

        self.rf.prev_first_field = self.isFirstField
        self.rf.prev_progressive_field = self.isProgressiveField

        # self.getLine0(validpulses, meanlinelen)

        return (
            line0loc,
            self.first_hsync_loc,
            self.first_hsync_loc_line,
            meanlinelen,
            vblank_pulses,
        )

    @property
    def compute_linelocs_issues(self):
        return self._compute_linelocs_issues

    def compute_linelocs(self):
        has_levels = self.rf.resync.has_levels()

        # Skip vsync serration/level detect if we already have levels from a previous field and
        # the option is enabled.
        # Also run level detection if we encountered issues in this function in the previous field.
        do_level_detect = (
            not self.rf.options.saved_levels
            or not has_levels
            or self.rf.compute_linelocs_issues is True
        )
        res = self._try_get_pulses(do_level_detect)
        if (
            res == NO_PULSES_FOUND or res[0] == None or self.sync_confidence == 0
        ) and not do_level_detect:
            # If we failed to fild valid pulses with the previous levels
            # and level detection was skipped, try again
            # running the full level detection
            ldd.logger.debug("Search for pulses failed, re-checking levels")
            res = self._try_get_pulses(True)

        self.rf.compute_linelocs_issues = True

        if res == NO_PULSES_FOUND:
            ldd.logger.error("Unable to find any sync pulses, jumping 100 ms")
            return None, None, int(self.rf.freq_hz / 10)

        line0loc, first_hsync_loc, first_hsync_loc_line, meanlinelen, vblank_pulses = (
            res
        )
        validpulses = self.validpulses

        # TODO: This is set here for NTSC, but in the PAL base class for PAL in process() it seems..
        # For 405-line it's done in fieldTypeC.process as of now to override that.
        self.linecount = 263 if self.isFirstField else 262

        # Number of lines to actually process.  This is set so that the entire following
        # VSYNC is processed
        proclines = self.outlinecount + self.lineoffset + 10

        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("raw_pulses"):
            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                threshold=self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2),
            )
        # threshold=self.rf.resync.last_pulse_threshold

        if first_hsync_loc is None:
            if self.initphase is False:
                ldd.logger.error("Unable to determine start of field - dropping field")
            return None, None, self.inlinelen * 100

        # If we don't have enough data at the end, move onto the next field
        lastline = (len(self.data["input"]) - line0loc) / meanlinelen - 1

        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("raw_pulses"):
            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                extra_lines=[line0loc],
            )

        if lastline < proclines:
            if self.prevfield is not None:
                ldd.logger.info(
                    "lastline = %s, proclines = %s, meanlinelen = %s, line0loc = %s)",
                    lastline,
                    proclines,
                    meanlinelen,
                    line0loc,
                )
                ldd.logger.info("Did not find the expected number of lines (lastline < proclines) , skipping a tiny bit")
            return None, None, max(line0loc - (meanlinelen * 20), self.inlinelen)

        linelocs, lineloc_errs, last_validpulse = sync.valid_pulses_to_linelocs(
            validpulses,
            first_hsync_loc,
            first_hsync_loc_line,
            meanlinelen,
            self.rf.hsync_tolerance,
            proclines,
            1.9,
        )

        self.linelocs0 = linelocs.copy()

        # ldd.logger.info("line0loc %s %s", int(line0loc), int(self.meanlinelen))

        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("line_locs"):
            line0loc_plot = -1 if line0loc == None else line0loc
            first_hsync_loc_plot = -1 if first_hsync_loc == None else first_hsync_loc
            vblank_next_plot = -1 if self.vblank_next == None else self.vblank_next

            print(
                "line0loc",
                line0loc_plot,
                "first_hsync_loc",
                first_hsync_loc_plot,
                "vblank_next",
                vblank_next_plot,
            )

            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                linelocs=linelocs,
                pulses=validpulses,
                vblank_lines=vblank_pulses,
                extra_lines=[line0loc_plot, first_hsync_loc_plot, vblank_next_plot],
            )

        if self.vblank_next is None:
            nextfield = linelocs[self.outlinecount - 7]
        else:
            nextfield = self.vblank_next - (self.inlinelen * 8)

        if np.count_nonzero(lineloc_errs) < 30:
            self.rf.compute_linelocs_issues = False
        elif self.rf.options.saved_levels:
            ldd.logger.debug(
                "Possible sync issues, re-running level detection on next field!"
            )

        return linelocs, lineloc_errs, nextfield

    def refine_linelocs_hsync(self):
        if not self.rf.options.skip_hsync_refine:
            threshold = (
                self.rf.resync.last_pulse_threshold
                if self.rf.options.hsync_refine_use_threshold
                else self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2)
            )

            return sync.refine_linelocs_hsync(self, self.linebad, threshold)
        else:
            return self.linelocs1.copy()

        if False:
            import timeit

            linebad = self.linebad.copy()
            linebad_b = self.linebad.copy()
            time = timeit.timeit(
                lambda: sync.refine_linelocs_hsync(self, linebad), number=100
            )
            time2 = timeit.timeit(
                lambda: self._refine_linelocs_hsync(linebad_b), number=100
            )
            print("time", time)
            print("time2", time2)
            linebad1 = self.linebad.copy()
            linebad2 = self.linebad.copy()
            # print(np.asanyarray(self._refine_linelocs_hsync(linebad1)) - np.asanyarray(refine_linelocs_hsync_t(self, linebad2)))
            assert np.all(
                np.asanyarray(self._refine_linelocs_hsync(linebad1))
                == np.asanyarray(sync.refine_linelocs_hsync(self, linebad2))
            )
        # return sync.refine_linelocs_hsync(self, self.linebad)

    def _refine_linelocs_hsync(self, linebad):
        """Refine the line start locations using horizontal sync data."""
        # Old python variant for dev comparisons, not used - will be removed later.
        # linelocs2 = np.asarray(self.linelocs1, dtype=np.float64)
        linelocs2 = self.linelocs1.copy()
        normal_hsync_length = self.usectoinpx(self.rf.SysParams["hsyncPulseUS"])

        demod_05 = self.data["video"]["demod_05"]
        one_usec = self.rf.freq

        for i in range(len(self.linelocs1)):
            # skip VSYNC lines, since they handle the pulses differently
            if inrange(i, 3, 6) or (self.rf.system == "PAL" and inrange(i, 1, 2)):
                linebad[i] = True
                continue

            # refine beginning of hsync

            # start looking 1 msec back
            ll1 = self.linelocs1[i] - one_usec
            # and locate the next time the half point between hsync and 0 is crossed.
            zc = lddu.calczc(
                demod_05,
                ll1,
                self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2),
                reverse=False,
                count=one_usec * 2,
            )

            right_cross = None

            if not self.rf.options.disable_right_hsync:
                right_cross = lddu.calczc(
                    demod_05,
                    ll1 + (normal_hsync_length) - one_usec,
                    self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2),
                    reverse=False,
                    count=one_usec * 3,
                )
            right_cross_refined = False

            # If the crossing exists, we can check if the hsync pulse looks normal and
            # refine it.
            if zc is not None and not linebad[i]:
                linelocs2[i] = zc

                # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
                hsync_area = demod_05[
                    int(zc - (one_usec * 0.75)) : int(zc + (one_usec * 8))
                ]
                if lddu.nb_min(hsync_area) < self.rf.iretohz(-55) or lddu.nb_max(
                    hsync_area
                ) > self.rf.iretohz(30):
                    # don't use the computed value here if it's bad
                    linebad[i] = True
                    linelocs2[i] = self.linelocs1[i]
                else:
                    porch_level = lddu.nb_median(
                        demod_05[int(zc + (one_usec * 8)) : int(zc + (one_usec * 9))]
                    )
                    sync_level = lddu.nb_median(
                        demod_05[int(zc + (one_usec * 1)) : int(zc + (one_usec * 2.5))]
                    )

                    # Re-calculate the crossing point using the mid point between the measured sync
                    # and porch levels
                    zc2 = lddu.calczc(
                        demod_05,
                        ll1,
                        (porch_level + sync_level) / 2,
                        reverse=False,
                        count=400,
                    )

                    # any wild variation here indicates a failure
                    if zc2 is not None and np.abs(zc2 - zc) < (one_usec / 2):
                        linelocs2[i] = zc2
                    else:
                        linebad[i] = True
            else:
                linebad[i] = True

            # Check right cross
            if right_cross is not None:
                zc2 = None

                zc_fr = right_cross - normal_hsync_length

                # The hsync area, burst, and porches should not leave -50 to 30 IRE (on PAL or NTSC)
                hsync_area = demod_05[
                    int(zc_fr - (one_usec * 0.75)) : int(zc_fr + (one_usec * 8))
                ]
                if lddu.nb_min(hsync_area) > self.rf.iretohz(-55) and lddu.nb_max(
                    hsync_area
                ) < self.rf.iretohz(30):
                    porch_level = lddu.nb_median(
                        demod_05[
                            int(zc_fr + (one_usec * 8)) : int(zc_fr + (one_usec * 9))
                        ]
                    )
                    sync_level = lddu.nb_median(
                        demod_05[
                            int(zc_fr + (one_usec * 1)) : int(zc_fr + (one_usec * 2.5))
                        ]
                    )

                    # Re-calculate the crossing point using the mid point between the measured sync
                    # and porch levels
                    zc2 = lddu.calczc(
                        demod_05,
                        ll1 + normal_hsync_length - one_usec,
                        (porch_level + sync_level) / 2,
                        reverse=False,
                        count=400,
                    )

                    # any wild variation here indicates a failure
                    if zc2 is not None and np.abs(zc2 - right_cross) < (one_usec / 2):
                        right_cross = zc2
                        right_cross_refined = True

            if linebad[i]:
                linelocs2[i] = self.linelocs1[
                    i
                ]  # don't use the computed value here if it's bad

            if right_cross is not None:
                # right_locs[i] = right_cross
                # hsync_from_right[i] = right_cross - normal_hsync_length + 2.25

                # If we get a good result from calculating hsync start from the
                # right side of the hsync pulse, we use that as it's less likely
                # to be messed up by overshoot.
                if right_cross_refined:
                    linebad[i] = False
                    linelocs2[i] = right_cross - normal_hsync_length + 2.25

        return linelocs2

    def getBlankRange(self, validpulses, start=0):
        """Look through pulses to fit a group that fit as a blanking area.
        Overridden to lower the threshold a little as the default
        discarded some distorted/non-standard ones.
        """
        vp_type = np.array([p[0] for p in validpulses])

        vp_vsyncs = np.where(vp_type[start:] == ldd.VSYNC)[0]
        firstvsync = vp_vsyncs[0] + start if len(vp_vsyncs) else None

        if firstvsync is None or firstvsync < 10:
            if start == 0:
                ldd.logger.debug("No vsync found!")
            return None, None

        for newstart in range(firstvsync - 10, firstvsync - 4):
            blank_locs = np.where(vp_type[newstart:] > 0)[0]
            if len(blank_locs) == 0:
                continue

            firstblank = blank_locs[0] + newstart
            hsync_locs = np.where(vp_type[firstblank:] == 0)[0]

            if len(hsync_locs) == 0:
                continue

            lastblank = hsync_locs[0] + firstblank - 1

            if (lastblank - firstblank) > formats.BLANK_LENGTH_THRESHOLD:
                return firstblank, lastblank

        # there isn't a valid range to find, or it's impossibly short
        return None, None

    def calc_burstmedian(self):
        # Set this to a constant value for now to avoid the comb filter messing with chroma levels.
        return 1.0

    def getpulses(self):
        """Find sync pulses in the demodulated video signal"""
        return self.rf.resync.get_pulses(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as tapes have large variations in line pos
        Due to e.g head switch.
        compute errors based off the second derivative - if it exceeds 1 something's wrong,
        and if 4 really wrong...
        """
        return baserr

    def dropout_detect(self):
        return detect_dropouts_rf(self, self.rf.dod_options)

    def get_timings(self):
        """Get the expected length and tolerance for sync pulses. Overriden to allow wider tolerance."""

        # Get the defaults - this works somehow because python.
        LT = super(FieldShared, self).get_timings()

        hsync_min = LT["hsync_median"] + self.usectoinpx(-0.7)
        hsync_max = LT["hsync_median"] + self.usectoinpx(0.7)

        LT["hsync"] = (hsync_min, hsync_max)

        eq_min = (
            self.usectoinpx(self.rf.SysParams["eqPulseUS"] - formats.EQ_PULSE_TOLERANCE)
            + LT["hsync_offset"]
        )
        eq_max = (
            self.usectoinpx(self.rf.SysParams["eqPulseUS"] + formats.EQ_PULSE_TOLERANCE)
            + LT["hsync_offset"]
        )

        LT["eq"] = (eq_min, eq_max)

        return LT

    def fix_badlines(self, linelocs_in, linelocs_backup_in=None):
        # No longer needed. Bad line locations are fixed in sync.pyx
        return linelocs_in
        """Go through the list of lines marked bad and guess something for the lineloc based on
        previous/next good lines"""
        # Overridden to add some further logic.
        self.linebad = self.compute_deriv_error(linelocs_in, self.linebad)
        linelocs = np.array(linelocs_in.copy())

        if linelocs_backup_in is not None:
            linelocs_backup = np.array(linelocs_backup_in.copy())
            badlines = np.isnan(linelocs)
            linelocs[badlines] = linelocs_backup[badlines]

        # If the next good line is this far down, don't use it to calculate guessed lineloc
        # as it may be below head switch and thus distort.
        last_from_bottom = len(linelocs) - 16

        for l in np.where(self.linebad)[0]:
            prevgood = l - 1
            nextgood = l + 1

            while prevgood >= 0 and self.linebad[prevgood]:
                prevgood -= 1

            while nextgood < len(linelocs) and self.linebad[nextgood]:
                nextgood += 1

            firstcheck = 0 if self.rf.system == "PAL" else 1

            if prevgood >= firstcheck and nextgood < (len(linelocs) + self.lineoffset):
                if nextgood > last_from_bottom:
                    # Don't use prev+next for these as that could cross head switch.
                    if prevgood > last_from_bottom + 4:
                        guess_len = linelocs[prevgood] - linelocs[prevgood - 1]
                        linelocs[l] = linelocs[l - 1] + guess_len
                else:
                    gap = (linelocs[nextgood] - linelocs[prevgood]) / (
                        nextgood - prevgood
                    )
                    linelocs[l] = (gap * (l - prevgood)) + linelocs[prevgood]

        return linelocs


class FieldPALShared(FieldShared, ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALShared, self).__init__(*args, **kwargs)
        self.ire0_backporch = (96, 160)

    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as most regular band tape formats does not use have a pilot burst.
        Tape formats that do have it will need separate logic for it anyhow.
        """
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs

    def determine_field_number(self):
        """Workaround to shut down phase id mismatch warnings, the actual code
        doesn't work properly with the vhs output at the moment."""
        return 1 + (self.rf.field_number % 8)


class FieldNTSCShared(FieldShared, ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCShared, self).__init__(*args, **kwargs)
        self.fieldPhaseID = 0
        self.ire0_backporch = (74, 124)

    def refine_linelocs_burst(self, linelocs=None):
        """Override this as it's LD specific
        At some point in the future we could maybe use the burst location to improve hsync accuracy,
        but ignore it for now.
        """
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        return linelocs


class FieldPALVHS(FieldPALShared):
    def __init__(self, *args, **kwargs):
        super(FieldPALVHS, self).__init__(*args, **kwargs)

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALVHS, self).downscale(
            final=final, *args, **kwargs
        )
        dschroma = decode_chroma(self, self.rf.DecoderParams["chroma_rotation"])

        return (dsout, dschroma), dsaudio, dsefm

    def try_detect_track(self):
        return try_detect_track_vhs_pal(self, self.rf.DecoderParams["chroma_rotation"])


class FieldPALSVHS(FieldPALVHS):
    """Add PAL SVHS-specific stuff (deemp, pilot burst etc here)"""

    def __init__(self, *args, **kwargs):
        super(FieldPALSVHS, self).__init__(*args, **kwargs)


class FieldPALUMatic(FieldPALShared):
    def __init__(self, *args, **kwargs):
        super(FieldPALUMatic, self).__init__(*args, **kwargs)

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALUMatic, self).downscale(
            final=final, *args, **kwargs
        )
        dschroma = decode_chroma_simple(self)

        return (dsout, dschroma), dsaudio, dsefm


class FieldPALBetamax(FieldPALShared):
    def __init__(self, *args, **kwargs):
        super(FieldPALBetamax, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        _ = try_detect_track_betamax_pal(self)
        return 0, False

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALBetamax, self).downscale(
            final=final, *args, **kwargs
        )

        dschroma = decode_chroma(
            self, chroma_rotation=self.rf.DecoderParams["chroma_rotation"]
        )

        return (dsout, dschroma), dsaudio, dsefm


class FieldPALVideo8(FieldPALShared):
    def __init__(self, *args, **kwargs):
        super(FieldPALVideo8, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        # PAL Video8 uses the same chroma phase rotation setup as PAL VHS.
        return try_detect_track_vhs_pal(self, self.rf.DecoderParams["chroma_rotation"])

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALVideo8, self).downscale(
            final=final, *args, **kwargs
        )

        dschroma = decode_chroma(
            self,
            chroma_rotation=self.rf.DecoderParams["chroma_rotation"],
            do_chroma_deemphasis=True,
        )

        return (dsout, dschroma), dsaudio, dsefm


class FieldPALTypeC(FieldPALShared, ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALTypeC, self).__init__(*args, **kwargs)

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldPALTypeC, self).downscale(
            final=final, *args, **kwargs
        )

        return (dsout, None), dsaudio, dsefm


class FieldNTSCVHS(FieldNTSCShared):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCVHS, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        return try_detect_track_ntsc(self, self.rf.DecoderParams["chroma_rotation"])

    def downscale(self, final=False, *args, **kwargs):
        """Downscale the channels and upconvert chroma to standard color carrier frequency."""
        dsout, dsaudio, dsefm = super(FieldNTSCVHS, self).downscale(
            final=final, *args, **kwargs
        )

        dschroma = decode_chroma(self, self.rf.DecoderParams["chroma_rotation"])

        self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, dschroma), dsaudio, dsefm


class FieldNTSCSVHS(FieldNTSCVHS):
    """Add NTSC SVHS-specific stuff (deemp etc here)"""

    def __init__(self, *args, **kwargs):
        super(FieldNTSCSVHS, self).__init__(*args, **kwargs)


class FieldNTSCBetamax(FieldNTSCShared):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCBetamax, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        return try_detect_track_ntsc(self, self.rf.DecoderParams["chroma_rotation"])

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldNTSCBetamax, self).downscale(
            final=final, *args, **kwargs
        )

        dschroma = decode_chroma(self, self.rf.DecoderParams["chroma_rotation"])

        return (dsout, dschroma), dsaudio, dsefm


class FieldMPALVHS(FieldNTSCVHS):
    def __init__(self, *args, **kwargs):
        super(FieldMPALVHS, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        return try_detect_track_vhs_pal(self, self.rf.DecoderParams["chroma_rotation"])


class FieldNTSCUMatic(FieldNTSCShared):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCUMatic, self).__init__(*args, **kwargs)

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldNTSCUMatic, self).downscale(
            final=final, *args, **kwargs
        )
        dschroma = decode_chroma_simple(self)

        self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, dschroma), dsaudio, dsefm


class FieldNTSCTypeC(FieldShared, ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCTypeC, self).__init__(*args, **kwargs)

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldNTSCTypeC, self).downscale(
            final=final, *args, **kwargs
        )

        # self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, None), dsaudio, dsefm


class FieldNTSCVideo8(FieldNTSCShared):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCVideo8, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        return try_detect_track_ntsc(self, self.rf.DecoderParams["chroma_rotation"])

    def downscale(self, final=False, *args, **kwargs):
        """Downscale the channels and upconvert chroma to standard color carrier frequency."""
        dsout, dsaudio, dsefm = super(FieldNTSCVideo8, self).downscale(
            final=final, *args, **kwargs
        )

        dschroma = decode_chroma(
            self, self.rf.DecoderParams["chroma_rotation"], do_chroma_deemphasis=True
        )

        self.fieldPhaseID = get_field_phase_id(self)

        return (dsout, dschroma), dsaudio, dsefm


class FieldMESECAMVHS(FieldPALShared):
    def __init__(self, *args, **kwargs):
        super(FieldMESECAMVHS, self).__init__(*args, **kwargs)

    def try_detect_track(self):
        return 0, False

    def downscale(self, final=False, *args, **kwargs):
        dsout, dsaudio, dsefm = super(FieldMESECAMVHS, self).downscale(
            final=final, *args, **kwargs
        )
        dschroma = decode_chroma_simple(self)

        return (dsout, dschroma), dsaudio, dsefm
