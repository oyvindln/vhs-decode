import numpy as np
from numba import njit

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
        if (
            tape_format == "UMATIC"
            or tape_format == "UMATIC_HI"
            or tape_format == "EIAJ"
            or tape_format == "VCR"
            or tape_format == "VCR_LP"
        ):
            # These use simple chroma downconversion and filters.
            field_class = FieldPALUMatic
        elif tape_format == "TYPEC" or tape_format == "TYPEB":
            field_class = FieldPALTypeC
        elif tape_format == "SVHS":
            field_class = FieldPALSVHS
        elif tape_format == "BETAMAX":
            field_class = FieldPALBetamax
        elif tape_format == "VIDEO8" or tape_format == "HI8":
            field_class = FieldPALVideo8
        else:
            if tape_format != "VHS" and tape_format != "VHSHQ":
                ldd.logger.info(
                    "Tape format unimplemented for PAL, using VHS field class."
                )
            field_class = FieldPALVHS
    elif system == "NTSC":
        if tape_format == "UMATIC":
            field_class = FieldNTSCUMatic
        elif tape_format == "TYPEC" or tape_format == "TYPEB":
            field_class = FieldNTSCTypeC
        elif tape_format == "SVHS":
            field_class = FieldNTSCSVHS
        elif tape_format == "BETAMAX" or tape_format == "BETAMAX_HIFI":
            field_class = FieldNTSCBetamax
        elif tape_format == "VIDEO8" or tape_format == "HI8":
            field_class = FieldNTSCVideo8
        else:
            if tape_format != "VHS" and tape_format != "VHSHQ":
                ldd.logger.info(
                    "Tape format unimplemented for NTSC, using VHS field class."
                )
            field_class = FieldNTSCVHS
    elif system == "MPAL" and tape_format == "VHS":
        field_class = FieldMPALVHS
    elif system == "MESECAM" and tape_format == "VHS":
        field_class = FieldMESECAMVHS

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


def get_line0_fallback(
    valid_pulses, raw_pulses, demod_05, lt_vsync, linelen, num_eq_pulses
):
    """
    Try a more primitive way of locating line 0 if the normal approach fails.
    This doesn't actually fine line 0, rather it locates the approx position of the last vsync before vertical blanking
    as the later code is designed to work off of that.
    Currently we basically just look for the first "long" pulse that could be start of vsync pulses in
    e.g a 240p/280p signal (that is, a pulse that is at least vsync pulse length.)
    """

    PULSE_START = 0
    PULSE_LEN = 1

    # TODO: get max len from field.
    long_pulses = list(
        filter(
            lambda p: inrange(p[PULSE_LEN], lt_vsync[0], lt_vsync[1] * 10), raw_pulses
        )
    )

    if False:
        # len(validpulses) > 300:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(demod_05)

        for raw_pulse in long_pulses:
            ax2.axvline(raw_pulse.start, color="#910000")
            ax2.axvline(raw_pulse.start + raw_pulse.len, color="#090909")

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
    if long_pulses:
        # Offset from start of first vsync to first line
        # NOTE: Not technically to first line but to the loc that would be expected for getLine0.
        # may need tweaking..

        first_long_pulse_pos = long_pulses[0][PULSE_START]

        line_0 = None
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
        return line_0, last_lineloc, True
    else:
        return None, None, None


def _run_vblank_state_machine(raw_pulses, line_timings, num_pulses, in_line_len):
    """Look though raw_pulses for a set valid vertical sync pulse seires.
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

    def hz_to_output(self, input):
        if type(input) == np.ndarray:
            if self.rf.options.export_raw_tbc:
                return input.astype(np.single)
            else:
                return hz_to_output_array(
                    input,
                    self.rf.DecoderParams["ire0"]
                    + self.rf.DecoderParams["track_ire0_offset"][
                        self.rf.track_phase ^ (self.field_number % 2)
                    ],
                    self.rf.DecoderParams["hz_ire"],
                    self.rf.SysParams["outputZero"],
                    self.rf.DecoderParams["vsync_ire"],
                    self.out_scale,
                )

        # Not sure what situations will cause input to not be a ndarray.

        if self.rf.options.export_raw_tbc:
            # Not sure if this will work.
            return np.single(input)

        reduced = (
            input
            - self.rf.DecoderParams["ire0"]
            - self.rf.DecoderParams["track_ire0_offset"][
                self.rf.track_phase ^ (self.field_number % 2)
            ]
        ) / self.rf.DecoderParams["hz_ire"]
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
        res = get_line0_fallback(
            valid_pulses,
            self.rawpulses,
            self.data["video"]["demod_05"],
            self.lt_vsync,
            self.inlinelen,
            self.rf.SysParams["numPulses"],
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
        # self.rawpulses = self.getpulses()
        if self.rawpulses is None or len(self.rawpulses) == 0:
            return NO_PULSES_FOUND, None

        self.validpulses = validpulses = self.refinepulses()
        meanlinelen = self.computeLineLen(validpulses)
        self.meanlinelen = meanlinelen

        # line0loc, lastlineloc, self.isFirstField = self.getLine0(validpulses)
        # NOTE: This seems to get the position of the last normal vsync before the vertical blanking interval rather than line0
        # (which is only the same thing on the top field of 525-line system signals)
        return self.getLine0(validpulses, meanlinelen), meanlinelen

    def compute_linelocs(self):
        has_levels = self.rf.resync.has_levels()
        # Skip vsync serration/level detect if we already have levels from a previous field and
        # the option is enabled.
        do_level_detect = not self.rf.options.saved_levels or not has_levels
        res, meanlinelen = self._try_get_pulses(do_level_detect)
        if (
            res == NO_PULSES_FOUND or res[0] == None or self.sync_confidence == 0
        ) and not do_level_detect:
            # If we failed to fild valid pulses with the previous levels
            # and level detection was skipped, try again
            # running the full level detection
            ldd.logger.debug("Search for pulses failed, re-checking levels")
            res, meanlinelen = self._try_get_pulses(True)

        if res == NO_PULSES_FOUND:
            ldd.logger.error("Unable to find any sync pulses, jumping 100 ms")
            return None, None, int(self.rf.freq_hz / 10)

        line0loc, lastlineloc, self.isFirstField = res
        validpulses = self.validpulses

        if self.rf.options.fallback_vsync and (
            not line0loc or self.sync_confidence == 0
        ):
            self.sync_confidence = 0
            line0loc_t, lastlineloc_t, is_first_field_t = self._get_line0_fallback(
                validpulses
            )
            if line0loc_t:
                ldd.logger.debug("Using fallback vsync, signal may be non-standard.")
                line0loc = line0loc_t
                lastlineloc = lastlineloc_t
                self.isFirstField = (
                    not self.prevfield.isFirstField if self.prevfield else True
                )
                self.sync_confidence = 10
        # Not sure if this is used for video.
        self.linecount = 263 if self.isFirstField else 262

        # Number of lines to actually process.  This is set so that the entire following
        # VSYNC is processed
        proclines = self.outlinecount + self.lineoffset + 10
        if self.rf.system == "PAL":
            proclines += 3

        lastlineloc_or_0 = lastlineloc

        # It's possible for getLine0 to return None for lastlineloc
        if lastlineloc is not None:
            numlines = (lastlineloc - line0loc) / self.inlinelen
            self.skipdetected = numlines < (self.linecount - 5)
        else:
            # Make sure we set this to 0 and not None for numba
            # to be able to compile valid_pulses_to_linelocs correctly.
            lastlineloc_or_0 = 0.0
            self.skipdetected = False

        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("raw_pulses"):
            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                threshold=self.rf.iretohz(self.rf.SysParams["vsync_ire"] / 2),
            )
        # threshold=self.rf.resync.last_pulse_threshold

        if line0loc is None:
            if self.initphase is False:
                ldd.logger.error("Unable to determine start of field - dropping field")
            return None, None, self.inlinelen * 100

        # If we don't have enough data at the end, move onto the next field
        lastline = (self.rawpulses[-1].start - line0loc) / meanlinelen
        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("raw_pulses"):
            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                extra_lines=[line0loc, lastlineloc_or_0],
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
                ldd.logger.info("lastline < proclines , skipping a tiny bit")
            return None, None, max(line0loc - (meanlinelen * 20), self.inlinelen)

        linelocs_dict, _ = sync.valid_pulses_to_linelocs(
            validpulses,
            line0loc,
            self.skipdetected,
            meanlinelen,
            self.linecount,
            self.rf.hsync_tolerance,
            lastlineloc_or_0,
        )

        rv_err = np.full(proclines, False)

        # Convert dictionary into array, then fill in gaps
        linelocs = np.asarray(
            [
                linelocs_dict[l] if l in linelocs_dict else -1
                for l in range(0, proclines)
            ]
        )
        linelocs_filled = linelocs.copy()

        self.linelocs0 = linelocs.copy()

        if linelocs_filled[0] < 0:
            next_valid = None
            for i in range(0, self.outlinecount + 1):
                if linelocs[i] > 0:
                    next_valid = i
                    break

            if next_valid is None:
                # If we don't find anything valid, guess something to avoid dropping fields
                prev_line0 = (
                    np.int64(self.prevfield.linelocs0[0])
                    if self.prevfield is not None
                    else 0
                )

                if prev_line0 > 0:
                    linelocs_filled = self.prevfield.linelocs0 - prev_line0 + line0loc
                else:
                    linelocs_filled[0] = line0loc
                linelocs = linelocs_filled
                ldd.logger.warning(
                    "no valid lines found! Guessing or using values for previous field so result will probably be garbled!"
                )
                rv_err[1:] = True
            else:
                linelocs_filled[0] = linelocs_filled[next_valid] - (
                    next_valid * meanlinelen
                )

            if linelocs_filled[0] < self.inlinelen:
                ldd.logger.info(
                    "linelocs_filled[0] too short! (%s) should be at least %s. Skipping a bit...",
                    linelocs_filled[0],
                    self.inlinelen,
                )
                # Skip a bit if no line positions were filled (which causes following code to fail for now).
                # Amount may need tweaking.
                return None, None, line0loc + (self.inlinelen * self.outlinecount - 7)

        for l in range(1, proclines):
            if linelocs_filled[l] < 0:
                rv_err[l] = True

                prev_valid = None
                next_valid = None

                for i in range(l, -1, -1):
                    if linelocs[i] > 0:
                        prev_valid = i
                        break
                for i in range(l, self.outlinecount + 1):
                    if linelocs[i] > 0:
                        next_valid = i
                        break

                if prev_valid is None:
                    avglen = self.inlinelen
                    linelocs_filled[l] = linelocs[next_valid] - (
                        avglen * (next_valid - l)
                    )
                elif next_valid is not None:
                    avglen = (linelocs[next_valid] - linelocs[prev_valid]) / (
                        next_valid - prev_valid
                    )
                    linelocs_filled[l] = linelocs[prev_valid] + (
                        avglen * (l - prev_valid)
                    )
                else:
                    avglen = self.inlinelen
                    linelocs_filled[l] = linelocs[prev_valid] + (
                        avglen * (l - prev_valid)
                    )

        # *finally* done :)

        rv_ll = np.asarray([linelocs_filled[l] for l in range(0, proclines)])

        # ldd.logger.info("line0loc %s %s", int(line0loc), int(self.meanlinelen))

        if self.rf.debug_plot and self.rf.debug_plot.is_plot_requested("line_locs"):
            plot_data_and_pulses(
                self.data["video"]["demod"],
                raw_pulses=self.rawpulses,
                linelocs=linelocs,
                pulses=validpulses,
            )

        if self.vblank_next is None:
            nextfield = linelocs_filled[self.outlinecount - 7]
        else:
            nextfield = self.vblank_next - (self.inlinelen * 8)

        return rv_ll, rv_err, nextfield

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

    def computewow(self, lineinfo):
        """Compute how much the line deviates fron expected
        Overridden to limit the result so we don't get super-bright lines at head switch.
        TODO: Better solution to that
        """
        wow = np.ones(len(lineinfo))

        for l in range(0, len(wow) - 1):
            computed = self.get_linelen(l) / self.inlinelen
            wow[l] = computed if inrange(computed, 0.9, 1.02) else 1.0

        for l in range(self.lineoffset, self.lineoffset + 10):
            wow[l] = np.median(wow[l : l + 4])

        return wow

    def fix_badlines(self, linelocs_in, linelocs_backup_in=None):
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
        test = try_detect_track_betamax_pal(self)
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
