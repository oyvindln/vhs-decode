import math
import os
import sqlite3
import numpy as np
import scipy.signal as sps

from collections import namedtuple
from unittest import mock
import itertools

import lddecode.core as ldd
import lddecode.utils as lddu
from lddecode.utils import inrange

import vhsdecode.formats as vhs_formats
import vhsdecode.sync as sync
from vhsdecode.field import FieldShared
from vhsdecode.process import VHSDecode, VHSRFDecode
from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.formats import parent_system

from lddecode.core import npfft


class FieldCVBSShared:
    refine_linelocs_hsync = FieldShared.refine_linelocs_hsync
    compute_deriv_error = FieldShared.compute_deriv_error

    def getpulses(self):
        """Find sync pulses in the demodulated video signal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def hz_to_output(self, input):
        if (
            self.rf.DecoderParams["clamp_agc"] is True
            and self.outlinecount * self.outlinelen == input.size
        ):
            return hz_to_output_override(self, input)
        else:
            return super(FieldCVBSShared, self).hz_to_output(input)

    def compute_linelocs(self):
        # Override to avoid mooving backwards if hitting lastline < proclines condition.
        # TODO: make shared function etc for vhs and cvbs and push some improvements
        # upstream.
        self.rawpulses = self.getpulses()
        if self.rawpulses is None or len(self.rawpulses) == 0:
            if self.fields_written:
                ldd.logger.error("Unable to find any sync pulses, skipping one field")
                return None, None, None
            else:
                ldd.logger.error("Unable to find any sync pulses, skipping one second")
                return None, None, int(self.rf.freq_hz)

        self.validpulses = validpulses = self.refinepulses()
        meanlinelen = self.computeLineLen(validpulses)
        line0loc, lastlineloc, self.isFirstField = self.getLine0(
            validpulses, meanlinelen
        )
        self.linecount = 263 if self.isFirstField else 262

        # Number of lines to actually process.  This is set so that the entire following
        # VSYNC is processed
        proclines = self.outlinecount + self.lineoffset + 10
        if self.rf.system == "PAL":
            proclines += 3

        # It's possible for getLine0 to return None for lastlineloc
        if lastlineloc is not None:
            numlines = (lastlineloc - line0loc) / self.inlinelen
            self.skipdetected = numlines < (self.linecount - 5)
        else:
            self.skipdetected = False
            lastlineloc = 0

        if line0loc is None:
            if self.initphase is False:
                ldd.logger.error("Unable to determine start of field - dropping field")
            return None, None, self.inlinelen * 200

        # If we don't have enough data at the end, move onto the next field
        lastline = (self.rawpulses[-1].start - line0loc) / meanlinelen
        if lastline < proclines:
            ldd.logger.error(
                "Missing data at the end of field, possibly dropped samples skipping a little."
            )
            # Make sore to not move backwards here
            return None, None, max(line0loc - (meanlinelen * 20), self.inlinelen)

        linelocs, lineloc_errs, last_validpulse = sync.valid_pulses_to_linelocs(
            validpulses,
            line0loc,
            0,
            meanlinelen,
            self.rf.hsync_tolerance,
            proclines,
            1.9,
        )

        self.linelocs0 = linelocs.copy()

        if self.vblank_next is None:
            nextfield = linelocs[self.outlinecount - 7]
        else:
            nextfield = self.vblank_next - (self.inlinelen * 8)

        return linelocs, lineloc_errs, nextfield


def generate_f05_filter(filters, freq_half, blocklen):
    F0_5 = sps.firwin(65, [0.5 / freq_half], pass_zero=True)
    F0_5_fft = lddu.filtfft((F0_5, [1.0]), blocklen)
    filters["F05_offset"] = 32
    filters["F05"] = F0_5_fft


def find_sync_levels(field):
    """Very crude sync level detection"""
    # Skip a few samples to avoid any possible edge distortion.
    data = field.data["video"]["demod_05"][10:]

    # Start with finding the minimum value of the input.
    sync_min = np.amin(data)
    max_val = np.amax(data)

    # Use the max for a temporary reference point which may be max ire or not.
    difference = max_val - sync_min

    # Find approximate sync areas.
    on_sync = data < (sync_min + (difference / 15))

    found_porch = False

    offset = 0
    blank_level = None

    while not found_porch:
        # Look for when we leave the approximate sync area next...
        search_start = np.argwhere(on_sync[offset:])[0][0]
        next_cross_raw = np.argwhere(1 - on_sync[search_start:])[0][0] + search_start
        # and a bit past that we ought to be in the back porch for blanking level.
        next_cross = next_cross_raw + int(field.usectoinpx(1.5))
        blank_level = data[next_cross] + offset
        if blank_level > sync_min + (difference / 15):
            found_porch = True
        else:
            # We may be in vsync, try to skip ahead a bit
            # TODO: This may not work yet.
            offset += int(field.usectoinpx(50))
            if offset > len(data) - 10:
                # Give up
                return None, None

    return sync_min, blank_level


def getpulses_override(field):
    """Find sync pulses in the demodulated video signal

    NOTE: TEMPORARY override until an override for the value itself is added upstream.
    """

    if field.rf.auto_sync:
        if "agc_blank_level" in field.rf.DecoderParams:
            sync_level = field.rf.DecoderParams["agc_sync_level"]
            blank_level = field.rf.DecoderParams["agc_blank_level"]
        else:
            sync_level, blank_level = find_sync_levels(field)

        if sync_level is not None and blank_level is not None:
            field.rf.DecoderParams["ire0"] = blank_level
            field.rf.DecoderParams["hz_ire"] = (blank_level - sync_level) / (
                -field.rf.SysParams["vsync_ire"]
            )

    # pass one using standard levels

    # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
    pulse_hz_min = field.rf.iretohz(field.rf.SysParams["vsync_ire"] - 15)
    pulse_hz_max = field.rf.iretohz(field.rf.SysParams["vsync_ire"] / 2)

    pulses = lddu.findpulses(
        field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
    )

    if len(pulses) == 0:
        # can't do anything about this
        return pulses

    # determine sync pulses from vsync
    vsync_locs = []
    vsync_means = []

    for i, p in enumerate(pulses):
        if p.len > field.usectoinpx(10):
            vsync_locs.append(i)
            vsync_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + field.rf.freq) : int(
                            p.start + p.len - field.rf.freq
                        )
                    ]
                )
            )

    if len(vsync_means) == 0:
        return None

    synclevel = np.median(vsync_means)

    if np.abs(field.rf.hztoire(synclevel) - field.rf.SysParams["vsync_ire"]) < 5:
        # sync level is close enough to use
        return pulses

    if vsync_locs is None or not len(vsync_locs):
        return None

    # Now compute black level and try again

    # take the eq pulses before and after vsync
    r1 = range(vsync_locs[0] - 5, vsync_locs[0])
    r2 = range(vsync_locs[-1] + 1, vsync_locs[-1] + 6)

    black_means = []

    for i in itertools.chain(r1, r2):
        if i < 0 or i >= len(pulses):
            continue

        p = pulses[i]
        if inrange(p.len, field.rf.freq * 0.75, field.rf.freq * 3):
            black_means.append(
                np.mean(
                    field.data["video"]["demod_05"][
                        int(p.start + (field.rf.freq * 5)) : int(
                            p.start + (field.rf.freq * 20)
                        )
                    ]
                )
            )

    blacklevel = np.median(black_means)

    pulse_hz_min = synclevel - (field.rf.SysParams["hz_ire"] * 10)
    pulse_hz_max = (blacklevel + synclevel) / 2

    return lddu.findpulses(field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max)


def hz_to_output_override(field, input):
    blank_levels = np.empty(field.outlinecount)
    sync_levels = np.empty(field.outlinecount)
    for i in range(0, field.outlinecount):
        blank_levels[i] = np.median(
            input[i * field.outlinelen + 96 : i * field.outlinelen + 164]
        )
        sync_levels[i] = np.median(
            input[i * field.outlinelen + 12 : i * field.outlinelen + 72]
        )

    field.rf.DecoderParams["agc_blank_level"] = np.median(
        blank_levels[(field.outlinecount // 3) * 2 :]
    )
    field.rf.DecoderParams["agc_sync_level"] = np.median(
        sync_levels[(field.outlinecount // 3) * 2 :]
    )

    reduced = input

    reduced[0 : 6 * field.outlinelen + 130] = input[
        0 : 6 * field.outlinelen + 130
    ] - np.median(blank_levels[7:12])

    for i in range(7, field.outlinecount - 5):
        reduced[
            i * field.outlinelen + 130 : (i + 1) * field.outlinelen + 130
        ] -= np.linspace(
            np.median(blank_levels[i - 2 : i + 3]),
            np.median(blank_levels[i - 1 : i + 4]),
            num=field.outlinelen,
        )

    reduced[(field.outlinecount - 5) * field.outlinelen + 130 :] = input[
        (field.outlinecount - 5) * field.outlinelen + 130 :
    ] - np.median(blank_levels[field.outlinecount - 9 : field.outlinecount - 5])

    if field.rf.DecoderParams["agc_set_gain"] == 0.0:
        vsyncs = blank_levels - sync_levels
        vsyncs = vsyncs[7 : field.outlinecount - 5]
        vsyncs.sort()
        new_gain = np.mean(vsyncs[(vsyncs.size // 4) : ((vsyncs.size * 3) // 4)]) / (
            -field.rf.SysParams["vsync_ire"]
        )
        if field.rf.DecoderParams["agc_gain"] is None:
            field.rf.DecoderParams["agc_gain"] = new_gain
            field.rf.DecoderParams["lowest_agc_gain"] = new_gain
            field.rf.DecoderParams["highest_agc_gain"] = new_gain
            field.rf.DecoderParams["lowest_used_agc_gain"] = new_gain
            field.rf.DecoderParams["highest_used_agc_gain"] = new_gain
        else:
            field.rf.DecoderParams["agc_gain"] = new_gain * field.rf.DecoderParams[
                "agc_speed"
            ] + field.rf.DecoderParams["agc_gain"] * (
                1.0 - field.rf.DecoderParams["agc_speed"]
            )
            field.rf.DecoderParams["lowest_agc_gain"] = min(
                field.rf.DecoderParams["lowest_agc_gain"], new_gain
            )
            field.rf.DecoderParams["highest_agc_gain"] = max(
                field.rf.DecoderParams["highest_agc_gain"], new_gain
            )
            field.rf.DecoderParams["lowest_used_agc_gain"] = min(
                field.rf.DecoderParams["lowest_used_agc_gain"],
                field.rf.DecoderParams["agc_gain"],
            )
            field.rf.DecoderParams["highest_used_agc_gain"] = max(
                field.rf.DecoderParams["highest_used_agc_gain"],
                field.rf.DecoderParams["agc_gain"],
            )
    else:
        field.rf.DecoderParams["agc_gain"] = field.rf.DecoderParams["agc_set_gain"]

    reduced /= (
        field.rf.DecoderParams["agc_gain"] * field.rf.DecoderParams["agc_gain_factor"]
    )
    reduced -= field.rf.SysParams["vsync_ire"]

    return np.uint16(
        np.clip(
            (reduced * field.out_scale) + field.rf.SysParams["outputZero"], 0, 65535
        )
        + 0.5
    )


class FieldPALCVBS(FieldCVBSShared, ldd.FieldPAL):
    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as most sources won't have a pilot burst."""
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs


class FieldNTSCCVBS(FieldCVBSShared, ldd.FieldNTSC):
    pass


class FieldMPALCVBS(FieldNTSCCVBS):
    def refine_linelocs_burst(self, linelocs=None):
        """Not used for PALM."""
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        self.fieldPhaseID = 0

        return linelocs


# Superclass to override laserdisc-specific parts of ld-decode with stuff that works for VHS
#
# We do this simply by using inheritance and overriding functions. This results in some redundant
# work that is later overridden, but avoids altering any ld-decode code to ease merging back in
# later as the ld-decode is in flux at the moment.
class CVBSDecode(ldd.LDdecode):
    def __init__(
        self,
        fname_in,
        fname_out,
        freader,
        logger,
        system="NTSC",
        threads=1,
        inputfreq=40,
        level_adjust=0.2,
        rf_options={},
        extra_options={},
    ):
        rf = CVBSDecodeInner(
            system=system,
            tape_format="UMATIC",
            inputfreq=inputfreq,
            rf_options=rf_options,
        )

        # The superclass constructs its own laserdisc RFDecode (and the demod cache around it),
        # so hand it the composite decoder instead. No output filename so it opens no files.
        with mock.patch.object(ldd, "RFDecode", lambda **_: rf):
            super(CVBSDecode, self).__init__(
                fname_in,
                None,
                freader,
                logger,
                analog_audio=False,
                system=parent_system(system),
                doDOD=False,
                threads=threads,
                extra_options=extra_options,
            )
        # Adjustment for output to avoid clipping.
        self.level_adjust = level_adjust

        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        if system == "PAL":
            self.FieldClass = FieldPALCVBS
        elif system == "NTSC":
            self.FieldClass = FieldNTSCCVBS
        elif system == "PALM":
            self.FieldClass = FieldMPALCVBS
        else:
            raise Exception("Unknown video system!", system)

        self.dbconn = None
        if extra_options.get("write_db"):
            if os.path.exists(fname_out + ".tbc.db"):
                os.unlink(fname_out + ".tbc.db")
            self.dbconn = sqlite3.connect(fname_out + ".tbc.db")
            self.create_db_schema()

        self.fname_out = fname_out

        if fname_out:
            self.outfile_video = open(fname_out + ".tbc", "wb")

    def buildmetadata(self, f):
        # Avoid crash if this is NaN
        if math.isnan(f.burstmedian):
            f.burstmedian = 0.0
        return super(CVBSDecode, self).buildmetadata(f)

    # For laserdisc this decodes frame numbers from VBI metadata, but there won't be such a thing on
    # other sources, so just skip it for now.
    def decodeFrameNumber(self, f1, f2):
        return None

    checkMTF = VHSDecode.checkMTF

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None

    def build_json(self):
        jout = super(CVBSDecode, self).build_json()
        if jout is None:
            return None

        if self.rf.color_system == "MPAL":
            jout["videoParameters"]["system"] = "PAL-M"

        return jout

    def writeout(self, dataset: tuple):
        if self.dbconn:
            super(CVBSDecode, self).writeout(dataset)
        else:
            f, fi, picture, audio, efm = dataset
            self.fieldinfo.append(fi)
            self.outfile_video.write(picture)
            self.fields_written += 1


class CVBSDecodeInner(ldd.RFDecode):
    def __init__(self, inputfreq=40, system="NTSC", tape_format="VHS", rf_options={}):
        # Make sure delays are populated with something
        # TODO: Fix this properly.
        self.computedelays()

        # First init the rf decoder normally.
        super(CVBSDecodeInner, self).__init__(
            inputfreq,
            parent_system(system),
            decode_analog_audio=False,
            has_analog_audio=False,
        )

        self._color_system = system

        self._chroma_trap = rf_options.get("chroma_trap", False)
        self.notch = rf_options.get("notch", None)
        self.notch_q = rf_options.get("notch_q", 10.0)
        self.auto_sync = rf_options.get("auto_sync", True)

        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        # Then we override the laserdisc parameters.
        self.SysParams, self.DecoderParams = vhs_formats.get_cvbs_params(system)

        # Make (intentionally) mutable copies of HZ<->IRE levels
        # (NOTE: used by upstream functions, we use a namedtuple to keep const values already)
        self.DecoderParams["ire0"] = self.SysParams["ire0"]
        self.DecoderParams["hz_ire"] = self.SysParams["hz_ire"]
        self.DecoderParams["vsync_ire"] = self.SysParams["vsync_ire"]

        # TEMP just set this high so it doesn't mess with anything.
        self.DecoderParams["video_lpf_freq"] = 6400000
        self.DecoderParams["video_deemp_strength"] = 1

        # Fill DecodarParams with additional options
        self.DecoderParams["clamp_agc"] = rf_options.get("clamp_agc", False)
        self.DecoderParams["agc_speed"] = rf_options.get("agc_speed", 0.1)
        self.DecoderParams["agc_gain_factor"] = rf_options.get("agc_gain_factor", 1.0)
        self.DecoderParams["agc_set_gain"] = rf_options.get("agc_set_gain", 0.0)
        self.DecoderParams["agc_gain"] = None

        # Lastly we re-create the filters with the new parameters.
        self.computevideofilters()

        generate_f05_filter(self.Filters, self.freq_half, self.blocklen)

        if self.notch is not None:
            self.Filters["FVideoNotch"] = sps.iirnotch(
                self.notch / self.freq_half, self.notch_q
            )

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024

        if self._chroma_trap:
            self._chroma_sep_class = ChromaSepClass(
                self.freq_hz, self.SysParams["fsc_mhz"]
            )
        self._options = namedtuple(
            "Options",
            [
                "disable_right_hsync",
                "skip_hsync_refine",
            ],
        )(
            not rf_options.get("rhs_hsync", False),
            rf_options.get("skip_hsync_refine", False),
        )

    @property
    def options(self):
        return self._options

    @property
    def color_system(self):
        return self._color_system

    computedelays = VHSRFDecode.computedelays

    def demodblock(self, data=None, mtf_level=0, fftdata=None, cut=False):
        datalen = len(fftdata)
        # We don't need the complex side here, should see if we could avoid even calculating it later.
        data = npfft.irfft(fftdata[: datalen + 1], datalen).real

        rv = {}

        # applies the Subcarrier trap
        # (this will remove most chroma info)
        if self._chroma_trap:
            luma = self._chroma_sep_class.work(data)
        else:
            luma = data

        if not self.auto_sync:
            luma += 0xFFFF / 2
            luma /= 4 * 0xFFFF
            luma *= self.iretohz(100)
            luma += self.iretohz(self.SysParams["vsync_ire"])

        if self.notch is not None:
            luma = sps.filtfilt(
                self.Filters["FVideoNotch"][0],
                self.Filters["FVideoNotch"][1],
                luma,
            )

        luma_fft = npfft.rfft(luma)

        luma05_fft = (
            luma_fft * self.Filters["F05"][: (len(self.Filters["F05"]) // 2) + 1]
        )
        luma05 = npfft.irfft(luma05_fft)
        luma05 = np.roll(luma05, -self.Filters["F05_offset"])
        videoburst = npfft.irfft(
            luma_fft * self.Filters["Fburst"][: (len(self.Filters["Fburst"]) // 2) + 1]
        ).astype(np.float32)

        video_out = np.rec.array(
            [luma, luma05, videoburst],
            names=["demod", "demod_05", "demod_burst"],
        )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        return rv
