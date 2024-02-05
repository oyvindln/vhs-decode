import math
import traceback
import numpy as np
import scipy.signal as sps

from collections import namedtuple
import itertools

import lddecode.core as ldd
import lddecode.utils as lddu
from lddecode.utils import inrange

import vhsdecode.formats as vhs_formats
import vhsdecode.sync as sync
from vhsdecode.addons.chromasep import ChromaSepClass
from vhsdecode.process import parent_system

# from vhsdecode.process import getpulses_override as vhs_getpulses_override
# from vhsdecode.addons.vsyncserration import VsyncSerration

from lddecode.core import npfft


def chroma_to_u16(chroma):
    """Scale the chroma output array to a 16-bit value for output."""
    S16_ABS_MAX = 32767

    if np.max(chroma) > S16_ABS_MAX or abs(np.min(chroma)) > S16_ABS_MAX:
        ldd.logger.warning("Chroma signal clipping.")
    return np.uint16(chroma + S16_ABS_MAX)


def generate_f05_filter(filters, freq_half, blocklen):
    F0_5 = sps.firwin(65, [0.5 / freq_half], pass_zero=True)
    F0_5_fft = lddu.filtfft((F0_5, [1.0]), blocklen)
    filters["F05_offset"] = 32
    filters["F05"] = F0_5_fft
    # filters["FVideo05"] = filters["Fvideo_lpf"] * filters["F05"]


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

    if False:
        import matplotlib.pyplot as plt

        data = field.data["video"]["demod_05"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
        #        ax1.plot(hilbert, color='#FF0000')
        # ax1.plot(data, color="#00FF00")
        ax1.axhline(sync_min, color="#0000FF")
        #        ax1.axhline(blank_level, color="#000000")
        ax1.axvline(search_start, color="#FF0000")
        ax1.axvline(next_cross_raw, color="#00FF00")
        ax1.axvline(next_cross, color="#0000FF")
        ax1.axhline(blank_level, color="#000000")
        #            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
        #            ax1.axhline(self.iretohz(7.5))
        #            ax1.axhline(self.iretohz(100))
        # print("Vsync IRE", self.SysParams["vsync_ire"])
        #            ax2 = ax1.twinx()
        #            ax3 = ax1.twinx()
        ax1.plot(data)
        ax2.plot(on_sync)
        #            ax2.plot(luma05[:2048])
        #            ax4.plot(env, color="#00FF00")
        #            ax3.plot(np.angle(hilbert))
        #            ax4.plot(hilbert.imag)
        #            crossings = find_crossings(env, 700)
        #            ax3.plot(crossings, color="#0000FF")
        plt.show()
        #            exit(0)

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

        if False:
            import matplotlib.pyplot as plt

            data = field.data["video"]["demod_05"]

            fig, ax1 = plt.subplots(1, 1, sharex=True)
            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.axhline(field.rf.iretohz(0), color="#000000")
            #        ax1.axhline(blank_level, color="#000000")
            ax1.axhline(field.rf.iretohz(field.rf.SysParams["vsync_ire"]))
            #            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
            #            ax1.axhline(self.iretohz(7.5))
            #            ax1.axhline(self.iretohz(100))
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            ax1.plot(data)
            #            ax2.plot(luma05[:2048])
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
            #            exit(0)

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


class FieldPALCVBS(ldd.FieldPAL):
    def __init__(self, *args, **kwargs):
        super(FieldPALCVBS, self).__init__(*args, **kwargs)

    def refine_linelocs_pilot(self, linelocs=None):
        """Override this as most sources won't have a pilot burst."""
        if linelocs is None:
            linelocs = self.linelocs2.copy()
        else:
            linelocs = linelocs.copy()

        return linelocs

    def refine_linelocs_hsync(self):
        if not self.rf.options.skip_hsync_refine:
            return sync.refine_linelocs_hsync(
                self, self.linebad, 0
            )  # TODO fix last param once it's actually used.
        else:
            return self.linelocs1.copy()

    def _determine_field_number(self):
        """Using LD code as it should work on stable sources, but may not work on stuff like vhs."""
        return 1 + (self.rf.field_number % 8)

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
            return super(FieldPALCVBS, self).hz_to_output(input)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as tapes have large variations in line pos
        Due to e.g head switch.
        compute errors based off the second derivative - if it exceeds 1 something's wrong,
        and if 4 really wrong...
        """
        return baserr

    def dropout_detect(self):
        return None


class FieldNTSCCVBS(ldd.FieldNTSC):
    def __init__(self, *args, **kwargs):
        super(FieldNTSCCVBS, self).__init__(*args, **kwargs)

    def refine_linelocs_hsync(self):
        if not self.rf.options.skip_hsync_refine:
            # TODO: test and use modified variant.
            return super(FieldNTSCCVBS, self).refine_linelocs_hsync()
        else:
            return self.linelocs1.copy()

    def _refine_linelocs_burst(self, linelocs=None):
        """Standard impl works for stable sources, we may need to override this for
        unstable ones though.
        """
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        return linelocs

    def dropout_detect(self):
        return None

    def getpulses(self):
        """Find sync pulses in the demodulated video signal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        return getpulses_override(self)

    def compute_deriv_error(self, linelocs, baserr):
        """Disabled this for now as line starts can vary widely."""
        return baserr


class FieldMPALCVBS(FieldNTSCCVBS):
    def __init__(self, *args, **kwargs):
        super(FieldMPALCVBS, self).__init__(*args, **kwargs)

    def refine_linelocs_burst(self, linelocs=None):
        """Not used for PALM."""
        if linelocs is None:
            linelocs = self.linelocs2
        else:
            linelocs = linelocs.copy()

        self.fieldPhaseID = 0

        return linelocs


def _demodcache_dummy(self, *args, **kwargs):
    self.ended = True
    pass


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
        # monkey patch init with a dummy to prevent calling set_start_method twice on macos
        # and not create extra threads.
        # This is kinda hacky and should be sorted in a better way ideally.
        temp_init = ldd.DemodCache.__init__
        ldd.DemodCache.__init__ = _demodcache_dummy

        super(CVBSDecode, self).__init__(
            fname_in,
            fname_out,
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
        # Overwrite the rf decoder with the VHS-altered one
        self.rf = CVBSDecodeInner(
            system=system,
            tape_format="UMATIC",
            inputfreq=inputfreq,
            rf_options=rf_options,
        )

        # Store reference to ourself in the rf decoder - needed to access data location for track
        # phase, may want to do this in a better way later.
        self.rf.decoder = self
        if system == "PAL":
            self.FieldClass = FieldPALCVBS
        elif system == "NTSC":
            self.FieldClass = FieldNTSCCVBS
        elif system == "MPAL":
            self.FieldClass = FieldMPALCVBS
        else:
            raise Exception("Unknown video system!", system)

        # Restore init functino now that superclass constructor is finished.
        ldd.DemodCache.__init__ = temp_init

        self.demodcache = ldd.DemodCache(
            self.rf, self.infile, self.freader, None, num_worker_threads=self.numthreads
        )

    # Override to avoid NaN in JSON.
    def calcsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        signal = np.mean(data)
        noise = np.std(data)

        # Make sure signal is positive so we don't try to do log on a negative value.
        if signal < 0.0:
            ldd.logger.info(
                "WARNING: Negative mean for SNR, changing to absolute value."
            )
            signal = abs(signal)
        if noise == 0:
            return 0
        return 20 * np.log10(signal / noise)

    def calcpsnr(self, f, snrslice):
        data = f.output_to_ire(f.dspicture[snrslice])

        #        signal = np.mean(data)
        noise = np.std(data)
        if noise == 0:
            return 0
        return 20 * np.log10(100 / noise)

    def buildmetadata(self, f):
        # Avoid crash if this is NaN
        if math.isnan(f.burstmedian):
            f.burstmedian = 0.0
        return super(CVBSDecode, self).buildmetadata(f)

    # For laserdisc this decodes frame numbers from VBI metadata, but there won't be such a thing on
    # other sources, so just skip it for now.
    def decodeFrameNumber(self, f1, f2):
        return None

    # Again ignored for non-ld sources.
    def checkMTF(self, field, pfield=None):
        return True

    def computeMetricsNTSC(self, metrics, f, fp=None):
        return None

    def build_json(self):
        # for f in self.fieldstack:
        #    if f:
        #        break
        ## TODO: Make some shared function/class for stuff that is the same in cvbs and vhs-decode
        try:
            # if not f:
            #    # Make sure we don't fail if the last attempted field failed to decode
            #    # Might be better to fix this elsewhere.
            #    f = self.prevfield
            jout = super(CVBSDecode, self).build_json()

            if self.rf.color_system == "MPAL":
                # jout["videoParameters"]["isSourcePal"] = True
                # jout["videoParameters"]["isSourcePalM"] = True
                jout["videoParameters"]["system"] = "PAL-M"

            return jout
        except TypeError as e:
            traceback.print_exc()
            print("Cannot build json: %s" % e)
            return None


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
        self.auto_sync = rf_options.get("auto_sync", False)

        self.hsync_tolerance = 0.8

        self.field_number = 0
        self.last_raw_loc = None

        # Then we override the laserdisc parameters.
        self.SysParams, self.DecoderParams = vhs_formats.get_format_params(
            system, "UMATIC", ldd.logger
        )

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

        self.Filters["FVideo"] = self.Filters["Fvideo_lpf"]
        generate_f05_filter(self.Filters, self.freq_half, self.blocklen)

        # Filter to pick out color-under chroma component.
        # filter at about twice the carrier. (This seems to be similar to what VCRs do)
        # TODO: Needs tweaking
        # Note: order will be doubled since we use filtfilt.
        # chroma_lowpass = sps.butter(
        #     2,
        #     [50000 / self.freq_hz_half, DP["chroma_bpf_upper"] / self.freq_hz_half],
        #     btype="bandpass",
        #     output="sos",
        # )
        # self.Filters["FVideoBurst"] = chroma_lowpass

        if self.notch is not None:
            self.Filters["FVideoNotch"] = sps.iirnotch(
                self.notch / self.freq_half, self.notch_q
            )
            # self.Filters["FVideoNotchF"] = lddu.filtfft(
            #     self.Filters["FVideoNotch"], self.blocklen
            # )

        # The following filters are for post-TBC:
        # The output sample rate is at approx 4fsc
        fsc_mhz = self.SysParams["fsc_mhz"]
        out_sample_rate_mhz = fsc_mhz * 4
        out_frequency_half = out_sample_rate_mhz / 2

        # Final band-pass filter for chroma output.
        # Mostly to filter out the higher-frequency wave that results from signal mixing.
        # Needs tweaking.
        # Note: order will be doubled since we use filtfilt.
        chroma_bandpass_final = sps.butter(
            1,
            [
                (fsc_mhz - 0.1) / out_frequency_half,
                (fsc_mhz + 0.1) / out_frequency_half,
            ],
            btype="bandpass",
            output="sos",
        )
        self.Filters["FChromaBpf"] = chroma_bandpass_final

        # Increase the cutoff at the end of blocks to avoid edge distortion from filters
        # making it through.
        self.blockcut_end = 1024
        self.demods = 0

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

    def computedelays(self, mtf_level=0):
        """Override computedelays
        It's normally used for dropout compensation, but the dropout compensation implementation
        in ld-decode assumes composite color. This function is called even if it's disabled, and
        seems to break with the VHS setup, so we disable it by overriding it for now.
        """
        # Set these to 0 for now, the metrics calculations look for them.
        self.delays = {}
        self.delays["video_sync"] = 0
        self.delays["video_white"] = 0

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

        if False:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
            #        ax1.plot(hilbert, color='#FF0000')
            # ax1.plot(data, color="#00FF00")
            ax1.axhline(self.iretohz(0))
            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
            ax1.axhline(self.iretohz(7.5))
            ax1.axhline(self.iretohz(100))
            # print("Vsync IRE", self.SysParams["vsync_ire"])
            #            ax2 = ax1.twinx()
            #            ax3 = ax1.twinx()
            ax1.plot(luma)
            ax2.plot(luma05)
            #            ax4.plot(env, color="#00FF00")
            #            ax3.plot(np.angle(hilbert))
            #            ax4.plot(hilbert.imag)
            #            crossings = find_crossings(env, 700)
            #            ax3.plot(crossings, color="#0000FF")
            plt.show()
        #            exit(0)

        video_out = np.rec.array(
            [luma, luma05, videoburst],
            names=["demod", "demod_05", "demod_burst"],
        )

        rv["video"] = (
            video_out[self.blockcut : -self.blockcut_end] if cut else video_out
        )

        return rv
